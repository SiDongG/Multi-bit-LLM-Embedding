import random
from typing import List, Tuple, Sequence, Optional, Iterable
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, PreTrainedModel, LogitsProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from utils import select_green_list, apply_random_edits
import numpy as np
from watermark import hash_context_with_key, construct_segments, allocate_bits_proportional_to_entropy
import math
from itertools import product   
from ldpc_encoder import LDPCCode, generate_ldpc_H

# ---------------------------
# Logits processors
# ---------------------------

class NoWatermarkLogitsProcessor(LogitsProcessor):
    """
    A logits processor that does absolutely nothing.
    This reproduces the behavior of the original LLM with no watermark.
    """
    def __call__(self, input_ids, scores):
        return scores  # unchanged
    

class MyEntropyHashWatermarkLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        secret_key,
        segments_per_bin,
        segment_bits,
        k,
        model,
        bias=6.0,
        entropy_bins=None,
    ):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.secret_key = str(secret_key)
        self.segments_per_bin = np.array(segments_per_bin, dtype=int)
        self.segment_bits = segment_bits
        self.k = k
        self.bias = bias
        self.model = model

        if entropy_bins is None:
            self.entropy_bins = np.array([1,2,3,4,5,6,7,8,9,10,11, np.inf])
        else:
            self.entropy_bins = entropy_bins

        assert len(self.segments_per_bin) == len(self.entropy_bins)-1

        # --- KL tracking ---
        self.total_kl = 0.0
        self.num_steps = 0

    def _compute_entropy_bits(self, logits):
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        H = -(probs * log_probs).sum().item()
        return H / np.log(2)

    def __call__(self, input_ids, scores):

        # ------------ shape handling -------------
        if scores.dim() == 2:
            logits = scores[0]
        else:
            logits = scores

        # ------------ 1. entropy -----------------
        # --- 1. entropy logits (using only last-k)
        entropy_ctx = input_ids[0][-self.k:].unsqueeze(0)

        with torch.no_grad():
            entropy_logits = self.model(entropy_ctx).logits[0, -1, :]

        H_bits = _compute_entropy_bits_from_logits(entropy_logits)


        # ------------ 2. bin ----------------------
        bin_idx = int(np.digitize(H_bits, self.entropy_bins) - 1)
        bin_idx = max(0, min(bin_idx, len(self.segments_per_bin)-1))

        if self.segments_per_bin[bin_idx] <= 0:
            return scores

        # ------------ 3. context ------------------
        ctx = input_ids[0].tolist()
        context_tokens = ctx[-self.k:] if len(ctx) >= self.k else ctx

        # ------------ 4. segment index ------------
        num_seg = self.segments_per_bin[bin_idx]
        seg_idx = hash_context_with_key(context_tokens, self.secret_key, num_seg)

        # ------------ 5. bit sequence -------------
        bit_seq = self.segment_bits.get((bin_idx, seg_idx), "")
        if len(bit_seq) == 0:
            return scores

        # ------------ 6. green list ---------------
        green_ids = select_green_list(
            secret_key=self.secret_key,
            bit_seq=bit_seq,
            prev_tokens=context_tokens,
            vocab_size=self.vocab_size,
        )
        green_ids = torch.tensor(green_ids, dtype=torch.long, device=logits.device)

        # ------------ 7. KL tracking BEFORE bias ---
        p = F.softmax(logits, dim=-1)
        # copy so later modification doesn't affect p
        p_detached = p.detach()

        # ------------ 8. apply bias ---------------
        biased = logits.clone()
        biased[green_ids] += self.bias

        # ------------ 9. KL(p || q) AFTER bias ----
        q = F.softmax(biased, dim=-1)
        kl = (p_detached * (p_detached.log() - q.log())).sum().item()

        # record
        self.total_kl += kl
        self.num_steps += 1

        # ------------ 10. return correct shape ----
        if scores.dim() == 2:
            new_scores = scores.clone()
            new_scores[0] = biased
            return new_scores
        else:
            return biased



def _compute_entropy_bits_from_logits(logits: torch.Tensor) -> float:
    """
    Compute H(p) in bits from logits (for a single step).
    logits: [vocab]
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)  # natural log
    entropy_nats = -(probs * log_probs).sum().item()
    entropy_bits = entropy_nats / np.log(2.0)
    return entropy_bits


def decode_watermark_bits(
    model,
    tokenizer,
    generated_ids: torch.LongTensor,
    secret_key,
    segments_per_bin,
    segment_bits,
    k: int,
    entropy_bins: np.ndarray = np.array([1,2,3,4,5,6,7,8,9,10,11, np.inf]),
):
    """
    Multi-hypothesis segment-level decoder.

    For each segment (b,s) with bit-length L:
      - consider ALL 2^L bit patterns as candidates
      - for each token assigned to (b,s), and each candidate pattern c:
          * build green_list(secret_key, c, context_tokens)
          * if token is in that green_list, increment count[b,s,c]
      - at the end, choose argmax_c count[b,s,c] as decoded bits for segment (b,s)

    Returns:
        recovered_stream: global decoded bitstream (str)
        segment_stats: dict[(bin_idx, seg_idx)] -> {
             'L': int,
             'trials': int,
             'counts': dict[bit_pattern_str] -> int,
             'best_bits': str
        }
        debug_info: optional per-position info
    """

    if generated_ids.dim() == 2:
        generated_ids = generated_ids[0]  # [seq_len]

    device = next(model.parameters()).device
    vocab_size = tokenizer.vocab_size
    segments_per_bin = np.array(segments_per_bin, dtype=int)
    seq_len = generated_ids.size(0)

    # ---------------------------
    # 0. Precompute candidate sets per segment
    # ---------------------------
    segment_stats = {}
    # segment_bits gives us the DESIGN length for each segment (b,s)
    for (bin_idx, seg_idx), bit_seq in segment_bits.items():
        L = len(bit_seq)
        if L == 0:
            continue

        # All 2^L candidate bitstrings
        candidates = ["".join(bits) for bits in product("01", repeat=L)]
        counts = {c: 0 for c in candidates}

        segment_stats[(bin_idx, seg_idx)] = {
            "L": L,
            "trials": 0,
            "counts": counts,
            "best_bits": None,
        }

    debug_info = []

    # ---------------------------
    # 1. Scan through positions
    # ---------------------------
    for t in range(1, seq_len):
        # 1.1 Extract previous k tokens
        start_ctx = max(0, t - k)
        context_tokens = generated_ids[start_ctx:t].tolist()
        if len(context_tokens) == 0:
            continue

        # 1.2 Recompute entropy at this step
        context_tensor = torch.tensor(
            context_tokens, dtype=torch.long, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            logits_full = model(context_tensor).logits[0, -1, :]
        H_bits = _compute_entropy_bits_from_logits(logits_full)

        # 1.3 Assign entropy bin
        bin_idx = int(np.digitize(H_bits, entropy_bins) - 1)
        bin_idx = max(0, min(bin_idx, len(segments_per_bin) - 1))

        num_seg_bin = segments_per_bin[bin_idx]
        token_id = int(generated_ids[t].item())

        if num_seg_bin <= 0:
            debug_info.append({
                "pos": t,
                "bin_idx": bin_idx,
                "segment_idx": None,
                "token_id": token_id,
                "considered": False,
            })
            continue

        # 1.4 Segment index (same as encoder)
        seg_idx = hash_context_with_key(context_tokens, secret_key, num_seg_bin)
        seg_key = (bin_idx, seg_idx)

        # This segment might have L=0 or might not exist if no bits allocated
        if seg_key not in segment_stats:
            debug_info.append({
                "pos": t,
                "bin_idx": bin_idx,
                "segment_idx": seg_idx,
                "token_id": token_id,
                "considered": False,
            })
            continue

        seg_info = segment_stats[seg_key]
        seg_info["trials"] += 1

        # 1.5 For each candidate bit pattern, rebuild its green list
        #     and see if token falls in it, then increment that candidate's count
        for cand_bits in seg_info["counts"].keys():
            green_list = select_green_list(
                secret_key=secret_key,
                bit_seq=cand_bits,
                prev_tokens=context_tokens,
                vocab_size=vocab_size,
            )
            if token_id in green_list:
                seg_info["counts"][cand_bits] += 1

        debug_info.append({
            "pos": t,
            "bin_idx": bin_idx,
            "segment_idx": seg_idx,
            "token_id": token_id,
            "considered": True,
        })

    # ---------------------------
    # 2. Finalize per-segment decisions
    # ---------------------------
    recovered_stream_parts = []

    for bin_idx in range(len(segments_per_bin)):
        for seg_idx in range(segments_per_bin[bin_idx]):
            seg_key = (bin_idx, seg_idx)
            if seg_key not in segment_stats:
                continue
            seg_info = segment_stats[seg_key]
            counts = seg_info["counts"]

            # argmax candidate by hits
            best_bits, best_hits = max(counts.items(), key=lambda kv: kv[1])
            seg_info["best_bits"] = best_bits
            recovered_stream_parts.append(best_bits)

    recovered_stream = "".join(recovered_stream_parts)

    return recovered_stream, segment_stats, debug_info


if __name__ == "__main__":

    # ---------------------------------------------------------------
    # EXPERIMENT SETTINGS
    # ---------------------------------------------------------------
    NUM_RUNS = 20
    BITSTREAM_LEN = 60   # number of payload bits per run
    N = 200               # number of tokens to generate
    k = 3                    # context window
    secret_key = "my_super_secret_key"
    model_name = "gpt2"

    # Entropy bins used by encoder/decoder
    entropy_bins = np.array([0,1,2,3,4,5,6,7,8,9,10,11,np.inf])

    # Empirical entropy distribution P_X
    P_X = np.array([0.042, 0.02, 0.024, 0.04, 0.07,
                    0.121, 0.139, 0.144, 0.14, 0.12, 0.068, 0.072])

    # ---------------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    print("\nLoaded model:", model_name)

    # ---------------------------------------------------------------
    # RUN EXPERIMENTS
    # ---------------------------------------------------------------
    successes = 0
    all_results = []

    for run in range(NUM_RUNS):
        print(f"\n==================== RUN {run+1}/{NUM_RUNS} ====================")

        # -----------------------------
        # 1. Generate random payload
        # -----------------------------
        bitstream = "".join(random.choice("01") for _ in range(BITSTREAM_LEN))
        print("Bitstream:", bitstream)

        # -----------------------------
        # 2. Construct segments_per_bin
        # -----------------------------
        segments_per_bin = construct_segments(P_X, 10)

        # -----------------------------
        # 3. Allocate per-segment bit sequences
        # -----------------------------
        segment_bits = allocate_bits_proportional_to_entropy(
            segments_per_bin,
            bitstream
        )

        # -----------------------------
        # 4. Build watermark logits processor
        # -----------------------------
        wm_processor = MyEntropyHashWatermarkLogitsProcessor(
            tokenizer=tokenizer,
            secret_key=secret_key,
            segments_per_bin=segments_per_bin,
            segment_bits=segment_bits,
            k=k,
            model=model,
        )
        processors = LogitsProcessorList([wm_processor])

        # -----------------------------
        # 5. Generate watermarked text
        # -----------------------------
        prompt = "He walks into the room"
        encoding = tokenizer(prompt, return_tensors="pt")
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]


        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=N,
                logits_processor=processors,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
            )

        generated_ids = out[0]
        print("Generated text (truncated):",
              tokenizer.decode(generated_ids[:80], skip_special_tokens=True), " ...")
        
        # -----------------------------
        # PRINT AVERAGE KL DIVERGENCE
        # -----------------------------
        if wm_processor.num_steps > 0:
            avg_kl = wm_processor.total_kl / wm_processor.num_steps
            print(f"Average KL divergence per token: {avg_kl:.6f}")
        else:
            print("No KL steps recorded.")
        

        # Apply random edits
        edited_ids = apply_random_edits(
            token_ids=generated_ids,
            vocab_size=tokenizer.vocab_size,
            p_ins=0,
            p_del=0,
            p_sub=0
        )

        print("Edited text (truncated):",
            tokenizer.decode(edited_ids[:80], skip_special_tokens=True), " ...")


        # -----------------------------
        # 6. Decode watermark
        # -----------------------------
        recovered_bits, segment_stats, debug_info= decode_watermark_bits(
            model=model,
            tokenizer=tokenizer,
            generated_ids=edited_ids,
            secret_key=secret_key,
            segments_per_bin=segments_per_bin,
            segment_bits=segment_bits,
            k=k,
        )

        print("Recovered bits:", recovered_bits)

        # -----------------------------
        # 7. Evaluate
        # -----------------------------
        success = (recovered_bits == bitstream)
        print("Success:", success)

        successes += int(success)
        all_results.append({
            "bitstream": bitstream,
            "recovered": recovered_bits,
            "success": success
        })

    # ---------------------------------------------------------------
    # FINAL REPORT
    # ---------------------------------------------------------------
    print("\n============================================================")
    print("FINAL EXPERIMENT REPORT")
    print("============================================================")
    print(f"Runs: {NUM_RUNS}")
    print(f"Successful decodings: {successes}/{NUM_RUNS}")
    print(f"Success rate: {successes/NUM_RUNS:.3f}")

    print("\nDetailed per-run results:")
    for i, r in enumerate(all_results):
        print(f"Run {i+1}: success={r['success']}, "
              f"bits={r['bitstream']}, decoded={r['recovered']}")


