

import random
from typing import List, Tuple, Sequence, Optional, Iterable
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, PreTrainedModel, LogitsProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from utils import select_green_list, apply_random_edits
import numpy as np
from watermarkutils import hash_context_with_key, construct_segments, allocate_bits_proportional_to_entropy
import math
from itertools import product   
from RS_OSD import rs_encode_bits, rs_decode_bits
import itertools

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

    """

    if generated_ids.dim() == 2:
        generated_ids = generated_ids[0] 

    device = next(model.parameters()).device
    vocab_size = tokenizer.vocab_size
    segments_per_bin = np.array(segments_per_bin, dtype=int)
    seq_len = generated_ids.size(0)

    # ---------------------------
    # 0. Precompute candidate sets per segment
    # ---------------------------
    segment_stats = {}

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

        # might have L=0 
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