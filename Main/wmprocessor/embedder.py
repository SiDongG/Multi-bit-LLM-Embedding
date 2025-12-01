
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
    log_probs = F.log_softmax(logits, dim=-1) 
    entropy_nats = -(probs * log_probs).sum().item()
    entropy_bits = entropy_nats / np.log(2.0)
    return entropy_bits

class NoWatermarkLogitsProcessor(LogitsProcessor):
    """
    A logits processor that does absolutely nothing.
    This reproduces the behavior of the original LLM with no watermark.
    """
    def __call__(self, input_ids, scores):
        return scores 
    

class MyEntropyHashWatermarkLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        secret_key,
        segments_per_bin,
        segment_bits,
        k,
        model,
        model_type="auto",    
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

        if model_type == "auto":
            name = model.__class__.__name__.lower()
            if "llama" in name:
                self.model_type = "llama"
            elif "gpt" in name:
                self.model_type = "gpt2"
            else:
                self.model_type = "other"
        else:
            self.model_type = model_type.lower()

        # ---------------------- ENTROPY BINS ----------------------------
        if entropy_bins is None:
            self.entropy_bins = np.array([1,2,3,4,5,6,7,8,9,10,11, np.inf])
        else:
            self.entropy_bins = entropy_bins

        assert len(self.segments_per_bin) == len(self.entropy_bins) - 1

        # ---------------------- KL STATS --------------------------------
        self.total_kl = 0.0
        self.num_steps = 0


    def _compute_entropy_bits(self, logits):
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        H = -(probs * log_probs).sum().item()
        return H / np.log(2)


    def __call__(self, input_ids, scores):

        # -------------------- shape unwrap ------------------------
        if scores.dim() == 2:
            logits = scores[0]
        else:
            logits = scores

        # -------------------- 1. build entropy context -------------------
        raw_ctx = input_ids[0][-self.k:]  

        if self.model_type == "llama":
            bos_id = self.tokenizer.bos_token_id
            if bos_id is not None:
                entropy_ctx = torch.cat([
                    torch.tensor([bos_id], device=raw_ctx.device),
                    raw_ctx
                ])
            else:
                entropy_ctx = raw_ctx
        else:
            entropy_ctx = raw_ctx

        entropy_ctx = entropy_ctx.unsqueeze(0) 

        # -------------------- 2. compute entropy ------------------------
        with torch.no_grad():
            entropy_logits = self.model(entropy_ctx).logits[0, -1, :]

        H_bits = self._compute_entropy_bits(entropy_logits)

        # -------------------- 3. map to entropy bin ---------------------
        bin_idx = int(np.digitize(H_bits, self.entropy_bins) - 1)
        bin_idx = max(0, min(bin_idx, len(self.segments_per_bin) - 1))

        if self.segments_per_bin[bin_idx] <= 0:
            return scores

        # -------------------- 4. segment index --------------------------
        ctx_list = input_ids[0].tolist()
        context_tokens = ctx_list[-self.k:] if len(ctx_list) >= self.k else ctx_list

        num_seg = self.segments_per_bin[bin_idx]
        seg_idx = hash_context_with_key(context_tokens, self.secret_key, num_seg)

        # -------------------- 5. bit sequence ---------------------------
        bit_seq = self.segment_bits.get((bin_idx, seg_idx), "")
        if len(bit_seq) == 0:
            return scores

        # -------------------- 6. compute greenlist ----------------------
        green_ids = select_green_list(
            secret_key=self.secret_key,
            bit_seq=bit_seq,
            prev_tokens=context_tokens,
            vocab_size=self.vocab_size,
        )
        green_ids = torch.tensor(green_ids, dtype=torch.long, device=logits.device)

        # -------------------- 7. KL BEFORE bias -------------------------
        p = F.softmax(logits, dim=-1)
        p_detached = p.detach()

        # -------------------- 8. apply watermark bias -------------------
        biased = logits.clone()
        biased[green_ids] += self.bias

        # -------------------- 9. KL AFTER bias --------------------------
        q = F.softmax(biased, dim=-1)
        kl = (p_detached * (p_detached.log() - q.log())).sum().item()

        self.total_kl += kl
        self.num_steps += 1

        # -------------------- 10. return correct shape ------------------
        if scores.dim() == 2:
            out = scores.clone()
            out[0] = biased
            return out
        else:
            return biased
