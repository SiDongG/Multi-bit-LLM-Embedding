import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import numpy as np
import hashlib
import utils

ENTROPY_BINS = np.array([1,2,3,4,5,6,7,8,9,10,11, np.inf])

INTERVALS = np.array(["[0,1)", "[1,2)", "[2,3)", "[3,4)", "[4,5)", "[5,6)", "[6,7)", "[7,8)", "[8,9)", "[9,10)", "[10,11)", "[11,∞)"])

P_X = np.array([0.042, 0.02, 0.024, 0.04, 0.07, 0.121, 0.139, 0.144, 0.14, 0.12, 0.068, 0.072])


def construct_segments(P_X, M):
    """
    Given probability vector P_X and an integer M, 
    ignore the first element and allocate M segments 
    proportionally to the remaining probabilities.
    
    Returns:
        segments: np.array of length len(P_X)-1 containing integer allocations
    """
    # Remove first element
    P = np.array(P_X[1:], dtype=float)  
    # Normalize remaining probabilities
    P = P / P.sum()
    # Initial proportional allocation
    raw = P * M
    # Round to integers
    segs = np.floor(raw).astype(int)
    # Correct rounding to ensure sum = M
    shortfall = M - segs.sum()
    # Distribute remaining segments to bins with largest fractional parts
    fractional_parts = raw - np.floor(raw)
    order = np.argsort(-fractional_parts)  # descending

    for idx in order[:shortfall]:
        segs[idx] += 1
    
    return segs


def allocate_bits_proportional_to_entropy(segments_per_bin, bitstream):
    """
    segments_per_bin: list[int] of length 11.
    bitstream: string of bits whose length = total_bits.

    Returns:
        mapping: dict[(bin_idx, seg_idx)] -> bit_subsequence (string)
    """
    import numpy as np

    segments_per_bin = np.array(segments_per_bin, dtype=int)
    total_bits = len(bitstream)

    # Representative entropies (midpoints): [1.5, 2.5, ..., 11.5]
    num_bins = len(segments_per_bin)
    left_edges = np.arange(1, num_bins + 1)
    right_edges = left_edges + 1
    bin_entropies = (left_edges + right_edges) / 2.0

    # Build segment list and weights
    weights = []
    keys = []
    for b, count in enumerate(segments_per_bin):
        for s in range(count):
            weights.append(bin_entropies[b])
            keys.append((b, s))

    weights = np.array(weights, dtype=float)
    if len(weights) == 0:
        return {}

    # Allocate lengths using same proportional logic as original
    W = weights.sum()
    ideal = total_bits * weights / W

    lengths = np.floor(ideal).astype(int)
    remainder = total_bits - lengths.sum()

    if remainder > 0:
        frac = ideal - lengths
        order = np.argsort(-frac)  # highest fractional part first
        for idx in order[:remainder]:
            lengths[idx] += 1

    # Now partition the bitstream into these lengths
    mapping = {}
    pos = 0
    for key, L in zip(keys, lengths):
        subseq = bitstream[pos : pos + L]
        mapping[key] = subseq
        pos += L

    return mapping


def hash_context_with_key(context_tokens, secret_key, modulus):
    # Convert secret key to string
    secret_key = str(secret_key)

    # Build hashing string
    s = secret_key + "_" + "_".join(str(t) for t in context_tokens)

    # SHA256 hash → integer
    h = hashlib.sha256(s.encode()).hexdigest()
    return int(h, 16) % modulus



