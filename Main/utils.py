# utils_watermark.py
import random
from typing import List, Iterable
import torch
import hashlib
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import itertools
import math
import json, os
import re
from datasets import load_dataset

def load_model(model_name="gpt2", device=None):
    """Load model/tokenizer once and reuse."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if device is not None:
        model.to(device)
    model.eval()
    return model, tokenizer


def bits_to_int(bit_seq):
    """Convert bit sequence ('0101' or [0,1,0,1]) to integer."""
    if isinstance(bit_seq, str):
        bits = [int(b) for b in bit_seq]
    else:
        bits = [int(b) for b in bit_seq]
    x = 0
    for b in bits:
        x = (x << 1) | b
    return x


def deterministic_rng(seed):
    """Create a numpy Generator from a SHA-256 seed."""
    h = hashlib.sha256(seed.encode()).digest()
    # convert first 8 bytes → uint64
    seed_int = int.from_bytes(h[:8], 'big')
    return np.random.default_rng(seed_int)


def select_green_list(secret_key, bit_seq, prev_tokens, vocab_size):
    """
    Secret-keyed deterministic partition of the vocabulary.

    For fixed (secret_key, prev_tokens) and bit-length k = len(bit_seq),
    the 2^k different bit patterns index disjoint slices of a *single*
    permuted vocabulary, giving a true partition.
    """
    # Ensure tokens list is stringified
    prev_tokens_str = "_".join(str(t) for t in prev_tokens)

    # --- Setup ---
    secret_key = str(secret_key)
    k = len(bit_seq)
    num_buckets = 2**k

    # Convert bit sequence to integer bucket index
    bucket_index = bits_to_int(bit_seq)

    # --- Compute uneven bucket sizes ---
    q = vocab_size // num_buckets
    r = vocab_size % num_buckets

    if bucket_index < r:
        bucket_size = q + 1
        start = bucket_index * (q + 1)
    else:
        bucket_size = q
        start = r * (q + 1) + (bucket_index - r) * q

    end = start + bucket_size

    # --- IMPORTANT: seed does NOT depend on bit_seq ---
    seed_string = f"{secret_key}|{prev_tokens_str}"
    rng = deterministic_rng(seed_string)

    # --- Single global permutation for this context ---
    perm = np.arange(vocab_size)
    rng.shuffle(perm)

    # --- Return this bucket's slice ---
    green_list = perm[start:end]
    return green_list.tolist()



def apply_random_edits(
    token_ids: torch.Tensor,
    vocab_size: int,
    p_ins: float = 0.0,
    p_del: float = 0.0,
    p_sub: float = 0.0,
):
    """
    Apply random insertion, deletion, and substitution to a sequence
    of token IDs to simulate adversarial editing.

    Args:
        token_ids: torch.LongTensor of shape [seq_len]
        vocab_size: size of tokenizer vocabulary
        p_ins: probability of inserting a random token BEFORE each token
        p_del: probability of deleting the current token
        p_sub: probability of substituting the current token with a random token

    Returns:
        edited_ids: torch.LongTensor with edits applied
    """

    edited = []

    for tok in token_ids.tolist():

        # ------------------------
        # 1. Random INSERT
        # ------------------------
        if random.random() < p_ins:
            rand_tok = random.randint(0, vocab_size - 1)
            edited.append(rand_tok)

        # ------------------------
        # 2. Random DELETE
        # ------------------------
        if random.random() < p_del:
            # skip adding current token → deletion
            continue

        # ------------------------
        # 3. Random SUBSTITUTE
        # ------------------------
        if random.random() < p_sub:
            rand_tok = random.randint(0, vocab_size - 1)
            edited.append(rand_tok)
        else:
            edited.append(tok)

    return torch.tensor(edited, dtype=torch.long)


def compact_json_arrays(json_str):
    """Post-process JSON string to put array elements on the same line."""
    lines = json_str.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this line starts an array (ends with just "[")
        if re.match(r'\s+"[^"]+":\s*\[\s*$', line):
            indent_match = re.match(r'(\s+)"([^"]+)":\s*\[\s*$', line)
            if indent_match:
                indent = indent_match.group(1)
                key = indent_match.group(2)
                
                # Collect array elements
                elements = []
                i += 1
                array_closed = False
                trailing_comma = False
                
                while i < len(lines):
                    elem_line = lines[i]
                    stripped = elem_line.strip()
                    
                    # Check for closing bracket
                    if stripped == ']':
                        array_closed = True
                        break
                    elif stripped == '],':
                        array_closed = True
                        trailing_comma = True
                        break
                    
                    # Check if this is an object (not a simple array element)
                    if stripped.startswith('{'):
                        # Abort - this array contains objects, don't compact
                        result.append(line)
                        i -= 1
                        break
                    
                    # Extract element value
                    if stripped:
                        elem_value = stripped.rstrip(',').strip()
                        # Only process if it looks like a simple value (number, boolean, or quoted string)
                        if elem_value and (elem_value.replace('-', '').replace('.', '').isdigit() or 
                                         elem_value in ['true', 'false', 'null'] or
                                         (elem_value.startswith('"') and elem_value.endswith('"'))):
                            elements.append(elem_value)
                    
                    i += 1
                
                if array_closed:
                    # Successfully compacted the array
                    if elements:
                        compact = f'{indent}"{key}": [{", ".join(elements)}]'
                    else:
                        compact = f'{indent}"{key}": []'
                    if trailing_comma:
                        compact += ','
                    result.append(compact)
                    i += 1
                    continue
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)

def numpy_to_python(obj):
    if isinstance(obj, (np.bool_, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def convert_segment_stats_for_json(segment_stats):
    """
    Converts tuple keys into string keys so that JSON can serialize them.
    Example: (5,0) -> "5,0"
    """
    new_dict = {}
    for key, value in segment_stats.items():
        if isinstance(key, tuple):
            new_key = ",".join(str(k) for k in key)
        else:
            new_key = str(key)
        new_dict[new_key] = value
    return new_dict