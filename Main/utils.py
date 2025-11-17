# utils_watermark.py
import random
from typing import List, Iterable
import torch
import hashlib
import numpy as np

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


# ---------------------------
# Example usage (skeleton)
# ---------------------------
# if __name__ == "__main__":
#     import numpy as np
#     import torch
#     from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

#     # ---------------------------------------------------------------
#     # 1. SETTINGS
#     # ---------------------------------------------------------------
#     entropy_bins = np.array(["[0,1)", "[1,2)", "[2,3)", "[3,4)", "[4,5)", "[5,6)", "[6,7)",
#                              "[7,8)", "[8,9)", "[9,10)", "[10,11)", "[11,12)", "[12,∞)"])

#     # Your P_X distribution
#     P_X = np.array([0.042, 0.02, 0.024, 0.04, 0.07,
#                     0.121, 0.139, 0.144, 0.14, 0.12, 0.068, 0.072])

#     model_name = "gpt2"
#     N = 200             # tokens to generate
#     M = np.round(N)

#     # Construct number of segments in each entropy bin
#     segments_per_bin = construct_segments(P_X, 15)

#     # Payload bits
#     bitstream = "100110111001010010100110101011110101010"

#     # Allocate per-segment bit subsequences
#     segment_bits = allocate_bits_proportional_to_entropy(
#         segments_per_bin,
#         bitstream
#     )

#     # ---------------------------------------------------------------
#     # 2. LOAD MODEL
#     # ---------------------------------------------------------------
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     model.eval()

#     secret_key = "my_super_secret_key"
#     k = 4  # context window size

#     # ---------------------------------------------------------------
#     # 3. BUILD WATERMARK PROCESSOR
#     # ---------------------------------------------------------------
#     wm_processor = MyEntropyHashWatermarkLogitsProcessor(
#         tokenizer=tokenizer,
#         secret_key=secret_key,
#         segments_per_bin=segments_per_bin,
#         segment_bits=segment_bits,
#         k=k,
#     )

#     processors = LogitsProcessorList([wm_processor])

#     # ---------------------------------------------------------------
#     # 4. GENERATE WATERMARKED TEXT
#     # ---------------------------------------------------------------
#     prompt = "He walks into the room"
#     input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

#     with torch.no_grad():
#         out = model.generate(
#             input_ids=input_ids,
#             max_new_tokens=N,
#             logits_processor=processors,
#             do_sample=True,   # use sampling so green-list bias works
#             top_p=0.9,
#             temperature=1.0
#         )

#     generated_ids = out[0]

#     generated_text = tokenizer.decode(out[0], skip_special_tokens=True)
#     print("\n==================== WATERMARKED TEXT ====================")
#     print(generated_text)

#     # ---------------------------------------------------------------
#     # GENERATE UNWATERMARKED (ORIGINAL) TEXT
#     # ---------------------------------------------------------------

#     no_wm_processor = NoWatermarkLogitsProcessor()
#     processors_none = LogitsProcessorList([no_wm_processor])

#     with torch.no_grad():
#         out_clean = model.generate(
#             input_ids=input_ids,
#             max_new_tokens=N,
#             logits_processor=processors_none,   # <--- no watermark processor
#             do_sample=True,                     # sampling, same as watermarked version
#             top_p=0.9,
#             temperature=1.0,
#         )

#     clean_text = tokenizer.decode(out_clean[0], skip_special_tokens=True)

#     print("\n==================== UNWATERMARKED TEXT ====================")
#     print(clean_text)

#     # ---------------------------------------------------------------
#     # 5. DECODE WATERMARK BITS
#     # ---------------------------------------------------------------
#     recovered_bits, segment_stats, debug_info = decode_watermark_bits(
#         model=model,
#         tokenizer=tokenizer,
#         generated_ids=generated_ids,
#         secret_key=secret_key,
#         segments_per_bin=segments_per_bin,
#         segment_bits=segment_bits,
#         k=k,
#     )


#     print("\n==================== DECODED BITSTREAM ====================")
#     print(recovered_bits)

#     print("\n==================== ORIGINAL BITSTREAM ====================")
#     print(bitstream)

#     # Optional: print first few debug steps
#     print("\n==================== DEBUG INFO (first 10 steps) ====================")
#     for d in debug_info[:10]:
#         print(d)