import argparse
import random
from typing import List, Tuple, Sequence, Optional, Iterable
import torch
import torch.nn.functional as F
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
    LogitsProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
)
from utils import select_green_list, apply_random_edits
import numpy as np
from watermarkutils import (
    hash_context_with_key,
    construct_segments,
    allocate_bits_proportional_to_entropy,
)
from RS_OSD import rs_decode_bits, rs_encode_bits, init_rs_codec
from wmprocessor.extractor import decode_watermark_bits
from wmprocessor.embedder import (
    MyEntropyHashWatermarkLogitsProcessor,
    NoWatermarkLogitsProcessor,
)
import itertools
import math
from datasets import load_dataset




def run_single_experiment(
    rng,
    BITSTREAM_LEN,
    P_X,
    seg_count,
    tokenizer,
    model,
    model_name,
    secret_key,
    k,
    N,
    rs_codec,
    n,
    k_rs,
    edit_prob,
    dataset,
    essay_dataset,
    bias
):
    """
    Runs one full watermarking + RS coding experiment.
    Returns all relevant outputs.
    """

    # -------------------------
    # 1. Generate random payload bits
    # -------------------------
    bitstream = rng.integers(0, 2, size=BITSTREAM_LEN, dtype=np.uint8)
    print("Bitstream:", bitstream)

    # -------------------------
    # 2. Construct segments per entropy bin
    # -------------------------
    segments_per_bin = construct_segments(P_X, seg_count)
    print("Segments per bin:", segments_per_bin)

    # -------------------------
    # 3. RS encoding
    # -------------------------
    code_bits = rs_encode_bits(bitstream, n, k_rs, rs_codec)

    # -------------------------
    # 4. Allocate bits to segments
    # -------------------------
    segment_bits = allocate_bits_proportional_to_entropy(
        segments_per_bin,
        code_bits
    )

    # -------------------------
    # 5. Initialize logits processor (embedding)
    # -------------------------
    wm_processor = MyEntropyHashWatermarkLogitsProcessor(
        tokenizer=tokenizer,
        secret_key=secret_key,
        segments_per_bin=segments_per_bin,
        segment_bits=segment_bits,
        k=k,
        model=model,
        model_type=model_name,
        bias=bias,
        entropy_bins=None,
    )
    processors = LogitsProcessorList([wm_processor])

    # -------------------------
    # 6. Generate watermarked text
    # -------------------------
     # -----------------------------
    # Build prompt depending on dataset choice
    # -----------------------------
    if dataset == "essay":
        if essay_dataset is None:
            raise ValueError("Dataset mode is 'essay' but no dataset was passed.")

        idx = int(rng.integers(0, len(essay_dataset)))
        sample = essay_dataset[idx]

        # Tokenize the essay
        encoded = tokenizer(sample["essays"], add_special_tokens=False)
        prompt_tokens = encoded["input_ids"][:100]      
        prompt = tokenizer.decode(prompt_tokens)

        # print("Prompt (tokens=100):", prompt, "...")
    else:
        prompt = "He walks into the room"


    encoding = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=100,
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # -----------------------------
    # 7. Generate CONTINUATION ONLY
    # -----------------------------
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

    # full text (prompt + continuation)
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)  
    continuation_ids = generated_ids[len(input_ids[0]):]
    continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    # print("Generated text (truncated):", generated_text, "...")

    if wm_processor.num_steps > 0:
        print(f"Average KL divergence per token: {wm_processor.total_kl / wm_processor.num_steps:.6f}")
    else:
        print("No KL steps recorded")

    # -------------------------
    # 7. Apply random edits (channel)
    # -------------------------
    edited_ids = apply_random_edits(
        token_ids=generated_ids,
        vocab_size=tokenizer.vocab_size,
        p_ins=edit_prob,
        p_del=edit_prob,
        p_sub=edit_prob,
    )
    edited_text = tokenizer.decode(edited_ids, skip_special_tokens=True)
    # print("Edited text (truncated):", edited_text, "...")

    # -------------------------
    # 8. Decode watermark
    # -------------------------
    recovered_bits, segment_stats, debug_info = decode_watermark_bits(
        model=model,
        tokenizer=tokenizer,
        generated_ids=edited_ids,
        secret_key=secret_key,
        segments_per_bin=segments_per_bin,
        segment_bits=segment_bits,
        k=k,
    )

    # Convert "010101" string → uint8 vector
    rec_bits = np.frombuffer(recovered_bits.encode(), dtype=np.uint8) - ord('0')
    rec_bits = rec_bits.astype(np.uint8)

    # -------------------------
    # 9. RS decoding
    # -------------------------
    decoded_bits = rs_decode_bits(rec_bits, segment_bits, segment_stats, n, k_rs, rs_codec)
    print("Decoded bits:", decoded_bits)

    # -------------------------
    # 10. Check success
    # -------------------------
    success = (decoded_bits == bitstream).all()
    print("Success:", success)

    return {
        "success": success,
        "bitstream": bitstream,
        "encoded_bits": code_bits,
        "extracted_bits": recovered_bits,
        "decoded_bits": decoded_bits,
        "segment_stats": segment_stats,
        "prompt": prompt,
        "generated_text": continuation_text,  
        "full_text": generated_text,        
        "edited_text": edited_text,
    }




