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
from utils import (
    select_green_list,
    apply_random_edits, 
    compact_json_arrays, 
    numpy_to_python, 
    convert_segment_stats_for_json
)
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
from runner import run_single_experiment
import json, os
import re
from datasets import load_dataset

def main():

    parser = argparse.ArgumentParser(description="LLM Multi-bit Embedding Experiment")

    parser.add_argument("--runs", type=int, default=1,
                        help="Number of experiment runs")
    parser.add_argument("--bits", type=int, default=24,
                        help="Length of raw payload bitstream BEFORE RS coding")
    parser.add_argument("--tokens", type=int, default=200,
                        help="Number of tokens to generate")
    parser.add_argument("--k", type=int, default=3,
                        help="Entropy prediction window size")
    parser.add_argument("--seg", type=int, default=10,
                        help="Total number of segments")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model name: gpt2, llama-3.1b")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--n", type=int, default=5,
                        help="RS(n,k) total number of symbols (bytes)")
    parser.add_argument("--k_rs", type=int, default=3,
                        help="RS(n,k) message symbol count (bytes)")
    parser.add_argument("--edit", type=float, default=0,
                        help="Edit probability for insertion/deletion/substitution")
    parser.add_argument("--dataset", type=str, default="essay",
                    choices=["none", "essay"],
                    help="Source of prompts: 'none' uses fixed prompt; 'essay' loads essay dataset")
    parser.add_argument("--bias", type=float, default=6.0,
                        help="Watermark bias parameter")
    args = parser.parse_args()

    # CONFIGURATION
    NUM_RUNS = args.runs
    BITSTREAM_LEN = args.bits          
    N = args.tokens
    k = args.k
    n = args.n                         
    k_rs = args.k_rs                    
    model_name = args.model
    secret_key = "my_super_secret_key"

    entropy_bins = np.array([0,1,2,3,4,5,6,7,8,9,10,11,np.inf])

    # Empirical entropy distribution for bin allocation
    P_X = np.array([
        0.042, 0.02, 0.024, 0.04, 0.07,
        0.121, 0.139, 0.144, 0.14, 0.12,
        0.068, 0.072
    ])

    # ---------------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------------
    print("\nLoading model:", model_name)

    # ---- 0. Auto-correct llama model names ----
    llama_aliases = ["llama", "llama3", "llama3.1", "llama3.1b", "llama-3", "llama-3.1"]

    def is_llama(name):
        n = name.lower()
        return any(alias in n for alias in llama_aliases) or "meta-llama" in n

    if is_llama(model_name):
        print("Detected LLaMA-like model name. Using official HF model id:")
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        print(" → Corrected model name:", model_name)
    else:
        print("Using model name as provided.")

    # ---- 1. Load tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # ---- 2. Load model ----
    model_is_llama = "meta-llama" in model_name.lower()

    if model_is_llama:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            token=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32
        )

    model.eval()
    print("Model loaded successfully.")

    # ---------------------------------------------------------------
    # LOAD DATASET
    # ---------------------------------------------------------------
    if args.dataset == "essay":
        print("Loading essay dataset...")
        essay_dataset = load_dataset(
            "ChristophSchuhmann/essays-with-instructions",
            split="train"
        )
        print(f"Loaded {len(essay_dataset)} essays.")
    else:
        essay_dataset = None
        print("Using random prompts (no dataset).")
    # INIT RS CODEC
    # ---------------------------------------------------------------
    print(f"\nUsing RS({n}, {k_rs}) code...")
    rs_codec, parity, t = init_rs_codec(n, k_rs)
    print(f"  parity symbols = {parity}, correctable t = {t}")

    # ---------------------------------------------------------------
    # RUN EXPERIMENTS
    # ---------------------------------------------------------------
    successes = 0
    all_results = []
    rng = np.random.default_rng(args.seed)

    for run in range(NUM_RUNS):
        print(f"\n==================== RUN {run+1}/{NUM_RUNS} ====================")

        result = run_single_experiment(
            rng=rng,
            BITSTREAM_LEN=BITSTREAM_LEN,
            P_X=P_X,
            seg_count=args.seg,   
            tokenizer=tokenizer,
            model=model,
            model_name=model_name,
            secret_key=secret_key,
            k=k,
            N=N,
            rs_codec=rs_codec,
            n=n,
            k_rs=k_rs,
            edit_prob=args.edit,     
            dataset=args.dataset,
            essay_dataset=essay_dataset,
            bias=args.bias     
        )

        successes += int(result["success"])
        all_results.append(result)

    
    # ---------------------------------------------------------------
    # SAVE JSON RESULTS
    # ---------------------------------------------------------------
    os.makedirs("results", exist_ok=True)

    output = {
        "runs": int(NUM_RUNS),
        "edit_probability": float(args.edit),
        "rs_params": {
            "n": int(n),
            "k_rs": int(k_rs),
            "parity": int(parity),
            "t_correctable": int(t)
        },
        "payload_bits": int(BITSTREAM_LEN),
        "successes": int(successes),
        "accuracy": float(successes / NUM_RUNS),
        "per_run": []
    }

    # ---------------------------
    # Global average bit accuracy
    # ---------------------------
    avg_bit_accuracy = np.mean([
        sum(int(a == b) for a, b in zip(r["bitstream"], r["decoded_bits"])) / BITSTREAM_LEN
        for r in all_results
    ])

    output["bit_accuracy"] = float(avg_bit_accuracy)

    # ---------------------------
    # Per-run results
    # ---------------------------
    for i, r in enumerate(all_results):

        correct_bits = sum(int(a == b) for a, b in zip(r["bitstream"], r["decoded_bits"]))
        bit_acc = correct_bits / BITSTREAM_LEN

        output["per_run"].append({
            "run": int(i + 1),
            "success": numpy_to_python(r["success"]),
            "bit_accuracy": float(bit_acc),
            "correct_bits": int(correct_bits),

            "bitstream": r["bitstream"].tolist(),
            "encoded_bits": r["encoded_bits"].tolist(),
            "decoded_bits": r["decoded_bits"].tolist(),

            "extracted_bits": [int(x) for x in r["extracted_bits"]],

            "segment_stats": convert_segment_stats_for_json(r["segment_stats"]),

            "prompt": r["prompt"],                            
            "generated_text": r["generated_text"],           
            "full_text": r["full_text"],                       
            "edited_text": r["edited_text"][:200],         
        })

    # ---------------------------
    # Save JSON
    # ---------------------------
    with open("results/results.json", "w") as f:
        json_str = json.dumps(output, indent=4)
        json_str = compact_json_arrays(json_str)
        f.write(json_str)

    print("\nSaved results to results/results.json")




if __name__ == "__main__":
    main()



