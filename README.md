# Covert LLM Embedding: A Resolvability Perspective

Preliminary Achievable Scheme for CS8803: MLS

## Features
- Covert multi-bit embedding in LLM outputs  
- Entropy-based and segmentation-based embedding  (Context window hashing and soft logits biasing)
- Resolvability-based distribution matching  
- Robust decoding under editing, paraphrasing, and noise  (RS over GF(256) and 1st order OSD)
- Statistical detection tools (KL, chi-square)  
- Experiment and evaluation pipelines  

## Installation
```bash
git clone https://github.com/SiDongG/Multi-bit-LLM-Embedding
cd Multi-bit-LLM-Embedding/Main
pip install -r requirements.txt
```

## Run
You can run experiments using either GPT-2 or LLaMA-3.1-8B-Instruct, with optional dataset prompts (essays) and configurable RS coding and editing noise.
Example: 
```bash
python main.py \
    --runs 10 \
    --bits 24 \
    --n 7 \
    --k_rs 3 \
    --seg 10 \
    --edit 0.07 \
```

Detailed output is printed in Results/results.json. This script runs 10 watermarking trials under required adversarial and coding rate setting. 