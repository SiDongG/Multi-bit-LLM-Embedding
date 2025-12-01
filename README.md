# Covert LLM Embedding: A Resolvability Perspective

Preliminary Achievable Scheme for CS8803: MLS

## Features
- Covert multi-bit embedding in LLM outputs  
- Entropy-based and segmentation-based embedding  (Context window hashing and soft logits biasing 
- Resolvability-based distribution matching  
- Robust decoding under editing, paraphrasing, and noise  (RS over GF(256) and 1st order OSD)
- Statistical detection tools (KL, chi-square)  
- Experiment and evaluation pipelines  

## Installation
```bash
git clone https://github.com/SiDongG/Multi-bit-LLM-Embedding
cd Multi-bit-LLM-Embedding/Main
pip install -r requirements.txt
