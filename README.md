covert-llm-embedding/
│
├── src/
│   ├── embedding/          # embedding algorithms (green-list, entropy-based, resolvability)
│   ├── decoding/           # decoders (hard/soft, OSD-like, MAP, sequential)
│   ├── stats/              # KL, chi-square, Chernoff, entropy analyzers
│   ├── llm/                # interfaces to HuggingFace models
│   ├── utils/              # helpers and shared utilities
│   └── evaluation/         # simulation experiments, benchmarking scripts
│
├── experiments/
│   ├── square_root_law/    # rate scaling experiments
│   ├── robustness/         # editing, paraphrasing, filtering robustness
│   ├── segmentation/       # entropy-segmentation results
│   └── watermarking/       # baselines (e.g., Kirchenbauer, Qu et al.)
│
├── data/
│   ├── prompts/
│   ├── corpora/
│   └── cached_models/
│
├── results/
│   ├── figures/
│   ├── logs/
│   └── processed/
│
├── slurm/
│   ├── run_experiment.sbatch
│   └── analysis_job.sbatch
│
└── README.md
