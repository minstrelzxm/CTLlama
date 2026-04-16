# CTLlama

**A language model for evaluating and synthesising clinical trials from their registrations and published results.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-minstrelzxm-yellow.svg)](https://huggingface.co/minstrelzxm)

---

## 🤗 Models & Data

All artifacts are released on Hugging Face.

### Models

| Stage | Model | Link |
| --- | --- | --- |
| Base (after CPT) | `ctllama-8b-base` | [🤗 minstrelzxm/CTLlama-8B-base](https://huggingface.co/minstrelzxm/CTLlama-8B-base) |
| Instruct (after SFT) | `CTLlama-8b-instruct-distill` | [🤗 minstrelzxm/CTLlama-8B-Instruct](https://huggingface.co/minstrelzxm/CTLlama-8B-Instruct) |
| Final (after RL/GRPO) | `CTLlama-8b-grpo` | [🤗 minstrelzxm/CTLlama-8B-Instruct-GRPO](https://huggingface.co/minstrelzxm/CTLlama-8B-Instruct-GRPO) |

### Training Data

| Stage | Dataset | Size | Link |
| --- | --- | --- | --- |
| CPT | Clinical trials + PubMed abstracts | ~75870 examples | [🤗 minstrelzxm/ctllama-cpt-corpus](https://huggingface.co/datasets/minstrelzxm/ctllama-cpt-corpus) |
| SFT | Expert-labelled clinical QA pairs | ~10K examples | [🤗 minstrelzxm/CTLlamaSFT](https://huggingface.co/datasets/minstrelzxm/CTLlamaSFT) |
| RL | Stabilisation Reinforcement QA pairs | ~2.5K examples | [🤗 minstrelzxm/CTLlamaRL](https://huggingface.co/datasets/minstrelzxm/CTLlamaRL) |

### Load any stage in three lines

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("minstrelzxm/CTLlama-8B-Instruct-GRPO")
model = AutoModelForCausalLM.from_pretrained("minstrelzxm/CTLlama-8B-Instruct-GRPO", torch_dtype="bfloat16", device_map="auto")
```

---

## 📖 Introduction

**CTLlama** is a domain-adapted LLaMA-3.1-8B released as a three-stage pipeline:

1. **Continual Pre-Training (CPT)** — further pre-trains LLaMA-3.1-8B on ~100M tokens of ClinicalTrials.gov reports and PubMed abstracts so the model's representations shift toward clinical language.
2. **Supervised Fine-Tuning (SFT)** — LoRA-tunes the CPT model on expert-labelled QA pairs with explicit `<think>` / `</think>` structure so the model follows clinical reasoning instructions.
3. **Reinforcement Learning (RL)** — applies Group Relative Policy Optimization (GRPO) with rule-based XML/CoT reward functions to stabilize output format and preserve reasoning quality across domains.

The result is a model that reads clinical trial registrations and published results, reasons about them step-by-step, and produces structured, auditable answers.

---

## ⚡ Quick Start

### Inference

```bash
pip install -r requirements.txt   # (or install torch, transformers, peft directly)
python evaluation/inference.py \
    --base-model minstrelzxm/ctllama-8b-base \
    --adapter-dir minstrelzxm/CTLlama-8b-instruct-distill \
    --prompts prompts.jsonl \
    --output predictions.jsonl
```

`prompts.jsonl` takes either `{"prompt": "..."}` or `{"messages": [{"role": "user", "content": "..."}]}` per line.

### Reproducing training

```bash
export HF_TOKEN=hf_...   # see .env.example

# 1. CPT
cd cpt && python run.py

# 2. SFT
cd ../sft && python run.py

# 3. RL (GRPO)
cd ../rl && python run.py
```

Each stage writes its LoRA adapter to its own output directory (`cpt_output/`, `lora_model/`, `grpo_saved_lora/`).

---

## ⚙️ Training Pipeline

### 1. Continual Pre-Training (CPT)

- **Data:** ~100M tokens from ClinicalTrials.gov and PubMed abstracts
- **Objective:** Causal (autoregressive) language modeling
- **Configuration:**
  - Learning rate: 5 × 10⁻⁵
  - Effective batch size: 128 (per-device 2 × grad-accum 8 × 8 GPUs)
  - Sequence length: 4 096
  - Precision: bf16 + gradient checkpointing
- **LoRA:** r=16, α=16, all attention + MLP projections

### 2. Supervised Fine-Tuning (SFT)

- **Data:** ~10K expert-labelled QA examples (system + instruction + reference + CoT + answer)
- **Objective:** Instruction-following via cross-entropy loss, **response-only** masking (loss computed on assistant output only, not on the prompt)
- **LoRA:** r=32, α=32, target modules = attention + MLP + `embed_tokens` + `lm_head`
- **Special tokens:** `<think>`, `</think>`

### 3. Reinforcement Learning (RL)

- **Algorithm:** GRPO (Group Relative Policy Optimization, via TRL)
- **Reward functions:**
  - XML-tag count and placement (`xmlcount_reward_func`)
  - Soft / strict `<think>…</think>` pattern match
  - Numeric answer validation (accepts negatives, decimals, commas)
  - Exact-match correctness
- **Hyperparameters:**
  - Learning rate: 5 × 10⁻⁶
  - Per-device batch size: 1; grad-accum: 1; num_generations: 6
  - Max prompt: 512; max completion: 3 584
  - Optimizer: `paged_adamw_8bit`

---

## 📁 Repository Layout

```text
CTLlama/
├── cpt/                 # Continual pre-training
│   ├── run.py           # Entry point
│   ├── setup.py         # Model + data preparation
│   ├── train.py         # Trainer wrapper
│   └── utils.py         # Tokenization
├── sft/                 # Supervised fine-tuning
│   ├── run.py
│   ├── setup.py         # Unsloth + LoRA + special tokens
│   ├── train.py
│   └── utils.py         # Chat-template formatting
├── rl/                  # GRPO RL
│   ├── run.py
│   ├── setup.py         # Unsloth + GRPO config
│   ├── train.py
│   └── utils.py         # Reward functions
├── evaluation/
│   └── inference.py     # CLI: base + adapter → predictions.jsonl
├── .env.example         # HF_TOKEN template
├── FIXES.md             # Audit log of review fixes
├── LICENSE
└── README.md
```

---

## 🛠️ Installation

```bash
git clone https://github.com/minstrelzxm/CTLlama.git
cd CTLlama
cp .env.example .env        # then edit with your HF_TOKEN
pip install torch transformers peft trl datasets unsloth wandb
```

Tested with Python 3.10+, CUDA 12.x, LLaMA-3.1 tokenizer.

---

## 🤝 Contributing

PRs welcome. Before submitting, please:

1. Run `pyflakes` on changed files — the main tree is lint-clean.
2. Don't touch the `*/draft/` folders — they are historical scratch space.
3. See [FIXES.md](FIXES.md) for the current quality-review audit log.

---

## 📄 License

MIT — see [LICENSE](LICENSE).
