# CTLlama: A language model for evaluating and synthesising clinical trials from their registrations and published results

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

> **CTLlama** is a domain-adapted LLaMA-3.1-8B model fine-tuned on clinical trials and biomedical abstracts, with a three-stage training pipeline: Continual Pre-Training (CPT), Supervised Fine-Tuning (SFT), and Reinforcement Learning (RL).

---

## 🚀 Table of Contents

- [🎯 Features](#-features)  
- [🗺️ Model Overview](#️-model-overview)  
- [⚙️ Training Pipeline](#️-training-pipeline)  
  - [1. Continual Pre-Training (CPT)](#1-continual-pre-training-cpt)  
  - [2. Supervised Fine-Tuning (SFT)](#2-supervised-fine-tuning-sft)  
  - [3. Reinforcement Learning (RL)](#3-reinforcement-learning-rl)  
  - [Training Pipeline Diagram](#training-pipeline-diagram)  
- [⚡ Quick Start](#-quick-start)  
- [📊 Results Overview](#-results-overview)  
- [🛠️ Installation & Usage](#️-installation--usage)  
- [🤝 Contributing](#-contributing)  
- [📄 License](#-license)  

---

## 🎯 Features

- **Domain adaptation**: Built on LLaMA-3.1-8B, specialized for clinical trial data and PubMed abstracts.  
- **Efficient fine-tuning**: Uses LoRA adapters for lightweight SFT and RL stages.  
- **Stable outputs**: GRPO-based reinforcement learning to enforce consistent XML/CoT formatting.  
- **Extensible**: Modular codebase with clear separation of setup, training, and utilities.

---

## 🗺️ Model Overview

CTLlama is trained in **three sequential stages** to maximize performance and stability in clinical NLP tasks:

1. **Continual Pre-Training (CPT)**  
   Further pre-trains the base LLaMA model on a large corpus of unlabelled clinical trial reports and biomedical article abstracts to adapt its representations to the clinical domain.

2. **Supervised Fine-Tuning (SFT)**  
   Trains on expert-annotated question-answer pairs (clinical reasoning + final answer) using LoRA adapters for domain-specific instruction following.

3. **Reinforcement Learning (RL)**  
   Applies Group Relative Policy Optimization (GRPO) with rule-based XML/CoT reward functions to stabilize and enforce the desired output format across domains.

---

## ⚙️ Training Pipeline

### 1. Continual Pre-Training (CPT)

- **Data**: ~100 M tokens from ClinicalTrials.gov and PubMed abstracts  
- **Objective**: Masked-language modeling to refine domain vocabulary and syntax  
- **Configuration**:  
  - Learning rate: 5 × 10⁻⁵  
  - Batch size: 128; sequence length: 4 096 

### 2. Supervised Fine-Tuning (SFT)

- **Data**: 10 000 expert-labelled QA examples  
- **Objective**: Instruction-following via cross-entropy loss on CoT + answer pairs  
- **LoRA settings**: rank=32, α=32, target modules = attention & head projections  

### 3. Reinforcement Learning (RL)

- **Algorithm**: GRPO (Group Relative Policy Optimization)  
- **Reward functions**:  
  - **Format compliance**: Strict/soft XML-CoT pattern matching  
  - **Answer correctness**: Exact match & numeric validation  
- **Hyperparameters**:  
  - Learning rate: 5 × 10⁻⁶  
  - Batch size: 1; gradient accumulation = 1; num_generations = 6  

---

### Training Pipeline Diagram

> _Replace this with your pipeline visualization_

