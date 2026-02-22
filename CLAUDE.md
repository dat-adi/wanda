# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **Wanda** (Pruning by Weights and Activations), a simple and effective pruning approach for Large Language Models. The project implements weight pruning methods including Wanda, SparseGPT, and magnitude pruning with support for structured and unstructured sparsity patterns.

## Core Architecture

### Main Components
- `main.py` - Primary entry point for running pruning experiments on LLaMA models
- `main_opt.py` - Entry point for pruning OPT models
- `lib/` - Core pruning algorithms and utilities:
  - `prune.py` - Main pruning functions (Wanda, magnitude, SparseGPT)
  - `sparsegpt.py` - SparseGPT implementation
  - `eval.py` - Perplexity and zero-shot evaluation
  - `data.py` - Dataset loading and calibration data sampling
  - `layerwrapper.py` - Layer wrapping utilities for pruning

### Specialized Components
- `lora_ft/` - LoRA fine-tuning for pruned models
- `dense_ft/` - Dense fine-tuning with sparse trainer (drop-in replacement for HuggingFace Trainer)
- `image_classifiers/` - Image classifier pruning implementation
- `scripts/` - Bash scripts for reproducing paper results

### Documentation
- `wanda.pdf` - Research paper with detailed methodology and results
- `README.md` - Main project documentation
- `INSTALL.md` - Installation instructions

## Common Commands

### Environment Setup
```bash
conda create -n prune_llm python=3.9
conda activate prune_llm
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install transformers==4.28.0 datasets==2.11.0 wandb sentencepiece accelerate==0.18.0
```

### Running Pruning Experiments

#### Basic Wanda Pruning (50% sparsity)
```bash
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_7b/unstructured/wanda/
```

#### LLaMA-2 Models
```bash
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama2_7b/unstructured/wanda/
```

#### Structured Sparsity (2:4 or 4:8)
```bash
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/llama_7b/2-4/wanda/
```

#### Zero-shot Evaluation
Add `--eval_zero_shot` flag to any pruning command to run zero-shot tasks in addition to WikiText perplexity.

### LoRA Fine-tuning
```bash
cd lora_ft
python finetune_lm.py \
    --model_name_or_path [PATH to pruned model] \
    --config_name "decapoda-research/llama-7b-hf" \
    --dataset_name c4 \
    --num_train_epochs 1 \
    --block_size 1024 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --max_train_samples 30000 \
    --output_dir [PATH to save LoRA weights]
```

### Dense Fine-tuning
The `dense_ft/` directory provides a sparse trainer that maintains sparsity during fine-tuning:
- `sparse_trainer.py` - Drop-in replacement for HuggingFace Trainer that zeros gradients for pruned weights
- Use when you want to fine-tune pruned models while preserving the sparsity pattern

### Reproducing Paper Results
Use the scripts in `scripts/` directory:
- `scripts/llama_7b.sh` - LLaMA-7B experiments
- `scripts/llama_13b.sh` - LLaMA-13B experiments
- `scripts/llama_30b.sh` - LLaMA-30B experiments
- `scripts/llama_65b.sh` - LLaMA-65B experiments
- `scripts/ablate_weight_update.sh` - Weight update ablation study

## Key Arguments

### Model and Data
- `--model` - HuggingFace model identifier or local path
- `--cache_dir` - Directory for storing model weights (default: "llm_weights")
- `--nsamples` - Number of calibration samples (default: 128)

### Pruning Configuration
- `--prune_method` - Pruning algorithm: ["magnitude", "wanda", "sparsegpt", "ablate_*"]
- `--sparsity_ratio` - Percentage of weights to prune (0.0-1.0)
- `--sparsity_type` - Sparsity pattern: ["unstructured", "2:4", "4:8"]
- `--use_variant` - Use Wanda variant from appendix

### Output
- `--save` - Directory to save pruning results and metrics
- `--save_model` - Directory to save the pruned model weights
- `--eval_zero_shot` - Run zero-shot evaluation tasks

## Model Support

- **LLaMA/LLaMA-2** (main focus): 7B, 13B, 30B, 65B, 70B variants
- **OPT**: Use `main_opt.py` instead of `main.py`
- **Image Classifiers**: See `image_classifiers/` directory

## Dependencies

This codebase requires specific versions:
- PyTorch 1.10.1 with CUDA 11.3
- Transformers 4.28.0 (specific version needed for LLaMA tokenizer)
- Datasets 2.11.0
- Accelerate 0.18.0