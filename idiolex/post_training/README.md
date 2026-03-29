# Post-Training with IdioleX Alignment

This module enables **LLM post-training using IdioleX representations** as an additional alignment signal.

The goal is to encourage generated outputs to better match **target dialect, style, and idiolectal variation**, beyond standard semantic alignment.

---

## Overview

Given:
- a base language model (e.g., LLaMA, Allam, SILMA)
- instruction-tuning data
- a trained IdioleX model

we optimize the LLM with an additional **style similarity objective**.
To account for length discrepencies, rather than simply calculating IdioleX representations over model outputs, we use a projection layer over the hidden state of the LLM and maximize similarity over that projection and the IdioleX representation of the ground truth.
This encourages outputs that are:
- stylistically aligned
- dialect-consistent
- closer to human-written text in idiolectal space

---
## Usage

### 1. Train IdioleX Model

See `idiolex/src/README.md`


### 2. Run post-training (SFT + IdioleX alignment)

```bash
accelerate launch \
  --config_file idiolex/post_training/accelerate_config.yaml \
  idiolex/post_training/sft.py \
  --model meta-llama/Llama-3.1-8B-Instruct \ # Or another model
  --train_jsonl data/post_training/train.jsonl \
  --reward_checkpoint models/idiolex_model/checkpoint.pth \
  --output_dir models/sft_idiolex
```

Training data should be in the format:

```
{"prompt": [{"role": "user", "content": "..."}], "ground_truth": "..."}
```

### 3. Evaluate

```
python -m idiolex.post_training.predict \
  --model models/sft_idiolex/final \
  --input eval/test.csv \
  --output eval/preds.tsv
```

We evaluate via the [AMIYA](https://aclanthology.org/2026.vardial-1.1.pdf) shared task from VarDial 2025.

---
## Hyperparameters

### Model

| Hyperparameter | Description | Example / Default |
|----------------|------------|------------------|
| `model` | Base language model | `meta-llama/Llama-3.1-8B-Instruct` |
| `reward_checkpoint` | Path to trained IdioleX model | `models/idiolex/checkpoint.pth` |
| `max_seq_length` | Maximum sequence length | `512` |

---

### Training

| Hyperparameter | Description | Default |
|----------------|------------|--------|
| `num_train_epochs` | Number of training epochs | `3` |
| `per_device_train_batch_size` | Batch size per GPU | `1–4` |
| `gradient_accumulation_steps` | Effective batch size scaling | `4–16` |
| `learning_rate` | Learning rate | `2e-5` |
| `lr_scheduler_type` | Learning rate schedule | `cosine` |
| `warmup_ratio` | Fraction of warmup steps | `0.03` |

---

### IdioleX Alignment

| Hyperparameter | Description | Default |
|----------------|------------|--------|
| `idiolex_weight` | Weight of IdioleX alignment loss | `0.1–1.0` |
| `similarity_metric` | Similarity function | `cosine` |
| `normalize_embeddings` | Use L2-normalized embeddings | `True` |

---

### Regularization

| Hyperparameter | Description | Default |
|----------------|------------|--------|
| `weight_decay` | Weight decay | `0.0` |
| `max_grad_norm` | Gradient clipping | `1.0` |

---

### Efficiency

| Hyperparameter | Description | Default |
|----------------|------------|--------|
| `fp16` / `bf16` | Mixed precision training | `True` |
| `gradient_checkpointing` | Memory optimization | `True` |
| `lora_rank` | LoRA rank | `8–64` |
| `lora_alpha` | LoRA scaling factor | `16–64` |


