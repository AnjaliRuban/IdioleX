# Post-Training with Dialect Embedding Rewards

This module enables post-training with rewards derived from trained dialect embedding models.

## Overview

The reward signal measures how well a generated completion matches the dialectal/stylistic characteristics of the prompt, encouraging the LLM to maintain consistent style.

```
Prompt → LLM → Completion
           ↓
    ┌──────────────────┐
    │   Reward Model   │  (Dialect Embedding Similarity)
    │                  │
    │  embed(ground_truth) · │
    │  embed(completion)│
    └──────────────────┘
           ↓
        Reward → GRPO Update
```

## Usage

### 1. Train a Dialect Embedding Model

First, train the dialect embedding model using the main training script:

```bash
torchrun --nproc_per_node=4 main.py \
    --tag dialect_model \
    --train_data data/processed/train_data \
    --dev_data data/processed/dev_data \
    --mean_center \
    --layerwise_pooling
```

### 2. Run RLHF Training

Use the trained embedding model as a reward function:

```bash
python rlhf/post_train.py \
    --reward_checkpoint models/dialect_model/checkpoint.pth \
    --llm_model silma-ai/SILMA-9B-Instruct-v1.0  \
    --dataset_name UBC-NLP/palm \
    --output_dir models/rlhf_finetuned \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --num_epochs 1
```

### 3. Use Reward Model Directly

You can also use the reward model programmatically:

```python
from rlhf import RewardModel

# Load from checkpoint
reward_model = RewardModel.from_checkpoint(
    "models/dialect_model/checkpoint.pth",
    device="cuda"
)

# Get reward function for TRL
reward_fn = reward_model.get_reward_function()

# Or compute rewards directly
rewards = reward_model(
    prompts=["How are you?", "What's up?"],
    completions=["I am fine, thank you.", "Not much, just chilling."]
)
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--reward_checkpoint` | (required) | Path to dialect embedding checkpoint |
| `--llm_model` | silma-ai/SILMA-9B-Instruct-v1.0  | LLM to fine-tune |
| `--dataset_name` | UBC-NLP/palm | HuggingFace dataset |
| `--prompt_field` | instruction | Field containing prompts |
| `--ground_truth_field` | output | Field containing ground truth completions |
| `--batch_size` | 4 | Training batch size |
| `--learning_rate` | 1e-5 | Learning rate |
| `--num_generations` | 4 | Generations per prompt (GRPO) |
| `--max_length` | 512 | Maximum sequence length |

## Requirements

Additional dependencies for RLHF:
```
trl>=0.7.0
datasets>=2.0.0
```

## How It Works

1. **Reward Computation**: For each prompt-completion pair, we:
   - Encode both texts using the trained dialect embedding model
   - Apply any centering/normalization from training
   - Compute cosine similarity as the reward

2. **GRPO Training**: Group Relative Policy Optimization:
   - Generates multiple completions per prompt
   - Ranks them by reward
   - Updates policy to prefer higher-reward completions

3. **Style Transfer**: The reward encourages completions that match the dialectal characteristics present in the prompt, enabling style-consistent generation.