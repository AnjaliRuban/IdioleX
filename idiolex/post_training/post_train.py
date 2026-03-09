"""
TRL GRPO training with IdioleX embedding similarity reward.

Usage:
    # Single GPU (testing):
    python train_grpo.py

    # Multi-GPU with accelerate:
    accelerate launch --config_file accelerate_config.yaml train_grpo.py

    # With vLLM generation (recommended for 9B models):
    accelerate launch --config_file accelerate_config.yaml train_grpo.py --use_vllm
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoModel, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


# ---------------------------------------------------------------------------
# IdioleX reward model (self-contained, same logic as idiolex_reward.py)
# ---------------------------------------------------------------------------

def average_pool(tokens, embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def last_token_pool(hidden_states, attention_mask):
    seq_len = attention_mask.sum(dim=1) - 1
    return hidden_states[torch.arange(hidden_states.size(0)), seq_len]


class IdioleXRewardModel(nn.Module):
    def __init__(self, model, tokenizer, embedding_model=None, centering_model=None, max_length=512):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.centering_model = centering_model
        self.max_length = max_length
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        if self.embedding_model is not None:
            self.embedding_model.eval()
            for p in self.embedding_model.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def encode(self, texts: list[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if self.embedding_model is not None:
            embeddings = average_pool(
                tokens=input_ids,
                embeddings=self.embedding_model(
                    hidden_states=outputs.hidden_states,
                    attention_mask=attention_mask,
                ),
                attention_mask=attention_mask,
            )
        else:
            embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)

        if self.centering_model is not None:
            embeddings = self.centering_model.apply(embeddings)
        else:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    @torch.no_grad()
    def score(self, completion: str, ground_truth: str) -> float:
        device = next(self.model.parameters()).device
        comp_emb = self.encode([completion], device)
        gt_emb = self.encode([ground_truth], device)
        return torch.sum(comp_emb * gt_emb, dim=-1).item()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu"):
        from src.centering import MeanCenterer
        from src.layer_pool import LayerwiseAttention

        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        args = checkpoint["args"]

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        config = AutoConfig.from_pretrained(args.base_model)
        config.output_hidden_states = True

        model = AutoModel.from_config(config)
        model.load_state_dict(checkpoint["model"])
        model = model.to(device)

        embedding_model = None
        if args.layerwise_pooling and checkpoint.get("embedding_model"):
            num_layers = model.config.num_hidden_layers + 1
            embedding_model = LayerwiseAttention(num_layers=num_layers).to(device)
            embedding_model.load_state_dict(checkpoint["embedding_model"])

        centering_model = None
        if args.mean_center and checkpoint.get("centering_model"):
            centering_model = MeanCenterer(dim=config.hidden_size).to(device)
            centering_model.load_state_dict(checkpoint["centering_model"])

        return cls(model=model, tokenizer=tokenizer,
                   embedding_model=embedding_model, centering_model=centering_model)


# ---------------------------------------------------------------------------
# Global reward model singleton
# ---------------------------------------------------------------------------
_reward_model: IdioleXRewardModel | None = None


def get_reward_model(checkpoint_path: str, device: str = "cpu") -> IdioleXRewardModel:
    global _reward_model
    if _reward_model is None:
        print(f"[idiolex_reward] Loading checkpoint: {checkpoint_path}")
        _reward_model = IdioleXRewardModel.from_checkpoint(checkpoint_path, device=device)
        print("[idiolex_reward] Reward model loaded.")
    return _reward_model


# ---------------------------------------------------------------------------
# TRL-compatible reward function
# ---------------------------------------------------------------------------
# TRL GRPOTrainer calls: reward_func(completions, ground_truth=..., **kwargs)
#   completions: list[str] (standard) or list[list[dict]] (conversational)
#   ground_truth: list[str] from the dataset column
#   Returns: list[float]

def make_reward_func(checkpoint_path: str, device: str = "cpu"):
    """Create a closure over the checkpoint path for TRL's reward_funcs API."""

    def idiolex_reward(completions, ground_truth, **kwargs) -> list[float]:
        rm = get_reward_model(checkpoint_path, device)
        rewards = []
        for completion, gt in zip(completions, ground_truth):
            # Handle conversational format: extract text from message dict
            if isinstance(completion, list):
                text = completion[-1]["content"] if completion else ""
            elif isinstance(completion, dict):
                text = completion.get("content", "")
            else:
                text = str(completion)
            rewards.append(rm.score(text, gt))
        return rewards

    return idiolex_reward


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_and_prepare_data(data_source: str, prompt_field: str, gt_field: str,
                          max_samples: int | None = None, val_ratio: float = 0.02):
    """Load a HF dataset and format for TRL GRPOTrainer.

    TRL expects:
      - prompt: list of message dicts [{"role": "user", "content": "..."}]
      - ground_truth: str (passed to reward function as kwarg)
    """
    print(f"Loading dataset: {data_source}")
    ds = load_dataset(data_source, split="train")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def format_for_trl(example):
        prompt_text = example[prompt_field]
        if isinstance(prompt_text, list):
            prompt_text = " ".join(str(p) for p in prompt_text)
        gt_text = example[gt_field]
        if isinstance(gt_text, list):
            gt_text = " ".join(str(g) for g in gt_text)
        return {
            "prompt": [{"role": "user", "content": str(prompt_text)}],
            "ground_truth": str(gt_text),
        }

    ds = ds.map(format_for_trl, remove_columns=ds.column_names)

    if val_ratio > 0:
        split = ds.train_test_split(test_size=val_ratio, seed=42)
        return split["train"], split["test"]
    return ds, None



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="TRL GRPO training with IdioleX reward")

    # Model
    parser.add_argument("--model", type=str, default="silma-ai/SILMA-9B-Instruct-v1.0")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Data
    parser.add_argument("--dataset_name", type=str, default="UBC-NLP/palm")
    parser.add_argument("--prompt_field", type=str, default="instruction")
    parser.add_argument("--gt_field", type=str, default="output")
    parser.add_argument("--max_samples", type=int, default=None)

    # Reward
    parser.add_argument("--reward_checkpoint", type=str, required=True)
    parser.add_argument("--reward_device", type=str, default="cpu")

    # Training
    parser.add_argument("--output_dir", type=str, default="./grpo_output")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4,
                        help="GRPO group size (completions per prompt)")
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # vLLM
    parser.add_argument("--use_vllm", action="store_true", default=False,
                        help="Use vLLM for fast generation")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.7)

    # Logging
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default="Idiolex_Post-train")
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # Set wandb
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # ---- Data ----
    train_ds, val_ds = load_and_prepare_data(
        args.dataset_name, args.prompt_field, args.gt_field,
        args.max_samples,
    )

    print(f"Train: {len(train_ds)} samples")
    if val_ds:
        print(f"Val:   {len(val_ds)} samples")

    # ---- Reward function ----
    reward_func = make_reward_func(args.reward_checkpoint, args.reward_device)

    # ---- LoRA config ----
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        bias="none",
    )

    # ---- GRPO config ----
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_timeout=1800,
        # GRPO-specific
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        # vLLM
        use_vllm=args.use_vllm,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        # Logging / saving
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="wandb",
        run_name=args.run_name or os.path.basename(args.output_dir),
    )

    # ---- Trainer ----
    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        reward_funcs=reward_func,
        peft_config=peft_config,
    )

    # ---- Train ----
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"Training complete. Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
