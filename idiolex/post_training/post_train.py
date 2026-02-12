"""Post-train an LLM with using IdioleX similarity-based rewards.

This script fine-tunes a language model using Group Relative Policy
Optimization (GRPO) with rewards from a trained IdioleX embedding model.

Usage:
    python post_training/post_train.py \
        --reward_checkpoint models/dialect_model/checkpoint.pth \
        --llm_model silma-ai/SILMA-9B-Instruct-v1.0 \
        --dataset_name UBC-NLP/palm \
        --output_dir models/post-train_finetuned
"""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from post_training.reward_model import RewardModel

os.environ["WANDB_PROJECT"] = "Idiolex Post-train"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Post-train an LLM using IdioleX embedding rewards."
    )

    # Model arguments
    parser.add_argument(
        "--reward_checkpoint",
        type=str,
        required=True,
        help="Path to trained IdioleX embedding checkpoint.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="silma-ai/SILMA-9B-Instruct-v1.0",
        help="Pre-trained LLM to fine-tune.",
    )

    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="UBC-NLP/palm",
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--prompt_field",
        type=str,
        default="instruction",
        help="Field name for prompts in dataset.",
    )
    parser.add_argument(
        "--ground_truth_field",
        type=str,
        default="output",
        help="Field name for ground truth completions in dataset.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (None for all).",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./post-train_output",
        help="Output directory for fine-tuned model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="Number of generations per prompt for GRPO.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )

    # Memory Efficiency
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for parameter-efficient fine-tuning.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )

    return parser.parse_args()


def prepare_dataset(args: argparse.Namespace):
    """Load and prepare the dataset for GRPO training.

    Args:
        args: Command line arguments.

    Returns:
        Prepared dataset with 'prompt' and 'ground_truth' fields.
    """
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    def ensure_string(example):
        prompt = example.get("prompt", "")
        ground_truth = example.get("ground_truth", "")
        if isinstance(prompt, list):
            prompt = " ".join(str(p) for p in prompt)
        elif not isinstance(prompt, str):
            prompt = str(prompt)
        if isinstance(ground_truth, list):
            ground_truth = " ".join(str(g) for g in ground_truth)
        elif not isinstance(ground_truth, str):
            ground_truth = str(ground_truth)
        return {"prompt": prompt, "ground_truth": ground_truth}

    # Rename prompt field if needed
    if args.prompt_field != "prompt":
        dataset = dataset.rename_column(args.prompt_field, "prompt")
    if args.ground_truth_field != "ground_truth":
        dataset = dataset.rename_column(args.ground_truth_field, "ground_truth")

    dataset = dataset.map(ensure_string)

    # Keep only the prompt and ground_truth fields
    dataset = dataset.select_columns(["prompt", "ground_truth"])

    print(f"Dataset size: {len(dataset)}")
    return dataset


def main():
    args = parse_args()

    # Determine device
    reward_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using reward device: {reward_device}")

    # Load reward model
    print(f"Loading reward model from: {args.reward_checkpoint}")
    reward_model = RewardModel.from_checkpoint(
        args.reward_checkpoint, device=reward_device
    )
    reward_fn = reward_model.get_reward_function()

    # Load LLM
    print(f"Loading LLM: {args.llm_model}")
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        llm_model = get_peft_model(llm_model, lora_config)

    # Ensure pad token is set
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    # Prepare dataset
    dataset = prepare_dataset(args)

    # Configure GRPO
    training_config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_completion_length=args.max_length,
        num_generations=args.num_generations,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=os.path.basename(args.output_dir),
    )

    # Initialize trainer
    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=llm_model,
        args=training_config,
        processing_class=llm_tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_fn,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    llm_tokenizer.save_pretrained(args.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
