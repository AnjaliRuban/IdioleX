"""
Generate predictions from a model given a CSV of prompts.

Usage:
    # Base HF model:
    python generate_predictions.py --model Qwen/Qwen2.5-7B-Instruct \
        --input eval/Egyptian.csv --output eval/Egyptian_preds.tsv

    # SFT adapter:
    python generate_predictions.py --model models/sft_dialect/final \
        --input eval/Egyptian.csv --output eval/Egyptian_preds.tsv

    # GRPO adapter:
    python generate_predictions.py --model models/grpo_dialect/final \
        --input eval/Egyptian.csv --output eval/Egyptian_preds.tsv

    # With 4-bit quantization:
    python generate_predictions.py --model Qwen/Qwen2.5-7B-Instruct \
        --input eval/Egyptian.csv --output eval/Egyptian_preds.tsv --load_in_4bit

    # Batch all eval files:
    for f in eval/Egyptian.csv eval/Morrocan.csv eval/Palestinian.csv eval/Saudi.csv eval/Syrian.csv; do
        name=$(basename "$f" .csv)
        python generate_predictions.py --model models/sft_dialect/final \
            --input "$f" --output "eval/${name}_preds.tsv"
    done
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_and_tokenizer(model_path: str, load_in_4bit: bool = False):
    """Load model, handling both HF model IDs and LoRA adapter directories."""

    is_adapter = os.path.isfile(os.path.join(model_path, "adapter_config.json"))

    load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    if is_adapter:
        from peft import PeftConfig, PeftModel

        adapter_config = PeftConfig.from_pretrained(model_path)
        base_model_name = adapter_config.base_model_name_or_path
        print(f"Loading base model: {base_model_name}")
        print(f"Loading adapter from: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)

        # Apply adapter (GRPO or SFT)
        print(f"Applying adapter: {model_path}")

        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        print("Adapter merged.")
    else:
        print(f"Loading model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    return model, tokenizer


def generate_predictions(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> list[str]:
    """Generate predictions for a list of prompts."""

    predictions = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]

        # Format as chat messages
        batch_messages = [
            [{"role": "user", "content": p}] for p in batch_prompts
        ]

        # Apply chat template
        batch_texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in batch_messages
        ]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated tokens (strip the prompt)
        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].shape[0]
            generated = output[prompt_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            predictions.append(text)

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate predictions from prompts CSV")
    parser.add_argument("--model", type=str, required=True,
                        help="HF model ID or path to adapter directory")
    parser.add_argument("--input", type=str, required=True,
                        help="Input CSV with 'prompt' column")
    parser.add_argument("--output", type=str, required=True,
                        help="Output TSV with prompt and prediction columns")
    parser.add_argument("--prompt_column", type=str, default=None,
                        help="Name of prompt column (auto-detected if not set)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for quick testing)")
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    args = parser.parse_args()

    prompts = []
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames

        # Auto-detect prompt column
        if args.prompt_column:
            col = args.prompt_column
        else:
            for candidate in ["prompt", "Prompt", "text", "Text", "instruction", "input", "question"]:
                if candidate in fields:
                    col = candidate
                    break
            else:
                # Fall back to first column
                col = fields[0]

        print(f"Using column '{col}' from {args.input}")

        for row in reader:
            prompts.append(row[col])

    if args.max_samples:
        prompts = prompts[: args.max_samples]

    print(f"Loaded {len(prompts)} prompts")

    model, tokenizer = load_model_and_tokenizer(args.model, args.load_in_4bit)

    predictions = generate_predictions(
        model,
        tokenizer,
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["prompt", "prediction"])
        for prompt, pred in zip(prompts, predictions):
            # Clean tabs/newlines from predictions to keep TSV valid
            clean_pred = pred.replace("\t", " ").replace("\n", " ").replace("\r", "")
            clean_prompt = prompt.replace("\t", " ").replace("\n", " ").replace("\r", "")
            writer.writerow([clean_prompt, clean_pred])

    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
