import argparse
import json
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(round(seconds))))

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tokenize_dataset(dataset, tokenizer, max_length: int):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy

def train_model(
    model,
    tokenizer,
    dataset,
    output_dir: Path,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    accumulation_steps: int,
    weight_decay: float,
    patience_limit: int,
    label_smoothing: float,
    verbose: bool,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        dataset["dev"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_dev_loss = float("inf")
    patience = 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        total_train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if verbose and epoch == 0 and step == 1:
                print(f"input_ids shape: {input_ids.shape}")
                print(f"attention_mask shape: {attention_mask.shape}")
                print(f"labels shape: {labels.shape}")

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            loss = criterion(logits, labels)
            total_train_loss += loss.item()
            (loss / accumulation_steps).backward()

            if step % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        train_time = format_time(time.time() - start_time)

        dev_loss, dev_accuracy = evaluate(model, dev_loader, criterion, device)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_accuracy,
            "train_time": train_time,
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"dev_loss={dev_loss:.4f} | "
            f"dev_acc={dev_accuracy:.4f} | "
            f"time={train_time}"
        )

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience = 0

            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")

            with open(output_dir / "best_metrics.json", "w") as f:
                json.dump(epoch_metrics, f, indent=2)

            print("Saved new best model.")
        else:
            patience += 1
            print(f"No improvement. Patience: {patience}/{patience_limit}")

            if patience >= patience_limit:
                print("Early stopping triggered.")
                break

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history

def main():
    parser = argparse.ArgumentParser(description="Train a closed-set text classifier.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    ###
    # Data files should be jsonl with the fields: {"text": "example text", "label": 0}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "dev": args.dev_file,
        },
    )
    dataset = tokenize_dataset(dataset, tokenizer, args.max_length)

    num_labels = len(set(dataset["train"]["labels"].tolist()))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
    )

    label_map = {int(i): int(i) for i in range(num_labels)}
    with open(output_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Training model: {args.model_name}")
    print(f"Number of labels: {num_labels}")

    train_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=output_dir,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        weight_decay=args.weight_decay,
        patience_limit=args.patience,
        label_smoothing=args.label_smoothing,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
