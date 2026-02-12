"""
Reformat external evaluation datasets for dialect embedding evaluation.

Supports multiple formats:
- DSL-ML: Tab-separated text with dialect labels
- PAN (authorship): Folder structure with authors and texts
- Reddit-style: Already in the expected format

Output format is a flat list of entries with:
- sentence: The text
- tags: List of metadata tags [dialect, author, etc.]
- input_ids: Tokenized sequence (added after reformatting)

Usage:
    python reformat_eval_data.py \
        --input data/vardial/test.txt \
        --output data/eval/vardial_test.json \
        --format dsl \
        --model FacebookAI/roberta-base
"""

import argparse
import json
import os
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reformat external evaluation datasets."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file.",
    )
    parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=["dsl", "pan", "madar", "reddit"],
        help="Input format type.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="FacebookAI/roberta-base",
        help="Model name for tokenizer.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum token sequence length.",
    )
    return parser.parse_args()


def reformat_dsl(input_path: str) -> list[dict[str, Any]]:
    """Reformat DSL-ML format (tab-separated: text<TAB>label1,label2,...).

    Args:
        input_path: Path to TSV file.

    Returns:
        List of entry dictionaries.
    """
    entries = []
    with open(input_path) as f:
        for line in tqdm(f, desc="Reading DSL data"):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            text = parts[0]
            tags = parts[1].split(",")

            entries.append(
                {
                    "sentence": text,
                    "tags": tags,
                }
            )

    return entries


def reformat_madar(input_path: str) -> list[dict[str, Any]]:
    """Reformat MADAR dataset format.

    Args:
        input_path: Path to TSV file.

    Returns:
        List of entry dictionaries.
    """

    entries = []
    with open(input_path) as f:
        for line in tqdm(f, desc="Reading MADAR data"):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            text = parts[0]
            dialect = parts[1]

            entries.append(
                {
                    "sentence": text,
                    "tags": [dialect],
                }
            )
    return entries


def reformat_pan(input_dir: str) -> tuple[list[dict], list[dict]]:
    """Reformat PAN authorship verification format.

    Expected structure:
        input_dir/
            problem001/
                ground-truth.json
                candidate001/
                    known01.txt
                    known02.txt
                unknown/
                    unknown01.txt

    Args:
        input_dir: Path to PAN dataset directory.

    Returns:
        Tuple of (train_entries, eval_entries).
    """
    train_entries = []
    eval_entries = []

    for problem_folder in os.listdir(input_dir):
        problem_path = os.path.join(input_dir, problem_folder)
        if not os.path.isdir(problem_path):
            continue

        # Load ground truth for unknown texts
        gt_path = os.path.join(problem_path, "ground-truth.json")
        text_to_author = {}
        if os.path.exists(gt_path):
            with open(gt_path) as f:
                gt_data = json.load(f).get("ground_truth", [])
            for entry in gt_data:
                text_to_author[entry["unknown-text"]] = entry["true-author"]

        # Process candidate folders
        for candidate_folder in os.listdir(problem_path):
            candidate_path = os.path.join(problem_path, candidate_folder)
            if not os.path.isdir(candidate_path):
                continue

            is_unknown = "unknown" in candidate_folder.lower()

            for filename in os.listdir(candidate_path):
                if not filename.endswith(".txt"):
                    continue

                file_path = os.path.join(candidate_path, filename)
                with open(file_path) as f:
                    lines = [l.strip() for l in f if l.strip()]

                for idx, line in enumerate(lines):
                    if is_unknown:
                        # Evaluation data
                        author = text_to_author.get(filename, "unknown")
                        entry = {
                            "sentence": line,
                            "tags": [problem_folder, author, f"line_{idx}"],
                        }
                        eval_entries.append(entry)
                    else:
                        # Training data (known authors)
                        entry = {
                            "sentence": line,
                            "tags": [
                                problem_folder,
                                candidate_folder,
                                filename,
                                f"line_{idx}",
                            ],
                        }
                        train_entries.append(entry)

    return train_entries, eval_entries


def reformat_reddit(input_dir: str) -> list[dict[str, Any]]:
    """Reformat Reddit-style data that's already been processed.

    Expected format: List of users, each with comments, each with sentences.

    Args:
        input_dir: Directory containing JSON files.

    Returns:
        Flat list of entry dictionaries.
    """
    entries = []

    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue

        dialect_name = filename.replace(".json", "")
        file_path = os.path.join(input_dir, filename)

        with open(file_path) as f:
            data = json.load(f)

        for user_data in data:
            for comment in user_data:
                for idx, sentence_data in enumerate(comment):
                    text = sentence_data.get("text", "")
                    if isinstance(text, list):
                        text = " ".join(text)

                    user = sentence_data.get("user", "unknown")

                    entry = {
                        "sentence": text,
                        "tags": [dialect_name, user, f"line_{idx}"],
                    }

                    # Preserve existing fields if present
                    if "text_ids" in sentence_data:
                        entry["input_ids"] = sentence_data["text_ids"]
                    if "feature" in sentence_data:
                        entry["feature"] = sentence_data["feature"]

                    entries.append(entry)

    return entries


def tokenize_entries(
    entries: list[dict],
    tokenizer,
    max_length: int,
) -> list[dict]:
    """Add tokenization to entries that don't have it.

    Args:
        entries: List of entry dictionaries.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.

    Returns:
        Entries with 'input_ids' field added.
    """
    for entry in tqdm(entries, desc="Tokenizing"):
        if "input_ids" not in entry:
            entry["input_ids"] = tokenizer(
                entry["sentence"],
                max_length=max_length,
                truncation=True,
            ).input_ids
    return entries


def main():
    args = parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Reformat based on input type
    print(f"Reformatting {args.format} data from {args.input}")

    if args.format == "dsl":
        entries = reformat_dsl(args.input)

    elif args.format == "madar":
        entries = reformat_madar(args.input)

    elif args.format == "pan":
        train_entries, eval_entries = reformat_pan(args.input)
        # Save both if PAN format
        entries = eval_entries  # Default to eval
        if train_entries:
            train_output = args.output.replace(".json", "_train.json")
            train_entries = tokenize_entries(train_entries, tokenizer, args.max_length)
            os.makedirs(os.path.dirname(train_output) or ".", exist_ok=True)
            with open(train_output, "w") as f:
                json.dump(train_entries, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(train_entries)} train entries -> {train_output}")

    elif args.format == "reddit":
        entries = reformat_reddit(args.input)

    # Tokenize
    entries = tokenize_entries(entries, tokenizer, args.max_length)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(entries)} entries -> {args.output}")


if __name__ == "__main__":
    main()
