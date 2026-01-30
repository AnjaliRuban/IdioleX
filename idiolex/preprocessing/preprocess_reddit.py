"""
Preprocess Reddit/subreddit data for dialect embedding training.

This script takes raw subreddit data (author -> comments dict) and:
1. Filters and cleans text
2. Tokenizes with the model's tokenizer
3. Creates train/dev/test splits

Usage:
    python preprocess_reddit.py \
        --input_dir data/raw \
        --output_dir data/processed \
        --model FacebookAI/roberta-base \
        --lang_filter ar  # optional language filter
"""

import argparse
import json
import os
import random
import re
from typing import Optional

from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Reddit data for dialect embedding training."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing raw JSON files (author -> comments dict).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed data.",
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
    parser.add_argument(
        "--min_words",
        type=int,
        default=5,
        help="Minimum words per sentence.",
    )
    parser.add_argument(
        "--min_sentences",
        type=int,
        default=2,
        help="Minimum sentences per comment.",
    )
    parser.add_argument(
        "--min_comments",
        type=int,
        default=2,
        help="Minimum comments per user.",
    )
    parser.add_argument(
        "--train_users",
        type=int,
        default=200,
        help="Number of users for training set (will receive LLM features).",
    )
    parser.add_argument(
        "--dev_users",
        type=int,
        default=5,
        help="Number of users for dev set.",
    )
    parser.add_argument(
        "--test_users",
        type=int,
        default=5,
        help="Number of users for test set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    """Clean a single piece of text.
    
    - Removes URLs
    - Removes HTML entities
    - Normalizes whitespace
    """
    # Remove URLs
    text = re.sub(r"http\S+", "URL", text)
    # Remove HTML entities like &amp;
    text = re.sub(r"&\w+;?", "", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return re.split(r"[.!?\n]+", text)


def process_comment(
    comment_text: str,
    min_words: int,
) -> list[dict]:
    """Process a single comment into sentence-level data.
    
    Args:
        comment_text: Raw comment text.
        min_words: Minimum words per sentence.
    
    Returns:
        List of sentence dictionaries with 'text' field.
    """
    sentences = []
    for sent in split_into_sentences(comment_text):
        cleaned = clean_text(sent)
        words = cleaned.split()
        if len(words) >= min_words:
            sentences.append({"text": words})
    return sentences


def process_user(
    user: str,
    comments: list[dict],
    min_words: int,
    min_sentences: int,
    min_comments: int,
) -> Optional[list[list[dict]]]:
    """Process all comments from a single user.
    
    Args:
        user: Username.
        comments: List of comment dictionaries with 'text' field.
        min_words: Minimum words per sentence.
        min_sentences: Minimum sentences per comment.
        min_comments: Minimum comments per user.
    
    Returns:
        List of comments, where each comment is a list of sentence dicts,
        or None if user doesn't meet minimum requirements.
    """
    if user == "[deleted]":
        return None
    
    processed_comments = []
    for comment in comments:
        text = comment.get("text", "")
        if text in ["[deleted]", "[removed]"]:
            continue
        
        sentences = process_comment(text, min_words)
        if len(sentences) >= min_sentences:
            # Add user info to each sentence
            for sent in sentences:
                sent["user"] = user
            processed_comments.append(sentences)
    
    if len(processed_comments) >= min_comments:
        return processed_comments
    return None


def tokenize_data(
    data: list[list[list[dict]]],
    tokenizer,
    max_length: int,
) -> list[list[list[dict]]]:
    """Tokenize all text data.
    
    Args:
        data: Nested list [users][comments][sentences].
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
    
    Returns:
        Same structure with 'text_ids' added to each sentence.
    """
    for user_data in tqdm(data, desc="Tokenizing"):
        for comment in user_data:
            for sentence in comment:
                text = " ".join(sentence["text"])
                sentence["text_ids"] = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                ).input_ids
    return data


def process_file(
    input_path: str,
    tokenizer,
    args: argparse.Namespace,
) -> dict[str, list]:
    """Process a single input file and create splits.
    
    Args:
        input_path: Path to input JSON file.
        tokenizer: HuggingFace tokenizer.
        args: Command line arguments.
    
    Returns:
        Dictionary with 'train', 'dev', 'test' splits.
    """
    with open(input_path) as f:
        raw_data = json.load(f)
    
    # Process all users
    processed_users = []
    for user, comments in tqdm(raw_data.items(), desc="Processing users"):
        user_data = process_user(
            user,
            comments,
            args.min_words,
            args.min_sentences,
            args.min_comments,
        )
        if user_data is not None:
            processed_users.append(user_data)
    
    print(f"  Processed {len(processed_users)} users from {len(raw_data)} total")
    
    # Shuffle and split
    random.shuffle(processed_users)
    
    test_end = args.test_users
    dev_end = test_end + args.dev_users
    train_end = min(dev_end + args.train_users, len(processed_users))
    
    splits = {
        "test_data": processed_users[:test_end],
        "dev_data": processed_users[test_end:dev_end],
        "train_data": processed_users[dev_end:train_end],
        "pretrain_data": processed_users[dev_end:], # Pretrain set includes train set
    }
    
    # Tokenize each split
    for split_name, split_data in splits.items():
        print(f"  Tokenizing {split_name}: {len(split_data)} users")
        tokenize_data(split_data, tokenizer, args.max_length)
    
    return splits


def main():
    args = parse_args()
    random.seed(args.seed)
    
    # Create output directories
    for split in ["pretrain_data", "train_data", "dev_data", "test_data"]:
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Process each input file
    for filename in os.listdir(args.input_dir):
        if not filename.endswith(".json"):
            continue
        
        print(f"\nProcessing {filename}...")
        input_path = os.path.join(args.input_dir, filename)
        
        # Get base name (e.g., "argentina" from "argentina_data.json")
        base_name = filename.replace("_data.json", "").replace(".json", "")
        
        splits = process_file(input_path, tokenizer, args)
        
        # Save splits
        for split_name, split_data in splits.items():
            output_path = os.path.join(args.output_dir, split_name, f"{base_name}.json")
            with open(output_path, "w") as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            print(f"  Saved {split_name}: {len(split_data)} users -> {output_path}")


if __name__ == "__main__":
    main()