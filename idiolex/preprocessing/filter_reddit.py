"""
Filter raw Pushshift Reddit data dumps by language.

This script reads compressed (.zst) Reddit comment dumps and filters
comments by language using fastText language identification.

Usage:
    python filter_reddit.py \
        --input_dir data/pushshift/Spanish/ \
        --output_dir data/raw/ \
        --lang_codes es

Requirements:
    - fasttext
    - zstandard
    - Download lid.176.bin from https://fasttext.cc/docs/en/language-identification.html
"""

import argparse
import json
import os
from collections import Counter, defaultdict

from tqdm import tqdm

import fasttext
import zstandard

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Reddit dumps by language."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing .zst Reddit dump files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for filtered data.",
    )
    parser.add_argument(
        "--lang_codes",
        type=str,
        nargs="+",
        required=True,
        help="FastText language codes to keep (e.g., 'es' for Spanish, 'ar' for Arabic).",
    )
    parser.add_argument(
        "--fasttext_model",
        type=str,
        default="lid.176.bin",
        help="Path to fastText language ID model.",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=10,
        help="Minimum comment length in characters.",
    )
    return parser.parse_args()


def read_zst_lines(file_path: str):
    """Generator to read lines from a .zst compressed file.
    
    Args:
        file_path: Path to .zst file.
    
    Yields:
        Decoded lines from the file.
    """
    with open(file_path, "rb") as f:
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(f)
        buffer = ""
        
        while True:
            chunk = reader.read(2**24)
            if not chunk:
                break
            
            try:
                decoded = chunk.decode("utf-8")
            except UnicodeDecodeError:
                # Try with larger chunk for multi-byte characters
                chunk += reader.read(2**20)
                try:
                    decoded = chunk.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            
            lines = (buffer + decoded).split("\n")
            buffer = lines[-1]
            
            for line in lines[:-1]:
                if line.strip():
                    yield line.strip()
        
        reader.close()


def filter_dump(
    input_path: str,
    lang_model,
    lang_codes: set[str],
    min_length: int,
) -> tuple[dict, Counter]:
    """Filter a single dump file by language.
    
    Args:
        input_path: Path to .zst dump file.
        lang_model: Loaded fastText model.
        lang_codes: Set of language codes to keep.
        min_length: Minimum comment length.
    
    Returns:
        Tuple of (author -> comments dict, author counts).
    """
    data = defaultdict(list)
    author_counts = Counter()
    total = 0
    kept = 0
    
    for line in tqdm(read_zst_lines(input_path), desc=os.path.basename(input_path)):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        total += 1
        
        author = entry.get("author", "")
        text = entry.get("body", "")
        
        # Skip deleted/removed
        if text in ["[deleted]", "[removed]"]:
            continue
        if author in ["[deleted]", "[removed]", "AutoModerator"]:
            continue
        if len(text) < min_length:
            continue
        
        # Language filter
        clean_text = text.replace("\n", " ").replace("\r", " ")
        pred = lang_model.predict(clean_text)[0][0]
        lang_code = pred.replace("__label__", "")
        
        if lang_code not in lang_codes:
            continue
        
        kept += 1
        data[author].append({
            "text": text,
            "id": entry.get("id", ""),
            "subreddit": entry.get("subreddit", ""),
        })
        author_counts[author] += 1
    
    print(f"  Kept {kept}/{total} comments ({100*kept/max(total,1):.1f}%)")
    return dict(data), author_counts


def main():    
    args = parse_args()
    
    # Load language model
    print(f"Loading fastText model: {args.fasttext_model}")
    lang_model = fasttext.load_model(args.fasttext_model)
    lang_codes = {f"__label__{c}" if not c.startswith("__label__") else c for c in args.lang_codes}
    print(f"Filtering for languages: {lang_codes}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each dump file
    for filename in os.listdir(args.input_dir):
        if not filename.endswith(".zst"):
            continue
        if filename.endswith(".zst.part"):
            continue
        
        print(f"\nProcessing {filename}...")
        input_path = os.path.join(args.input_dir, filename)
        
        # Get subreddit name
        subreddit = filename.split("_")[0].lower()
        
        data, author_counts = filter_dump(
            input_path,
            lang_model,
            lang_codes,
            args.min_length,
        )
        
        print(f"  {len(data)} authors, {sum(author_counts.values())} comments")
        
        # Save data
        output_path = os.path.join(args.output_dir, f"{subreddit}_data.json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved -> {output_path}")
        
        # Save author stats
        stats_path = os.path.join(args.output_dir, f"{subreddit}_authors.json")
        with open(stats_path, "w") as f:
            json.dump(author_counts.most_common(), f, indent=2)


if __name__ == "__main__":
    main()