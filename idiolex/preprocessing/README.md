# Data Preprocessing Pipeline

This directory contains scripts for preparing data for dialect embedding training.

## Pipeline Overview

```
Raw Reddit Dumps (.zst)
        │
        ▼
┌─────────────────────┐
│ filter_reddit_dump  │  Filter by language using fastText
└─────────────────────┘
        │
        ▼
Filtered Data (author → comments)
        │
        ▼
┌─────────────────────┐
│ preprocess_reddit   │  Clean, tokenize, split
└─────────────────────┘
        │
        ▼
Pretrain/Train/Dev/Test Splits
        │
        ▼
┌─────────────────────┐
│ add_features        │  Add LLM-generated features to train/dev/test splits
└─────────────────────┘
        │
        ▼
Data with Features → Ready for Training
```

For external evaluation datasets:
```
External Data (VarDial, PAN, etc.)
        │
        ▼
┌─────────────────────┐
│ reformat_eval_data  │  Convert to standard format
└─────────────────────┘
        │
        ▼
Evaluation Data → Ready for Evaluation
```

## Scripts

### 1. filter_reddit_dump.py

Filter raw Pushshift Reddit dumps by language.

```bash
# Download fastText language ID model first:
# wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

export ROOT=/path/to/repo/root
python preprocessing/filter_reddit_dump.py \
    --input_dir $ROOT/data/pushshift/Spanish \
    --output_dir $ROOT/data/raw \
    --lang_codes es \
    --fasttext_model lid.176.bin
```

**Input**: Directory with `.zst` compressed Reddit dumps
**Output**: JSON files with `{author: [comments]}` structure

### 2. preprocess_reddit.py

Process filtered Reddit data: clean text, tokenize, create splits.

```bash
python preprocessing/preprocess_reddit.py \
    --input_dir $ROOT/data/raw \
    --output_dir $ROOT/data/processed \
    --model bertin-project/bertin-roberta-base-spanish\
    --dev_users 5 \
    --test_users 5
```

**Input**: JSON files with `{author: [comments]}` structure
**Output**: Train/dev/test splits in hierarchical format:
```
data/processed/
├── train_data/
│   ├── argentina.json
│   └── mexico.json
├── dev_data/
│   └── ...
└── test_data/
    └── ...
```

### 3. add_features.py

Add LLM-generated linguistic feature vectors.

```bash
export LITELLM_API_KEY=your_key
export LITELLM_API_BASE_URL=https://api.openai.com/v1  # optional

python preprocessing/add_features.py \
    --input_dir data/processed/train_data \
    --output_dir data/processed/train_data_feats \
    --model gpt-4o-mini \
    --feature
    --batch_size 50

python preprocessing/add_features.py \
    --input_dir $ROOT/data/processed/dev_data \
    --output_dir $ROOT/data/processed/dev_data_feats \
    --model gpt-4o-mini \
    --batch_size 50
```

## Data Formats

### Training Data Format

Hierarchical JSON structure: `[users][comments][sentences]`

```json
[
  [  // User 1
    [  // Comment 1
      {
        "user": "username",
        "text": ["word1", "word2", "..."],
        "text_ids": [101, 2003, 102],
        "feature": [0.1, 0.2, ...]  // optional
      }
    ]
  ]
]
```

### Evaluation Data Format

Flat list of entries:

```json
[
  {
    "sentence": "The original text",
    "tags": ["dialect", "author", "line_0"],
    "input_ids": [101, 2003, 102]
  }
]
```

## Requirements

Core:
- transformers
- tqdm

For filter_reddit_dump.py:
- fasttext
- zstandard

For add_features.py:
- litellm