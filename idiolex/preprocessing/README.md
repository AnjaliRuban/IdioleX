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
export LANGUAGE=arabic
export LANG_CODES=ar

python preprocessing/filter_reddit_dump.py \
    --input_dir $ROOT/data/pushshift/$LANGUAGE \
    --output_dir $ROOT/data/$LANGUAGE \
    --lang_codes $LANG_CODES \
    --fasttext_model lid.176.bin
```

**Input**: Directory with `.zst` compressed Reddit dumps
**Output**: JSON files with `{author: [comments]}` structure

### 2. preprocess_reddit.py

Process filtered Reddit data: clean text, tokenize, create splits.

```bash
export BASEMODEL=intfloat/multilingual-e5-large
export BASEMODELTAG=multilingual-e5-large

python preprocessing/preprocess_reddit.py \
    --input_dir $ROOT/data/${LANGUAGE} \
    --output_dir $ROOT/data/${LANGUAGE}_${BASEMODELTAG} \
    --model $BASEMODEL \
    --dev_users 10 \
    --test_users 10
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

### 3. generate_features.py

Add LLM-generated linguistic feature vectors.

```bash
export LITELLM_API_KEY=your_key
export LITELLM_API_BASE_URL=https://api.openai.com/v1  # optional

export FEATMODEL=openai/gpt-5-mini
export SPLIT=train 

python preprocessing/generate_features.py \
    --input_dir $ROOT/data/${LANGUAGE}_${BASEMODELTAG}/${SPLIT}_data \
    --output_dir $ROOT/data/${LANGUAGE}_${BASEMODELTAG}/${SPLIT}_data_with_feats \
    --model $FEATMODEL \
    --features $ROOT/idiolex/preprocessing/feature_list/$LANGUAGE.json
    --batch_size 50
```

## Data Formats

### Training Data Format

Hierarchical JSON structure: `[users][comments][sentences]`

```json
[
  [  // User X
    [  // Comment Y
      { // Sentence Z
        "user": "username",
        "text": ["word1", "word2", "..."],
        "text_ids": [101, 2003, 102],
        "feature": [0.1, 0.2, ...]
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
