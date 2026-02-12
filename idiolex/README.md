# IdioleX

A PyTorch framework for learning hierarchical idiolectal representation using contrastive learning with margin ranking optimization.

## Overview

This project implements a method for learning text representations that capture linguistic similarity. 
The approach uses hierarchical sampling based on the natural structure of text data (dialect → user → comment → sentence) combined with margin ranking loss to optimize embedding quality.

### Key Features

- **Hierarchical Contrastive Learning**: Samples batches that maintain dialect/user/comment relationships
- **Margin Ranking Loss**: Pairwise ranking optimization for embedding similarity
- **VICReg Regularization**: Prevents embedding collapse through variance and decorrelation constraints
- **Layerwise Attention Pooling**: Learns to combine information across transformer layers
- **Mean Centering**: Improves embedding distribution for better similarity computation
- **Distributed Training**: Full DDP support for multi-GPU training

## Installation

```bash
git clone https://github.com/AnjaliRuban/IdioleX.git
cd IdioleX
pip install -r requirements.txt
cd idiolex
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- transformers >= 4.30
- scikit-learn
- wandb
- numpy

## Data Preparation

See [`preprocessing/README.md`](preprocessing/README.md) for the full data pipeline.

Quick start:
```bash
# 1. Filter pushshift data dump

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

# 2. Process raw Reddit data
export BASEMODEL=intfloat/multilingual-e5-large
export BASEMODELTAG=multilingual-e5-large

python preprocessing/preprocess_reddit.py \
    --input_dir $ROOT/data/${LANGUAGE} \
    --output_dir $ROOT/data/${LANGUAGE}_${BASEMODELTAG} \
    --model $BASEMODEL \
    --dev_users 10 \
    --input_dir data/pushshift/\
    --output_dir data/raw \
    --lang_codes CODE \
    --fasttext_model lid.176.bin

# 3. Add LLM features for pretraining
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
    --input_dir data/processed/train_data \
    --output_dir data/processed/train_data_feats
```

## Data Format

Training data should be organized as JSON files with hierarchical structure:

```json
[
  [  // User 1
    [  // Comment 1
      {"text_ids": [1, 2, 3], "feature": [0.1, 0.2, ...]},  // Sentence 1
      {"text_ids": [4, 5, 6], "feature": [0.3, 0.4, ...]}   // Sentence 2
    ]
  ]
]
```

## Usage

### Training

```bash
# Single GPU
torchrun --nproc_per_node=1 --module idiolex.src.main \
    --tag $EXPERIMENT_NAME \
    --pretrain_data $ROOT/data/${LANGUAGE}_${BASEMODELTAG}/train_data \
    --train_data $ROOT/data/${LANGUAGE}_${BASEMODELTAG}/train_data_with_feats \
    --dev_data $ROOT/data/${LANGUAGE}_${BASEMODELTAG}/dev_data_with_feats \
    --base_model $BASEMODEL \
    --batch_size 16 \
    --dev_size 1024 \
    --feat_dim $FEATLEN \
    --pretrained \
    --pretrain \
    --mean_center \
    --layerwise_pooling \
    --verbose


# Multi-GPU
torchrun --nproc_per_node=4 --module idiolex.src.main \
    --tag $EXPERIMENT_NAME \
    --pretrain_data $ROOT/data/${LANGUAGE}_${BASEMODELTAG}/train_data \
    --train_data $ROOT/data/${LANGUAGE}_${BASEMODELTAG}/train_data_with_feats \
    --dev_data $ROOT/data/${LANGUAGE}_${BASEMODELTAG}/dev_data_with_feats \
    --base_model $BASEMODEL \
    --batch_size 16 \
    --dev_size 1024 \
    --feat_dim $FEATLEN \
    --pretrained \
    --pretrain \
    --mean_center \
    --layerwise_pooling \
    --verbose
```

### Evaluation

```bash
torchrun --nproc_per_node=1 main.py \
    --tag eval_run \
    --dev_data data/test \
    --checkpoint models/experiment_name/checkpoint.pth \
    --evaluate
```

## Post-Training

Use trained embeddings as rewards for LLM fine-tuning. See [`post_training/README.md`](post_training/README.md) for details.

```bash
python post_training/post_train.py \
    --reward_checkpoint models/experiment_name/checkpoint.pth \
    --llm_model silma-ai/SILMA-9B-Instruct-v1.0 \
    --dataset_name UBC-NLP/palm \
    --output_dir models/post-train_finetuned
```


### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--base_model` | roberta-base | Pre-trained model for text encoding |
| `--batch_size` | 16 | Batch size (must be divisible by 4 or 16) |
| `--mini` | False | Use mini batches (4) vs full (16) |
| `--sigma_margin` | 0.5 | Margin for ranking loss |
| `--mean_center` | False | Apply mean centering |
| `--layerwise_pooling` | False | Use layerwise attention |
| `--lr` | 1e-5 | Learning rate |
| `--epoch` | 10 | Number of training epochs |
| `--pretrain` | False | Enable feature pre-training phase |

## Project Structure

```
dialect-embeddings/
├── main.py                # Training script
├── src/
│   ├── __init__.py        # Package exports
│   ├── centering.py       # Mean centering (DDP)
│   ├── data_utils.py      # Data loading utilities
│   ├── evaluation.py      # Evaluation functions
│   ├── feature_head.py    # Feature/projection heads
│   ├── layer_pool.py      # Layerwise attention
│   ├── process.py         # Batch processing
│   └── utils.py           # Utility functions
├── preprocessing/
│   ├── README.md               # Preprocessing documentation
│   ├── filter_reddit.py        # Filter raw Reddit dumps
│   ├── preprocess_reddit.py    # Process & tokenize data
│   ├── generate_features.py    # Add LLM features
│   └── reformat_eval_data.py   # Reformat eval datasets
├── requirements.txt
└── README.md
```

## License

MIT License
