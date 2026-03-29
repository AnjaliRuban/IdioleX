# IdioleX: Idiolectal Representation Learning

This repository implements **IdioleX**, a framework for learning sentence representations that encode **stylistic and dialectal variation**, demphasizing semantic content.

The method combines:
- Hierarchical weak supervision (author, dialect, context)
- LLM-extracted linguistic features
- Contrastive and ranking objectives

These representations can be used for:
- Dialect identification
- Authorship attribution
- Style similarity
- Post-training alignment of LLMs

---

## 🔧 Key Components

- **Hierarchical training** with graded relevance (same comment → same author → same dialect → different dialect)
- **Feature supervision** via LLM-extracted linguistic features
- **Contrastive + ranking losses**
- **Mean centering + VICReg regularization** to reduce anisotropy
- **Layer-wise pooling** for better representation extraction

---

## 🚀 Quick Start

### Install dependencies
```bash
pip install torch transformers wandb ijson scikit-learn
```

### Train Model

```bash
python idiolex.src.main \
  --train_data path/to/train \
  --dev_data path/to/dev \
  --tag experiment_name \
  --base_model model \ # Huggingface model name 
  --pretrained \ # Use pre-trained model
  --pretrain \ # Do two-stage training
  --feat_dim num_features \
  --layerwise_pooling \
  --mean_center
```

Multi-GPU training is supported using torchrun:

```bash
torchrun --nproc_per_node=N idiolex.src.main \
  --train_data path/to/train \
  --dev_data path/to/dev \
  --tag experiment_name \
  --base_model model \
  --pretrained \
  --pretrain \
  --feat_dim num_features \
  --layerwise_pooling \
  --mean_center
```

### Evaluate

```bash
python idiolex.src.main \
  --checkpoint models/exp/checkpoint.pth \
  --dev_data path/to/dev \
  --evaluate \
  --tag eval_run
```

---

## Data Preprocessing

All preprocessing is handled via scripts in `idiolex/preprocessing`.

These scripts take raw text data (e.g., Reddit dumps) and produce the **hierarchical JSON format** required for training.

---

## Pipeline Overview

The preprocessing pipeline consists of:

1. Cleaning + filtering raw data
2. Sentence segmentation
3. Hierarchical structuring (dialect → user → comment → sentence)
4. Tokenization
5. (Optional) linguistic feature extraction

---

## Running Preprocessing

### 1. Run preprocessing pipeline

```bash
python -m idiolex.preprocessing.preprocess_reddit \
  --input_dir data/raw \
  --output_dir data/processed \
  --model_name model # Huggingface model name
```

This will:
- clean and filter data
- split into sentences
- tokenize text
- build hierarchical structure

### 2. Extract linguistic features

```
python -m idiolex.preprocessing.generate_features \
  --input_dir data/processed \
  --output_dir data/processed_with_features
  --features idiolex/preprocessing/feature_lists/language.json
```

This step adds binary feature vectors per sentence using LLM-based extraction. 

## Data Format

The pre-processed training data files should have the following structure to support the hierarchical supervision:

```json
[
  [ // User 1
    [  // Comment 1
      {"text_ids": [1, 2, 3], "feature": [0.1, 0.2, ...]},  // Sentence 1
      {"text_ids": [4, 5, 6], "feature": [0.3, 0.4, ...]}   // Sentence 2
      ...
    ],
    ...
  ],
  ...
]
```

## Downstream Classification

The `idiolex/classification` module fine-tunes an IdioleX checkpoint for task-specific uses such as dialect identification and authorship attribution.
It supports:
- single-label classification
- multi-label classification
- optional TF-IDF + logistic regression lexical ensembling
- optional open-set <UNK> prediction

### Run classification

```bash
python -m idiolex.classification.main \
  --train data/classification/train.json \
  --val data/classification/val.json \
  --test data/classification/test.json \
  --model models/idiolex/checkpoint.pth \
  --output-dir runs/classification
```

Multi-label classification will be attempted with the additional `--multi-label` tag.

Lexical ensembling will be added with the additional ` --use-lexical` tag. In this case, a portion of the validation set will be witheld from the finetuning for use in determining the ratio with which the IdioleX probabilities and lexical probabilities are combined.

### Outputs
This produces:
```
best_model/
    results.json
    label_map.json
```
and, if lexical ensembling is enabled:

```
best_model/
    fidf.joblib
    lr_clf.joblib
    lex_weight.json
```

## Post-Training

The `idiolex/post_training` module uses trained IdioleX embeddings as an additional alignment objective during LLM supervised fine-tuning.
This is useful when you want a model to generate outputs that better match the target dialect or style.
This section presumes you have trained a relevant IdioleX model already.

### Generating Instruction-Tuning Data (Optional)

To maintain instruction-following capabilities, we finetune with instruction-answer pairs.
However, the majority of dialectal data is not instruction-tuning data, so we augment the existing data using an LLM.

```bash
export LITELLM_API_KEY="your_key"
export HF_TOKEN="your_token"

python3 idiolex/preprocesing/generate_post-train_instructions.py \
    --output_dir data/post_training \
    --madar_path local/download/madar \
    --saudial_path local/download/saudial \
    --joda_path local/download/joda \
    --masc_path local/download/masc 
```

Other datasets are automatically downloaded.


### Supervised Fine-Tuning

```bash
accelerate launch --config_file idiolex/post_training/accelerate_config.yaml \
    idiolex/post_training/sft.py \
    --train_jsonl data/post_training/train.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct \ # or any Hugginface LLM
    --reward_checkpoint models/idiolex_model/checkpoint.pth \
    --output_dir models/sft_idiolex
```

LLM is trained using LoRA so the output directory will include the adapter checkpoints.

### Generate Predictions

```bash
python -m idiolex.post_training.predict \
  --model models/sft_idiolex/final \
  --input eval/language.csv \ # AMIYA eval set
  --output eval/language_preds.tsv
```




