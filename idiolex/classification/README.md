# Downstream Classification with IdioleX

This module provides utilities for evaluating IdioleX embeddings on downstream classification tasks such as:

- Dialect Identification (DID)
- Authorship Attribution (AA)
- Multi-label dialect classification
- Ensembling with lexical features

---

## Overview

Given:
- a trained IdioleX model (or pretrained encoder)
- labeled classification data

this module trains a classifier on top of sentence representations.
---

## Usage

### A. Basic Training

```bash
python -m idiolex.classification.main \
  --train data/classification/train.json \
  --val data/classification/val.json \
  --test data/classification/test.json \
  --model models/idiolex/checkpoint.pth \
  --output_dir runs/classification
```

### B. Multi-Label Classification

```bash
python -m idiolex.classification.main \
  --train data/classification/train.json \
  --val data/classification/val.json \
  --test data/classification/test.json \
  --model models/idiolex/checkpoint.pth \
  --multi_label \
  --output_dir runs/classification_multilabel
```

### B. Classification with "Unknown" Class

```bash
python -m idiolex.classification.main \
  --train data/classification/train.json \
  --val data/classification/val.json \
  --test data/classification/test.json \
  --model models/idiolex/checkpoint.pth \
  --has-unk \
  --output_dir runs/classification_multilabel
```


### C. Training Only Prediction Head

```bash
python -m idiolex.classification.main \
  --train data/classification/train.json \
  --val data/classification/val.json \
  --test data/classification/test.json \
  --model models/idiolex/checkpoint.pth \
  --freeze_encoder \
  --output_dir runs/classification
```

### D. Lexical Ensembling

```bash
python -m idiolex.classification.main \
  --train data/classification/train.json \
  --val data/classification/val.json \
  --test data/classification/test.json \
  --model models/idiolex/checkpoint.pth \
  --use_lexical \
  --output_dir runs/classification_lexical
```

After training, mutliple metrics are reported on the test set:
- Accuracy (exact match for multi-label)
- Macro F1
- Per-Class F1


## Data Format

Data should be in JSON:
```json
{
  "sentence": "text",
  "label": ["Spain"]
}
```
where the label array can have either one or many entries.

## Hyperparameters

### Model

| Hyperparameter | Description | Default |
|----------------|------------|--------|
| `model` | Path to IdioleX checkpoint (optional) | `None` |
| `base_model` | Encoder backbone | `FacebookAI/roberta-base` |
| `freeze_encoder` | Freeze encoder weights during training | `False` |

---

### Training

| Hyperparameter | Description | Default |
|----------------|------------|--------|
| `batch_size` | Batch size | `32` |
| `epochs` | Number of training epochs | `5–10` |
| `learning_rate` | Learning rate | `1e-4` |
| `weight_decay` | Weight decay | `0.0` |

---

### Classification

| Hyperparameter | Description | Default |
|----------------|------------|--------|
| `num_labels` | Number of classes | inferred |
| `multi_label` | Enable multi-label classification | `False` |
| `threshold` | Decision threshold for multi-label | `0.5` |

---

### Lexical Features (Optional)

| Hyperparameter | Description | Default |
|----------------|------------|--------|
| `use_lexical` | Enable TF-IDF + logistic regression ensembling | `False` |
| `tfidf_max_features` | Maximum vocabulary size | `50,000` |
| `ngram_range` | N-gram range | `(1, 2)` |

---

### Evaluation

| Hyperparameter | Description | Default |
|----------------|------------|--------|
| `metric` | Evaluation metric | `macro_f1` |
| `compute_exact_match` | Compute exact match (multi-label) | `True` |


