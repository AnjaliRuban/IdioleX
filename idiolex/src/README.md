# IdioleX Core Modeling (`src/`)

This directory contains the core implementation of the IdioleX framework for learning
idiolectal and dialect-aware sentence representations.

It includes:
- embedding extraction
- hierarchical contrastive training
- feature supervision
- evaluation metrics

---

## Model Overview

IdioleX learns sentence embeddings that reflect **stylistic and dialectal similarity**
rather than purely semantic similarity.

Each input sentence is mapped to a vector:
```
x → encoder → pooling → mean centering → embedding
```

The model is trained using:
- **hierarchical ranking supervision**
- **linguistic feature prediction**
- **contrastive learning**

---

## Parameters

### Model & Representation

| Hyperparameter | Description | Default |
|----------------|------------|--------|
| `base_model` | Pretrained encoder backbone | `FacebookAI/roberta-base` |
| `model_len` | Maximum sequence length | `512` |
| `layerwise_pooling` | Use layer-wise attention pooling | `False` |
| `mean_center` | Apply mean-centering to embeddings | `False` |
| `embedding_dim` | Hidden size of encoder | `768` |

---

### Training

| Hyperparameter | Description | Default |
|----------------|------------|--------|
| `batch_size` | Total batch size | `16` |
| `mini` | Use mini-batches (size 4 groups) | `False` |
| `epoch` | Number of training epochs | `10` |
| `pretrain_epoch` | Number of pretraining epochs | `3` |
| `pretrain` | Enable ranking-only pretraining | `False` |
| `dev_step` | Steps between validation | `250` |
| `dev_size` | Validation subset size | `None` |
| `patience` | Early stopping patience | `25` |

---

### Optimization

| Hyperparameter | Description | Default |
|----------------|------------|--------|
| `lr` | Learning rate | `1e-5` |
| `warmup_lr` | LR warmup steps | `25,000` |
| `optimizer` | Optimizer | `Adam` |

---

### Ranking Loss

| Hyperparameter | Description | Default |
|----------------|------------|--------|
| `sigma_margin` | Margin for ranking loss | `0.5` |
| `warmup_margin` | Steps to warm up margin | `25,000` |

---

### Feature Learning

| Hyperparameter | Description | Default |
|----------------|------------|--------|
| `feat_dim` | Number of linguistic features | `56` |
| `alpha` | Weight of feature loss | `0.8` |
| `beta` | Decay rate of α | `0.0` |
| `supcon_tau` | Temperature (contrastive loss) | `0.07` |
| `supcon_topk` | Top-k positives (Jaccard) | `5` |

---

### Regularization

| Hyperparameter | Description | Default |
|----------------|------------|--------|
| `vicreg_weight` | Weight on VICReg term | `0.25` |
| `eps` | Stability constant (VICReg) | `1e-4` |

---



