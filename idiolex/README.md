# IdioleX Python Package

This directory contains the core implementation of the IdioleX framework for
idiolectal and dialect-aware sentence representation learning.

The codebase is organized into modular components covering:

- embedding training
- preprocessing
- downstream classification
- post-training alignment

---

## Module Overview

### `src/`
Core modeling and training logic.

Includes:
- encoder + pooling
- ranking + contrastive losses
- feature prediction heads
- evaluation pipeline

👉 Entry point:
```bash
python -m idiolex.src.main
```

### `preprocessing/`

Data preprocessing pipeline.

Includes:
- raw data cleaning
- sentence segmentation
- hierarchical structuring
- tokenization
- feature extraction
- dataset splitting

👉 Used before any training.

### `classification/`

Downstream classification utilities.

Supports:
- dialect identification
- authorship attribution
- multi-label classification
- lexical ensembling

👉 Used after embeddings are trained.

### `post_training/`

LLM alignment using IdioleX representations.

Uses embedding similarity as an additional training signal for supervised fine-tuning (SFT) for style/dialect alignment.

👉 Advanced use case (not required for core training).




