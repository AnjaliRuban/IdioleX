"""classification/utils.py — shared utilities."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def freeze_encoder(model):
    """Freeze the entire encoder, training only the classification head."""
    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        is_head = name.startswith("classifier") or name.startswith("layer_pooler") or name.startswith("pooler")
        param.requires_grad = is_head
        if is_head:
            trainable += param.numel()
        else:
            frozen += param.numel()
    print(f"  Frozen encoder: {frozen:,} params | Trainable head: {trainable:,} params")

def freeze_bottom_layers(model, n: int):
    """Freeze the bottom n transformer layers (BERT/RoBERTa family)."""
    if n == 0:
        return
    for attr in ("bert", "roberta", "arabert"):
        encoder = getattr(model, attr, None)
        if encoder is not None:
            layers = encoder.encoder.layer
            for layer in layers[:n]:
                for param in layer.parameters():
                    param.requires_grad = False
            print(f"  Froze bottom {n}/{len(layers)} transformer layers.")
            return
    print("  Warning: could not identify encoder layers to freeze.")

def get_linear_schedule(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 1.0 - progress)
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    multi_label: bool,
    label_names: list[str],
    unk_index: int = None,
) -> dict:
    display_names = list(label_names)
    report_labels = None

    if unk_index is not None:
        display_names = display_names + ["<UNK>"]
        report_labels = list(range(len(label_names) + 1))

    print(classification_report(labels, preds, target_names=label_names,
                                labels=report_labels, digits=3, zero_division=0))
    if multi_label:
        exact_match = float((preds == labels).all(axis=1).mean())
        print(f"Exact match accuracy: {exact_match:.4f}")
        return {
            "exact_match": exact_match,
            "macro_f1": f1_score(labels, preds, average="macro",  zero_division=0),
            "micro_f1": f1_score(labels, preds, average="micro",  zero_division=0),
        }
    else:
        exact_match = float((preds == labels).mean())
        print(f"Exact match accuracy: {exact_match:.4f}")
        kw = dict(labels=report_labels, zero_division=0) if report_labels else dict(zero_division=0)
        return {
            "exact_match":  exact_match,
            "macro_f1":  f1_score(labels, preds, average="macro", **kw),
        }

def tune_unk_threshold(
    proba:       np.ndarray,
    true_labels: np.ndarray,
    num_classes: int,
    lo:    float = 0.0,
    hi:    float = 1.0,
    steps: int   = 100,
) -> float:
    """
    Sweep confidence thresholds on the dev set.
    Predictions below the threshold are assigned <UNK> (index = num_classes).
    Returns the threshold that maximises macro F1 over all classes including UNK.

    Parameters
    ----------
    proba       : (N, num_classes) probability matrix from the neural model
    true_labels : (N,) integer labels; num_classes encodes the true UNK entries
    num_classes : number of known classes (UNK sentinel = this value)
    """
    print("\nTuning <UNK> threshold on dev set...")
    best_t, best_f1 = 0.0, 0.0

    for t in np.linspace(lo, hi, steps + 1):
        preds = _apply_threshold(proba, t, num_classes)
        # Include UNK as a real class in the F1 calculation
        f1 = f1_score(true_labels, preds, average="macro",
                      labels=list(range(num_classes + 1)), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    print(f"  Best threshold={best_t:.3f}  dev macro F1={best_f1:.4f}")
    return best_t


def apply_unk_threshold(
    proba:       np.ndarray,
    threshold:   float,
    num_classes: int,
) -> np.ndarray:
    """Apply a confidence threshold; low-confidence predictions become UNK."""
    return _apply_threshold(proba, threshold, num_classes)


def _apply_threshold(proba, threshold, num_classes):
    preds = proba.argmax(axis=1).copy()
    preds[proba.max(axis=1) < threshold] = num_classes
    return preds

def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    return f1_score(labels, preds, average="macro", zero_division=0)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

class BestModelTracker:
    """Saves the best model to disk based on val macro F1."""

    def __init__(self, output_dir: str, model, tokenizer):
        self.output_dir = Path(output_dir)
        self.model      = model
        self.tokenizer  = tokenizer
        self.best_f1    = 0.0
        self.no_improve = 0

    def update(self, val_f1: float) -> bool:
        """Returns True if improved, False otherwise."""
        if val_f1 > self.best_f1:
            self.best_f1    = val_f1
            self.no_improve = 0
            self.model.save_pretrained(self.output_dir / "best_model")
            self.tokenizer.save_pretrained(self.output_dir / "best_model")
            print(f"  ✓ Saved best model (macro F1={self.best_f1:.4f})")
            return True
        else:
            self.no_improve += 1
            return False


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def save_results(output_dir: str, results: dict, args):
    out = Path(output_dir)
    results["args"] = vars(args)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}/results.json")
