"""
classification/main.py — entry point for fine-tuning with optional TF-IDF ensemble.
"""

import os
import argparse
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import (
    build_encoder, encode_labels, get_texts,
    load_json, TextDataset,
)
from .lexical import (
    build_lexical_model,
    lexical_proba, save_lexical, tune_ensemble_weight,
)
from .model import load_model, LayerPoolClassifier
from .train import evaluate, get_probas, run_training
from .utils import (
    BestModelTracker, compute_metrics, freeze_bottom_layers, freeze_encoder,
    get_linear_schedule, macro_f1, save_results, set_seed,
    tune_unk_threshold, apply_unk_threshold
)


def get_args():
    p = argparse.ArgumentParser(
        description="Fine-tune an encoder for sequence classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- data ---
    p.add_argument("--train",  required=True)
    p.add_argument("--val",    default=None)
    p.add_argument("--test",   required=True)

    # --- model ---
    p.add_argument("--model",        required=True,
                   help="HF model name, local HF dir, or idiolex .pth checkpoint")
    p.add_argument("--base-model",   default=None,
                   help="Override base model name (only for .pth loading if auto-detect fails)")
    p.add_argument("--max-length",   type=int, default=512)
    p.add_argument("--freeze-encoder", action="store_true",
                   help="Freeze entire encoder, train classification head only. "
                        "Mutually exclusive with --freeze-layers.")
    p.add_argument("--freeze-layers", type=int, default=0,
                   help="Freeze bottom N transformer layers")

    # --- task ---
    p.add_argument("--multi-label",  action="store_true")
    p.add_argument("--has-unk", action="store_true")
    p.add_argument("--threshold",    type=float, default=0.5,
                   help="Decision threshold for multi-label")

    # --- lexical ensemble ---
    p.add_argument("--use-lexical",         action="store_true",
                   help="Add a TF-IDF + LR component and ensemble with the neural model")
    p.add_argument("--tfidf-analyzer",      default="char_wb")
    p.add_argument("--tfidf-ngram-range",   default="2,6",
                   help="Comma-separated min,max n-gram range e.g. '2,6'")
    p.add_argument("--tfidf-max-features",  type=int, default=80_000)
    p.add_argument("--lr-C",                type=float, default=3.0,
                   help="Regularisation for the lexical LR")
    p.add_argument("--lex-weight",          type=float, default=None,
                   help="Fixed lexical weight (0-1). If omitted, tuned on val set.")

    # --- training ---
    p.add_argument("--epochs",           type=int,   default=10)
    p.add_argument("--batch-size",       type=int,   default=32)
    p.add_argument("--lr",               type=float, default=2e-5)
    p.add_argument("--warmup-ratio",     type=float, default=0.1)
    p.add_argument("--weight-decay",     type=float, default=0.01)
    p.add_argument("--label-smoothing",  type=float, default=0.0,
                   help="Label smoothing (single-label only)")
    p.add_argument("--patience",         type=int,   default=3)

    # --- misc ---
    p.add_argument("--output-dir", default="runs/classification")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--fp16",       action="store_true")

    args = p.parse_args()
    # parse ngram range string → tuple
    lo, hi = args.tfidf_ngram_range.split(",")
    args.tfidf_ngram_range = (int(lo), int(hi))
    return args

def main():
    args = get_args()
    set_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Task   : {'multi-label' if args.multi_label else 'single-label'}")
    print(f"Lexical: {args.use_lexical}")

    train_data = load_json(args.train)
    if args.val:
        val_data   = load_json(args.val)
    else:
        random.shuffle(train_data)
        val_data = train_data[:len(train_data) // 10]
        train_data = train_data[len(train_data) // 10:]

    test_data  = load_json(args.test)

    enc         = build_encoder(train_data, args.multi_label)
    label_names = list(enc.classes_)
    num_classes = len(label_names)
    print(f"Classes ({num_classes}): {label_names}")

    import json
    with open(out / "label_map.json", "w") as f:
        json.dump({"labels": label_names, "multi_label": args.multi_label},
                  f, ensure_ascii=False, indent=2)

    train_labels = encode_labels(train_data, enc, args.multi_label, num_classes)
    val_labels   = encode_labels(val_data,   enc, args.multi_label, num_classes)

    print(f"\nLoading tokenizer from {args.model}...")
    tok_path  = args.model if not args.model.endswith(".pth") else args.base_model if args.base_model else vars(torch.load(args.model, map_location="cpu", weights_only=False)["args"]).get("base_model")

    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    train_ds = TextDataset(train_data, train_labels, tokenizer, args.max_length)
    val_ds   = TextDataset(val_data, val_labels, tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"\nLoading model from {args.model}...")
    model = load_model(args.model, num_classes, args.multi_label, args.base_model)
    model = model.to(device)

    if args.freeze_encoder:
        freeze_encoder(model)
    elif args.freeze_layers > 0:
        freeze_bottom_layers(model, args.freeze_layers)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = (
        torch.nn.BCEWithLogitsLoss() if args.multi_label
        else torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    )
    optimizer    = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule(optimizer, warmup_steps, total_steps)
    scaler       = torch.cuda.amp.GradScaler() if (args.fp16 and torch.cuda.is_available()) else None

    tracker = BestModelTracker(str(out), model, tokenizer)
    history = run_training(
        model, train_loader, val_loader,
        optimizer, scheduler, criterion,
        tracker, args, device, scaler,
    )

    print("\nLoading best neural checkpoint...")
    best_path = str(out / "best_model")
    if os.path.exists(os.path.join(best_path, "head.pt")):
        model = LayerPoolClassifier.from_pretrained(best_path, num_classes).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(out / "best_model").to(device)

    lex_weight  = None
    vectorizer  = None
    lr_clf      = None

    if args.use_lexical:
        print("\n=== Lexical component (TF-IDF + LR) ===")
        vectorizer, lr_clf, _ = build_lexical_model(
            train_texts  = get_texts(train_data),
            train_labels = train_labels,
            val_texts    = get_texts(val_data),
            val_labels   = val_labels,
            multi_label  = args.multi_label,
            tfidf_analyzer     = args.tfidf_analyzer,
            tfidf_ngram_range  = args.tfidf_ngram_range,
            tfidf_max_features = args.tfidf_max_features,
            lr_C               = args.lr_C,
        )

        # Get val probabilities from both components for weight tuning
        P_neural = get_probas(model, val_loader, device, args.multi_label)
        P_lex    = lexical_proba(vectorizer, lr_clf, get_texts(val_data), args.multi_label)

        if args.lex_weight is not None:
            lex_weight = args.lex_weight
            print(f"  Using fixed lex_weight={lex_weight}")
        else:
            lex_weight = tune_ensemble_weight(P_lex, P_neural, val_labels, args.multi_label)

        save_lexical(str(out), vectorizer, lr_clf, lex_weight)

    unk_threshold = None
    unk_index     = None


    def predict_ensemble(data, loader, threshold=None):
        P_neural = get_probas(model, loader, device, args.multi_label)
        if args.use_lexical:
            P_lex = lexical_proba(vectorizer, lr_clf, get_texts(data), args.multi_label)
            P = lex_weight * P_lex + (1 - lex_weight) * P_neural
        else:
            P = P_neural
        if args.multi_label:
            return (P >= args.threshold).astype(int), P
        else:
            if threshold is not None:
                return apply_unk_threshold(P, threshold, num_classes), P
            return P.argmax(axis=1), P

    results = {"history": history}

    print("\n--- Validation ---")
    preds, P_val = predict_ensemble(val_data, val_loader)

    if not args.multi_label and args.has_unk:
        unk_index = num_classes
        unk_threshold = tune_unk_threshold(P_val, val_labels, num_classes)
        preds = apply_unk_threshold(P_val, unk_threshold, num_classes)
        results["unk_threshold"] = unk_threshold

    results["val"] = compute_metrics(preds, val_labels, args.multi_label, label_names, unk_index)

    if test_data is not None:
        test_labels = encode_labels(test_data, enc, args.multi_label, num_classes)
        test_ds     = TextDataset(test_data, test_labels, tokenizer, args.max_length)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        print("\n--- Test ---")
        preds, P_test = predict_ensemble(test_data, test_loader, threshold=unk_threshold)
        results["test"] = compute_metrics(preds, test_labels, args.multi_label, label_names, unk_index)

        # Save per-sample predictions and class probabilities
        all_label_names = label_names + ["<UNK>"] if args.has_unk else label_names
        per_sample = []
        for i, item in enumerate(test_data):
            true_label = [all_label_names[j] for j, v in enumerate(test_labels[i]) if v] if args.multi_label else all_label_names[test_labels[i]]
            pred_label = [all_label_names[j] for j, v in enumerate(preds[i]) if v] if args.multi_label else all_label_names[preds[i]]
            per_sample.append({
                "sentence":   item["sentence"],
                "true_label": true_label,
                "pred_label": pred_label,
                "probs":      {all_label_names[j]: round(float(P_test[i, j]), 4)
                               for j in range(P_test.shape[1])},
            })

        preds_path = out / "test_predictions.json"
        with open(preds_path, "w", encoding="utf-8") as f:
            json.dump(per_sample, f, ensure_ascii=False, indent=2)
        print(f"  Per-sample predictions saved to {preds_path}")

    save_results(str(out), results, args)
    print(f"\nBest val macro F1: {tracker.best_f1:.4f}")


if __name__ == "__main__":
    main()
