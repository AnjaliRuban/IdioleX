"""classification/lexical.py — TF-IDF + Logistic Regression component."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score


def build_lexical_model(
    train_texts:  list[str],
    train_labels: np.ndarray,
    val_texts:    list[str],
    val_labels:   np.ndarray,
    multi_label:  bool,
    tfidf_analyzer:     str   = "char_wb",
    tfidf_ngram_range:  tuple = (2, 6),
    tfidf_max_features: int   = 80_000,
    lr_C:               float = 3.0,
) -> tuple:
    """
    Fit TF-IDF + LR on training data.
    Returns (vectorizer, classifier, val_f1).
    """
    print("  Fitting TF-IDF...")
    vectorizer = TfidfVectorizer(
        analyzer=tfidf_analyzer,
        ngram_range=tfidf_ngram_range,
        max_features=tfidf_max_features,
        sublinear_tf=True,
    )
    X_tr = vectorizer.fit_transform(train_texts)
    X_v  = vectorizer.transform(val_texts)

    print("  Fitting LR...")
    base_lr = LogisticRegression(C=lr_C, max_iter=2000, solver="saga", n_jobs=-1)
    clf     = OneVsRestClassifier(base_lr) if multi_label else base_lr
    clf.fit(X_tr, train_labels)

    if multi_label:
        preds  = clf.predict(X_v)
        val_f1 = f1_score(val_labels, preds, average="macro", zero_division=0)
    else:
        val_f1 = f1_score(val_labels, clf.predict(X_v), average="macro", zero_division=0)

    print(f"  LR val macro F1: {val_f1:.4f}")
    return vectorizer, clf, val_f1


def lexical_proba(
    vectorizer: TfidfVectorizer,
    clf,
    texts: list[str],
    multi_label: bool,
) -> np.ndarray:
    """Returns (N, num_classes) probability matrix."""
    X = vectorizer.transform(texts)
    if multi_label:
        # OneVsRestClassifier.predict_proba returns (N, C) sigmoid scores
        return clf.predict_proba(X)
    else:
        return clf.predict_proba(X)


def tune_ensemble_weight(
    P_lex:    np.ndarray,
    P_neural: np.ndarray,
    labels:   np.ndarray,
    multi_label: bool,
) -> float:
    """
    Grid-search the lexical weight w in [0, 1] that maximises val macro F1.
    Returns the best weight.
    """
    print("\nTuning ensemble weight...")
    best_w, best_f1 = 0.5, 0.0
    for w in np.arange(0.0, 1.05, 0.05):
        P     = w * P_lex + (1 - w) * P_neural
        preds = (P >= 0.5).astype(int) if multi_label else P.argmax(axis=1)
        f1    = f1_score(labels, preds, average="macro", zero_division=0)
        print(f"  lex_weight={w:.2f}  val_macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_w = f1, float(w)
    print(f"  Best lex_weight={best_w:.2f}  val_macro_f1={best_f1:.4f}")
    return best_w


def save_lexical(output_dir: str, vectorizer, clf, lex_weight: float):
    out = Path(output_dir)
    joblib.dump(vectorizer, out / "tfidf.joblib")
    joblib.dump(clf,        out / "lr_clf.joblib")
    with open(out / "lex_weight.json", "w") as f:
        json.dump({"lex_weight": lex_weight}, f)
    print(f"  Lexical model saved to {out}/")


def load_lexical(model_dir: str):
    d = Path(model_dir)
    vectorizer = joblib.load(d / "tfidf.joblib")
    clf        = joblib.load(d / "lr_clf.joblib")
    with open(d / "lex_weight.json") as f:
        lex_weight = json.load(f)["lex_weight"]
    return vectorizer, clf, lex_weight
