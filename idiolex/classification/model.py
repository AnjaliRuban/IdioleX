"""classification/model.py — model loading (.pth or HF directory)."""

import torch
from transformers import AutoModelForSequenceClassification


def load_model(
    model_path: str,
    num_classes: int,
    multi_label: bool,
    base_model_override: str = None,
):
    """
    Load a sequence classification model from either:
      - An idiolex .pth checkpoint  (contains 'model' state dict + 'args')
      - A HuggingFace model directory

    Parameters
    ----------
    model_path : str
        Path to .pth file or HF model directory.
    num_classes : int
        Number of output classes (classification head size).
    multi_label : bool
        Affects the problem_type passed to from_pretrained.
    base_model_override : str, optional
        Explicitly pass the HF base model name (e.g. 'aubmindlab/bert-base-arabertv02').
        Only needed for .pth loading if the name can't be found in the checkpoint args.
    """
    problem_type = (
        "multi_label_classification" if multi_label
        else "single_label_classification"
    )

    if model_path.endswith(".pth"):
        return _load_from_pth(model_path, num_classes, problem_type, base_model_override)
    else:
        return AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            problem_type=problem_type,
        )


def _load_from_pth(path, num_classes, problem_type, base_model_override):
    ckpt= torch.load(path, map_location="cpu", weights_only=False)
    saved_args  = vars(ckpt["args"])

    base_model = base_model_override if base_model_override else saved_args.get("base_model")

    if base_model is None:
        raise ValueError(
            f"Could not find base model name in checkpoint args.\n"
            f"Available keys: {list(saved_args.keys())}\n"
            f"Pass it explicitly with --base-model."
        )
    print(f"  Base model : {base_model}")
    print(f"  Checkpoint : epoch {ckpt.get('epoch', '?')}, step {ckpt.get('step', '?')}")

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        problem_type=problem_type,
    )

    # ckpt["model"] is the encoder weights only — no classifier head
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    _report_key_mismatches(missing, unexpected)
    return model


def _report_key_mismatches(missing, unexpected):
    # Classifier keys being missing is expected — new randomly-init head
    non_classifier_missing = [k for k in missing if not k.startswith("classifier")]
    if non_classifier_missing:
        print(f"  Warning — unexpected missing keys: {non_classifier_missing}")
    else:
        print(f"  Encoder loaded cleanly. "
              f"({len(missing)} classifier keys init from scratch, as expected)")
    if unexpected:
        print(f"  Warning — unexpected keys in checkpoint: {unexpected}")
