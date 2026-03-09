"""classification/train.py — training and evaluation loops."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import macro_f1

def train_epoch(
    model,
    loader:    DataLoader,
    optimizer,
    scheduler,
    criterion,
    device,
    scaler=None,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model,
    loader:     DataLoader,
    device,
    multi_label: bool,
    threshold:   float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (preds, labels) as numpy arrays."""
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        if multi_label:
            preds = (torch.sigmoid(logits).cpu() >= threshold).int().numpy()
        else:
            preds = logits.argmax(-1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(batch["labels"].numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


@torch.no_grad()
def get_probas(
    model,
    loader:      DataLoader,
    device,
    multi_label: bool,
) -> np.ndarray:
    """Returns (N, num_classes) probability matrix."""
    model.eval()
    all_probs = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        if multi_label:
            probs = torch.sigmoid(logits).cpu().numpy()
        else:
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        all_probs.append(probs)

    return np.concatenate(all_probs)



def run_training(
    model,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    optimizer,
    scheduler,
    criterion,
    tracker,        # BestModelTracker from utils.py
    args,
    device,
    scaler=None,
) -> list[dict]:
    """
    Runs the full training loop with early stopping.
    Returns history (list of per-epoch dicts).
    """
    history   = []
    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...\n")

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler,
                              criterion, device, scaler)

        preds, targets = evaluate(model, val_loader, device,
                                  args.multi_label, args.threshold)
        val_f1 = macro_f1(preds, targets)

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_macro_f1={val_f1:.4f}")
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_macro_f1": val_f1})

        tracker.update(val_f1)

        if tracker.no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    return history
