"""classification/data.py — data loading, label encoding, Dataset."""

import json

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import Dataset


def load_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_encoder(train_data: list[dict], multi_label: bool):
    """Fit and return a label encoder on training tags."""
    if multi_label:
        enc = MultiLabelBinarizer()
        enc.fit([item["tags"] for item in train_data])
    else:
        enc = LabelEncoder()
        enc.fit([item["tags"][0] for item in train_data])
    return enc


UNK_LABEL = "<UNK>"

def encode_labels(
    data: list[dict],
    enc,
    multi_label: bool,
    num_classes: int = None,
) -> np.ndarray:
    if multi_label:
        return enc.transform([item["tags"] for item in data]).astype(np.float32)
    else:
        labels = []
        for item in data:
            tag = item["tags"][0]
            if tag == UNK_LABEL or tag not in enc.classes_:
                labels.append(num_classes)   # sentinel for unknown author
            else:
                labels.append(int(enc.transform([tag])[0]))
        return np.array(labels, dtype=np.int64)


def get_texts(data: list[dict]) -> list[str]:
    return [item["sentence"] for item in data]


class TextDataset(Dataset):
    def __init__(self, data, labels, tokenizer, max_length):
        texts = get_texts(data)
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }
