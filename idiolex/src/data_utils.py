"""Data loading utilities for dialect embedding training."""

import argparse
import json
import os
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler, Sampler


class HierarchicalDataset(Dataset):
    """Dataset for hierarchical text data with dialect/user/comment structure.

    Loads JSON files containing text data organized by dialect, user, and comment.
    Each item includes indices for hierarchical sampling.
    """

    def __init__(self, directory: str, feat_len: int = 50) -> None:
        """Initialize the dataset.

        Args:
            directory: Path to directory containing JSON data files.
            feat_len: Length of feature vectors.
        """
        self.directory = directory
        self.feat_len = feat_len
        self.data: list[dict] = []

        files = [f for f in os.listdir(directory) if f.endswith(".json")]
        for filename in files:
            self._load_file(filename)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.data[idx]
        return {
            "input_ids": (
                item["text_ids"][0]
                if len(item["text_ids"]) == 1
                else item["text_ids"]
            ),
            "feature_vec": (
                item["feature"]
                if "feature" in item and item["feature"] is not None
                else [0] * self.feat_len
            ),
            "comment_idx": item["comment_idx"],
            "user_idx": item["user_idx"],
            "dialect_idx": item["dialect_idx"],
            "idx": idx,
        }

    def _load_file(self, filename: str) -> None:
        """Load and process a single JSON file."""
        with open(os.path.join(self.directory, filename)) as f:
            raw_data = json.load(f)

        idx = len(self.data)
        dialect_data = []
        dialect_idx = [idx, idx]

        for user in raw_data:
            user_data = []
            user_idx = [idx, idx]

            for comment in user:
                comment_data = []
                comment_idx = [idx, idx]

                for line in comment:
                    comment_data.append(line)
                    comment_idx[1] += 1
                    user_idx[1] += 1
                    dialect_idx[1] += 1
                    idx += 1

                for item in comment_data:
                    item["comment_idx"] = comment_idx

                user_data.extend(comment_data)

            for item in user_data:
                item["user_idx"] = user_idx

            dialect_data.extend(user_data)

        for item in dialect_data:
            item["dialect_idx"] = dialect_idx

        self.data.extend(dialect_data)

class StandardDataset(Dataset):
    """Standard dataset for text data without hierarchical structure or feature vectors (for evaluation).

    Loads JSON files containing text data.
    """

    def __init__(self, filepath: str) -> None:
        """Initialize the dataset.

        Args:
            filepath: Path to JSON data file.
        """
        with open(filepath) as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.data[idx]
        return {
            "input_ids": (
                item["input_ids"][0]
                if len(item["input_ids"]) == 1
                else item["input_ids"]
            ),
            "idx": idx,
            "tags": item.get("tags", []),
        }


class TripletSampler(BatchSampler):
    """Batch sampler for hierarchical triplet mining.

    Creates batches with related samples based on the dialect/user/comment
    hierarchy to enable contrastive learning.
    """

    def __init__(
        self,
        sampler: Sampler[int] | Iterable[int],
        batch_size: int,
        drop_last: bool,
        mini: bool = False,
    ) -> None:
        """Initialize the triplet sampler.

        Args:
            sampler: Base sampler to draw indices from.
            batch_size: Number of samples per batch.
            drop_last: Whether to drop the last incomplete batch.
            mini: If True, use mini batches (size 4), else full (size 16).

        Raises:
            AssertionError: If batch_size not divisible by mini batch size.
            TypeError: If sampler is not a valid type.
        """
        group_size = 4 if mini else 16
        assert batch_size % group_size == 0, (
            f"batch_size must be divisible by {group_size}"
        )

        if isinstance(sampler, DistributedSampler):
            self.data_source = sampler.dataset
        elif isinstance(sampler, Sampler):
            self.data_source = sampler.data_source
        else:
            raise TypeError("sampler must be a Sampler or DistributedSampler")

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.mini = mini

    def __len__(self) -> int:
        return len(self.sampler)

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []

        for idx in self.sampler:
            batch.extend(self._get_related_indices(idx))

            if len(batch) >= self.batch_size:
                yield batch[: self.batch_size]
                batch = batch[self.batch_size :]

        if batch and not self.drop_last:
            yield batch

    def _get_related_indices(self, idx: int) -> list[int]:
        """Get indices for samples related to the anchor."""
        if self.mini:
            indices = [idx]
            item = self._get_item(idx)
            indices.append(
                np.random.randint(item["dialect_idx"][0], item["dialect_idx"][1])
            )

            new_idx = self._get_same_language(idx)
            indices.append(new_idx)

            item = self._get_item(new_idx)
            indices.append(
                np.random.randint(item["dialect_idx"][0], item["dialect_idx"][1])
            )
        else:
            indices = []
            indices.extend(self._get_cluster(idx))
            indices.extend(self._get_cluster(self._get_same_dialect(indices[-1])))
            indices.extend(self._get_cluster(self._get_same_language(indices[-1])))
            indices.extend(self._get_cluster(self._get_same_dialect(indices[-1])))

        return indices

    def _get_item(self, idx: int) -> dict[str, Any]:
        """Get item from dataset."""
        if isinstance(self.data_source, Dataset):
            item = self.data_source[idx]
        else:
            item = self.data_source.data[idx]
        return item

    def _get_cluster(self, idx: int) -> list[int]:
        """Get a cluster of related indices."""
        indices = [idx]
        indices.append(self._get_same_comment(indices[-1]))
        indices.append(self._get_same_author(indices[-1]))
        indices.append(self._get_same_comment(indices[-1]))
        return indices

    def _get_same_comment(self, idx: int) -> int:
        """Get index from the same comment."""
        item = self._get_item(idx)
        start, end = item["comment_idx"]
        new_idx = int(np.random.randint(start, end - 1))
        return new_idx + 1 if new_idx >= idx else new_idx

    def _get_same_author(self, idx: int) -> int:
        """Get index from the same author."""
        item = self._get_item(idx)
        comment_length = item["comment_idx"][1] - item["comment_idx"][0]
        start, end = item["user_idx"]
        new_idx = int(np.random.randint(start, end - comment_length))
        return (
            new_idx + comment_length
            if new_idx >= item["comment_idx"][0]
            else new_idx
        )

    def _get_same_dialect(self, idx: int) -> int:
        """Get index from the same dialect."""
        item = self._get_item(idx)
        author_length = item["user_idx"][1] - item["user_idx"][0]
        start, end = item["dialect_idx"]
        new_idx = int(np.random.randint(start, end - author_length))
        return new_idx + author_length if new_idx >= item["user_idx"][0] else new_idx

    def _get_same_language(self, idx: int) -> int:
        """Get index from the same language (different dialect)."""
        item = self._get_item(idx)
        dialect_length = item["dialect_idx"][1] - item["dialect_idx"][0]
        end = len(self.data_source) - dialect_length
        new_idx = int(np.random.randint(0, end))
        return (
            new_idx + dialect_length
            if new_idx >= item["dialect_idx"][0]
            else new_idx
        )


def make_collator(
    args: argparse.Namespace,
) -> Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]:
    """Create a collate function for training batches.

    Args:
        args: Arguments containing model_len and mini settings.

    Returns:
        Collate function for DataLoader.
    """

    if args.evaluate:
        def collate_fn(batch: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, torch.Tensor]:
            input_ids = pad_sequence(
                [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch], batch_first=True
            )
            input_ids = input_ids[:, : min(input_ids.size(1), args.model_len)]
            input_attn_mask = input_ids != pad_token_id

            idxs = [b["idx"] for b in batch]
            tags = [b["tags"] for b in batch]
            return {
                "idxs": idxs,
                "tags": tags,
                "input_ids": input_ids,
                "input_attn_mask": input_attn_mask,
            }
    else:
        def collate_fn(
            batch: list[dict[str, Any]],
            pad_token_id: int = 0,
        ) -> dict[str, torch.Tensor]:
            input_ids = pad_sequence(
                [torch.tensor(b["input_ids"], dtype=torch.long) for b in batch],
                batch_first=True,
            )
            input_ids = input_ids[:, : min(input_ids.size(1), args.model_len)]
            attention_mask = input_ids != pad_token_id

            try:
                feat_ids = torch.stack(
                    [torch.tensor(b["feature_vec"], dtype=torch.float) for b in batch],
                    dim=0,
                )
            except Exception:
                feat_ids = torch.zeros(len(batch), len(batch[0]["feature_vec"]))

            if args.mini:
                graded_relevance = torch.tensor([2, 1, 0, 0], dtype=torch.float)
            else:
                graded_relevance = torch.tensor(
                    [4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    dtype=torch.float,
                )

            return {
                "input_ids": input_ids,
                "input_attn_mask": attention_mask,
                "feat_ids": feat_ids,
                "graded_relevance": graded_relevance,
                "idxs": [b["idx"] for b in batch],
            }

    return collate_fn