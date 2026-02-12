"""Mean centering modules for embedding normalization."""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class MeanCenterer(nn.Module):
    """Distributed mean centerer for multi-GPU training.

    Maintains a global running mean across all distributed ranks.
    """

    def __init__(self, dim: int, dtype: torch.dtype = torch.float32) -> None:
        """Initialize the distributed mean centerer.

        Args:
            dim: Embedding dimension.
            dtype: Data type for the mean buffer.
        """
        super().__init__()
        self.register_buffer("mu", torch.zeros(dim, dtype=dtype))
        self.register_buffer("n", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _global_sum_and_count(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute global sum and count across all ranks."""
        if embeddings.numel() == 0:
            sum_local = torch.zeros_like(self.mu)
            count_local = torch.zeros((), device=self.mu.device, dtype=torch.long)
        else:
            sum_local = embeddings.detach().sum(dim=0).to(self.mu.device)
            count_local = torch.tensor(
                embeddings.size(0), device=embeddings.device, dtype=torch.long
            )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_local, op=dist.ReduceOp.SUM)

        return sum_local, count_local

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor) -> None:
        """Update global running mean with local batch.

        Args:
            embeddings: Pre-normalization embeddings of shape [batch_size, dim].
        """
        global_sum, global_count = self._global_sum_and_count(embeddings)
        if global_count.item() == 0:
            return

        batch_mean = global_sum / global_count.to(global_sum.dtype)
        weight = global_count.to(global_sum.dtype) / (
            self.n.to(global_sum.dtype) + global_count.to(global_sum.dtype)
        )
        self.mu.add_((batch_mean - self.mu) * weight)
        self.n.add_(global_count)

    def apply(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply centering and L2 normalization.

        Args:
            embeddings: Embeddings of shape [batch_size, dim].

        Returns:
            Centered and normalized embeddings of shape [batch_size, dim].
        """
        return F.normalize(embeddings - self.mu, p=2, dim=-1)
