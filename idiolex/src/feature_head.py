"""Feature prediction and projection head modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureHead(nn.Module):
    """MLP head for predicting linguistic features from embeddings."""

    def __init__(self, input_dim: int, feat_dim: int) -> None:
        """Initialize the feature head.

        Args:
            input_dim: Dimension of input embeddings.
            feat_dim: Number of output features to predict.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.ReLU(),
            nn.Linear(2 * input_dim, feat_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict features from embeddings.

        Args:
            embeddings: Input embeddings of shape [batch_size, input_dim].

        Returns:
            Feature logits of shape [batch_size, feat_dim].
        """
        return self.net(embeddings)


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""

    def __init__(self, input_dim: int, out_dim: int = 256) -> None:
        """Initialize the projection head.

        Args:
            input_dim: Dimension of input embeddings.
            out_dim: Dimension of output projections.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, out_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project and normalize embeddings.

        Args:
            embeddings: Input embeddings of shape [batch_size, input_dim].

        Returns:
            L2-normalized projections of shape [batch_size, out_dim].
        """
        return F.normalize(self.net(embeddings), dim=-1)
