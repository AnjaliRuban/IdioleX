"""Layer-wise attention pooling for transformer embeddings."""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Parameter, ParameterList


class LayerwiseAttention(nn.Module):
    """Layer-wise attention mechanism for combining transformer layer outputs.

    Learns a weighted combination of all transformer layer outputs.
    Based on Rei et al. 2020.
    """

    def __init__(
        self,
        num_layers: int,
        layer_norm: bool = False,
        layer_weights: Optional[list[float]] = None,
        dropout: Optional[float] = None,
    ) -> None:
        """Initialize the layer-wise attention module.

        Args:
            num_layers: Number of transformer layers to combine.
            layer_norm: Whether to apply layer normalization before combining.
            layer_weights: Initial weights for each layer. Defaults to zeros.
            dropout: Dropout probability for layer weights during training.

        Raises:
            ValueError: If layer_weights length doesn't match num_layers.
        """
        super().__init__()
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        self.dropout = dropout

        if layer_weights is None:
            layer_weights = [0.0] * num_layers
        elif len(layer_weights) != num_layers:
            raise ValueError(
                f"layer_weights length ({len(layer_weights)}) must match "
                f"num_layers ({num_layers})"
            )

        self.scalar_parameters = ParameterList(
            [
                Parameter(torch.FloatTensor([layer_weights[i]]), requires_grad=True)
                for i in range(num_layers)
            ]
        )
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        if self.dropout:
            self.register_buffer("dropout_mask", torch.zeros(num_layers))
            self.register_buffer(
                "dropout_fill", torch.empty(num_layers).fill_(-1e20)
            )

    def forward(
        self,
        hidden_states: list[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Combine layer embeddings using learned attention weights.

        Args:
            hidden_states: List of layer outputs, each of shape
                [batch_size, seq_len, hidden_dim].
            attention_mask: Optional mask of shape [batch_size, seq_len].

        Returns:
            Combined embeddings of shape [batch_size, seq_len, hidden_dim].

        Raises:
            ValueError: If number of hidden states doesn't match num_layers.
        """
        if len(hidden_states) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} hidden states, got {len(hidden_states)}"
            )

        weights = torch.cat(list(self.scalar_parameters))

        if self.training and self.dropout:
            weights = torch.where(
                self.dropout_mask.uniform_() > self.dropout,
                weights,
                self.dropout_fill,
            )

        normed_weights = torch.softmax(weights, dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        if not self.layer_norm:
            combined = sum(w * h for w, h in zip(normed_weights, hidden_states))
            return self.gamma * combined

        mask_float = attention_mask.float()
        broadcast_mask = mask_float.unsqueeze(-1)

        combined = sum(
            w * self._layer_norm(h, broadcast_mask, mask_float)
            for w, h in zip(normed_weights, hidden_states)
        )
        return self.gamma * combined

    def _layer_norm(
        self,
        tensor: torch.Tensor,
        broadcast_mask: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply layer normalization with masking."""
        tensor_masked = tensor * broadcast_mask
        batch_size, _, hidden_dim = tensor.size()

        num_elements = mask.sum(1) * hidden_dim
        mean = tensor_masked.view(batch_size, -1).sum(1)
        mean = (mean / num_elements).view(batch_size, 1, 1)

        variance = (
            (((tensor_masked - mean) * broadcast_mask) ** 2)
            .view(batch_size, -1)
            .sum(1)
            / num_elements
        )

        return (tensor - mean) / torch.sqrt(variance + 1e-12).view(batch_size, 1, 1)