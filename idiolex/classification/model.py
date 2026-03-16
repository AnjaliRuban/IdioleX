"""classification/model.py — model loading (.pth or HF directory)."""
import os
from typing import Union 

import torch
import torch.nn as nn
from torch.nn import Parameter, ParameterList
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig


# ---------------------------------------------------------------------------
# Layer pooling module
# ---------------------------------------------------------------------------

class LayerPooler(nn.Module):
    """Layer-wise attention mechanism for combining transformer layer outputs.

    Learns a weighted combination of all transformer layer outputs.
    Based on Rei et al. 2020.
    """

    def __init__(
        self,
        num_layers: int,
        layer_norm: bool = False,
        layer_weights: Union[list[float], None] = None,
        dropout: Union[float, None] = None,
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
            self.register_buffer("dropout_fill", torch.empty(num_layers).fill_(-1e20))

    def forward(
        self,
        hidden_states: list[torch.Tensor],
        attention_mask: Union[torch.Tensor, None] = None,
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
            combined = sum(
                w * h for w, h in zip(normed_weights, hidden_states)
            )
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

        variance = (((tensor_masked - mean) * broadcast_mask) ** 2).view(
            batch_size, -1
        ).sum(1) / num_elements


def last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool the last non-padding token from a sequence of hidden states.

    Handles both left-padded and right-padded sequences.

    Args:
        last_hidden_states: Hidden states of shape [batch_size, seq_len, hidden_dim].
        attention_mask: Attention mask of shape [batch_size, seq_len].

    Returns:
        Last token embeddings of shape [batch_size, hidden_dim].
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def average_pool(
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    padding_index: int = 1,
) -> torch.Tensor:
    """Mean pool over non-padding token embeddings.

    Args:
        tokens: Token IDs of shape [batch_size, seq_len].
        embeddings: Token embeddings of shape [batch_size, seq_len, hidden_dim].
        attention_mask: Attention mask of shape [batch_size, seq_len].
        padding_index: Index used for padding tokens.

    Returns:
        Mean-pooled embeddings of shape [batch_size, hidden_dim].
    """
    masked_embeddings = _mask_fill(0.0, tokens, embeddings, padding_index)
    sentence_embeddings = torch.sum(masked_embeddings, dim=1)
    sum_mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float().sum(1)
    return sentence_embeddings / sum_mask


def _mask_fill(
    fill_value: float,
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    padding_index: int,
) -> torch.Tensor:
    """Mask embeddings at padding positions with a fill value.

    Args:
        fill_value: Value to fill at padding positions.
        tokens: Token IDs of shape [batch_size, seq_len].
        embeddings: Token embeddings of shape [batch_size, seq_len, hidden_dim].
        padding_index: Index used for padding tokens.

    Returns:
        Masked embeddings with padding positions filled.
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)


# ---------------------------------------------------------------------------
# Full classification model with layer pooling
# ---------------------------------------------------------------------------

class LayerPoolClassifier(nn.Module):
    """
    Encoder + optional layer pooler + linear classification head.
    Mirrors the idiolex architecture when layerwise_pooling=True.
    """
    def __init__(self, encoder, layer_pooler: LayerPooler, num_classes: int, hidden_size: int):
        super().__init__()
        self.encoder      = encoder
        self.layer_pooler = layer_pooler
        self.classifier   = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Skip embedding layer (index 0), use transformer layer outputs only
        hidden_states = out.hidden_states
        pooled        = average_pool(
            tokens=input_ids,
            embeddings = self.layer_pooler(hidden_states, attention_mask),
            attention_mask=attention_mask
        )
        logits        = self.classifier(pooled)            # (batch, num_classes)

        # Return an object with .logits so training code is unchanged
        return _LogitsWrapper(logits)

    def save_pretrained(self, path):
        """Save encoder + pooler + head so we can reload for inference."""
        import os, json
        os.makedirs(path, exist_ok=True)
        self.encoder.save_pretrained(path)
        torch.save({
            "layer_pooler": self.layer_pooler.state_dict(),
            "classifier":   self.classifier.state_dict(),
        }, os.path.join(path, "head.pt"))

    @classmethod
    def from_pretrained(cls, path, num_classes: int):
        encoder     = AutoModel.from_pretrained(path)
        hidden_size = encoder.config.hidden_size
        num_layers  = encoder.config.num_hidden_layers

        pooler = LayerPooler(num_layers + 1)
        model  = cls(encoder, pooler, num_classes, hidden_size)

        head_path = os.path.join(path, "head.pt")
        if os.path.exists(head_path):
            state = torch.load(head_path, map_location="cpu")
            model.layer_pooler.load_state_dict(state["layer_pooler"])
            model.classifier.load_state_dict(state["classifier"])
        return model


class _LogitsWrapper:
    """Thin wrapper so LayerPoolClassifier output is compatible with training loop."""
    def __init__(self, logits):
        self.logits = logits


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(
    model_path:          str,
    num_classes:         int,
    multi_label:         bool,
    base_model_override: str  = None,
    use_layer_pooling:   bool = True,
):
    """
    Load a classification model from either:
      - An idiolex .pth checkpoint  (contains 'model' + optional 'embedding_model')
      - A saved LayerPoolClassifier directory (contains head.pt)
      - A standard HuggingFace model directory

    Parameters
    ----------
    use_layer_pooling : bool
        If True and loading from .pth, use layer pooling matching idiolex.
        If the checkpoint has no embedding_model, falls back to equal-weight
        pooling (still better than CLS-only). Default True.
    """
    if model_path.endswith(".pth"):
        return _load_from_pth(model_path, num_classes, base_model_override, use_layer_pooling)

    # Saved LayerPoolClassifier
    import os
    if os.path.exists(os.path.join(model_path, "head.pt")):
        print(f"  Loading LayerPoolClassifier from {model_path}")
        return LayerPoolClassifier.from_pretrained(model_path, num_classes)

    # Standard HF checkpoint (fallback)
    problem_type = (
        "multi_label_classification" if multi_label else "single_label_classification"
    )
    print(f"  Loading standard HF model from {model_path} (no layer pooling)")
    return AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        problem_type=problem_type,
    )


def _load_from_pth(path, num_classes, base_model_override, use_layer_pooling):
    ckpt       = torch.load(path, map_location="cpu", weights_only=False)
    saved_args = vars(ckpt["args"])

    base_model = base_model_override or saved_args.get("base_model")
    if base_model is None:
        raise ValueError(
            f"Could not find base model name in checkpoint args.\n"
            f"Available keys: {list(saved_args.keys())}\n"
            f"Pass it explicitly with --base-model."
        )
    print(f"  Base model        : {base_model}")
    print(f"  Checkpoint        : epoch {ckpt.get('epoch', '?')}, step {ckpt.get('step', '?')}")
    print(f"  Layerwise pooling : {saved_args.get('layerwise_pooling', False)}")

    encoder = AutoModel.from_pretrained(base_model)
    missing, unexpected = encoder.load_state_dict(ckpt["model"], strict=False)
    _report_key_mismatches(missing, unexpected)

    hidden_size = encoder.config.hidden_size
    num_layers  = encoder.config.num_hidden_layers + 1
    pooler      = LayerPooler(num_layers)

    # Load idiolex's trained layer weights if available
    if use_layer_pooling and ckpt.get("embedding_model") is not None:
        print("  Loading idiolex layer pooler weights...")
        pooler.load_state_dict(ckpt["embedding_model"])
    else:
        if use_layer_pooling:
            print("  No embedding_model in checkpoint — using uniform layer weights.")
        else:
            print("  use_layer_pooling=False — using uniform layer weights.")

    return LayerPoolClassifier(encoder, pooler, num_classes, hidden_size)


def _report_key_mismatches(missing, unexpected):
    non_trivial_missing = [k for k in missing if not k.startswith("pooler")]
    if non_trivial_missing:
        print(f"  Warning — missing keys: {non_trivial_missing}")
    else:
        print(f"  Encoder loaded cleanly.")
    if unexpected:
        print(f"  Warning — unexpected keys in checkpoint: {unexpected}")
