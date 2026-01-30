"""Utility functions for embedding operations, loss computation, and metrics."""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F


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


def anchor_on_k(out: torch.Tensor, n: int) -> torch.Tensor:
    """Reorder tensor columns to place the n-th column first.

    Args:
        out: Input tensor of shape [batch_size, seq_len].
        n: Index of column to move to front (0 <= n < seq_len).

    Returns:
        Reordered tensor with column n at position 0.

    Raises:
        RuntimeError: If n is out of valid range.
    """
    if not (0 <= n < out.shape[1]):
        raise RuntimeError(f"n={n} must be in range [0, {out.shape[1]})")

    if not out.is_contiguous():
        out = out.contiguous()

    idx = out.shape[1] // 2
    head = 0

    while head != n:
        if (head + idx) <= n:
            out = torch.cat(
                (out[:, idx : idx * 2], out[:, :idx], out[:, idx * 2 :]),
                dim=1,
            )
            head += idx
        idx = idx // 2

    if not out.is_contiguous():
        out = out.contiguous()

    return out


def get_anchor_indices(mini: bool, n: int) -> torch.Tensor:
    """Get indices for anchor positions in a batch.

    Args:
        mini: If True, use mini batches (size 4), else full batches (size 16).
        n: Offset for each group.

    Returns:
        Tensor of anchor indices.
    """
    batch_size = 4 if mini else 16
    return torch.tensor(
        [i + n for i in range(0, batch_size, batch_size) for _ in range(batch_size)]
    )


def margin_ranking_loss(
    predictions: torch.Tensor,
    graded_relevance: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Compute pairwise margin ranking loss.

    Args:
        predictions: Predicted scores of shape [batch_size, num_items].
        graded_relevance: Relevance labels of shape [num_items].
        margin: Margin for the ranking loss.

    Returns:
        Scalar margin ranking loss.
    """
    batch_size, num_items = predictions.shape
    device = predictions.device

    pred_i = predictions.unsqueeze(2).expand(batch_size, num_items, num_items)
    pred_j = predictions.unsqueeze(1).expand(batch_size, num_items, num_items)

    rel_diffs = graded_relevance.view(1, num_items, 1) - graded_relevance.view(
        1, 1, num_items
    )
    rel_diffs = rel_diffs.expand(batch_size, -1, -1).to(device)
    mask = rel_diffs != 0

    target = torch.sign(rel_diffs)

    input1 = pred_i[mask]
    input2 = pred_j[mask]
    target = target[mask]

    return F.margin_ranking_loss(input1, input2, target, margin=margin, reduction="mean")


def mean_reciprocal_rank(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
) -> float:
    """Compute Mean Reciprocal Rank (MRR) for a batch of queries.

    Args:
        y_true: Ground truth relevance of shape [batch_size, num_items].
        y_score: Predicted scores of shape [batch_size, num_items].

    Returns:
        MRR score averaged over the batch.
    """
    batch_size = y_true.shape[0]
    mrr = 0.0

    for i in range(batch_size):
        relevant_indices = (y_true[i] > 0).nonzero(as_tuple=True)[0]
        if relevant_indices.numel() == 0:
            continue

        relevant_scores = y_score[i, relevant_indices]
        sorted_indices = torch.argsort(relevant_scores, descending=True)
        rank = (sorted_indices[0] + 1).float()
        mrr += 1.0 / rank

    return mrr / batch_size if batch_size > 0 else 0.0


def vicreg_regularizer(
    embeddings: torch.Tensor,
    eps: float = 1e-4,
    gamma: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """VICReg regularization loss for embedding diversity.

    Args:
        embeddings: Embeddings of shape [batch_size, dim].
        eps: Small constant for numerical stability.
        gamma: Weight for variance loss term.
        beta: Weight for decorrelation loss term.

    Returns:
        Combined variance and decorrelation loss.
    """
    std = embeddings.std(dim=0) + eps
    var_loss = F.relu(1.0 - std).mean()

    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (centered.size(0) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    decor_loss = (off_diag**2).mean()

    return gamma * var_loss + beta * decor_loss


def concat_all_gather_no_grad(x: torch.Tensor) -> torch.Tensor:
    """Gather tensors across DDP ranks without gradient propagation."""
    if not (dist.is_available() and dist.is_initialized()):
        return x
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    with torch.no_grad():
        dist.all_gather(gathered, x)
    return torch.cat(gathered, dim=0)


def jaccard_weights(
    binary_features: torch.Tensor,
    topk: int = 5,
) -> torch.Tensor:
    """Compute Jaccard similarity weights between samples.

    Args:
        binary_features: Binary feature vectors of shape [batch_size, num_features].
        topk: Number of top similar pairs to keep per sample.

    Returns:
        Weight matrix of shape [batch_size, batch_size] with top-k per row.
    """
    inter = binary_features @ binary_features.T
    sums = binary_features.sum(dim=1, keepdim=True)
    union = sums + sums.T - inter + 1e-8
    weights = inter / union
    weights.fill_diagonal_(0.0)

    if topk > 0:
        k = min(topk, weights.size(1) - 1)
        kth_values = torch.topk(weights, k=k, dim=1).values[:, -1].unsqueeze(1)
        weights = (weights >= kth_values).float() * weights

    return weights


def supervised_contrastive(
    embeddings: torch.Tensor,
    weights: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss with weighted positives.

    Args:
        embeddings: L2-normalized embeddings of shape [batch_size, dim].
        weights: Non-negative pairwise weights of shape [batch_size, batch_size].
        tau: Temperature parameter.

    Returns:
        Scalar contrastive loss.
    """
    sim = (embeddings @ embeddings.T) / tau
    log_den = sim.logsumexp(dim=1, keepdim=True)
    log_prob = sim - log_den
    loss = -(weights * log_prob).sum() / (weights.sum() + 1e-8)
    return loss


def bce_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy loss with logits."""
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")