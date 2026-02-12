"""Batch processing for training and evaluation."""

from typing import Union

import torch
import torch.nn.functional as F
from sklearn.metrics import ndcg_score

from idiolex.src.utils import (
    anchor_on_k,
    average_pool,
    get_anchor_indices,
    last_token_pool,
    margin_ranking_loss,
    mean_reciprocal_rank,
    vicreg_regularizer,
)


def process_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device_id: Union[int, str],
    mini: bool = False,
    sigma_margin: float = 0.1,
    embedding_model: Union[torch.nn.Module, None] = None,
    centering_model: Union[torch.nn.Module, None] = None,
    verbose: bool = False,
    eval_mode: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Union[torch.Tensor, None],
    Union[torch.Tensor, None],
    Union[dict[str, torch.Tensor], None],
]:
    """Process a single batch and compute embeddings, loss, and metrics.

    Args:
        model: The base encoder model.
        batch: Dictionary containing:
            - input_ids: Token IDs [batch_size, seq_len]
            - input_attn_mask: Attention mask [batch_size, seq_len]
            - graded_relevance: Relevance scores [batch_size]
        device_id: Device to run computation on.
        mini: Use mini batches (size 4) vs full batches (size 16).
        sigma_margin: Margin for margin ranking loss.
        embedding_model: Optional layerwise attention model.
        centering_model: Optional mean centering model.
        verbose: Print debug information.
        eval_mode: If True, only compute embeddings.

    Returns:
        Tuple of (normalized_embeddings, raw_embeddings, loss, mrr, metrics).
        In eval_mode, loss/mrr/metrics are None.
    """
    input_ids = batch["input_ids"].to(device_id)
    attention_mask = batch["input_attn_mask"].to(device_id)

    model_out = model(input_ids=input_ids, attention_mask=attention_mask)

    if embedding_model:
        raw_out = average_pool(
            tokens=input_ids,
            embeddings=embedding_model(
                hidden_states=model_out.hidden_states,
                attention_mask=attention_mask,
            ),
            attention_mask=attention_mask,
        )
    else:
        raw_out = last_token_pool(
            model_out.last_hidden_state,
            attention_mask=attention_mask,
        )

    # Apply centering and normalization
    if centering_model and not eval_mode:
        with torch.no_grad():
            centering_model.update(raw_out)

    if centering_model:
        txt_out = centering_model.apply(raw_out)
    else:
        txt_out = F.normalize(raw_out, p=2, dim=-1)

    if eval_mode:
        return txt_out, raw_out, None, None, None

    # Initialize loss computation
    batch_mrr = []
    ndcg_scores = {"ndcg": [], "ndcg@1": [], "ndcg@3": []}
    batch_loss = []
    graded_relevance = batch["graded_relevance"].to(device_id)

    # VICReg regularization
    var_loss = vicreg_regularizer(txt_out)

    # Process each anchor position
    N = 4 if mini else 16
    assert txt_out.shape[0] % N == 0, "Batch size must be divisible by N"

    batch_size = txt_out.shape[0] // N

    for n in range(N):
        anchor_idxs = get_anchor_indices(batch_size, mini, n)
        anchors = txt_out[anchor_idxs].to(device_id)

        # Compute similarities using dot product
        candidates = txt_out
        similarity = torch.sum(anchors * candidates, dim=-1)

        similarity = similarity.view(-1, N)
        similarity = anchor_on_k(similarity, n)

        # Compute margin loss
        margin_loss = (
            margin_ranking_loss(similarity, graded_relevance, margin=sigma_margin)
            / batch_size
        )
        batch_loss.append(margin_loss)

        # Compute metrics
        y_true = graded_relevance.unsqueeze(0).repeat(batch_size, 1).detach().cpu()
        y_score = similarity.detach().cpu()

        if y_true.shape[1] > 1:
            y_true_no_anchor = y_true[:, 1:]
            y_score_no_anchor = y_score[:, 1:]

            ndcg_scores["ndcg"].append(ndcg_score(y_true_no_anchor, y_score_no_anchor))
            ndcg_scores["ndcg@1"].append(
                ndcg_score(y_true_no_anchor, y_score_no_anchor, k=1)
            )
            ndcg_scores["ndcg@3"].append(
                ndcg_score(y_true_no_anchor, y_score_no_anchor, k=3)
            )

        batch_mrr.append(
            mean_reciprocal_rank(
                graded_relevance.unsqueeze(0)[:, 1:], similarity[:, 1:]
            )
        )

    # Aggregate metrics
    metrics = {
        k: torch.tensor(sum(v) / len(v) if v else 0.0).to(device_id)
        for k, v in ndcg_scores.items()
    }
    batch_mrr = (
        (sum(batch_mrr) / len(batch_mrr)).to(device_id)
        if batch_mrr
        else torch.tensor(0.0).to(device_id)
    )

    total_loss = torch.stack(batch_loss).sum() + 0.25 * var_loss

    return txt_out, raw_out, total_loss, batch_mrr, metrics
