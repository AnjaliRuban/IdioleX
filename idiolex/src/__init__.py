"""Dialect embedding learning with hierarchical contrastive training."""

from .centering import MeanCenterer
from .data_utils import (
    HierarchicalDataset,
    StandardDataset,
    TripletSampler,
    make_collator,
)
from .evaluation import evaluate_model, get_dataloader
from .feature_head import FeatureHead, ProjectionHead
from .layer_pool import LayerwiseAttention
from .process import mean_reciprocal_rank, process_batch
from .utils import (
    anchor_on_k,
    average_pool,
    bce_logits,
    concat_all_gather_no_grad,
    get_anchor_indices,
    jaccard_weights,
    last_token_pool,
    margin_ranking_loss,
    mean_reciprocal_rank,
    supervised_contrastive,
    vicreg_regularizer,
)

__all__ = [
    # Centering
    "MeanCenterer",
    # Data utilities
    "HierarchicalDataset",
    "StandardDataset",
    "TripletSampler",
    "make_collator",
    # Evaluation
    "evaluate_model",
    "get_dataloader",
    # Feature heads
    "FeatureHead",
    "ProjectionHead",
    # Layer pooling
    "LayerwiseAttention",
    # Processing
    "process_batch",
    # Utilities
    "anchor_on_k",
    "average_pool",
    "bce_logits",
    "concat_all_gather_no_grad",
    "get_anchor_indices",
    "jaccard_weights",
    "last_token_pool",
    "margin_ranking_loss",
    "mean_reciprocal_rank",
    "supervised_contrastive",
    "vicreg_regularizer",
]
