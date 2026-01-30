"""Evaluation utilities for model assessment."""

import argparse
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.sampler import Sampler

from .data_utils import TripletSampler, make_collator
from .process import process_batch
from .utils import (
    bce_logits,
    concat_all_gather_no_grad,
    jaccard_weights,
    supervised_contrastive,
)


def get_dataloader(
    dataset: Dataset,
    args: argparse.Namespace,
    sampler_cls: type[Sampler] = RandomSampler,
) -> DataLoader:
    """Create a DataLoader for training or evaluation.

    Args:
        dataset: Dataset to load from.
        args: Arguments containing batch_size, mini, and model_len.
        sampler_cls: Sampler class to use.

    Returns:
        Configured DataLoader.
    """
    if args.evaluate:
        return DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=make_collator(args),
            num_workers=4,
        )
    return DataLoader(
        dataset=dataset,
        batch_sampler=TripletSampler(
            sampler=sampler_cls(dataset),
            batch_size=args.batch_size,
            drop_last=False,
            mini=args.mini,
        ),
        collate_fn=make_collator(args),
        num_workers=4,
    )


def evaluate_model(
    model: torch.nn.Module,
    dataset: Dataset,
    args: argparse.Namespace,
    device_id: int | str,
    embedding_model: Optional[torch.nn.Module] = None,
    centering_model: Optional[torch.nn.Module] = None,
    feat_head: Optional[torch.nn.Module] = None,
    proj_head: Optional[torch.nn.Module] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate model on a dataset.

    Args:
        model: Text encoder model.
        dataset: Dataset to evaluate on.
        args: Training arguments.
        device_id: Device for computation.
        embedding_model: Optional layerwise attention model.
        centering_model: Optional mean centering model.
        feat_head: Optional feature prediction head.
        proj_head: Optional projection head.

    Returns:
        Tuple of (avg_loss, avg_mrr, metrics_dict).
    """
    model.eval()
    dataloader = get_dataloader(dataset, args, sampler_cls=RandomSampler)

    if args.evaluate:
        outputs = []
    else:
        mrr_values = []
        loss_values = []
        all_metrics: dict[str, list[float]] = {
            "ndcg": [],
            "ndcg@1": [],
            "ndcg@3": [],
        }

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if args.evaluate:
                txt_out, raw_out, _, _, _, _ = process_batch(
                    model=model,
                    batch=batch,
                    embedding_model=embedding_model,
                    centering_model=centering_model,
                    nu=args.nu,
                    device_id=device_id,
                    eval_mode=True,
                    verbose=args.verbose,
                )
                outputs.extend([{
                    "idx": batch["idxs"][i],
                    "tags": batch["tags"][i],
                    "input_ids": batch["input_ids"][i].cpu().tolist(),
                    "txt_out": t.cpu().tolist(),
                    "raw_out": r.cpu().tolist(),
                } for i, t, r in zip(range(len(batch["idxs"])), txt_out, raw_out)])
                
            else:
                txt_out, raw_out, batch_loss, batch_mrr, batch_metrics = process_batch(
                    model=model,
                    batch=batch,
                    mini=args.mini,
                    sigma_margin=args.sigma_margin,
                    embedding_model=embedding_model,
                    centering_model=centering_model,
                    device_id=device_id,
                    verbose=args.verbose,
                )

                if batch_mrr is not None:
                    mrr_values.append(batch_mrr.item())

                if batch_metrics is not None:
                    for k, v in batch_metrics.items():
                        all_metrics[k].append(v)

                # Compute feature loss if heads are provided
                if feat_head and proj_head:
                    feat_out = feat_head(raw_out)
                    pred_loss = bce_logits(feat_out, batch["feat_ids"].to(device_id))

                    with torch.no_grad():
                        feat_bin_local = (batch["feat_ids"].to(device_id) > 0.5).float()
                        feat_bin_all = concat_all_gather_no_grad(feat_bin_local)

                        if dist.is_available() and dist.is_initialized():
                            world_size = dist.get_world_size()
                            rank = dist.get_rank()
                            local_batch = feat_bin_local.size(0)
                            start = rank * local_batch
                            end = start + local_batch

                            w_all = jaccard_weights(feat_bin_all, topk=args.supcon_topk)
                            w = w_all[start:end, start:end].contiguous()
                        else:
                            w = jaccard_weights(feat_bin_local, topk=args.supcon_topk)

                    proj_out = proj_head(raw_out)
                    supcon_loss = supervised_contrastive(proj_out, w, tau=args.supcon_tau)
                    feature_loss = 0.25 * pred_loss + supcon_loss
                else:
                    feature_loss = torch.tensor(0.0, device=device_id)

                loss = args.alpha * feature_loss + (1 - args.alpha) * batch_loss
                if loss is not None:
                    loss_values.append(loss.item())

                if args.dev_size is not None and i >= args.dev_size:
                    break

    avg_loss = torch.mean(torch.tensor(loss_values))
    avg_mrr = torch.mean(torch.tensor(mrr_values))
    avg_metrics = {k: torch.mean(torch.tensor(v)) for k, v in all_metrics.items()}

    if args.evaluate:
        return outputs
    return avg_loss, avg_mrr, avg_metrics