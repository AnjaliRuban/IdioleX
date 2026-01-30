"""Main training script for idiolectal representation learning."""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src import (
    FeatureHead,
    LayerwiseAttention,
    MeanCenterer,
    ProjectionHead,
    HierarchicalDataset,
    StandardDataset,
    bce_logits,
    concat_all_gather_no_grad,
    evaluate_model,
    get_dataloader,
    jaccard_weights,
    process_batch,
    supervised_contrastive,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train dialect embedding models with hierarchical contrastive learning."
    )

    # Model configuration
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--base_model",
        type=str,
        default="FacebookAI/roberta-base",
        help="Pre-trained model name for encoder.",
    )
    model_group.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pre-trained weights for encoder.",
    )
    model_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint for resuming training.",
    )
    model_group.add_argument(
        "--model_len",
        type=int,
        default=256,
        help="Maximum sequence length for input.",
    )

    # Data configuration
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--pretrain_data",
        type=str,
        help="Directory containing pre-training data.",
    )
    data_group.add_argument(
        "--train_data",
        type=str,
        help="Directory containing training data.",
    )
    data_group.add_argument(
        "--dev_data",
        type=str,
        required=True,
        help="Directory containing validation data.",
    )

    # Training configuration
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--pretrain",
        action="store_true",
        help="Enable ranking-only pre-training phase.",
    )
    train_group.add_argument(
        "--mini",
        action="store_true",
        help="Use mini batches (size 4) instead of full (size 16).",
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (must be divisible by mini batch size).",
    )
    train_group.add_argument(
        "--pretrain_epoch",
        type=int,
        default=3,
        help="Number of pre-training epochs.",
    )
    train_group.add_argument(
        "--epoch",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    train_group.add_argument(
        "--dev_step",
        type=int,
        default=250,
        help="Steps between validation runs.",
    )
    train_group.add_argument(
        "--dev_size",
        type=int,
        default=None,
        help="Number of validation samples (None for full set).",
    )

    # Optimizer configuration
    optim_group = parser.add_argument_group("Optimizer")
    optim_group.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    optim_group.add_argument(
        "--warmup_lr",
        type=int,
        default=25_000,
        help="Steps for learning rate warmup.",
    )
    optim_group.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience.",
    )

    # Loss configuration
    loss_group = parser.add_argument_group("Loss")
    loss_group.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Weight for feature loss vs ranking loss.",
    )
    loss_group.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="Decay rate for alpha.",
    )
    loss_group.add_argument(
        "--sigma_margin",
        type=float,
        default=0.5,
        help="Margin for margin ranking loss.",
    )
    loss_group.add_argument(
        "--warmup_margin",
        type=int,
        default=25_000,
        help="Steps for margin warmup.",
    )

    # Feature learning configuration
    feat_group = parser.add_argument_group("Features")
    feat_group.add_argument(
        "--feat_dim",
        type=int,
        default=56,
        help="Number of linguistic features.",
    )
    feat_group.add_argument(
        "--supcon_tau",
        type=float,
        default=0.07,
        help="Temperature for supervised contrastive loss.",
    )
    feat_group.add_argument(
        "--supcon_topk",
        type=int,
        default=5,
        help="Top-k positives for Jaccard weighting.",
    )

    # Model variants
    variant_group = parser.add_argument_group("Model Variants")
    variant_group.add_argument(
        "--layerwise_pooling",
        action="store_true",
        help="Use layerwise attention pooling.",
    )
    variant_group.add_argument(
        "--mean_center",
        action="store_true",
        help="Apply mean centering to embeddings.",
    )

    # Experiment configuration
    exp_group = parser.add_argument_group("Experiment")
    exp_group.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Experiment name for logging and saving.",
    )
    exp_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    exp_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    exp_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation only (requires --checkpoint).",
    )

    return parser.parse_args()


def setup_distributed() -> None:
    """Initialize distributed training environment."""
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")


def init_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_logging(args: argparse.Namespace) -> None:
    """Initialize wandb logging."""
    wandb.init(
        project="dialect-metric",
        name=args.tag,
        config={
            "command": f"{sys.executable} {' '.join(sys.argv)}",
            "model": args.txt_model,
            "lr": args.lr,
            "model_len": args.model_len,
            "batch_size": args.batch_size,
            "mini": args.mini,
            "alpha": args.alpha,
            "beta": args.beta,
            "sigma_margin": args.sigma_margin,
            "layerwise_pooling": args.layerwise_pooling,
            "mean_center": args.mean_center,
            "num_devices": dist.get_world_size(),
        },
    )

    os.makedirs(f"models/{args.tag}", exist_ok=True)


def init_models(
    args: argparse.Namespace,
    device_id: int,
    no_grad: bool = False,
) -> dict:
    """Initialize all models and optimizer."""
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        checkpoint_id = args.checkpoint
        args = checkpoint["args"]
        args.checkpoint = checkpoint_id

    parameters = [] if not no_grad else None

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    config = AutoConfig.from_pretrained(args.base_model)
    config.vocab_size = len(tokenizer)
    config.output_hidden_states = True

    if args.pretrained:
        model = AutoModel.from_pretrained(args.base_model).to(device_id)
        model.config.output_hidden_states = True
    else:
        model = AutoModel.from_config(config).to(device_id)

    if checkpoint:
        model.load_state_dict(checkpoint["model"])

    txt_model = DistributedDataParallel(
        model, device_ids=[device_id], find_unused_parameters=True
    )

    if not no_grad:
        parameters.extend(model.parameters())

    # Initialize optional components
    embedding_model = None
    if args.layerwise_pooling:
        num_layers = txt_model.module.config.num_hidden_layers + 1
        embedding_model = LayerwiseAttention(num_layers=num_layers).to(device_id)
        if checkpoint:
            embedding_model.load_state_dict(checkpoint["embedding_model"])
        embedding_model = DistributedDataParallel(
            embedding_model, device_ids=[device_id], find_unused_parameters=True
        )
        if not no_grad:
            parameters.extend(embedding_model.parameters())

    centering_model = None
    if args.mean_center:
        centering_model = MeanCenterer(dim=config.hidden_size).to(device_id)
        if checkpoint:
            centering_model.load_state_dict(checkpoint["centering_model"])
        if not no_grad:
            parameters.extend(centering_model.parameters())

    feat_head = FeatureHead(
        input_dim=config.hidden_size, feat_dim=args.feat_dim
    ).to(device_id)
    proj_head = ProjectionHead(input_dim=config.hidden_size, out_dim=256).to(
        device_id
    )
    if checkpoint:
        feat_head.load_state_dict(checkpoint["feat_head"])
        proj_head.load_state_dict(checkpoint["proj_head"])
    feat_head = DistributedDataParallel(
        feat_head, device_ids=[device_id], find_unused_parameters=True
    )
    proj_head = DistributedDataParallel(
        proj_head, device_ids=[device_id], find_unused_parameters=True
    )
    if not no_grad:
        parameters.extend(proj_head.parameters())

    optimizer = None if parameters is None else torch.optim.Adam(parameters, lr=args.lr)
    if checkpoint and not no_grad:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return {
        "models": {
            "model": model,
            "embedding_model": embedding_model,
            "centering_model": centering_model,
            "feat_head": feat_head,
            "proj_head": proj_head,
        },
        "metrics": {"best_mrr": 0, "best_loss": float("inf")},
        "optimizer": optimizer,
        "args": args,
    }


def train(
    model_dict: dict,
    data: HierarchicalDataset,
    dev_data: HierarchicalDataset,
    args: argparse.Namespace,
    device_id: int,
    world_size: int,
    pretrain: bool = False,
) -> None:
    """Run training loop."""
    models = model_dict["models"]
    model = models["model"]
    embedding_model = models["embedding_model"]
    centering_model = models["centering_model"]
    feat_head = models["feat_head"]
    proj_head = models["proj_head"]

    optimizer = model_dict["optimizer"]
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_lr
    )

    model.train()
    data_sampler = DistributedSampler(
        data, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=0
    )

    best_loss = model_dict["metrics"]["best_loss"]
    patience = 0
    stop_flag = torch.tensor([0], device=device_id)
    num_epochs = args.pretrain_epoch if pretrain else args.epoch
    args.step = 0

    for epoch in range(num_epochs):
        if args.debug:
            init_seed(0)
        else:
            data_sampler.set_epoch(epoch)

        dataloader = get_dataloader(data, args, sampler_cls=data_sampler.__class__)

        for i, batch in enumerate(dataloader):
            if pretrain and stop_flag.item() == 1:
                return

            # Compute current margin with warmup
            progress = (epoch * len(data) + i * world_size) / args.warmup_margin
            curr_margin = (
                min(args.sigma_margin * progress, args.sigma_margin)
                if pretrain
                else args.sigma_margin
            )

            # Forward pass
            _, raw_out, batch_loss, batch_mrr, batch_metrics = process_batch(
                model=model,
                batch=batch,
                mini=args.mini,
                sigma_margin=curr_margin,
                embedding_model=embedding_model,
                centering_model=centering_model,
                device_id=device_id,
                verbose=args.verbose,
            )

            # Compute feature loss if not in pretrain phase
            if pretrain:
                feature_loss = torch.tensor(0.0, device=device_id)
                loss = batch_loss
            else:
                feat_out = feat_head(raw_out)
                pred_loss = bce_logits(feat_out, batch["feat_ids"].to(device_id))

                with torch.no_grad():
                    feat_bin = (batch["feat_ids"].to(device_id) > 0.5).float()
                    feat_bin_all = concat_all_gather_no_grad(feat_bin)

                    if dist.is_available() and dist.is_initialized():
                        rank = dist.get_rank()
                        local_batch = feat_bin.size(0)
                        start, end = rank * local_batch, (rank + 1) * local_batch
                        w_all = jaccard_weights(feat_bin_all, topk=args.supcon_topk)
                        w = w_all[start:end, start:end].contiguous()
                    else:
                        w = jaccard_weights(feat_bin, topk=args.supcon_topk)

                proj_out = proj_head(raw_out)
                supcon_loss = supervised_contrastive(proj_out, w, tau=args.supcon_tau)
                feature_loss = 0.25 * pred_loss + supcon_loss
                loss = args.alpha * feature_loss + (1 - args.alpha) * batch_loss
                

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Logging
            if device_id == 0:
                print(
                    f"Epoch {epoch} | Step {i}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f} | MRR: {batch_mrr.item():.4f}"
                )
                args.step += world_size
                wandb.log(
                    {
                        "epoch": epoch,
                        "step": args.step,
                        "train/loss": loss.item(),
                        "train/rank_loss": batch_loss.item(),
                        "train/feat_loss": feature_loss.item(),
                        "train/alpha": args.alpha,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/mrr": batch_mrr.item(),
                        "train/ndcg": batch_metrics["ndcg"].item(),
                        "train/ndcg@1": batch_metrics["ndcg@1"].item(),
                        "train/ndcg@3": batch_metrics["ndcg@3"].item(),
                        "train/margin": curr_margin,
                    }
                )

            args.alpha = max(0, args.alpha - args.beta)

            # Validation
            if i % args.dev_step == 0:
                metrics = validate(args, model_dict, dev_data, device_id, world_size)
                model.train()

                if device_id == 0:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "step": args.step,
                            "val/loss": metrics["loss"].item(),
                            "val/mrr": metrics["mrr"].item(),
                            "val/ndcg": metrics["metrics"]["ndcg"].item(),
                            "val/ndcg@1": metrics["metrics"]["ndcg@1"].item(),
                            "val/ndcg@3": metrics["metrics"]["ndcg@3"].item(),
                            "patience": patience,
                        }
                    )

                    if metrics["loss"].item() < best_loss:
                        patience = 0
                        best_loss = metrics["loss"].item()
                        save_checkpoint(args, model_dict, epoch, i, world_size)
                    else:
                        patience += 1
                        stop_flag[0] = 1 if pretrain and patience >= args.patience else 0

                if dist.is_initialized():
                    dist.broadcast(stop_flag, src=0)


def validate(
    args: argparse.Namespace,
    model_dict: dict,
    data: HierarchicalDataset,
    device_id: int,
    world_size: int,
) -> dict:
    """Run validation and aggregate metrics across ranks."""
    models = model_dict["models"]

    with torch.no_grad():
        loss, mrr, metrics = evaluate_model(
            models["model"].module,
            data,
            args,
            device_id,
            embedding_model=models["embedding_model"],
            centering_model=models["centering_model"],
            proj_head=models["proj_head"],
            feat_head=models["feat_head"],
        )

    # Aggregate across ranks
    for tensor in [loss, mrr]:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size

    for v in metrics.values():
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
        v /= world_size

    return {"loss": loss, "mrr": mrr, "metrics": metrics}


def save_checkpoint(
    args: argparse.Namespace,
    model_dict: dict,
    epoch: int,
    step: int,
    world_size: int,
) -> None:
    """Save model checkpoint."""
    models = model_dict["models"]
    torch.save(
        {
            "model": models["model"].module.state_dict(),
            "embedding_model": (
                models["embedding_model"].module.state_dict()
                if args.layerwise_pooling
                else None
            ),
            "centering_model": (
                models["centering_model"].state_dict() if args.mean_center else None
            ),
            "feat_head": (
                models["feat_head"].module.state_dict()
            ),
            "proj_head": (
                models["proj_head"].module.state_dict()
            ),
            "optimizer": model_dict["optimizer"].state_dict(),
            "args": args,
            "epoch": epoch,
            "step": args.step + step * world_size,
        },
        f"models/{args.tag}/checkpoint.pth",
    )


def run(args: argparse.Namespace) -> None:
    """Main training/evaluation entry point."""
    setup_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()

    if device_id == 0 and not args.evaluate:
        init_logging(args)

    print(f"World size: {world_size}, Rank: {rank}, Device: {device_id}")

    model_dict = init_models(args, device_id)
    args = model_dict["args"]

    # Evaluation mode
    if args.evaluate:
        eval_data = StandardDataset(args.dev_data)
        with torch.no_grad():
            loss, mrr, metrics = evaluate_model(
                model=model_dict["models"]["model"].module,
                dataset=eval_data,
                args=args,
                device_id=device_id,
                embedding_model=model_dict["models"]["embedding_model"],
                centering_model=model_dict["models"]["centering_model"],
            )

        # Aggregate across ranks
        for tensor in [loss, mrr]:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= world_size

        for v in metrics.values():
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
            v /= world_size

        if device_id == 0:
            results = {
                "loss": loss.item(),
                "mrr": mrr.item(),
                **{k: v.item() for k, v in metrics.items()},
            }
            print(f"Evaluation results: {results}")
            os.makedirs("evals", exist_ok=True)
            with open(f"evals/{args.tag}.json", "w") as f:
                json.dump(results, f, indent=2)

        dist.destroy_process_group()
        return

    # Load datasets
    train_data = HierarchicalDataset(args.train_data)
    dev_data = HierarchicalDataset(args.dev_data)

    # Training phases
    train(model_dict, train_data, dev_data, args, device_id, world_size, pretrain=True)

    if args.pretrain:
        pretrain_data = HierarchicalDataset(args.pretrain_data, feat_len=args.feat_dim)
        args.model_len = args.model_len * 2
        train(model_dict, pretrain_data, dev_data, args, device_id, world_size)

    dist.destroy_process_group()


def main() -> None:
    """Entry point."""
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()