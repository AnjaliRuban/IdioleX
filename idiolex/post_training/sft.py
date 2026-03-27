"""
Stage 1: SFT with IdioleX embedding alignment loss (multi-GPU).

Loss = CE_loss + alpha * (1 - cosine_sim(projected_hidden_states, idiolex_embedding))

Usage:
    accelerate launch --config_file accelerate_config.yaml train_sft_idiolex.py \
        --train_jsonl data/dialect_instructions/train.jsonl \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --reward_checkpoint models/idiolex/checkpoint.pth \
        --output_dir models/sft_idiolex
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from idiolex.src.centering import MeanCenterer
from idiolex.src.layer_pool import LayerwiseAttention


# ---------------------------------------------------------------------------
# IdioleX encoder (frozen, for precomputing target embeddings only)
# ---------------------------------------------------------------------------

def average_pool(tokens, embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def last_token_pool(hidden_states, attention_mask):
    seq_len = attention_mask.sum(dim=1) - 1
    return hidden_states[torch.arange(hidden_states.size(0)), seq_len]


class IdioleXEncoder(nn.Module):
    def __init__(self, model, tokenizer, embedding_model=None, centering_model=None, max_length=512):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.centering_model = centering_model
        self.max_length = max_length
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        if self.embedding_model is not None:
            self.embedding_model.eval()
            for p in self.embedding_model.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def encode(self, texts: list[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if self.embedding_model is not None:
            embeddings = average_pool(
                tokens=input_ids,
                embeddings=self.embedding_model(
                    hidden_states=outputs.hidden_states,
                    attention_mask=attention_mask,
                ),
                attention_mask=attention_mask,
            )
        else:
            embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)

        if self.centering_model is not None:
            embeddings = self.centering_model.apply(embeddings)
        else:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu"):
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        args = checkpoint["args"]

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        config = AutoConfig.from_pretrained(args.base_model)
        config.output_hidden_states = True

        model = AutoModel.from_config(config)
        model.load_state_dict(checkpoint["model"])
        model = model.to(device)

        embedding_model = None
        if args.layerwise_pooling and checkpoint.get("embedding_model"):
            num_layers = model.config.num_hidden_layers + 1
            embedding_model = LayerwiseAttention(num_layers=num_layers).to(device)
            embedding_model.load_state_dict(checkpoint["embedding_model"])

        centering_model = None
        if args.mean_center and checkpoint.get("centering_model"):
            centering_model = MeanCenterer(dim=config.hidden_size).to(device)
            centering_model.load_state_dict(checkpoint["centering_model"])

        return cls(model=model, tokenizer=tokenizer,
                   embedding_model=embedding_model, centering_model=centering_model)


# ---------------------------------------------------------------------------
# Projection head: LLM hidden dim -> IdioleX embedding dim
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    def __init__(self, llm_hidden_dim: int, idiolex_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(llm_hidden_dim, llm_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(llm_hidden_dim // 2, idiolex_dim),
        )

    def forward(self, x):
        weight_dtype = self.proj[0].weight.dtype
        x = x.to(weight_dtype)
        return F.normalize(self.proj(x), p=2, dim=-1)


# ---------------------------------------------------------------------------
# Combined model for DeepSpeed
# ---------------------------------------------------------------------------

class LLMWithProjection(nn.Module):
    def __init__(self, llm, projection):
        super().__init__()
        self.llm = llm
        self.projection = projection

    def forward(self, input_ids, attention_mask, response_starts, target_embeddings, alpha=0.5):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # CE loss on response tokens
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].clone()
        for i in range(shift_labels.shape[0]):
            start = response_starts[i].item() - 1
            if start > 0:
                shift_labels[i, :start] = -100

        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Embedding alignment loss
        last_hidden = outputs.hidden_states[-1]
        pooled = pool_response_hidden_states(last_hidden, attention_mask, response_starts)
        projected = self.projection(pooled)
        emb_loss = 1.0 - F.cosine_similarity(projected, target_embeddings.to(projected.dtype)).mean()

        combined_loss = ce_loss + alpha * emb_loss
        return combined_loss, ce_loss, emb_loss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DialectSFTDataset(Dataset):
    def __init__(self, llm_tokenizer, idiolex_embeddings: torch.Tensor,
                 raw_samples: list[dict], max_seq_length: int = 1024):
        self.samples = []
        self.idiolex_embeddings = []

        for idx, sample in enumerate(tqdm(raw_samples, desc="Tokenizing")):
            prompt = sample["prompt"]
            gt = sample["ground_truth"]

            if isinstance(prompt, list):
                messages = list(prompt)
            elif isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": str(prompt)}]

            messages.append({"role": "assistant", "content": gt})

            full_text = llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            full_tokens = llm_tokenizer(
                full_text, truncation=True, max_length=max_seq_length,
                return_tensors="pt",
            )

            prompt_messages = messages[:-1]
            prompt_text = llm_tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = llm_tokenizer(
                prompt_text, truncation=True, max_length=max_seq_length,
            )
            response_start = len(prompt_tokens["input_ids"])

            self.samples.append({
                "input_ids": full_tokens["input_ids"].squeeze(0),
                "attention_mask": full_tokens["attention_mask"].squeeze(0),
                "response_start": response_start,
            })
            self.idiolex_embeddings.append(idiolex_embeddings[idx])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.samples[idx]["input_ids"],
            "attention_mask": self.samples[idx]["attention_mask"],
            "response_start": self.samples[idx]["response_start"],
            "idiolex_embedding": self.idiolex_embeddings[idx],
        }


def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids, attention_mask, response_starts, idiolex_embeddings = [], [], [], []

    for b in batch:
        pad_len = max_len - b["input_ids"].shape[0]
        input_ids.append(F.pad(b["input_ids"], (pad_len, 0), value=0))
        attention_mask.append(F.pad(b["attention_mask"], (pad_len, 0), value=0))
        response_starts.append(b["response_start"] + pad_len)
        idiolex_embeddings.append(b["idiolex_embedding"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "response_starts": torch.tensor(response_starts),
        "idiolex_embeddings": torch.stack(idiolex_embeddings),
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def pool_response_hidden_states(hidden_states, attention_mask, response_starts):
    batch_size = hidden_states.shape[0]
    pooled = []
    for i in range(batch_size):
        start = response_starts[i].item()
        end = attention_mask[i].sum().item()
        if start >= end:
            pooled.append(hidden_states[i, end - 1, :])
        else:
            pooled.append(hidden_states[i, start:end, :].mean(dim=0))
    return torch.stack(pooled)


# ---------------------------------------------------------------------------
# Precompute IdioleX embeddings
# ---------------------------------------------------------------------------

def precompute_idiolex_embeddings(raw_samples, checkpoint_path, device="cuda", batch_size=64):
    print(f"Loading IdioleX from {checkpoint_path}")
    idiolex = IdioleXEncoder.from_checkpoint(checkpoint_path, device=device)
    idiolex_dim = idiolex.embedding_dim

    all_ground_truths = [s["ground_truth"] for s in raw_samples]
    all_embeddings = []

    print(f"Precomputing IdioleX embeddings for {len(all_ground_truths)} samples...")
    for i in tqdm(range(0, len(all_ground_truths), batch_size), desc="IdioleX encode"):
        batch_texts = all_ground_truths[i:i + batch_size]
        embs = idiolex.encode(batch_texts, torch.device(device))
        all_embeddings.append(embs.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    del idiolex
    torch.cuda.empty_cache()
    return embeddings, idiolex_dim


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Precompute IdioleX embeddings BEFORE accelerator ----
    with open(args.train_jsonl, encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f]
    print(f"Loaded {len(raw_samples)} samples")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if os.path.exists(args.precomputed_representations):
        print(f"[rank {local_rank}] Loading cached IdioleX embeddings")
        cached = torch.load(args.precomputed_representations, weights_only=True, map_location="cpu")
        idiolex_embeddings = cached["embeddings"]
        idiolex_dim = cached["dim"]
    elif local_rank == 0:
        idiolex_embeddings, idiolex_dim = precompute_idiolex_embeddings(
            raw_samples, args.reward_checkpoint, device="cuda",
        )
        torch.save({"embeddings": idiolex_embeddings, "dim": idiolex_dim}, cache_path)
        print(f"Cached embeddings to {cache_path}")
    else:
        import time
        print(f"[rank {local_rank}] Waiting for rank 0 to precompute embeddings...")
        while not os.path.exists(cache_path):
            time.sleep(2)
        time.sleep(1)
        cached = torch.load(cache_path, weights_only=True, map_location="cpu")
        idiolex_embeddings = cached["embeddings"]
        idiolex_dim = cached["dim"]

    # ---- Initialize accelerator ----
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if args.wandb_project else None,
        mixed_precision="bf16",
    )
    is_main = accelerator.is_main_process

    if is_main and args.wandb_project:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {
                "name": args.run_name or f"sft-idiolex-{os.path.basename(args.output_dir)}",
            }},
        )

    # ---- Tokenizer ----
    llm_tokenizer = AutoTokenizer.from_pretrained(args.model)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_tokenizer.padding_side = "left"

    # ---- Dataset ----
    if is_main:
        print("Building dataset...")
    dataset = DialectSFTDataset(
        llm_tokenizer, idiolex_embeddings, raw_samples,
        max_seq_length=args.max_seq_length,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.per_device_batch_size,
        shuffle=True, collate_fn=collate_fn, drop_last=True,
    )

    # ---- Load LLM + LoRA ----
    if is_main:
        print(f"Loading LLM: {args.model}")

    load_kwargs = {"torch_dtype": torch.bfloat16}
    if args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    llm = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    llm_hidden_dim = llm.config.hidden_size

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        bias="none",
    )
    llm = get_peft_model(llm, peft_config)
    if is_main:
        llm.print_trainable_parameters()

    if args.gradient_checkpointing:
        llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # ---- Model ----
    projection = ProjectionHead(llm_hidden_dim, idiolex_dim)
    model = LLMWithProjection(llm, projection)

    # ---- Optimizer ----
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(all_trainable, lr=args.lr, weight_decay=0.01)

    num_update_steps = args.num_epochs * (len(dataloader) // args.gradient_accumulation_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(num_update_steps, 1),
    )

    # ---- Prepare ----
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler,
    )

    # ---- Train ----
    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.num_epochs):
        model.train()
        total_ce_loss = 0
        total_emb_loss = 0
        total_combined = 0
        num_logged = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}",
                     disable=not is_main)

        if epoch > 0:
            for param in projection.parameters():
                param.requires_grad = False
            print("Freezing projection head.")

        for batch_idx, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                loss, ce_loss, emb_loss = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["response_starts"],
                    batch["idiolex_embeddings"],
                    alpha=args.alpha,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(all_trainable, max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                if is_main:
                    ce_val = ce_loss.item()
                    emb_val = emb_loss.item()
                    combined = ce_val + args.alpha * emb_val

                    total_ce_loss += ce_val
                    total_emb_loss += emb_val
                    total_combined += combined
                    num_logged += 1

                    pbar.set_postfix({
                        "ce": f"{ce_val:.3f}",
                        "emb": f"{emb_val:.3f}",
                        "total": f"{combined:.3f}",
                        "step": global_step,
                    })

                    if args.wandb_project and global_step % args.logging_steps == 0:
                        accelerator.log({
                            "train/ce_loss": ce_val,
                            "train/emb_loss": emb_val,
                            "train/total_loss": combined,
                            "train/lr": scheduler.get_last_lr()[0],
                        }, step=global_step)

                if global_step > 0 and global_step % args.save_steps == 0:
                    if is_main:
                        spath = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.unwrap_model(model).llm.save_pretrained(spath)
                        llm_tokenizer.save_pretrained(spath)
                        torch.save(
                            accelerator.unwrap_model(model).projection.state_dict(),
                            os.path.join(spath, "projection.pth"),
                        )
                        print(f"\n  Saved checkpoint-{global_step}")

        if is_main and num_logged > 0:
            avg_ce = total_ce_loss / num_logged
            avg_emb = total_emb_loss / num_logged
            avg_total = total_combined / num_logged
            print(f"  Epoch {epoch+1}: ce={avg_ce:.4f}, emb={avg_emb:.4f}, total={avg_total:.4f}")

            if avg_total < best_loss:
                best_loss = avg_total
                spath = os.path.join(args.output_dir, "best")
                accelerator.unwrap_model(model).llm.save_pretrained(spath)
                llm_tokenizer.save_pretrained(spath)
                torch.save(
                    accelerator.unwrap_model(model).projection.state_dict(),
                    os.path.join(spath, "projection.pth"),
                )
                print(f"  New best! Saved to {spath}")

    if is_main:
        spath = os.path.join(args.output_dir, "final")
        accelerator.unwrap_model(model).llm.save_pretrained(spath)
        llm_tokenizer.save_pretrained(spath)
        torch.save(
            accelerator.unwrap_model(model).projection.state_dict(),
            os.path.join(spath, "projection.pth"),
        )
        print(f"\nTraining complete. Final model saved to {spath}")

    accelerator.end_training()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SFT with IdioleX embedding alignment loss")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    parser.add_argument("--reward_checkpoint", type=str, required=True)
    parser.add_argument("--precomputed_representations", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=1024)

    parser.add_argument("--output_dir", type=str, default="./sft_idiolex_output")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--wandb_project", type=str, default="Idiolex_Post-train")
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
