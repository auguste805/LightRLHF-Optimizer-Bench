#!/usr/bin/env python3
"""
Main script: Bradley-Terry reward model training on HH-RLHF (GPT-2 backbone).
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import Batch, build_dataloaders, build_tokenizer
from model import GPT2RewardModel
from optimizer import get_optimizer
from utils import plot_training_curves, save_checkpoint


def bradley_terry_loss(
    reward_winner: torch.Tensor, reward_loser: torch.Tensor
) -> torch.Tensor:
    """L = -log σ(r_w - r_l), stable via logsigmoid."""
    diff = reward_winner - reward_loser
    return (-F.logsigmoid(diff)).mean()


def batch_to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        winner_input_ids=batch.winner_input_ids.to(device),
        winner_attention_mask=batch.winner_attention_mask.to(device),
        loser_input_ids=batch.loser_input_ids.to(device),
        loser_attention_mask=batch.loser_attention_mask.to(device),
    )


def _forward_loss(
    model: torch.nn.Module,
    batch: Batch,
) -> torch.Tensor:
    r_w = model(batch.winner_input_ids, batch.winner_attention_mask)
    r_l = model(batch.loser_input_ids, batch.loser_attention_mask)
    return bradley_terry_loss(r_w, r_l)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float,
    *,
    use_bf16: bool,
    trainable_params: List[torch.nn.Parameter],
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = use_bf16 and device.type == "cuda"

    for batch in loader:
        batch = batch_to_device(batch, device)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = _forward_loss(model, batch)
        else:
            loss = _forward_loss(model, batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_bf16: bool,
) -> Tuple[float, float]:
    """Mean validation loss and aggregate pairwise accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pairs = 0
    n_batches = 0
    use_amp = use_bf16 and device.type == "cuda"

    for batch in loader:
        batch = batch_to_device(batch, device)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                r_w = model(batch.winner_input_ids, batch.winner_attention_mask)
                r_l = model(batch.loser_input_ids, batch.loser_attention_mask)
                loss = bradley_terry_loss(r_w, r_l)
        else:
            r_w = model(batch.winner_input_ids, batch.winner_attention_mask)
            r_l = model(batch.loser_input_ids, batch.loser_attention_mask)
            loss = bradley_terry_loss(r_w, r_l)

        total_correct += int((r_w > r_l).sum().item())
        total_pairs += r_w.numel()

        total_loss += loss.item()
        n_batches += 1

    mean_loss = total_loss / max(n_batches, 1)
    accuracy = total_correct / max(total_pairs, 1)
    return mean_loss, accuracy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GPT-2 reward model (Bradley-Terry).")
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--plot-path", type=str, default="outputs/rm_training_curves.png")
    p.add_argument("--checkpoint-path", type=str, default="outputs/rm_checkpoint.pt")
    p.add_argument("--optim-name", type=str, default="adam")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable BF16 autocast (default: BF16 on when CUDA is available).",
    )
    p.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA; full fine-tune backbone + value head (heavy).",
    )
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank (default: 8).")
    p.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (default: 16).")
    p.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout on adapters (default: 0).",
    )
    p.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        help='GPT-2 attention backend, e.g. "sdpa" (PyTorch SDPA) or "eager". Empty = HF default.',
    )
    p.add_argument(
        "--debug_mode",
        action="store_true",
        help=(
            "Use tiny subsets: first 100 train pairs and 20 val pairs "
            "(for overfit / sanity checks)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    use_bf16 = not args.no_bf16 and device.type == "cuda"
    attn_impl = args.attn_implementation.strip() or None

    tokenizer = build_tokenizer(args.model_name)
    train_loader, val_loader = build_dataloaders(
        tokenizer,
        args.max_length,
        args.batch_size,
        args.num_workers,
        cache_dir=args.cache_dir,
        debug_mode=args.debug_mode,
    )

    if args.debug_mode:
        print(
            f"Debug mode: train batches ≈ {len(train_loader)}, val batches ≈ {len(val_loader)}"
        )

    model = GPT2RewardModel(
        args.model_name,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        attn_implementation=attn_impl,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(
        model,
        args.optim_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
        params=trainable_params,
    )

    print(
        f"BF16 autocast: {use_bf16} | LoRA: {not args.no_lora} | "
        f"attn: {attn_impl or 'default'}"
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accs: list[float] = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.max_grad_norm,
            use_bf16=use_bf16,
            trainable_params=trainable_params,
        )
        va_loss, va_acc = evaluate(
            model, val_loader, device, use_bf16=use_bf16
        )
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        val_accs.append(va_acc)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={va_acc:.4f}"
        )

    save_checkpoint(
        args.checkpoint_path,
        model,
        optimizer=optimizer,
        epoch=args.epochs,
        extra_state={
            "optim_name": args.optim_name,
            "use_bf16": use_bf16,
            "use_lora": not args.no_lora,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "attn_implementation": attn_impl,
        },
    )
    print(f"Saved checkpoint to {args.checkpoint_path}")

    plot_training_curves(train_losses, val_losses, val_accs, args.plot_path)
    print(f"Saved plot to {args.plot_path}")


if __name__ == "__main__":
    main()
