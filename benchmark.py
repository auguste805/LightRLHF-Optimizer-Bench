#!/usr/bin/env python3
"""
One-epoch optimizer benchmark: AdamW, SGD, RMSprop on RM training.

Dual-track CSV logging (high-frequency train, medium-frequency val), periodic
snapshots per optimizer, graceful Ctrl+C; global comparison from
``train_metrics.csv`` / ``val_metrics.csv`` (legacy ``metrics.csv`` supported).
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
from collections import deque
from typing import Any, Dict, List, Literal, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from data import Batch, build_dataloaders, build_tokenizer
from model import GPT2RewardModel
from optimizer import get_optimizer
from train import batch_to_device, bradley_terry_loss, evaluate
from utils import (
    TRAIN_METRICS_FIELDS,
    VAL_METRICS_FIELDS,
    append_benchmark_csv_row,
    ensure_benchmark_optimizer_dirs,
    plot_global_benchmark_comparison,
    plot_optimizer_dual_snapshot,
    plot_train_loss_ema_only,
    plot_val_acc_only,
)

# (CSV / folder label, get_optimizer key)
BENCHMARK_OPTIMIZERS: List[Tuple[str, str]] = [
    ("AdamW", "adamw"),
    ("SGD", "sgd"),
    ("RMSprop", "rmsprop"),
]

RunOutcome = Literal["continue_benchmark", "exit_benchmark"]


class BenchmarkDualLogger:
    """Append-only dual-track metrics (train vs validation) with immediate fsync."""

    def __init__(self, train_csv_path: str, val_csv_path: str) -> None:
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path

    def append_train(self, step: int, train_loss: float, learning_rate: float) -> None:
        append_benchmark_csv_row(
            self.train_csv_path,
            TRAIN_METRICS_FIELDS,
            {
                "step": step,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
            },
        )

    def append_val(self, step: int, val_loss: float, val_acc: float) -> None:
        append_benchmark_csv_row(
            self.val_csv_path,
            VAL_METRICS_FIELDS,
            {
                "step": step,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _forward_batch_metrics(
    model: torch.nn.Module,
    batch: Batch,
    *,
    use_bf16: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, float, float]:
    batch = batch_to_device(batch, device)
    use_amp = use_bf16 and device.type == "cuda"
    if use_amp:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            r_w = model(batch.winner_input_ids, batch.winner_attention_mask)
            r_l = model(batch.loser_input_ids, batch.loser_attention_mask)
            loss = bradley_terry_loss(r_w, r_l)
    else:
        r_w = model(batch.winner_input_ids, batch.winner_attention_mask)
        r_l = model(batch.loser_input_ids, batch.loser_attention_mask)
        loss = bradley_terry_loss(r_w, r_l)
    train_acc = (r_w > r_l).float().mean().item()
    return loss, loss.item(), train_acc


def _maybe_snapshot_plot(
    train_csv_path: str,
    val_csv_path: str,
    plots_dir: str,
    display_name: str,
    step: int,
    *,
    tag: str,
    ema_alpha: float,
) -> None:
    """Render dual-track CSVs to ``plots/{tag}_step_{step}.png``."""
    out_png = os.path.join(plots_dir, f"{tag}_step_{step}.png")
    plot_optimizer_dual_snapshot(
        train_csv_path,
        val_csv_path,
        out_png,
        optimizer_label=display_name,
        title_suffix=f"({tag} @ step {step})",
        ema_alpha=ema_alpha,
    )
    prefix = os.path.join(plots_dir, f"{tag}_step_{step}")
    plot_train_loss_ema_only(
        train_csv_path,
        f"{prefix}_train_loss_ema.png",
        optimizer_label=display_name,
        title_suffix=f"({tag} @ step {step})",
        ema_alpha=ema_alpha,
    )
    plot_val_acc_only(
        val_csv_path,
        f"{prefix}_val_acc.png",
        optimizer_label=display_name,
        title_suffix=f"({tag} @ step {step})",
    )


def _resolve_interrupt_behavior(on_interrupt: str) -> RunOutcome:
    """
    After Ctrl+C: ``prompt`` asks on TTY; ``next`` always continues;
    ``exit`` always stops the whole benchmark.
    """
    if on_interrupt == "exit":
        print("Interrupt policy: exiting benchmark (no further optimizers).")
        return "exit_benchmark"
    if on_interrupt == "next":
        print("Interrupt policy: continuing with next optimizer.")
        return "continue_benchmark"
    if not sys.stdin.isatty():
        print(
            "Non-interactive shell: continuing with next optimizer "
            "(use --on-interrupt exit to stop)."
        )
        return "continue_benchmark"
    try:
        ans = input("Continue with next optimizer? [Y/n]: ").strip().lower()
    except EOFError:
        return "continue_benchmark"
    if ans in ("n", "no"):
        print("Stopping benchmark.")
        return "exit_benchmark"
    print("Continuing with next optimizer.")
    return "continue_benchmark"


def run_one_optimizer(
    display_name: str,
    optim_key: str,
    *,
    benchmark_dir: str,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
    max_grad_norm: float,
    train_log_interval: int,
    val_log_interval: int,
    progress_interval: int,
    plot_interval: int,
    on_interrupt: str,
    learning_rate: float,
    warmup_ratio: float,
    weight_decay: float,
    seed: int,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    attn_impl: str | None,
    ema_alpha: float,
) -> RunOutcome:
    """
    Train one epoch; append ``train_metrics.csv`` every ``train_log_interval``
    steps and ``val_metrics.csv`` every ``val_log_interval`` steps; snapshots on schedule.
    Returns whether the outer loop should exit (user chose stop) or continue.
    """
    _, plots_dir, train_csv_path, val_csv_path = ensure_benchmark_optimizer_dirs(
        benchmark_dir, display_name
    )
    logger = BenchmarkDualLogger(train_csv_path, val_csv_path)
    set_seed(seed)

    model = GPT2RewardModel(
        model_name,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        attn_implementation=attn_impl,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt_kwargs: Dict[str, Any] = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "params": trainable_params,
    }
    if optim_key == "sgd":
        opt_kwargs["momentum"] = 0.9

    optimizer = get_optimizer(model, optim_key, **opt_kwargs)
    total_training_steps = len(train_loader)
    warmup_steps = int(total_training_steps * warmup_ratio)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    min_lr = learning_rate * 0.1

    step = 0
    last_val_logged_step = 0
    train_loss_f = 0.0
    train_acc_f = 0.0
    val_loss, val_acc = 0.0, 0.0
    interrupted = False
    recent_losses: deque[float] = deque(maxlen=train_log_interval)

    def log_train_row() -> None:
        if len(recent_losses) < train_log_interval:
            return
        avg_loss = sum(recent_losses) / float(train_log_interval)
        lr_now = lr_scheduler.get_last_lr()[0]
        logger.append_train(step, avg_loss, lr_now)

    def log_val_row_and_print() -> None:
        nonlocal last_val_logged_step, val_loss, val_acc
        model.eval()
        val_loss, val_acc = evaluate(
            model, val_loader, device, use_bf16=use_bf16
        )
        model.train()
        logger.append_val(step, val_loss, val_acc)
        print(
            f"[{display_name}] step={step} "
            f"train_loss(last {train_log_interval})="
            f"{sum(recent_losses) / max(len(recent_losses), 1):.4f} "
            f"train_acc={train_acc_f:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={lr_scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )
        last_val_logged_step = step
        if plot_interval > 0 and step % plot_interval == 0:
            _maybe_snapshot_plot(
                train_csv_path,
                val_csv_path,
                plots_dir,
                display_name,
                step,
                tag="snapshot",
                ema_alpha=ema_alpha,
            )

    try:
        print(
            f"[{display_name}] total_steps={total_training_steps} "
            f"warmup_steps={warmup_steps} peak_lr={learning_rate:.2e}",
            flush=True,
        )
        print(
            f"[{display_name}] train_metrics.csv every {train_log_interval} steps; "
            f"val_metrics.csv every {val_log_interval} steps (full validation). "
            f"Train-only console every {progress_interval} steps if >0.",
            flush=True,
        )
        model.train()
        for batch in train_loader:
            loss_t, train_loss_f, train_acc_f = _forward_batch_metrics(
                model, batch, use_bf16=use_bf16, device=device
            )

            optimizer.zero_grad(set_to_none=True)
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            # Keep cosine floor at 10% peak LR after warmup.
            if step + 1 > warmup_steps:
                for param_group in optimizer.param_groups:
                    if param_group["lr"] < min_lr:
                        param_group["lr"] = min_lr

            step += 1
            recent_losses.append(train_loss_f)

            if step % train_log_interval == 0:
                log_train_row()

            if (
                progress_interval > 0
                and step % progress_interval == 0
                and step % val_log_interval != 0
            ):
                print(
                    f"[{display_name}] step={step} (train) "
                    f"train_loss={train_loss_f:.4f} train_acc={train_acc_f:.4f} "
                    f"lr={lr_scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
            if step % val_log_interval == 0:
                log_val_row_and_print()

    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted manually by user (KeyboardInterrupt).")
        if (
            os.path.isfile(train_csv_path)
            and os.path.getsize(train_csv_path) > 0
        ) or (
            os.path.isfile(val_csv_path) and os.path.getsize(val_csv_path) > 0
        ):
            inter_path = os.path.join(plots_dir, f"interrupted_step_{step}.png")
            plot_optimizer_dual_snapshot(
                train_csv_path,
                val_csv_path,
                inter_path,
                optimizer_label=display_name,
                title_suffix=f"(interrupted @ step {step})",
                ema_alpha=ema_alpha,
            )
            iprefix = os.path.join(plots_dir, f"interrupted_step_{step}")
            plot_train_loss_ema_only(
                train_csv_path,
                f"{iprefix}_train_loss_ema.png",
                optimizer_label=display_name,
                title_suffix=f"(interrupted @ step {step})",
                ema_alpha=ema_alpha,
            )
            plot_val_acc_only(
                val_csv_path,
                f"{iprefix}_val_acc.png",
                optimizer_label=display_name,
                title_suffix=f"(interrupted @ step {step})",
            )
            print(f"Saved interrupt snapshot to {inter_path}")

    finally:
        if not interrupted and step != last_val_logged_step and step > 0:
            model.eval()
            val_loss, val_acc = evaluate(
                model, val_loader, device, use_bf16=use_bf16
            )
            model.train()
            logger.append_val(step, val_loss, val_acc)
            print(
                f"[{display_name}] step={step} (final val) "
                f"train_loss={train_loss_f:.4f} train_acc={train_acc_f:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"lr={lr_scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )
            last_val_logged_step = step
            if plot_interval > 0 and step % plot_interval == 0:
                _maybe_snapshot_plot(
                    train_csv_path,
                    val_csv_path,
                    plots_dir,
                    display_name,
                    step,
                    tag="snapshot",
                    ema_alpha=ema_alpha,
                )

        del model, optimizer, lr_scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if interrupted:
        return _resolve_interrupt_behavior(on_interrupt)
    return "continue_benchmark"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RM optimizer benchmark (1 epoch, multiple optimizers).")
    p.add_argument("--model-name", type=str, default="gpt2")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--train-log-interval",
        type=int,
        default=50,
        help="Append train_metrics.csv every N steps (mean loss over last N steps).",
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=2000,
        help="Run full validation and append val_metrics.csv every N steps.",
    )
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=0.1,
        help="EMA smoothing weight on train_loss in plots (higher = more responsive).",
    )
    p.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help=(
            "Every N training steps, print train_loss/train_acc/lr only (no val). "
            "0 disables. Skips steps that coincide with --log-interval to avoid duplicate lines."
        ),
    )
    p.add_argument(
        "--plot-interval",
        type=int,
        default=10000,
        help="After each logged row, if step %% plot_interval == 0, save plot from CSV.",
    )
    p.add_argument(
        "--benchmark-dir",
        type=str,
        default="benchmark_results",
        help=(
            "Base dir: benchmark_results/{Optimizer}/train_metrics.csv, "
            "val_metrics.csv, and plots/."
        ),
    )
    p.add_argument(
        "--global-plot-name",
        type=str,
        default="global_comparison.png",
        help="Written under --benchmark-dir after all (or partial) runs.",
    )
    p.add_argument(
        "--on-interrupt",
        choices=["prompt", "next", "exit"],
        default="prompt",
        help="Ctrl+C: ask interactively, always continue, or always exit.",
    )
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--attn-implementation", type=str, default="sdpa")
    p.add_argument(
        "--debug_mode",
        action="store_true",
        help="Use tiny train/val subsets (same as train.py).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print(
        f"DataLoaders: train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"(batch_size={args.batch_size})",
        flush=True,
    )

    global_plot_path = os.path.join(
        os.path.abspath(args.benchmark_dir), args.global_plot_name
    )

    try:
        for display_name, optim_key in BENCHMARK_OPTIMIZERS:
            print(f"\n=== Benchmark: {display_name} ({optim_key}) ===")
            outcome = run_one_optimizer(
                display_name,
                optim_key,
                benchmark_dir=args.benchmark_dir,
                model_name=args.model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                use_bf16=use_bf16,
                max_grad_norm=args.max_grad_norm,
                train_log_interval=args.train_log_interval,
                val_log_interval=args.log_interval,
                progress_interval=args.progress_interval,
                plot_interval=args.plot_interval,
                on_interrupt=args.on_interrupt,
                learning_rate=args.learning_rate,
                warmup_ratio=args.warmup_ratio,
                weight_decay=args.weight_decay,
                seed=args.seed,
                use_lora=not args.no_lora,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                attn_impl=attn_impl,
                ema_alpha=args.ema_alpha,
            )
            if outcome == "exit_benchmark":
                break
    finally:
        plot_global_benchmark_comparison(
            args.benchmark_dir,
            global_plot_path,
            ema_alpha=args.ema_alpha,
        )
        base = os.path.abspath(args.benchmark_dir)
        print("\nSaved global comparison plots (dual-track or legacy CSVs):")
        print(f"- {os.path.join(base, 'train_loss_comparison.png')}")
        print(f"- {os.path.join(base, 'val_loss_comparison.png')}")
        print(f"- {os.path.join(base, 'val_acc_comparison.png')}")
        print(f"- {os.path.join(base, 'learning_rate_comparison.png')}")
        print(f"- {global_plot_path} (2×2 summary)")


if __name__ == "__main__":
    main()
