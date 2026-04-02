"""
Metrics, checkpoint I/O, and training curve visualization.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Legacy single-file benchmark logger (optional / older runs).
METRICS_CSV_FIELDNAMES = [
    "optimizer",
    "step",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "lr",
]

# Dual-track benchmark CSVs (per optimizer directory).
TRAIN_METRICS_FIELDS = ["step", "train_loss", "learning_rate"]
VAL_METRICS_FIELDS = ["step", "val_loss", "val_acc"]


def pairwise_accuracy(
    reward_winner: torch.Tensor, reward_loser: torch.Tensor
) -> float:
    """
    Fraction of pairs with r_w > r_l (strict inequality).

    Tensors may be any shape as long as they broadcast equally; typically [B].
    """
    return (reward_winner > reward_loser).float().mean().item()


def save_checkpoint(
    path: str,
    model: nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model / optimizer state for reproducibility and inference."""
    state: Dict[str, Any] = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if extra_state:
        state.update(extra_state)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load weights into ``model`` (and optionally optimizer). Returns full payload."""
    map_loc = device if device is not None else "cpu"
    payload = torch.load(path, map_location=map_loc)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_accs: List[float],
    out_path: str,
) -> None:
    """Plot train/val Bradley-Terry loss and validation pairwise accuracy vs epoch."""
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Bradley-Terry Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(epochs, val_accs, color="tab:green", label="Val Pairwise Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Pairwise Accuracy (r_w > r_l)")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def ensure_benchmark_optimizer_dirs(
    benchmark_base_dir: str, optimizer_display_name: str
) -> tuple[str, str, str, str]:
    """
    Create ``benchmark_base_dir/{Optimizer}/``, ``.../plots/``, and dual CSV paths.

    Returns
    -------
    run_dir, plots_dir, train_metrics_csv, val_metrics_csv
    """
    safe_name = optimizer_display_name.replace(os.sep, "_").strip() or "run"
    run_dir = os.path.join(os.path.abspath(benchmark_base_dir), safe_name)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    train_csv = os.path.join(run_dir, "train_metrics.csv")
    val_csv = os.path.join(run_dir, "val_metrics.csv")
    return run_dir, plots_dir, train_csv, val_csv


def append_metrics_row_disk(row: Dict[str, Any], csv_path: str) -> None:
    """
    Append one metrics row with header-on-first-write; flush + fsync for durability.
    """
    directory = os.path.dirname(csv_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    exists = os.path.isfile(csv_path)
    line = {k: row[k] for k in METRICS_CSV_FIELDNAMES}
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(line)
        f.flush()
        os.fsync(f.fileno())


def append_benchmark_csv_row(
    csv_path: str, fieldnames: List[str], row: Dict[str, Any]
) -> None:
    """Append one row; header on first write; flush + fsync."""
    directory = os.path.dirname(csv_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    exists = os.path.isfile(csv_path)
    line = {k: row[k] for k in fieldnames}
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(line)
        f.flush()
        os.fsync(f.fileno())


def ema_series(values: Sequence[float], *, alpha: float = 0.1) -> np.ndarray:
    """
    Exponential moving average with smoothing weight ``alpha`` on the new sample:
    s[t] = (1-alpha)*s[t-1] + alpha*x[t].
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x
    out = np.zeros_like(x)
    out[0] = x[0]
    one_m = 1.0 - alpha
    for i in range(1, len(x)):
        out[i] = one_m * out[i - 1] + alpha * x[i]
    return out


def plot_optimizer_snapshot_from_csv(
    csv_path: str,
    out_path: str,
    *,
    optimizer_label: str,
    title_suffix: str = "",
) -> None:
    """
    Legacy: snapshot from a single ``metrics.csv``. Prefer
    :func:`plot_optimizer_dual_snapshot` for dual-track runs.
    """
    if not os.path.isfile(csv_path):
        return
    df = pd.read_csv(csv_path)
    if df.empty or "step" not in df.columns:
        return

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    sub = df.sort_values("step")
    color = "tab:blue"

    ax_loss.plot(
        sub["step"],
        sub["train_loss"],
        color=color,
        linestyle="-",
        label="train",
    )
    ax_loss.plot(
        sub["step"],
        sub["val_loss"],
        color=color,
        linestyle="--",
        label="val",
    )
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"{optimizer_label} — Train/Val Loss {title_suffix}".strip())
    ax_loss.legend()
    ax_loss.grid(True, linestyle="--", alpha=0.35)

    ax_acc.plot(sub["step"], sub["train_acc"], color=color, linestyle="-", label="train")
    ax_acc.plot(sub["step"], sub["val_acc"], color=color, linestyle="--", label="val")
    idx_max = sub["val_acc"].idxmax()
    row = sub.loc[idx_max]
    ax_acc.scatter(
        [row["step"]],
        [row["val_acc"]],
        marker="*",
        s=280,
        color="gold",
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
        label=f"max val = {row['val_acc']:.4f}",
    )
    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title(f"{optimizer_label} — Train/Val Acc {title_suffix}".strip())
    ax_acc.set_ylim(0.0, 1.02)
    ax_acc.legend()
    ax_acc.grid(True, linestyle="--", alpha=0.35)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_optimizer_dual_snapshot(
    train_csv_path: str,
    val_csv_path: str,
    out_path: str,
    *,
    optimizer_label: str,
    title_suffix: str = "",
    ema_alpha: float = 0.1,
) -> None:
    """
    Snapshot from ``train_metrics.csv`` + ``val_metrics.csv``:
    train loss (raw + EMA) + LR on twin axis; val loss + val acc.
    """
    if not os.path.isfile(train_csv_path) and not os.path.isfile(val_csv_path):
        return
    train_df = pd.read_csv(train_csv_path) if os.path.isfile(train_csv_path) else pd.DataFrame()
    val_df = pd.read_csv(val_csv_path) if os.path.isfile(val_csv_path) else pd.DataFrame()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    color = "tab:blue"

    if not train_df.empty and "step" in train_df.columns:
        tr = train_df.sort_values("step")
        raw = tr["train_loss"].to_numpy(dtype=float)
        sm = ema_series(raw, alpha=ema_alpha)
        ax_top.plot(
            tr["step"],
            raw,
            color=color,
            alpha=0.35,
            linewidth=1.2,
            label="train_loss (raw)",
        )
        ax_top.plot(
            tr["step"],
            sm,
            color=color,
            linewidth=2.2,
            label=f"train_loss (EMA α={ema_alpha})",
        )
        if "learning_rate" in tr.columns:
            ax_lr = ax_top.twinx()
            ax_lr.plot(
                tr["step"],
                tr["learning_rate"],
                color="tab:orange",
                linewidth=1.4,
                alpha=0.85,
                label="LR",
            )
            ax_lr.set_ylabel("Learning rate")
            ax_lr.tick_params(axis="y", labelcolor="tab:orange")
    ax_top.set_xlabel("Step")
    ax_top.set_ylabel("Loss")
    ax_top.set_title(f"{optimizer_label} — Train loss & LR {title_suffix}".strip())
    ax_top.grid(True, linestyle="--", alpha=0.35)
    ax_top.legend(loc="upper right", fontsize=8)

    if not val_df.empty and "step" in val_df.columns:
        va = val_df.sort_values("step")
        ax_bot.plot(va["step"], va["val_loss"], color=color, linewidth=1.8, label="val_loss")
        ax_a = ax_bot.twinx()
        ax_a.plot(
            va["step"],
            va["val_acc"],
            color="tab:green",
            linewidth=1.8,
            alpha=0.9,
            label="val_acc",
        )
        ax_a.set_ylabel("Accuracy")
        ax_a.set_ylim(0.0, 1.02)
        idx_max = va["val_acc"].idxmax()
        row = va.loc[idx_max]
        ax_a.scatter(
            [row["step"]],
            [row["val_acc"]],
            marker="*",
            s=220,
            color="gold",
            edgecolors="black",
            zorder=10,
        )
    ax_bot.set_xlabel("Step")
    ax_bot.set_ylabel("Val loss")
    ax_bot.set_title(f"{optimizer_label} — Validation {title_suffix}".strip())
    ax_bot.legend(loc="upper right", fontsize=8)
    ax_bot.grid(True, linestyle="--", alpha=0.35)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_train_loss_ema_only(
    train_csv_path: str,
    out_path: str,
    *,
    optimizer_label: str,
    title_suffix: str = "",
    ema_alpha: float = 0.1,
) -> None:
    """
    One figure: 50-step mean train loss — raw (light) + EMA (bold). No LR axis.
    """
    if not os.path.isfile(train_csv_path):
        return
    train_df = pd.read_csv(train_csv_path)
    if train_df.empty or "step" not in train_df.columns or "train_loss" not in train_df.columns:
        return
    tr = train_df.sort_values("step")
    raw = tr["train_loss"].to_numpy(dtype=float)
    sm = ema_series(raw, alpha=ema_alpha)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    color = "tab:blue"
    ax.plot(tr["step"], raw, color=color, alpha=0.35, linewidth=1.2, label="train_loss (raw)")
    ax.plot(
        tr["step"],
        sm,
        color=color,
        linewidth=2.4,
        label=f"train_loss (EMA α={ema_alpha})",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss (mean over last N steps)")
    ax.set_title(f"{optimizer_label} — Train loss {title_suffix}".strip())
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_val_acc_only(
    val_csv_path: str,
    out_path: str,
    *,
    optimizer_label: str,
    title_suffix: str = "",
) -> None:
    """One figure: validation accuracy vs step (e.g. every 2000 steps)."""
    if not os.path.isfile(val_csv_path):
        return
    val_df = pd.read_csv(val_csv_path)
    if val_df.empty or "step" not in val_df.columns or "val_acc" not in val_df.columns:
        return
    va = val_df.sort_values("step")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    color = "tab:green"
    ax.plot(va["step"], va["val_acc"], color=color, linewidth=2.0, marker="o", label="val_acc")
    idx_max = va["val_acc"].idxmax()
    row = va.loc[idx_max]
    ax.scatter(
        [row["step"]],
        [row["val_acc"]],
        marker="*",
        s=240,
        color="gold",
        edgecolors="black",
        zorder=10,
        label=f"best {float(row['val_acc']):.4f}",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation accuracy")
    ax.set_title(f"{optimizer_label} — Val accuracy {title_suffix}".strip())
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_global_benchmark_comparison(
    benchmark_base_dir: str,
    out_path: Optional[str] = None,
    *,
    ema_alpha: float = 0.1,
) -> None:
    """
    Load dual-track CSVs from each ``benchmark_base_dir/{Optimizer}/`` and save
    comparative figures. Falls back to legacy ``metrics.csv`` if dual files absent.
    """
    base = os.path.abspath(benchmark_base_dir)
    if not os.path.isdir(base):
        return
    train_frames: List[pd.DataFrame] = []
    val_frames: List[pd.DataFrame] = []
    legacy_frames: List[pd.DataFrame] = []

    for name in sorted(os.listdir(base)):
        sub = os.path.join(base, name)
        if not os.path.isdir(sub):
            continue
        tp = os.path.join(sub, "train_metrics.csv")
        vp = os.path.join(sub, "val_metrics.csv")
        mp = os.path.join(sub, "metrics.csv")
        if os.path.isfile(tp) or os.path.isfile(vp):
            if os.path.isfile(tp):
                tdf = pd.read_csv(tp)
                if not tdf.empty:
                    tdf = tdf.copy()
                    tdf["optimizer"] = name
                    train_frames.append(tdf)
            if os.path.isfile(vp):
                vdf = pd.read_csv(vp)
                if not vdf.empty:
                    vdf = vdf.copy()
                    vdf["optimizer"] = name
                    val_frames.append(vdf)
        elif os.path.isfile(mp):
            part = pd.read_csv(mp)
            if not part.empty:
                legacy_frames.append(part)

    if train_frames or val_frames:
        train_all = (
            pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
        )
        val_all = pd.concat(val_frames, ignore_index=True) if val_frames else pd.DataFrame()
        plot_benchmarks_four_dual(train_all, val_all, base, ema_alpha=ema_alpha)
        if out_path:
            plot_benchmarks_dual(train_all, val_all, out_path, ema_alpha=ema_alpha)
        return

    if not legacy_frames:
        return
    combined = pd.concat(legacy_frames, ignore_index=True)
    plot_benchmarks_four(combined, base)
    if out_path:
        plot_benchmarks(combined, out_path)


def _plot_train_loss_ema_comparison(
    df: pd.DataFrame,
    out_path: str,
    *,
    ema_alpha: float = 0.1,
) -> None:
    """Raw (light) + EMA (bold) train loss for each optimizer."""
    if df is None or df.empty or "train_loss" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    optimizers = list(df["optimizer"].unique())
    cmap = plt.cm.get_cmap("tab10", max(len(optimizers), 3))
    for i, opt in enumerate(optimizers):
        sub = df[df["optimizer"] == opt].sort_values("step")
        raw = sub["train_loss"].to_numpy(dtype=float)
        sm = ema_series(raw, alpha=ema_alpha)
        color = cmap(i % 10)
        ax.plot(
            sub["step"],
            raw,
            color=color,
            alpha=0.35,
            linewidth=1.1,
            label=f"{opt} (raw)",
        )
        ax.plot(
            sub["step"],
            sm,
            color=color,
            linewidth=2.2,
            label=f"{opt} (EMA)",
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss (50-step mean)")
    ax.set_title("Train Loss vs Step — raw + EMA (per optimizer)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=7, ncol=2)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_benchmarks_four_dual(
    train_all: pd.DataFrame,
    val_all: pd.DataFrame,
    out_dir: str,
    *,
    ema_alpha: float = 0.1,
) -> None:
    """Four standalone figures from dual-track data."""
    out_dir_abs = os.path.abspath(out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    _plot_train_loss_ema_comparison(
        train_all,
        os.path.join(out_dir_abs, "train_loss_comparison.png"),
        ema_alpha=ema_alpha,
    )
    if not val_all.empty:
        _plot_single_metric(
            val_all,
            y_col="val_loss",
            out_path=os.path.join(out_dir_abs, "val_loss_comparison.png"),
            title="Validation Loss vs Step (Optimizer Comparison)",
            ylabel="Validation Loss",
        )
        _plot_single_metric(
            val_all,
            y_col="val_acc",
            out_path=os.path.join(out_dir_abs, "val_acc_comparison.png"),
            title="Validation Accuracy vs Step (Optimizer Comparison)",
            ylabel="Validation Accuracy",
            mark_best=True,
        )
    if not train_all.empty and "learning_rate" in train_all.columns:
        _plot_single_metric(
            train_all,
            y_col="learning_rate",
            out_path=os.path.join(out_dir_abs, "learning_rate_comparison.png"),
            title="Learning Rate vs Step (Optimizer Comparison)",
            ylabel="Learning rate",
        )


def plot_benchmarks_dual(
    train_all: pd.DataFrame,
    val_all: pd.DataFrame,
    out_path: str,
    *,
    ema_alpha: float = 0.1,
) -> None:
    """2×2 summary: train loss+EMA, LR, val loss, val acc."""
    if (train_all is None or train_all.empty) and (val_all is None or val_all.empty):
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax_tl, ax_tr, ax_bl, ax_br = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    cmap = plt.cm.get_cmap("tab10", 10)

    if train_all is not None and not train_all.empty:
        for i, opt in enumerate(train_all["optimizer"].unique()):
            sub = train_all[train_all["optimizer"] == opt].sort_values("step")
            raw = sub["train_loss"].to_numpy(dtype=float)
            sm = ema_series(raw, alpha=ema_alpha)
            c = cmap(i % 10)
            ax_tl.plot(sub["step"], raw, color=c, alpha=0.35, linewidth=1.0)
            ax_tl.plot(sub["step"], sm, color=c, linewidth=2.0, label=opt)
        ax_tl.set_title("Train loss (raw + EMA)")
        ax_tl.set_xlabel("Step")
        ax_tl.set_ylabel("Loss")
        ax_tl.legend(loc="best", fontsize=7)
        ax_tl.grid(True, linestyle="--", alpha=0.35)

        for i, opt in enumerate(train_all["optimizer"].unique()):
            sub = train_all[train_all["optimizer"] == opt].sort_values("step")
            if "learning_rate" not in sub.columns:
                break
            c = cmap(i % 10)
            ax_tr.plot(sub["step"], sub["learning_rate"], color=c, linewidth=1.8, label=opt)
        ax_tr.set_title("Learning rate")
        ax_tr.set_xlabel("Step")
        ax_tr.set_ylabel("LR")
        ax_tr.legend(loc="best", fontsize=7)
        ax_tr.grid(True, linestyle="--", alpha=0.35)

    if val_all is not None and not val_all.empty:
        for i, opt in enumerate(val_all["optimizer"].unique()):
            sub = val_all[val_all["optimizer"] == opt].sort_values("step")
            c = cmap(i % 10)
            ax_bl.plot(sub["step"], sub["val_loss"], color=c, linewidth=1.8, label=opt)
        ax_bl.set_title("Validation loss")
        ax_bl.set_xlabel("Step")
        ax_bl.set_ylabel("Val loss")
        ax_bl.legend(loc="best", fontsize=7)
        ax_bl.grid(True, linestyle="--", alpha=0.35)

        for i, opt in enumerate(val_all["optimizer"].unique()):
            sub = val_all[val_all["optimizer"] == opt].sort_values("step")
            c = cmap(i % 10)
            ax_br.plot(sub["step"], sub["val_acc"], color=c, linewidth=1.8, label=opt)
        ax_br.set_title("Validation accuracy")
        ax_br.set_xlabel("Step")
        ax_br.set_ylabel("Val acc")
        ax_br.set_ylim(0.0, 1.02)
        ax_br.legend(loc="best", fontsize=7)
        ax_br.grid(True, linestyle="--", alpha=0.35)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_single_metric(
    df: pd.DataFrame,
    *,
    y_col: str,
    out_path: str,
    title: str,
    ylabel: str,
    mark_best: bool = False,
) -> None:
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    optimizers = list(df["optimizer"].unique())
    cmap = plt.cm.get_cmap("tab10", max(len(optimizers), 3))

    for i, opt in enumerate(optimizers):
        sub = df[df["optimizer"] == opt].sort_values("step")
        color = cmap(i % 10)
        ax.plot(sub["step"], sub[y_col], color=color, linewidth=1.9, label=opt)
        if mark_best:
            idx = sub[y_col].idxmax()
            row = sub.loc[idx]
            ax.scatter(
                [row["step"]],
                [row[y_col]],
                marker="*",
                s=260,
                color=color,
                edgecolors="black",
                linewidths=0.9,
                zorder=10,
            )
            ax.annotate(
                f"{row[y_col]:.3f}",
                (row["step"], row[y_col]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color=color,
            )

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_col.endswith("_acc"):
        ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_benchmarks_four(df: pd.DataFrame, out_dir: str) -> None:
    """
    Save four standalone comparison plots:
      train_loss, val_loss, train_acc, val_acc.
    """
    out_dir_abs = os.path.abspath(out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    _plot_single_metric(
        df,
        y_col="train_loss",
        out_path=os.path.join(out_dir_abs, "train_loss_comparison.png"),
        title="Train Loss vs Step (Optimizer Comparison)",
        ylabel="Train Loss",
    )
    _plot_single_metric(
        df,
        y_col="val_loss",
        out_path=os.path.join(out_dir_abs, "val_loss_comparison.png"),
        title="Validation Loss vs Step (Optimizer Comparison)",
        ylabel="Validation Loss",
    )
    _plot_single_metric(
        df,
        y_col="train_acc",
        out_path=os.path.join(out_dir_abs, "train_acc_comparison.png"),
        title="Train Accuracy vs Step (Optimizer Comparison)",
        ylabel="Train Accuracy",
    )
    _plot_single_metric(
        df,
        y_col="val_acc",
        out_path=os.path.join(out_dir_abs, "val_acc_comparison.png"),
        title="Validation Accuracy vs Step (Optimizer Comparison)",
        ylabel="Validation Accuracy",
        mark_best=True,
    )


def plot_benchmarks(df: pd.DataFrame, out_path: str) -> None:
    """
    Two stacked subplots: train/val loss vs step, train/val acc vs step;
    one color per optimizer; dashed = validation; stars mark max val_acc.
    """
    if df is None or df.empty:
        return
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    optimizers = list(df["optimizer"].unique())
    cmap = plt.cm.get_cmap("tab10", max(len(optimizers), 3))

    for i, opt in enumerate(optimizers):
        sub = df[df["optimizer"] == opt].sort_values("step")
        color = cmap(i % 10)
        ax_loss.plot(
            sub["step"],
            sub["train_loss"],
            color=color,
            linestyle="-",
            linewidth=1.8,
            label=f"{opt} train",
        )
        ax_loss.plot(
            sub["step"],
            sub["val_loss"],
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=f"{opt} val",
        )

        ax_acc.plot(
            sub["step"],
            sub["train_acc"],
            color=color,
            linestyle="-",
            linewidth=1.8,
            label=f"{opt} train",
        )
        ax_acc.plot(
            sub["step"],
            sub["val_acc"],
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=f"{opt} val",
        )

        idx_max = sub["val_acc"].idxmax()
        row = sub.loc[idx_max]
        ax_acc.scatter(
            [row["step"]],
            [row["val_acc"]],
            marker="*",
            s=320,
            color=color,
            edgecolors="black",
            linewidths=1.0,
            zorder=10,
            label=f"{opt} max val acc",
        )
        ax_acc.annotate(
            f"{row['val_acc']:.3f}",
            (row["step"], row["val_acc"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color=color,
        )

    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Train / Val Loss (Bradley-Terry) vs Step")
    ax_loss.grid(True, linestyle="--", alpha=0.35)
    ax_loss.legend(loc="upper right", fontsize=8, ncol=2)

    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Train / Val Pairwise Accuracy vs Step")
    ax_acc.set_ylim(0.0, 1.02)
    ax_acc.grid(True, linestyle="--", alpha=0.35)
    ax_acc.legend(loc="lower right", fontsize=8, ncol=2)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
