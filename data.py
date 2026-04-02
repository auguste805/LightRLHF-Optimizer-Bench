"""
HH-RLHF data loading, tokenizer setup, and batch collation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from datasets import Dataset, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import GPT2Tokenizer

DEBUG_TRAIN_MAX_SAMPLES = 100
DEBUG_VAL_MAX_SAMPLES = 20


@dataclass
class Batch:
    """Winner (chosen) / loser (rejected) token tensors for one batch."""

    winner_input_ids: torch.Tensor
    winner_attention_mask: torch.Tensor
    loser_input_ids: torch.Tensor
    loser_attention_mask: torch.Tensor


def build_tokenizer(model_name: str) -> GPT2Tokenizer:
    """Load tokenizer and set pad token for GPT-2 (no native PAD)."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


class HHPreferenceDataset(TorchDataset):
    """Preference pairs from `Anthropic/hh-rlhf` (chosen vs rejected)."""

    def __init__(
        self,
        hf_split: Dataset,
        tokenizer: GPT2Tokenizer,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ds = hf_split

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.ds[idx]
        chosen = row["chosen"]
        rejected = row["rejected"]

        win = self.tokenizer(
            chosen,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
            add_special_tokens=True,
        )
        lose = self.tokenizer(
            rejected,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return {
            "winner_input_ids": win["input_ids"].squeeze(0),
            "winner_attention_mask": win["attention_mask"].squeeze(0),
            "loser_input_ids": lose["input_ids"].squeeze(0),
            "loser_attention_mask": lose["attention_mask"].squeeze(0),
        }


def collate_preference_batch(
    batch: List[Dict[str, torch.Tensor]], pad_token_id: int
) -> Batch:
    """Pad variable-length sequences to rectangular batch tensors."""

    def stack_pad(key_ids: str, key_mask: str) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = [b[key_ids] for b in batch]
        masks = [b[key_mask] for b in batch]
        input_ids = pad_sequence(ids, batch_first=True, padding_value=pad_token_id)
        attention_masks = pad_sequence(masks, batch_first=True, padding_value=0)
        return input_ids, attention_masks

    w_ids, w_mask = stack_pad("winner_input_ids", "winner_attention_mask")
    l_ids, l_mask = stack_pad("loser_input_ids", "loser_attention_mask")
    return Batch(
        winner_input_ids=w_ids,
        winner_attention_mask=w_mask,
        loser_input_ids=l_ids,
        loser_attention_mask=l_mask,
    )


def _load_and_maybe_trim_split(
    split: str,
    cache_dir: str | None,
    *,
    debug_mode: bool,
    debug_max_samples: int | None,
) -> Dataset:
    ds = load_dataset(
        "Anthropic/hh-rlhf",
        split=split,
        cache_dir=cache_dir,
    )
    if debug_mode and debug_max_samples is not None:
        n = min(debug_max_samples, len(ds))
        ds = ds.select(range(n))
    return ds


def build_dataloaders(
    tokenizer: GPT2Tokenizer,
    max_length: int,
    batch_size: int,
    num_workers: int,
    *,
    cache_dir: str | None = None,
    debug_mode: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Train loader (shuffle=True) and validation loader (HH `test` split).

    If ``debug_mode`` is True, use the first ``DEBUG_TRAIN_MAX_SAMPLES`` train
    rows and ``DEBUG_VAL_MAX_SAMPLES`` validation rows for fast sanity checks.
    """
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, "Tokenizer must have pad_token_id set."

    train_raw = _load_and_maybe_trim_split(
        "train",
        cache_dir,
        debug_mode=debug_mode,
        debug_max_samples=DEBUG_TRAIN_MAX_SAMPLES if debug_mode else None,
    )
    val_raw = _load_and_maybe_trim_split(
        "test",
        cache_dir,
        debug_mode=debug_mode,
        debug_max_samples=DEBUG_VAL_MAX_SAMPLES if debug_mode else None,
    )

    train_set = HHPreferenceDataset(train_raw, tokenizer, max_length)
    val_set = HHPreferenceDataset(val_raw, tokenizer, max_length)

    collate_fn = lambda samples: collate_preference_batch(samples, pad_id)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
