"""
GPT-2 backbone with a scalar value head for reward modeling.

Default: PyTorch SDPA attention + optional PEFT LoRA on the backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2Model

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover
    LoraConfig = None  # type: ignore
    TaskType = None  # type: ignore
    get_peft_model = None  # type: ignore


class GPT2RewardModel(nn.Module):
    """
    GPT-2 without LM head + linear value head.

    Pools the hidden state at the last non-padded token for each sequence and
    projects to a scalar reward.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        *,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        attn_implementation: str | None = "sdpa",
    ) -> None:
        super().__init__()
        backbone = _load_gpt2_backbone(model_name, attn_implementation=attn_implementation)

        if use_lora:
            if get_peft_model is None or LoraConfig is None:
                raise ImportError("LoRA requires `peft`. Install with: pip install peft")
            # FEATURE_EXTRACTION -> PeftModelForFeatureExtraction (no
            # prepare_inputs_for_generation). CAUSAL_LM targets GPT2LMHeadModel only.
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=["c_attn", "c_fc", "c_proj"],
                fan_in_fan_out=True,
            )
            self.backbone = get_peft_model(backbone, lora_config)
        else:
            self.backbone = backbone

        hidden_size = self.backbone.config.n_embd
        self.value_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.value_head.weight, std=0.02)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        lengths = attention_mask.long().sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_indices, lengths]
        reward = self.value_head(last_hidden).squeeze(-1)
        return reward


def _load_gpt2_backbone(model_name: str, attn_implementation: str | None) -> GPT2Model:
    """Load GPT2Model; prefer SDPA (`scaled_dot_product_attention`) when supported."""
    if not attn_implementation:
        return GPT2Model.from_pretrained(model_name)
    try:
        return GPT2Model.from_pretrained(
            model_name,
            attn_implementation=attn_implementation,
        )
    except (TypeError, ValueError, OSError):
        return GPT2Model.from_pretrained(model_name)
