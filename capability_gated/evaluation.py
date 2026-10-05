"""Evaluation helpers for capability-gated models.

:func:`evaluate_perplexity_at_rank` measures next-token-prediction perplexity at
a chosen active rank -- the primary way to *observe* the effect of gating.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_perplexity_at_rank(
    model: Any,
    tokenizer: Any,  # noqa: ARG001 - kept for signature parity / future use
    dataset: Any,
    rank: int | None,
    batch_size: int = 8,
    max_seq_length: int = 256,  # noqa: ARG001 - dataset is assumed pre-packed
) -> tuple[float, float]:
    """Compute perplexity of ``model`` on ``dataset``.

    If ``rank`` is an int, it is applied to every low-rank layer via
    ``model._set_rank_all(rank)`` before evaluating. Pass ``rank=None`` to
    evaluate the *current* rank configuration untouched -- useful after
    per-block or per-module gating, which a global ``set_rank_all`` would
    otherwise overwrite.

    ``dataset`` must already provide ``input_ids`` and ``labels`` (e.g. from
    :func:`~capability_gated.training.prepare_lm_dataset`).

    Returns:
        ``(perplexity, avg_loss)``.
    """
    model.eval()
    if rank is not None:
        model._set_rank_all(rank)

    def collate_fn(examples: list[dict]) -> dict:
        input_ids = torch.stack([torch.tensor(ex["input_ids"]) for ex in examples])
        labels = torch.stack([torch.tensor(ex["labels"]) for ex in examples])
        return {"input_ids": input_ids, "labels": labels}

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        desc = "Perplexity @ current ranks" if rank is None else f"Perplexity @ rank {rank}"
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            outputs = model(input_ids=input_ids, labels=labels)
            num_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / max(1, total_tokens)
    return math.exp(min(20.0, avg_loss)), avg_loss
