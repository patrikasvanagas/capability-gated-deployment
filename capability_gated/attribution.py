"""Rank-ablation attribution: mechanism-native localisation of capabilities.

The privilege mechanism in this package gates a ``DynamicLowRankLinear``
(``o = B h + bias``, ``h = A x``) to rank ``r`` by slicing the factors. Because
only the module itself changes, the effect on its output is *exact*:

    ``do = -B[:, r:] h[r:]``

First order in any scalar metric ``m`` (here: the correct-answer logit margin of
a multiple-choice item), the effect of gating coordinate ``k = (block, module)``
to rank ``r`` is

    ``dm(k, r) ~= -sum_{j>=r} c_j(k)``  with  ``c_j = sum_t <g_t, B[:, j]> h_{t,j}``

where ``g = ∂m/∂o``. One forward + one backward per batch therefore yields the
per-component attribution ``c ∈ R^R`` for **every** coordinate simultaneously,
and cumulative sums give the predicted effect at every rank of any ladder. The
measured sweep of Rauba et al. (2026) -- one full benchmark evaluation per
(coordinate, rank) -- needs hundreds of evaluations for a strictly coarser map.

Sign convention used everywhere downstream: scores are **drops** --
``s_D(k, r) = sum_{j>=r} c_j`` averaged over items, positive when gating removes
correct-answer margin. Selectivity follows the paper:
``sigma_D(k) = s~_D(k) - max_{D' != D} s~_{D'}(k)`` on normalised scores.

Everything here is first-order and must be validated against reality; the
module also ships the probe machinery (``select_probes`` / ``measure_probes``)
that measures true rank-drop effects for a spanning sample of coordinates.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Self, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .lowrank import DynamicLowRankLinear
from .mcq import (
    DEFAULT_PROMPT_STYLE,
    MCQEvaluator,
    MCQItem,
    MCQResult,
    PromptStyle,
    TokenisedBatch,
    answer_logits,
    answer_margin,
    build_batches,
    letter_token_ids,
    pearsonr,
    spearmanr,
    subject_for,
    two_sided_p_from_z,
)
from .rank_control import get_transformer_blocks

if TYPE_CHECKING:
    from .suppression_search import Coord

# Categorical colors for the three MLP projections, validated for CVD
# separation, lightness band, chroma and surface contrast (dataviz six checks).
MODULE_COLORS: dict[str, str] = {
    "gate_proj": "#33689E",
    "up_proj": "#B3852A",
    "down_proj": "#B65566",
    # GPT-NeoX / Pythia projection names, same fixed order.
    "dense_h_to_4h": "#33689E",
    "dense_4h_to_h": "#B65566",
}
_FALLBACK_COLORS = ("#33689E", "#B3852A", "#B65566", "#5C5C5C")
DOMAIN_MARKERS: dict[str, str] = {"wmdp-bio": "o", "wmdp-chem": "s", "wmdp-cyber": "^"}

_PAPER_RC: Any = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
}
_INK, _MUTED, _HAIR = "0.15", "0.45", "0.80"
# Public alias so run-report scripts can share the paper style without touching
# a private name.
PAPER_RC: Any = _PAPER_RC


def module_color(name: str, position: int = 0) -> str:
    return MODULE_COLORS.get(name, _FALLBACK_COLORS[position % len(_FALLBACK_COLORS)])


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AttributionSpec:
    """The instrument definition, serialised verbatim into the run record.

    The statistical protocol requires the localisation method to be frozen and
    recorded before adaptive use; this object *is* that record's instrument
    section, so anything that changes the scores must live here.
    """

    name: str = "rank_ablation_attribution"
    order: int = 1
    metric: str = "letter_margin"
    token_scope: str = "all"  # noqa: S105 - not a secret; "all"|"last" tokens
    accum_dtype: str = "float32"
    expansion_point: str = "max_rank"
    freeze_params: bool = True
    module_names: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "order": self.order,
            "metric": self.metric,
            "delta_o": "-B[:, r:] @ h[r:]  (exact: the module input is unchanged)",
            "estimator": (
                "dm(k, r) ~= -sum_{j>=r} c_j ;  c_j = sum_t <g_t, B[:, j]> h_{t,j}"
            ),
            "token_scope": self.token_scope,
            "accum_dtype": self.accum_dtype,
            "expansion_point": self.expansion_point,
            "freeze_params": self.freeze_params,
            "module_names": list(self.module_names),
        }


class RankAblationAttributor:
    """Context manager owning the attribution hooks. Always use ``with``.

    On ``__enter__`` every parameter's ``requires_grad`` is saved and cleared --
    the backward then skips all weight-gradient matmuls (costing ~1x forward
    instead of 2x) and allocates no parameter gradients; the graph still exists
    because the *input embeddings* require grad. Every targeted module is
    asserted to sit at ``max_rank``, the Taylor expansion point.

    The forward hook recomputes ``h = A x`` under ``no_grad`` (the module's own
    forward uses ``F.linear`` on sliced weights, so ``self.A`` never fires as a
    submodule) and registers a *tensor* hook on the output. That hook consumes
    the incoming gradient the moment backward produces it -- the backward visits
    the last block first, so hidden states are freed early and only one gradient
    tensor is ever live.

    The hook body returns immediately unless armed (``start_batch``) and the
    output actually requires grad: ``Tensor.register_hook`` raises on tensors
    without ``grad_fn``, so without the guard a leaked attributor would crash
    every later inference-mode evaluation.
    """

    def __init__(
        self,
        model: Any,
        *,
        spec: AttributionSpec | None = None,
        max_rank: int | None = None,
    ) -> None:
        self.model = model
        self.spec = spec or AttributionSpec()
        blocks = get_transformer_blocks(model)
        self.n_blocks = len(blocks)
        self.module_names = tuple(self.spec.module_names)
        self._modules: dict[Coord, DynamicLowRankLinear] = {}
        for block_idx, block in enumerate(blocks):
            for name, mod in block.named_modules():
                local = name.split(".")[-1]
                if isinstance(mod, DynamicLowRankLinear) and local in self.module_names:
                    self._modules[(block_idx, local)] = mod
        expected = self.n_blocks * len(self.module_names)
        if len(self._modules) != expected:
            raise ValueError(
                f"Found {len(self._modules)} gated modules but expected "
                f"{expected} ({self.n_blocks} blocks x {self.module_names}); "
                f"the coordinate grid must be rectangular.",
            )
        ranks = {m.max_rank for m in self._modules.values()}
        if len(ranks) != 1:
            raise ValueError(f"Mixed max_rank across modules: {sorted(ranks)}")
        self.max_rank = int(max_rank or ranks.pop())
        self.coords: list[Coord] = [
            (b, m) for b in range(self.n_blocks) for m in self.module_names
        ]
        self._coord_pos = {
            (b, m): (b, mi)
            for b in range(self.n_blocks)
            for mi, m in enumerate(self.module_names)
        }
        self._handles: list[Any] = []
        self._saved_requires_grad: list[tuple[Any, bool]] = []
        self._recording = False
        self._hidden: dict[Coord, torch.Tensor] = {}
        self._visited: set[Coord] = set()
        self._batch_c: torch.Tensor | None = None
        self.debug_keep = False
        self.last_hidden: dict[Coord, torch.Tensor] = {}
        self.last_grad: dict[Coord, torch.Tensor] = {}

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self) -> Self:
        self._saved_requires_grad = [
            (p, p.requires_grad) for p in self.model.parameters()
        ]
        if self.spec.freeze_params:
            for p, _ in self._saved_requires_grad:
                p.requires_grad_(requires_grad=False)
        for coord, mod in self._modules.items():
            if mod.active_rank != self.max_rank:
                raise RuntimeError(
                    f"{coord} is at rank {mod.active_rank}, not max_rank "
                    f"{self.max_rank}; the expansion point is the ungated model.",
                )
            self._handles.append(mod.register_forward_hook(self._make_hook(coord)))
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            for handle in self._handles:
                handle.remove()
        finally:
            self._handles.clear()
            for p, flag in self._saved_requires_grad:
                p.requires_grad_(requires_grad=flag)
            self._saved_requires_grad.clear()
            self._recording = False
            self._hidden.clear()
            self._batch_c = None

    # -- hooks ---------------------------------------------------------------
    def _make_hook(self, coord: Coord) -> Any:
        mod = self._modules[coord]

        def forward_hook(_mod: Any, args: tuple, output: torch.Tensor) -> None:
            if not self._recording or not output.requires_grad:
                return
            if coord in self._visited or coord in self._hidden:
                raise RuntimeError(
                    f"{coord} fired twice in one batch -- gradient checkpointing "
                    f"or a re-entrant forward would double-count attributions.",
                )
            with torch.no_grad():
                hidden = F.linear(args[0].detach(), mod.A.weight)  # (rows, T, R)
            self._hidden[coord] = hidden
            output.register_hook(lambda grad: self._consume(coord, grad))

        return forward_hook

    def _consume(self, coord: Coord, grad: torch.Tensor) -> None:
        hidden = self._hidden.pop(coord)
        mod = self._modules[coord]
        if self.debug_keep:
            self.last_hidden[coord] = hidden.detach().float().cpu()
            self.last_grad[coord] = grad.detach().float().cpu()
        # (rows, T, out) @ (out, R) -> (rows, T, R); cuBLAS accumulates the long
        # reduction in fp32 internally, so the matmul may stay in model dtype.
        projected = torch.matmul(grad.detach(), mod.B.weight)
        contrib = (projected.float() * hidden.float()).sum(dim=-2)  # (rows, R)
        block_idx, module_idx = self._coord_pos[coord]
        assert self._batch_c is not None
        self._batch_c[:, block_idx, module_idx, :] = contrib
        self._visited.add(coord)

    # -- per-batch driver interface -----------------------------------------
    def start_batch(self, n_rows: int) -> None:
        device = next(iter(self._modules.values())).A.weight.device
        self._batch_c = torch.zeros(
            (n_rows, self.n_blocks, len(self.module_names), self.max_rank),
            dtype=torch.float32,
            device=device,
        )
        self._hidden.clear()
        self._visited = set()
        self._recording = True

    def finish_batch(self) -> torch.Tensor:
        self._recording = False
        missing = set(self.coords) - self._visited
        if missing:
            raise RuntimeError(
                f"{len(missing)} coordinates never received a gradient (e.g. "
                f"{sorted(missing)[:3]}). The forward likely ran under "
                f"no_grad/inference_mode, or the metric did not depend on them.",
            )
        if self._hidden:
            raise RuntimeError(
                "Hidden states left unconsumed -- the backward pass never ran.",
            )
        assert self._batch_c is not None
        batch_c, self._batch_c = self._batch_c, None
        return batch_c


@dataclass
class FluencyPriceMap:
    """First-order NLL cost of gating every coordinate at any integer rank.

    ``component_nll[b, m, j]`` is ``d NLL / d component_j`` at full privilege.
    Gating to rank ``r`` removes components ``j >= r``, so the predicted NLL
    change is ``-sum(component_nll[..., r:])``.  E3 sums the positive part of
    these singleton changes as a deliberately conservative, composable price;
    its final measured perplexity gate remains authoritative because first-order
    prices cannot capture interactions between coordinates.
    """

    blocks: np.ndarray
    modules: np.ndarray
    component_nll: np.ndarray
    baseline_nll: float
    n_blocks: int
    n_tokens: int
    seconds: float
    dataset: str = ""
    split: str = ""

    @property
    def max_rank(self) -> int:
        return int(self.component_nll.shape[-1])

    def _positions(self, coord: Coord) -> tuple[int, int]:
        block, module = coord
        bi = np.nonzero(self.blocks == int(block))[0]
        mi = np.nonzero(self.modules == str(module))[0]
        if len(bi) != 1 or len(mi) != 1:
            raise KeyError(f"Unknown fluency-price coordinate {coord!r}")
        return int(bi[0]), int(mi[0])

    def predicted_delta_nll(self, coord: Coord, rank: int) -> float:
        """Signed first-order ``NLL(gated) - NLL(full)`` for one move."""
        rank = int(rank)
        if not 0 <= rank <= self.max_rank:
            raise ValueError(f"rank must lie in [0, {self.max_rank}], got {rank}")
        if rank == self.max_rank:
            return 0.0
        bi, mi = self._positions(coord)
        return -float(self.component_nll[bi, mi, rank:].sum(dtype=np.float64))

    def price(self, coord: Coord, rank: int) -> float:
        """Non-negative singleton price used by the search Lagrangian."""
        return max(0.0, self.predicted_delta_nll(coord, rank))

    def config_price(self, cfg: Mapping[Coord, int]) -> float:
        """Additive predicted NLL cost, with negative singleton credits clipped."""
        return float(sum(self.price(coord, rank) for coord, rank in cfg.items()))

    def save_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema": "capability_gated.attribution/fluency_prices/v1",
            "baseline_nll": self.baseline_nll,
            "n_blocks": self.n_blocks,
            "n_tokens": self.n_tokens,
            "seconds": self.seconds,
            "dataset": self.dataset,
            "split": self.split,
            "sign_convention": (
                "predicted_delta_nll(coord,r)=-sum(component_nll[coord,r:]); "
                "search price=max(0,predicted_delta_nll)"
            ),
        }
        np.savez_compressed(
            path,
            blocks=self.blocks,
            modules=self.modules,
            component_nll=self.component_nll,
            meta=np.array(json.dumps(meta)),
        )
        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> FluencyPriceMap:
        with np.load(path, allow_pickle=False) as payload:
            meta = json.loads(str(payload["meta"]))
            return cls(
                blocks=payload["blocks"],
                modules=payload["modules"],
                component_nll=payload["component_nll"],
                baseline_nll=float(meta["baseline_nll"]),
                n_blocks=int(meta["n_blocks"]),
                n_tokens=int(meta["n_tokens"]),
                seconds=float(meta["seconds"]),
                dataset=str(meta.get("dataset", "")),
                split=str(meta.get("split", "")),
            )


def estimate_fluency_prices(
    model: Any,
    dataset: Any,
    *,
    max_rank: int,
    batch_size: int = 4,
    spec: AttributionSpec | None = None,
    dataset_name: str = "",
    split: str = "",
    progress: bool = True,
) -> FluencyPriceMap:
    """Estimate every singleton's NLL price with one LM backward pass.

    The packed blocks all have the same length, but accumulation is still
    token-weighted so a future dataset with ignored labels or a short last block
    has the correct corpus-average scale.  No parameter gradients are formed;
    :class:`RankAblationAttributor` freezes weights and records only activation
    derivatives at the full-rank expansion point.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    def collate(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        keys = [k for k in ("input_ids", "labels", "attention_mask") if k in examples[0]]
        return {k: torch.stack([torch.as_tensor(ex[k]) for ex in examples]) for k in keys}

    model.eval()
    model._set_rank_all(int(max_rank))
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    started = time.perf_counter()
    total_tokens = 0
    total_loss = 0.0

    with RankAblationAttributor(model, spec=spec, max_rank=max_rank) as attributor:
        weighted_components = torch.zeros(
            (attributor.n_blocks, len(attributor.module_names), max_rank),
            dtype=torch.float64,
            device=device,
        )
        for batch_i, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            # Causal-LM loss predicts labels[..., 1:] from positions [..., :-1].
            n_tokens = int((labels[..., 1:] != -100).sum().item())
            with torch.enable_grad():
                attributor.start_batch(int(input_ids.shape[0]))
                embeds = (
                    model.get_input_embeddings()(input_ids)
                    .detach()
                    .requires_grad_(requires_grad=True)
                )
                outputs = model(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                torch.autograd.grad(outputs.loss, embeds)
                batch_c = attributor.finish_batch()
            # outputs.loss is the mean over this batch's prediction tokens.
            weighted_components += batch_c.sum(dim=0).double() * n_tokens
            total_loss += float(outputs.loss.detach()) * n_tokens
            total_tokens += n_tokens
            if progress and (batch_i % 20 == 0 or batch_i == len(loader)):
                elapsed = time.perf_counter() - started
                print(
                    f"[fluency-prices] {batch_i}/{len(loader)} batches | "
                    f"{elapsed:.0f}s | {total_tokens / max(elapsed, 1e-9):,.0f} tok/s",
                )
            del batch_c, outputs, embeds, input_ids, labels

        components = (weighted_components / max(1, total_tokens)).float().cpu().numpy()

    return FluencyPriceMap(
        blocks=np.arange(attributor.n_blocks, dtype=np.int32),
        modules=np.array(attributor.module_names),
        component_nll=components,
        baseline_nll=total_loss / max(1, total_tokens),
        n_blocks=len(dataset),
        n_tokens=total_tokens,
        seconds=time.perf_counter() - started,
        dataset=dataset_name,
        split=split,
    )


def _split_batch(batch: TokenisedBatch) -> list[TokenisedBatch]:
    """Halve a batch row-wise (used by the OOM backoff)."""
    n = int(batch.rows.shape[0])
    if n <= 1:
        raise torch.OutOfMemoryError("Cannot split a single-row batch further.")
    halves = []
    for sl in (slice(0, n // 2), slice(n // 2, n)):
        mask = batch.attention_mask[sl]
        halves.append(
            TokenisedBatch(
                input_ids=batch.input_ids[sl],
                attention_mask=mask,
                last_index=batch.last_index[sl],
                answers=batch.answers[sl],
                rows=batch.rows[sl],
                n_real_tokens=int(mask.sum()),
            ),
        )
    return halves


@dataclass
class AttributionResult:
    """The score cube of one task: per-component attributions + per-item tails.

    ``components_*`` are means over items of ``c_j`` -- ``(n_blocks, n_modules,
    R)``. ``per_item_tails`` stores, per item, the predicted margin **drop**
    ``sum_{j>=r} c_j`` at each ladder rank -- ``(N, n_blocks, n_modules, K)`` --
    which is small (a few MB) and gives item-level bootstrap CIs for any
    downstream statistic; per-item *components* would be ~100x larger and are
    deliberately not kept.
    """

    task: str
    spec: AttributionSpec
    blocks: np.ndarray  # (n_blocks,) int32
    modules: np.ndarray  # (n_modules,) str
    ranks: np.ndarray  # (K,) int32 descending, includes max_rank
    components_all: np.ndarray  # (n_blocks, n_modules, R) float32
    components_correct: np.ndarray
    per_item_tails: np.ndarray  # (N, n_blocks, n_modules, K) float32
    item_index: np.ndarray  # (N,) source-split row numbers
    item_margin: np.ndarray  # (N,) float32, logsumexp margins at full privilege
    item_decision: np.ndarray  # (N,) float32, z_correct - max(z_incorrect)
    item_correct: np.ndarray  # (N,) bool, full-privilege correctness
    n_items: int
    n_correct: int
    n_tokens: int
    n_truncated: int
    seconds: float
    peak_bytes: int
    model_id: str = ""
    checkpoint: str = ""

    # -- views ---------------------------------------------------------------
    def _item_mask(self, items: str) -> np.ndarray:
        if items == "all":
            return np.ones(self.n_items, dtype=bool)
        if items == "correct":
            return self.item_correct
        raise ValueError(f"items must be 'all' or 'correct', got {items!r}")

    def mean_components(self, *, items: str = "correct") -> np.ndarray:
        return self.components_correct if items == "correct" else self.components_all

    def mean_margin(self, *, items: str = "correct") -> float:
        mask = self._item_mask(items)
        return float(self.item_margin[mask].mean()) if mask.any() else float("nan")

    def mean_decision(self, *, items: str = "correct") -> float:
        """Mean decision margin ``z_correct - max(z_incorrect)``.

        Unlike the logsumexp margin (which is negative on low-confidence items
        even when the argmax is right, because logsumexp of three close wrong
        options exceeds their max), this is positive on every correctly answered
        item -- which makes it the right *scale* for cross-task normalisation.
        """
        mask = self._item_mask(items)
        return float(self.item_decision[mask].mean()) if mask.any() else float("nan")

    def _rank_pos(self, rank: int) -> int:
        pos = np.nonzero(self.ranks == rank)[0]
        if len(pos) == 0:
            raise KeyError(f"rank {rank} not on the stored ladder {self.ranks.tolist()}")
        return int(pos[0])

    def tail(self, rank: int, *, items: str = "correct") -> np.ndarray:
        """Predicted margin drop ``sum_{j>=r} c_j`` per coordinate, ``(n_blocks, n_modules)``.

        Computed from the cumulative sum so that ``tail(max_rank)`` is *exactly*
        zero -- the expansion point anchors every curve.
        """
        comp = self.mean_components(items=items).astype(np.float64)
        csum = comp.cumsum(axis=-1)
        total = csum[..., -1]
        if rank >= self.ranks.max():
            return np.zeros_like(total, dtype=np.float32)
        return (total - csum[..., rank - 1]).astype(np.float32)

    def predicted_delta_margin(self, rank: int, *, items: str = "correct") -> np.ndarray:
        return -self.tail(rank, items=items)

    def curve(self, coord: Coord, *, items: str = "correct") -> np.ndarray:
        """Predicted drop at every ladder rank for one coordinate, ``(K,)``."""
        block_idx, module_name = coord
        module_idx = int(np.nonzero(self.modules == module_name)[0][0])
        mask = self._item_mask(items)
        return self.per_item_tails[mask, block_idx, module_idx, :].mean(axis=0)

    def spectrum(
        self, coord: Coord, *, cumulative: bool = True, items: str = "correct",
    ) -> np.ndarray:
        block_idx, module_name = coord
        module_idx = int(np.nonzero(self.modules == module_name)[0][0])
        comp = self.mean_components(items=items)[block_idx, module_idx]
        return comp.cumsum() if cumulative else comp

    def bootstrap(
        self,
        rank: int,
        *,
        items: str = "correct",
        n_boot: int = 2000,
        seed: int = 0,
    ) -> dict[str, np.ndarray]:
        """Item-resampled uncertainty of the mean drop at ``rank``.

        Returns ``mean/se/lo/hi/z/p`` arrays of shape ``(n_blocks, n_modules)``;
        feed ``p`` to :func:`capability_gated.analysis.benjamini_hochberg` for an
        FDR mask over the coordinate grid.
        """
        mask = self._item_mask(items)
        values = self.per_item_tails[mask][:, :, :, self._rank_pos(rank)]
        rng = np.random.default_rng(seed)
        n = values.shape[0]
        idx = rng.integers(0, n, size=(n_boot, n))
        means = values[idx].mean(axis=1)  # (n_boot, n_blocks, n_modules)
        mean = values.mean(axis=0)
        se = means.std(axis=0, ddof=1)
        z = np.divide(mean, se, out=np.zeros_like(mean), where=se > 0)
        return {
            "mean": mean,
            "se": se,
            "lo": np.quantile(means, 0.025, axis=0),
            "hi": np.quantile(means, 0.975, axis=0),
            "z": z,
            "p": np.asarray(two_sided_p_from_z(z)),
        }

    # -- persistence ---------------------------------------------------------
    def save_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "task": self.task,
            "spec": self.spec.to_json(),
            "n_items": self.n_items,
            "n_correct": self.n_correct,
            "n_tokens": self.n_tokens,
            "n_truncated": self.n_truncated,
            "seconds": self.seconds,
            "peak_bytes": self.peak_bytes,
            "model_id": self.model_id,
            "checkpoint": self.checkpoint,
        }
        np.savez_compressed(
            path,
            blocks=self.blocks,
            modules=self.modules,
            ranks=self.ranks,
            components_all=self.components_all,
            components_correct=self.components_correct,
            per_item_tails=self.per_item_tails,
            item_index=self.item_index,
            item_margin=self.item_margin,
            item_decision=self.item_decision,
            item_correct=self.item_correct,
            meta=np.array(json.dumps(meta)),
        )
        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> AttributionResult:
        with np.load(path, allow_pickle=False) as payload:
            meta = json.loads(str(payload["meta"]))
            spec_json = dict(meta["spec"])
            spec = AttributionSpec(
                name=spec_json["name"],
                order=int(spec_json["order"]),
                metric=spec_json["metric"],
                token_scope=spec_json["token_scope"],
                accum_dtype=spec_json["accum_dtype"],
                expansion_point=spec_json["expansion_point"],
                freeze_params=bool(spec_json["freeze_params"]),
                module_names=tuple(spec_json["module_names"]),
            )
            return cls(
                task=meta["task"],
                spec=spec,
                blocks=payload["blocks"],
                modules=payload["modules"],
                ranks=payload["ranks"],
                components_all=payload["components_all"],
                components_correct=payload["components_correct"],
                per_item_tails=payload["per_item_tails"],
                item_index=payload["item_index"],
                item_margin=payload["item_margin"],
                item_decision=payload["item_decision"],
                item_correct=payload["item_correct"],
                n_items=int(meta["n_items"]),
                n_correct=int(meta["n_correct"]),
                n_tokens=int(meta["n_tokens"]),
                n_truncated=int(meta["n_truncated"]),
                seconds=float(meta["seconds"]),
                peak_bytes=int(meta["peak_bytes"]),
                model_id=meta["model_id"],
                checkpoint=meta["checkpoint"],
            )


def run_attribution(
    model: Any,
    tokenizer: Any,
    items: Sequence[MCQItem],
    *,
    task: str,
    max_rank: int,
    ranks: Sequence[int],
    spec: AttributionSpec | None = None,
    prompt_style: PromptStyle = DEFAULT_PROMPT_STYLE,
    subject: str | None = None,
    shots: Sequence[MCQItem] = (),
    max_seq_len: int = 1024,
    max_tokens_per_batch: int = 2048,
    max_rows_per_batch: int = 16,
    oom_backoff: bool = True,
    progress: bool = True,
    model_id: str = "",
    checkpoint: str = "",
) -> AttributionResult:
    """Score every coordinate on ``items`` with one forward+backward per batch.

    ``max_tokens_per_batch`` defaults to a quarter of the evaluator's budget:
    the gradient-enabled pass keeps ~1.4 MB of activations per token alive, so
    2048 tokens is roughly a 3 GB graph on top of the weights. On CUDA OOM the
    failing batch is split in half and retried (`oom_backoff`), so a long run
    degrades instead of dying.

    The pass also yields full-privilege margins/correctness as by-products;
    callers should check them against the plain evaluator's numbers -- any
    mismatch means the two code paths diverged and every score is suspect.
    """
    spec = spec or AttributionSpec()
    ladder = sorted({int(r) for r in ranks} | {int(max_rank)}, reverse=True)
    if min(ladder) < 1 or max(ladder) > max_rank:
        raise ValueError(f"ranks must lie in [1, {max_rank}]; got {ladder}")
    device = next(model.parameters()).device
    letter_ids = letter_token_ids(tokenizer)
    batches, stats = build_batches(
        items,
        tokenizer,
        prompt_style=prompt_style,
        subject=subject or subject_for(task),
        shots=shots,
        max_seq_len=max_seq_len,
        max_tokens_per_batch=max_tokens_per_batch,
        max_rows_per_batch=max_rows_per_batch,
        device=device,
    )

    n_items = len(items)
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    per_item_tails: np.ndarray | None = None
    item_margin = np.zeros(n_items, dtype=np.float32)
    item_decision = np.zeros(n_items, dtype=np.float32)
    item_correct = np.zeros(n_items, dtype=bool)

    with RankAblationAttributor(model, spec=spec, max_rank=max_rank) as attributor:
        n_blocks = attributor.n_blocks
        n_modules = len(attributor.module_names)
        per_item_tails = np.zeros(
            (n_items, n_blocks, n_modules, len(ladder)), dtype=np.float32,
        )
        sum_all = torch.zeros(
            (n_blocks, n_modules, max_rank), dtype=torch.float32, device=device,
        )
        sum_correct = torch.zeros_like(sum_all)
        # Ladder tails read the inclusive cumsum at index r-1; taking the total
        # from the same cumsum keeps tail(max_rank) == 0 to the last bit.
        rank_index = torch.tensor([r - 1 for r in ladder], device=device)

        queue = list(batches)
        done = 0
        while queue:
            batch = queue.pop(0)
            try:
                with torch.enable_grad():
                    attributor.start_batch(int(batch.rows.shape[0]))
                    embeds = (
                        model.get_input_embeddings()(batch.input_ids)
                        .detach()
                        .requires_grad_(requires_grad=True)
                    )
                    logits = answer_logits(
                        model, batch, letter_ids, inputs_embeds=embeds,
                    )
                    margins = answer_margin(logits, batch.answers)
                    torch.autograd.grad(margins.sum(), embeds)
                    batch_c = attributor.finish_batch()
            except torch.OutOfMemoryError:
                if not oom_backoff:
                    raise
                torch.cuda.empty_cache()
                halves = _split_batch(batch)
                print(
                    f"[attribution] OOM at {int(batch.rows.shape[0])} rows; "
                    f"retrying as 2 x {[int(h.rows.shape[0]) for h in halves]}",
                )
                queue = halves + queue
                continue

            with torch.no_grad():
                detached = logits.detach().float()
                preds = detached.argmax(dim=1)
                correct = preds == batch.answers
                row_idx = torch.arange(detached.shape[0], device=detached.device)
                z_correct = detached[row_idx, batch.answers]
                wrong = detached.clone()
                wrong[row_idx, batch.answers] = float("-inf")
                decision = z_correct - wrong.max(dim=1).values
                csum = batch_c.double().cumsum(dim=-1)
                total = csum[..., -1]
                tails = total.unsqueeze(-1) - csum[..., rank_index]
                tails[..., 0] = 0.0  # ladder[0] == max_rank, exactly zero
                rows = batch.rows.cpu().numpy()
                per_item_tails[rows] = tails.float().cpu().numpy()
                item_margin[rows] = margins.detach().float().cpu().numpy()
                item_decision[rows] = decision.cpu().numpy()
                item_correct[rows] = correct.cpu().numpy()
                sum_all += batch_c.sum(dim=0)
                if bool(correct.any()):
                    sum_correct += batch_c[correct].sum(dim=0)
            del batch_c, logits, margins, embeds
            done += 1
            if progress and (done % 20 == 0 or not queue):
                elapsed = time.perf_counter() - started
                print(
                    f"[attribution:{task}] {done} batches | "
                    f"{elapsed:.0f}s | {stats['total_tokens'] / max(1e-9, elapsed):,.0f} tok/s",
                )

        n_correct = int(item_correct.sum())
        components_all = (sum_all / max(1, n_items)).cpu().numpy()
        components_correct = (sum_correct / max(1, n_correct)).cpu().numpy()

    return AttributionResult(
        task=task,
        spec=spec,
        blocks=np.arange(attributor.n_blocks, dtype=np.int32),
        modules=np.array(attributor.module_names),
        ranks=np.array(ladder, dtype=np.int32),
        components_all=components_all,
        components_correct=components_correct,
        per_item_tails=per_item_tails,
        item_index=np.array([it.index for it in items], dtype=np.int64),
        item_margin=item_margin,
        item_decision=item_decision,
        item_correct=item_correct,
        n_items=n_items,
        n_correct=n_correct,
        n_tokens=int(stats["total_tokens"]),
        n_truncated=int(stats["n_truncated"]),
        seconds=time.perf_counter() - started,
        peak_bytes=(
            int(torch.cuda.max_memory_allocated(device))
            if torch.cuda.is_available()
            else 0
        ),
        model_id=model_id,
        checkpoint=checkpoint,
    )


# ---------------------------------------------------------------------------
# Score maps and selectivity
# ---------------------------------------------------------------------------
@dataclass
class ScoreMaps:
    """Per-task score maps and the cross-task selectivity map.

    Axes of every array: ``(task, n_blocks, n_modules, K ladder ranks)``.
    ``sigma`` uses each task's configured competitor set (default: every other
    scored task, which is what a retention-constrained search cares about);
    ``sigma_wmdp`` always carries the paper's literal variant where competitors
    are the *other WMDP domains only*.
    """

    tasks: list[str]
    competitors: dict[str, list[str]]
    blocks: np.ndarray
    modules: np.ndarray
    ranks: np.ndarray
    s_raw: np.ndarray  # predicted margin drop, margin units
    s_norm: np.ndarray
    sigma: np.ndarray
    sigma_wmdp: np.ndarray
    baseline_margin: dict[str, float]
    n_items: dict[str, int]
    normalisation: str
    item_filter: str

    def _task_pos(self, task: str) -> int:
        return self.tasks.index(task)

    def _rank_pos(self, rank: int) -> int:
        pos = np.nonzero(self.ranks == rank)[0]
        if len(pos) == 0:
            raise KeyError(f"rank {rank} not on ladder {self.ranks.tolist()}")
        return int(pos[0])

    def score(self, task: str, rank: int) -> np.ndarray:
        """Normalised drop map at one rank, ``(n_blocks, n_modules)``."""
        return self.s_norm[self._task_pos(task), :, :, self._rank_pos(rank)]

    def selectivity(
        self, task: str, rank: int, *, variant: str = "all",
    ) -> np.ndarray:
        source = self.sigma if variant == "all" else self.sigma_wmdp
        return source[self._task_pos(task), :, :, self._rank_pos(rank)]

    def coord_list(self) -> list[Coord]:
        return [
            (int(b), str(m))
            for b in self.blocks
            for m in self.modules
        ]

    def top_coords(
        self,
        task: str,
        rank: int,
        k: int,
        *,
        by: str = "selectivity",
    ) -> list[tuple[Coord, float]]:
        grid = (
            self.selectivity(task, rank) if by == "selectivity" else self.score(task, rank)
        )
        flat = [
            ((int(b), str(m)), float(grid[bi, mi]))
            for bi, b in enumerate(self.blocks)
            for mi, m in enumerate(self.modules)
        ]
        flat.sort(key=lambda pair: pair[1], reverse=True)
        return flat[:k]

    def rank_reference_stability(self, task: str) -> np.ndarray:
        """Spearman correlation of the score map between every pair of ranks.

        High values mean the choice of reference rank barely matters; a low
        off-diagonal band would say the map itself depends on the operating
        point, which the pre-registered ``rank_ref`` must then acknowledge.
        """
        ti = self._task_pos(task)
        k = len(self.ranks)
        out = np.ones((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                rho = spearmanr(
                    self.s_norm[ti, :, :, i].ravel(),
                    self.s_norm[ti, :, :, j].ravel(),
                )
                out[i, j] = out[j, i] = rho
        return out

    def save_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "tasks": self.tasks,
            "competitors": self.competitors,
            "baseline_margin": self.baseline_margin,
            "n_items": self.n_items,
            "normalisation": self.normalisation,
            "item_filter": self.item_filter,
        }
        np.savez_compressed(
            path,
            blocks=self.blocks,
            modules=self.modules,
            ranks=self.ranks,
            s_raw=self.s_raw,
            s_norm=self.s_norm,
            sigma=self.sigma,
            sigma_wmdp=self.sigma_wmdp,
            meta=np.array(json.dumps(meta)),
        )
        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> ScoreMaps:
        with np.load(path, allow_pickle=False) as payload:
            meta = json.loads(str(payload["meta"]))
            return cls(
                tasks=list(meta["tasks"]),
                competitors={k: list(v) for k, v in meta["competitors"].items()},
                blocks=payload["blocks"],
                modules=payload["modules"],
                ranks=payload["ranks"],
                s_raw=payload["s_raw"],
                s_norm=payload["s_norm"],
                sigma=payload["sigma"],
                sigma_wmdp=payload["sigma_wmdp"],
                baseline_margin={k: float(v) for k, v in meta["baseline_margin"].items()},
                n_items={k: int(v) for k, v in meta["n_items"].items()},
                normalisation=meta["normalisation"],
                item_filter=meta["item_filter"],
            )


def build_score_maps(
    results: Mapping[str, AttributionResult],
    *,
    competitors: Mapping[str, Sequence[str]] | None = None,
    normalisation: str = "margin_frac",
    item_filter: str = "correct",
) -> ScoreMaps:
    """Assemble per-task maps and selectivity from attribution results.

    Normalisation matters more than it looks: margins are not comparable across
    tasks (a harder task has smaller margins), so differencing raw scores
    across tasks would mostly measure task difficulty.

    * ``"margin_frac"`` (default): ``s / mean_decision_margin_task`` -- "the
      fraction of the decision margin this coordinate carries". The scale is the
      mean of ``z_correct - max(z_incorrect)`` over the filtered items, which is
      positive on every correctly answered item by definition; the logsumexp
      margin is *not* usable here because it goes negative on low-confidence
      correct answers (weak domains at small model scale). Dimensionless,
      preserves zero, directly interpretable.
    * ``"zscore"``: z across the coordinate grid within (task, rank); robustness
      check for the shortlist.
    * ``"none"``: raw margin units.
    """
    tasks = list(results)
    first = results[tasks[0]]
    for task in tasks[1:]:
        res = results[task]
        if not (
            np.array_equal(res.ranks, first.ranks)
            and np.array_equal(res.blocks, first.blocks)
            and np.array_equal(res.modules, first.modules)
        ):
            raise ValueError(f"{task} was scored on a different coordinate/rank grid.")

    shape = (len(tasks), len(first.blocks), len(first.modules), len(first.ranks))
    s_raw = np.zeros(shape, dtype=np.float32)
    baseline_margin: dict[str, float] = {}
    n_items: dict[str, int] = {}
    for ti, task in enumerate(tasks):
        res = results[task]
        mask = res._item_mask(item_filter)
        s_raw[ti] = res.per_item_tails[mask].mean(axis=0)
        baseline_margin[task] = res.mean_decision(items=item_filter)
        n_items[task] = int(mask.sum())

    if normalisation == "margin_frac":
        s_norm = np.empty_like(s_raw)
        for ti, task in enumerate(tasks):
            margin = baseline_margin[task]
            if not margin > 0:
                raise ValueError(
                    f"{task}: mean decision margin {margin:.4f} <= 0 under "
                    f"filter {item_filter!r}; margin_frac is undefined. With "
                    f"item_filter='correct' this cannot happen unless the task "
                    f"has no correct items -- which the headroom gate screens "
                    f"out. Run the gate first.",
                )
            s_norm[ti] = s_raw[ti] / margin
    elif normalisation == "zscore":
        flat = s_raw.reshape(len(tasks), -1, shape[-1])
        mean = flat.mean(axis=1)[:, None, :]
        std = flat.std(axis=1)[:, None, :]
        s_norm = (
            (flat - mean) / np.where(std > 0, std, np.float32(1.0))
        ).reshape(shape).astype(np.float32)
    elif normalisation == "none":
        s_norm = s_raw.copy()
    else:
        raise ValueError(f"Unknown normalisation {normalisation!r}")

    competitor_map = {
        task: list((competitors or {}).get(task, [x for x in tasks if x != task]))
        for task in tasks
    }
    for task, comps in competitor_map.items():
        unknown = [c for c in comps if c not in tasks]
        if unknown:
            raise ValueError(f"{task}: competitors {unknown} were not scored.")

    def sigma_against(chosen: Mapping[str, Sequence[str]]) -> np.ndarray:
        out = np.full(shape, np.nan, dtype=np.float32)
        for ti, task in enumerate(tasks):
            comps = [tasks.index(c) for c in chosen.get(task, []) if c != task]
            if not comps:
                continue
            out[ti] = s_norm[ti] - s_norm[comps].max(axis=0)
        return out

    wmdp_map = {
        task: [t for t in tasks if t != task and t.startswith("wmdp")]
        for task in tasks
        if task.startswith("wmdp")
    }
    return ScoreMaps(
        tasks=tasks,
        competitors={k: list(v) for k, v in competitor_map.items()},
        blocks=first.blocks.copy(),
        modules=first.modules.copy(),
        ranks=first.ranks.copy(),
        s_raw=s_raw,
        s_norm=s_norm,
        sigma=sigma_against(competitor_map),
        sigma_wmdp=sigma_against(wmdp_map),
        baseline_margin=baseline_margin,
        n_items=n_items,
        normalisation=normalisation,
        item_filter=item_filter,
    )


# ---------------------------------------------------------------------------
# Shortlists -- the search-ready artifact
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShortlistEntry:
    coord: Coord
    ranks: tuple[int, ...]
    score: float
    selectivity: float
    collateral: float
    predicted_drop_frac: dict[int, float]
    fluency_price_nll: dict[int, float] | None = None
    ci_lo: float = float("nan")
    ci_hi: float = float("nan")
    fdr_reject: bool = False


@dataclass
class Shortlist:
    domain: str
    strategy: str
    rank_ref: int
    normalisation: str
    entries: list[ShortlistEntry]

    def candidate_ranks(self) -> dict[Coord, list[int]]:
        """Exactly ``suppression_search.beam_search``'s ``candidate_ranks`` input."""
        return {e.coord: list(e.ranks) for e in self.entries}

    def to_json(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "strategy": self.strategy,
            "rank_ref": self.rank_ref,
            "normalisation": self.normalisation,
            "candidate_ranks": {
                f"{b}_{m}": list(e.ranks) for (b, m), e in
                ((entry.coord, entry) for entry in self.entries)
            },
            "entries": [
                {
                    "block": e.coord[0],
                    "module": e.coord[1],
                    "ranks": list(e.ranks),
                    "score": e.score,
                    "selectivity": e.selectivity,
                    "collateral": e.collateral,
                    "predicted_drop_frac": {str(r): v for r, v in e.predicted_drop_frac.items()},
                    "fluency_price_nll": (
                        {str(r): v for r, v in e.fluency_price_nll.items()}
                        if e.fluency_price_nll is not None else None
                    ),
                    "ci_lo": e.ci_lo,
                    "ci_hi": e.ci_hi,
                    "fdr_reject": e.fdr_reject,
                }
                for e in self.entries
            ],
        }


def build_shortlist(
    maps: ScoreMaps,
    domain: str,
    *,
    rank_ref: int,
    top_k: int = 12,
    ranks_per_coord: int = 2,
    strategy: str = "selectivity",
    mass_targets: Sequence[float] = (0.5, 0.9),
    seed: int = 0,
    exclude: Iterable[Coord] = (),
    significance: Mapping[str, np.ndarray] | None = None,
    fluency_prices: FluencyPriceMap | None = None,
) -> Shortlist:
    """Top-k coordinates with per-coordinate candidate ranks.

    Candidate ranks are chosen per coordinate rather than from a global ladder:
    for each mass target ``q``, take the *mildest* ladder rank whose predicted
    drop reaches ``q x`` that coordinate's **peak** drop over the ladder. A
    coordinate whose mass sits in the last components gets a gentle rank; one
    whose mass is spread gets an aggressive one -- the point of having a
    per-component score rather than a per-coordinate scalar. Targets are
    relative to the peak, not to the rank-1 total, because per-component
    attributions can be negative and the drop is therefore not monotone in
    rank -- the peak is what a suppression search can actually reach.

    ``strategy`` doubles as the matched-budget ablation arms the paper needs:
    ``"magnitude"`` ranks by raw importance (ignores selectivity),
    ``"random"`` is a seeded uniform draw, and ``"anti"`` takes the *least*
    selective coordinates -- a control that should fail. ``"all"`` keeps every
    coordinate (no sign filter, ordered by selectivity) and is the candidate
    grid for a measured single-intervention sweep.

    The suppression-oriented strategies (``selectivity``/``magnitude``) only
    admit coordinates whose predicted drop is positive at ``rank_ref`` *and*
    at the peak: a coordinate that gating *helps* can carry a large selectivity
    value purely because its competitors are even more negative, and it cannot
    suppress anything. ``max_rank`` is never a candidate rank -- it is the
    identity intervention.

    ``significance`` (optional) carries ``lo``/``hi``/``reject`` grids from
    ``AttributionResult.bootstrap`` + BH-FDR to annotate entries.
    """
    ti = maps._task_pos(domain)
    ki = maps._rank_pos(rank_ref)
    score_grid = maps.s_norm[ti, :, :, ki]
    sigma_grid = maps.sigma[ti, :, :, ki]
    collateral_grid = score_grid - sigma_grid  # max competitor score, by definition
    raw = maps.s_raw[ti]  # (n_blocks, n_modules, K)
    # Candidate ranks exclude max_rank: setting it is a no-op the search would
    # only waste evaluations on.
    ladder_desc = [
        r for r in sorted(int(r) for r in maps.ranks)[::-1] if r < int(maps.ranks.max())
    ]
    ladder_pos = [maps._rank_pos(r) for r in ladder_desc]
    peak_drop = raw[:, :, ladder_pos].max(axis=-1)  # (n_blocks, n_modules)

    order_value = {
        "selectivity": lambda bi, mi: sigma_grid[bi, mi],
        "magnitude": lambda bi, mi: score_grid[bi, mi],
        "anti": lambda bi, mi: -sigma_grid[bi, mi],
    }
    excluded = set(exclude)
    indexed = [
        (bi, mi, coord)
        for bi in range(len(maps.blocks))
        for mi in range(len(maps.modules))
        if (coord := (int(maps.blocks[bi]), str(maps.modules[mi]))) not in excluded
    ]
    if strategy in ("selectivity", "magnitude"):
        indexed = [
            t for t in indexed
            if score_grid[t[0], t[1]] > 0 and peak_drop[t[0], t[1]] > 0
        ]
    if strategy == "all":
        # Full coverage, no filtering: the candidate grid for a *measured*
        # single-intervention sweep, where attribution's only job is choosing
        # which ranks are worth an evaluation.
        chosen = sorted(indexed, key=lambda t: sigma_grid[t[0], t[1]], reverse=True)
        chosen = chosen[:top_k]
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(indexed)
        chosen = indexed[:top_k]
    elif strategy in order_value:
        key = order_value[strategy]
        chosen = sorted(indexed, key=lambda t: key(t[0], t[1]), reverse=True)[:top_k]
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    entries = []
    for bi, mi, coord in chosen:
        peak = float(peak_drop[bi, mi])
        picked: list[int] = []
        for q in mass_targets:
            candidate = None
            if peak > 0:
                for rank in ladder_desc:  # mildest (largest rank) first
                    if float(raw[bi, mi, maps._rank_pos(rank)]) >= q * peak:
                        candidate = rank
                        break
            picked.append(candidate if candidate is not None else min(ladder_desc))
        ranks = tuple(sorted(set(picked), reverse=True))[: max(1, ranks_per_coord)]
        entries.append(
            ShortlistEntry(
                coord=coord,
                ranks=ranks,
                score=float(score_grid[bi, mi]),
                selectivity=float(sigma_grid[bi, mi]),
                collateral=float(collateral_grid[bi, mi]),
                predicted_drop_frac={
                    int(r): float(raw[bi, mi, maps._rank_pos(r)] / peak)
                    if peak > 0
                    else float("nan")
                    for r in ranks
                },
                fluency_price_nll=(
                    {int(r): fluency_prices.price(coord, r) for r in ranks}
                    if fluency_prices is not None else None
                ),
                ci_lo=float(significance["lo"][bi, mi]) if significance else float("nan"),
                ci_hi=float(significance["hi"][bi, mi]) if significance else float("nan"),
                fdr_reject=bool(significance["reject"][bi, mi]) if significance else False,
            ),
        )
    return Shortlist(
        domain=domain,
        strategy=strategy,
        rank_ref=rank_ref,
        normalisation=maps.normalisation,
        entries=entries,
    )


def shortlists_to_json(
    shortlists: Mapping[str, Mapping[str, Shortlist]],
    *,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "capability_gated.attribution/shortlist/v1",
        **dict(meta),
        "domains": {
            domain: {strategy: sl.to_json() for strategy, sl in by_strategy.items()}
            for domain, by_strategy in shortlists.items()
        },
    }


def load_candidate_ranks(
    path: str | Path,
    domain: str,
    strategy: str = "selectivity",
) -> dict[Coord, list[int]]:
    """Read a saved shortlist back into ``beam_search``'s ``candidate_ranks``."""
    payload = json.loads(Path(path).read_text())
    entry = payload["domains"][domain][strategy]["candidate_ranks"]
    out: dict[Coord, list[int]] = {}
    for key, ranks in entry.items():
        block_str, module = key.split("_", 1)
        out[(int(block_str), module)] = [int(r) for r in ranks]
    return out


# ---------------------------------------------------------------------------
# Validation probes: predicted vs measured
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Probe:
    domain: str
    coord: Coord
    rank: int
    kind: str  # top | mid | random | anti | ladder


def select_probes(
    maps: ScoreMaps,
    *,
    domains: Sequence[str],
    rank_ref: int,
    n_top: int = 3,
    n_mid: int = 2,
    n_random: int = 2,
    n_anti: int = 1,
    ladder_coord_per_domain: bool = True,
    ladder: Sequence[int] = (),
    seed: int = 0,
) -> list[Probe]:
    """15-25 (coordinate, rank) probes spanning the predicted-effect range.

    Spanning is the point: a scatter built only from top coordinates measures
    correlation over a truncated range and flatters the instrument. ``mid``
    draws from the interquartile band of selectivity, ``random`` is seeded
    uniform, ``anti`` takes the least selective coordinates, and one coordinate
    per domain is swept over ``ladder`` so the first-order error can be shown
    growing with intervention size rather than at one operating point.
    """
    rng = np.random.default_rng(seed)
    probes: list[Probe] = []
    seen: set[tuple[str, Coord, int]] = set()

    def add(domain: str, coord: Coord, rank: int, kind: str) -> None:
        key = (domain, coord, int(rank))
        if key not in seen:
            seen.add(key)
            probes.append(Probe(domain=domain, coord=coord, rank=int(rank), kind=kind))

    for domain in domains:
        ranked = maps.top_coords(domain, rank_ref, len(maps.coord_list()))
        coords_only = [c for c, _ in ranked]
        for coord, _ in ranked[:n_top]:
            add(domain, coord, rank_ref, "top")
        quartile = len(coords_only) // 4
        mid_band = coords_only[quartile: len(coords_only) - quartile] or coords_only
        for i in rng.choice(len(mid_band), size=min(n_mid, len(mid_band)), replace=False):
            add(domain, mid_band[int(i)], rank_ref, "mid")
        for i in rng.choice(
            len(coords_only), size=min(n_random, len(coords_only)), replace=False,
        ):
            add(domain, coords_only[int(i)], rank_ref, "random")
        for coord, _ in ranked[::-1][:n_anti]:
            add(domain, coord, rank_ref, "anti")
        if ladder_coord_per_domain and ranked:
            sweep_coord = ranked[0][0]
            sweep_ranks = [int(r) for r in (ladder or maps.ranks.tolist())]
            for rank in sweep_ranks:
                if 1 <= rank < int(maps.ranks.max()):
                    add(domain, sweep_coord, rank, "ladder")
    return probes


@dataclass
class ProbeTable:
    """Predicted-vs-measured rows plus the correlation summary."""

    rows: list[dict[str, Any]]

    def correlations(self, *, noise_floor: float = 0.0) -> dict[str, dict[str, float]]:
        pred = np.array([r["pred_drop_margin"] for r in self.rows], dtype=float)
        meas_m = np.array([r["meas_drop_margin"] for r in self.rows], dtype=float)
        meas_a = np.array([r["meas_drop_acc"] for r in self.rows], dtype=float)
        keep = np.abs(pred) > noise_floor
        out: dict[str, Any] = {}
        for name, meas in (("margin", meas_m), ("accuracy", meas_a)):
            sign = (
                float((np.sign(pred[keep]) == np.sign(meas[keep])).mean())
                if keep.any()
                else float("nan")
            )
            out[name] = {
                "pearson": pearsonr(pred, meas),
                "spearman": spearmanr(pred, meas),
                "sign_agreement": sign,
                "n": int(len(pred)),
                "n_above_floor": int(keep.sum()),
            }
        rel = [
            abs(r["pred_drop_margin"] - r["meas_drop_margin"]) / abs(r["mass_total"])
            for r in self.rows
            if r.get("mass_total")
        ]
        out["median_rel_error_vs_mass"] = float(np.median(rel)) if rel else float("nan")
        return out

    def to_json(self, *, noise_floor: float = 0.0) -> dict[str, Any]:
        """Rows plus correlations, recording the floor they were computed under.

        ``sign_agreement`` and ``n_above_floor`` are meaningless without the
        threshold that produced them, and a saved file computed at the default
        floor next to a printed one computed at the measured jitter floor is two
        different numbers under one name. Pass the same floor you report.
        """
        return {
            "noise_floor": noise_floor,
            "rows": self.rows,
            "correlations": self.correlations(noise_floor=noise_floor),
        }

    def to_single_intervention_json(
        self,
        baseline: Mapping[str, MCQResult],
        max_rank: int,
    ) -> dict[str, Any]:
        """The beam search's ``"{block}_{rank}_{module}"`` schema -- **measured only**.

        Predicted margins never enter this file: its fields are literally
        ``acc,none``, and a converted prediction would be indistinguishable from
        a measurement to every downstream consumer.
        """
        out: dict[str, Any] = {
            "_note": (
                "Sparse measured subset from attribution-validation probes; "
                "NOT a full sweep. All accuracies are real evaluations on the "
                "search split (A)."
            ),
            "max_rank": {t: r.lm_eval_row() for t, r in baseline.items()},
        }
        for row in self.rows:
            key = f"{row['block']}_{row['rank']}_{row['module']}"
            out[key] = {
                "results": row["measured_results"],
                "rank": row["rank"],
                "rank_percentage": round(100 * row["rank"] / max_rank),
            }
        return out


def measure_probes(
    evaluator: MCQEvaluator,
    probes: Sequence[Probe],
    results: Mapping[str, AttributionResult],
    *,
    collateral_tasks: Sequence[str] = (),
    items: str = "correct",
    progress: bool = True,
    item_sink: dict[str, dict[str, MCQResult]] | None = None,
) -> ProbeTable:
    """Measure each probe's true effect on the identical cached batches.

    Setting the rank *is* the additive patch ``o <- o + do`` exactly, so the
    only thing under test is the first-order truncation. Margin drops are
    computed over the same item filter the prediction used; accuracy drops are
    over the full item set (accuracy is what the search will optimise).

    ``item_sink``, when given, collects each probe's per-task ``MCQResult`` under
    a ``probe_NN`` label. These are the only measurements of a *gated* model on
    split A that the notebook makes, so their per-item vectors are worth keeping:
    they cost a GPU pass to reproduce and support any paired test later.
    """
    baseline = evaluator.baseline()
    rows = []
    for i, probe in enumerate(probes):
        res = results[probe.domain]
        block_idx, module_name = probe.coord
        module_idx = int(np.nonzero(res.modules == module_name)[0][0])
        eval_items = evaluator.items[probe.domain]
        if [it.index for it in eval_items] != res.item_index.tolist():
            raise ValueError(
                f"{probe.domain}: evaluator items differ from attribution items; "
                f"predicted and measured drops would not be comparable.",
            )
        mask = res._item_mask(items)
        pred_drop = float(res.tail(probe.rank, items=items)[block_idx, module_idx])
        mass_total = float(res.tail(1, items=items)[block_idx, module_idx])

        tasks = [probe.domain, *collateral_tasks]
        measured = evaluator.evaluate_config({probe.coord: probe.rank}, tasks)
        if item_sink is not None:
            item_sink[f"probe_{i:02d}"] = dict(measured)
        base = baseline[probe.domain]
        new = measured[probe.domain]
        meas_drop_margin = float(
            base.margins[mask].mean() - new.margins[mask].mean(),
        )
        rows.append(
            {
                "domain": probe.domain,
                "block": probe.coord[0],
                "module": probe.coord[1],
                "rank": probe.rank,
                "kind": probe.kind,
                "pred_drop_margin": pred_drop,
                "meas_drop_margin": meas_drop_margin,
                "meas_drop_acc": float(base.acc - new.acc),
                "mass_total": mass_total,
                "base_acc": base.acc,
                "new_acc": new.acc,
                "collateral_drop_acc": {
                    t: float(baseline[t].acc - measured[t].acc)
                    for t in collateral_tasks
                },
                "measured_results": {t: measured[t].lm_eval_row() for t in tasks},
            },
        )
        if progress:
            print(
                f"[probe {i + 1}/{len(probes)}] {probe.domain} {probe.coord} "
                f"r={probe.rank} ({probe.kind}): pred {pred_drop:+.4f} "
                f"meas {meas_drop_margin:+.4f} margin | "
                f"acc {base.acc:.3f}->{new.acc:.3f}",
            )
    return ProbeTable(rows=rows)


# ---------------------------------------------------------------------------
# Figures (matplotlib imported inside each function so headless search runs
# never pay for pyplot -- same policy as the lazy lm-eval imports elsewhere)
# ---------------------------------------------------------------------------
def _save_or_show(fig: Any, save_stem: str | Path | None, data: Any = None) -> Any:
    """Write the figure and, beside it, exactly the series that was plotted.

    A figure whose numbers exist only inside a PDF has to be regenerated -- and
    on a GPU -- to be reused in another style or another paper. ``data`` costs
    kilobytes and removes that dependency.
    """
    if save_stem is not None:
        from .artifacts import save_figure

        save_figure(fig, save_stem, data=data)
    return fig


def plot_coordinate_heatmaps(
    grids: Mapping[str, np.ndarray],
    blocks: np.ndarray,
    modules: np.ndarray,
    *,
    cbar_label: str,
    masks: Mapping[str, np.ndarray] | None = None,
    fig_title: str | None = None,
    save_stem: str | Path | None = None,
) -> Any:
    """One (module x block) heatmap per task, shared diverging scale centred at 0.

    Signed drops need a diverging map (two hues, neutral midpoint); masked cells
    (e.g. FDR non-significant) are drawn hatched grey rather than dropped, so
    absence of evidence stays visually distinct from evidence of absence.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    with plt.rc_context(_PAPER_RC):
        n = len(grids)
        fig, axes = plt.subplots(
            n, 1, figsize=(0.38 * len(blocks) + 2.2, 1.05 * len(modules) * n + 0.8),
            dpi=200, constrained_layout=True, squeeze=False,
        )
        vmax = max(float(np.nanmax(np.abs(g))) for g in grids.values()) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        for ax, (task, grid) in zip(axes[:, 0], grids.items()):
            shown = grid.T  # (modules, blocks)
            image = ax.imshow(shown, aspect="auto", cmap="RdBu_r", norm=norm)
            if masks is not None and task in masks:
                mask = ~masks[task].T
                overlay = np.where(mask, 1.0, np.nan)
                ax.imshow(
                    overlay, aspect="auto", cmap="Greys", alpha=0.55, vmin=0, vmax=2,
                )
            ax.set_yticks(range(len(modules)), [str(m) for m in modules])
            ax.set_xticks(range(0, len(blocks), max(1, len(blocks) // 12)))
            ax.set_title(task, loc="left", fontsize=11, color=_INK)
            ax.tick_params(colors=_MUTED, length=3)
            for side in ax.spines.values():
                side.set_color(_HAIR)
        axes[-1, 0].set_xlabel("transformer block", color=_INK)
        cbar = fig.colorbar(image, ax=axes[:, 0], shrink=0.85, pad=0.01)
        cbar.set_label(cbar_label, color=_INK)
        cbar.outline.set_color(_HAIR)
        if fig_title:
            fig.suptitle(fig_title, fontsize=13, fontweight="bold", color=_INK)
    return _save_or_show(fig, save_stem, {
        "what": "one (block x module) grid per task; rows of the image are modules",
        "cbar_label": cbar_label,
        "title": fig_title,
        "blocks": blocks.tolist(),
        "modules": [str(m) for m in modules],
        "grids": {task: np.asarray(grid).tolist() for task, grid in grids.items()},
        "masks": (
            {t: np.asarray(m).tolist() for t, m in masks.items()}
            if masks is not None else None
        ),
    })


def plot_predicted_drop_curves(
    result: AttributionResult,
    coords: Sequence[Coord],
    *,
    items: str = "correct",
    title: str | None = None,
    save_stem: str | Path | None = None,
) -> Any:
    """Predicted margin drop vs rank for chosen coordinates (log2 x, reversed)."""
    import matplotlib.pyplot as plt

    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200, constrained_layout=True)
        ranks = result.ranks.astype(float)
        for pos, coord in enumerate(coords):
            curve = result.curve(coord, items=items)
            color = module_color(coord[1], pos)
            ax.plot(
                ranks, curve, color=color, lw=2, solid_capstyle="round",
                marker="o", markersize=4.5, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=1.0,
                label=f"block {coord[0]} · {coord[1]}",
            )
        ax.set_xscale("log", base=2)
        ax.set_xlim(ranks.max() * 1.3, max(ranks.min() / 1.3, 0.7))
        ax.set_xticks(ranks)
        ax.set_xticklabels([f"{int(r)}" for r in ranks], rotation=45, ha="right")
        ax.axhline(0.0, color=_HAIR, lw=0.8)
        ax.set_xlabel("active rank   (privilege decreases →)", color=_INK)
        ax.set_ylabel("predicted margin drop", color=_INK)
        ax.grid(visible=True, which="major", color="0.91", lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        if title:
            ax.set_title(title, loc="left", fontweight="bold", color=_INK, pad=12)
    return _save_or_show(fig, save_stem, {
        "what": "predicted margin drop vs active rank, one curve per coordinate",
        "task": result.task,
        "items": items,
        "ranks": ranks.tolist(),
        "curves": {
            f"{c[0]}_{c[1]}": result.curve(c, items=items).tolist() for c in coords
        },
    })


def plot_component_spectrum(
    result: AttributionResult,
    coords: Sequence[Coord],
    *,
    cumulative: bool = True,
    items: str = "correct",
    title: str | None = None,
    save_stem: str | Path | None = None,
) -> Any:
    """Where along the component index does the attributable mass live?

    Cumulative-from-the-tail view: y at component j = predicted drop if every
    component >= j were removed. Mass concentrated at high j means gating can be
    surgical; mass at low j means it cannot -- this is the mechanistic claim
    behind the whole gate, so it gets its own figure.
    """
    import matplotlib.pyplot as plt

    plotted: dict[str, list[float]] = {}
    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200, constrained_layout=True)
        for pos, coord in enumerate(coords):
            spectrum = result.spectrum(coord, cumulative=False, items=items)
            values = spectrum[::-1].cumsum()[::-1] if cumulative else spectrum
            plotted[f"{coord[0]}_{coord[1]}"] = values.tolist()
            ax.plot(
                np.arange(len(values)), values, lw=1.6,
                color=module_color(coord[1], pos),
                label=f"block {coord[0]} · {coord[1]}",
            )
        ax.axhline(0.0, color=_HAIR, lw=0.8)
        ax.set_xlabel("component index j", color=_INK)
        ax.set_ylabel(
            "tail mass  sum over j' >= j of c_j'" if cumulative else "component c_j",
            color=_INK,
        )
        ax.grid(visible=True, which="major", color="0.91", lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
        if title:
            ax.set_title(title, loc="left", fontweight="bold", color=_INK, pad=12)
    return _save_or_show(fig, save_stem, {
        "what": (
            "attributable mass along the component index; cumulative-from-tail "
            "means y at j = predicted drop if every component >= j were removed"
        ),
        "task": result.task,
        "items": items,
        "cumulative": cumulative,
        "series": plotted,
    })


def plot_score_vs_selectivity(
    maps: ScoreMaps,
    domain: str,
    rank: int,
    *,
    shortlist: Shortlist | None = None,
    save_stem: str | Path | None = None,
) -> Any:
    """Is selectivity achievable at all, or does importance imply collateral?"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=200, constrained_layout=True)
        score = maps.score(domain, rank)
        sigma = maps.selectivity(domain, rank)
        for mi, module in enumerate(maps.modules):
            ax.scatter(
                score[:, mi], sigma[:, mi], s=34,
                color=module_color(str(module), mi), alpha=0.85,
                edgecolors="white", linewidths=0.8, label=str(module),
            )
        if shortlist is not None:
            module_pos = {str(m): i for i, m in enumerate(maps.modules)}
            for entry in shortlist.entries:
                bi = int(np.nonzero(maps.blocks == entry.coord[0])[0][0])
                mi = module_pos[entry.coord[1]]
                ax.scatter(
                    score[bi, mi], sigma[bi, mi], s=110, facecolors="none",
                    edgecolors=_INK, linewidths=1.2, zorder=4,
                )
        ax.axhline(0.0, color=_HAIR, lw=0.8)
        ax.axvline(0.0, color=_HAIR, lw=0.8)
        ax.set_xlabel(f"importance  $s$({domain})", color=_INK)
        ax.set_ylabel(r"selectivity  $\sigma$", color=_INK)
        ax.grid(visible=True, which="major", color="0.91", lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        handles = [
            Line2D([], [], marker="o", ls="", color=module_color(str(m), i),
                   markeredgecolor="white", label=str(m))
            for i, m in enumerate(maps.modules)
        ]
        if shortlist is not None:
            handles.append(
                Line2D([], [], marker="o", ls="", markerfacecolor="none",
                       markeredgecolor=_INK, label="shortlisted"),
            )
        ax.legend(handles=handles, frameon=False, fontsize=9)
        ax.set_title(
            f"{domain} @ rank {rank}", loc="left", fontweight="bold",
            color=_INK, pad=12,
        )
    return _save_or_show(fig, save_stem, {
        "what": "importance s vs selectivity sigma, one point per coordinate",
        "domain": domain,
        "rank": int(rank),
        "blocks": maps.blocks.tolist(),
        "modules": [str(m) for m in maps.modules],
        "score": score.tolist(),
        "selectivity": sigma.tolist(),
        "shortlisted": (
            [[e.coord[0], e.coord[1]] for e in shortlist.entries]
            if shortlist is not None else None
        ),
    })


def plot_validation_scatter(
    table: ProbeTable,
    *,
    noise_floor: float = 0.0,
    save_stem: str | Path | None = None,
) -> Any:
    """Predicted vs measured: margin drop (left, with y=x) and accuracy drop.

    Pass the same ``noise_floor`` the analysis reports: the sign-agreement in
    the annotation is floor-dependent, and a figure quoting it at a different
    threshold than ``probes.json`` is two numbers under one name.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with plt.rc_context(_PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=200, constrained_layout=True)
        pred = np.array([r["pred_drop_margin"] for r in table.rows])
        meas_m = np.array([r["meas_drop_margin"] for r in table.rows])
        meas_a = np.array([r["meas_drop_acc"] for r in table.rows])
        corr = table.correlations(noise_floor=noise_floor)
        seen_domains: dict[str, int] = {}
        for row in table.rows:
            seen_domains.setdefault(row["domain"], len(seen_domains))
        for ax, meas, label, key in (
            (axes[0], meas_m, "measured margin drop", "margin"),
            (axes[1], meas_a, "measured accuracy drop", "accuracy"),
        ):
            for row, x, y in zip(table.rows, pred, meas):
                ax.scatter(
                    x, y, s=42,
                    color=module_color(row["module"]),
                    marker=DOMAIN_MARKERS.get(row["domain"], "D"),
                    edgecolors="white", linewidths=0.8, alpha=0.9,
                )
            if key == "margin":
                lim = max(1e-9, float(np.max(np.abs(np.r_[pred, meas]))))
                ax.plot([-lim, lim], [-lim, lim], color=_HAIR, lw=1.0, ls=(0, (4, 3)))
                ax.set_xlim(-lim * 1.1, lim * 1.1)
                ax.set_ylim(-lim * 1.1, lim * 1.1)
            ax.axhline(0.0, color=_HAIR, lw=0.6)
            ax.axvline(0.0, color=_HAIR, lw=0.6)
            ax.set_xlabel("predicted margin drop", color=_INK)
            ax.set_ylabel(label, color=_INK)
            stats = corr[key]
            ax.text(
                0.03, 0.97,
                f"Pearson {stats['pearson']:.2f} · Spearman {stats['spearman']:.2f}\n"
                f"sign agreement {stats['sign_agreement']:.2f} (n={stats['n']})",
                transform=ax.transAxes, va="top", fontsize=9, color=_MUTED,
            )
            ax.grid(visible=True, which="major", color="0.91", lw=0.7)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        modules = sorted({r["module"] for r in table.rows})
        handles = [
            Line2D([], [], marker="o", ls="", color=module_color(m),
                   markeredgecolor="white", label=m)
            for m in modules
        ] + [
            Line2D([], [], marker=DOMAIN_MARKERS.get(d, "D"), ls="", color=_MUTED,
                   markeredgecolor="white", label=d)
            for d in seen_domains
        ]
        axes[1].legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
        fig.suptitle(
            "First-order attribution vs measured rank-gating effects",
            fontsize=12.5, fontweight="bold", color=_INK,
        )
    return _save_or_show(fig, save_stem, {
        "what": "predicted vs measured drop, one point per probe",
        "noise_floor": noise_floor,
        "correlations": corr,
        "rows": table.rows,
    })


def plot_first_order_error(
    table: ProbeTable,
    *,
    save_stem: str | Path | None = None,
) -> Any:
    """Relative error of the ladder-sweep probes vs rank: how first-order decays."""
    import matplotlib.pyplot as plt

    ladder_rows = [r for r in table.rows if r["kind"] == "ladder"]
    with plt.rc_context(_PAPER_RC):
        fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=200, constrained_layout=True)
        by_key: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
        for row in ladder_rows:
            by_key.setdefault((row["domain"], row["block"], row["module"]), []).append(row)
        plotted: dict[str, dict[str, list[float]]] = {}
        for pos, (key, key_rows) in enumerate(sorted(by_key.items())):
            ordered = sorted(key_rows, key=lambda r: -r["rank"])
            ranks = [r["rank"] for r in ordered]
            err = [
                abs(r["pred_drop_margin"] - r["meas_drop_margin"])
                / max(1e-12, abs(r["mass_total"]))
                for r in ordered
            ]
            plotted[f"{key[0]}|{key[1]}_{key[2]}"] = {"rank": ranks, "rel_error": err}
            ax.plot(
                ranks, err, marker="o", lw=1.8, markersize=5,
                color=module_color(key[2], pos),
                markeredgecolor="white", markeredgewidth=0.9,
                label=f"{key[0]} · block {key[1]} {key[2]}",
            )
        ax.set_xscale("log", base=2)
        if ladder_rows:
            all_ranks = sorted({r["rank"] for r in ladder_rows}, reverse=True)
            ax.set_xlim(max(all_ranks) * 1.3, min(all_ranks) / 1.3)
            ax.set_xticks(all_ranks)
            ax.set_xticklabels([str(r) for r in all_ranks], rotation=45, ha="right")
        ax.set_xlabel("active rank   (intervention grows →)", color=_INK)
        ax.set_ylabel("|pred - meas| / coordinate mass", color=_INK)
        ax.grid(visible=True, which="major", color="0.91", lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(
            "First-order error grows with intervention size",
            loc="left", fontweight="bold", color=_INK, pad=12,
        )
    return _save_or_show(fig, save_stem, {
        "what": "|pred - meas| / coordinate mass vs rank, per ladder-swept coordinate",
        "series": plotted,
    })


def plot_headroom(
    results: Mapping[str, MCQResult],
    *,
    chance: float = 0.25,
    gate_pp: float = 5.0,
    save_stem: str | Path | None = None,
) -> Any:
    """Full-privilege accuracy per task with Wilson CIs against the chance floor."""
    import matplotlib.pyplot as plt

    with plt.rc_context(_PAPER_RC):
        tasks = list(results)
        fig, ax = plt.subplots(
            figsize=(6.6, 0.42 * len(tasks) + 1.6), dpi=200, constrained_layout=True,
        )
        y = np.arange(len(tasks))[::-1]
        accent = "#33689E"
        for yi, task in zip(y, tasks):
            res = results[task]
            lo, hi = res.wilson
            passed = (res.acc - chance) >= gate_pp / 100 and lo > chance
            ax.plot([lo, hi], [yi, yi], color=_MUTED, lw=1.6, solid_capstyle="round")
            ax.scatter(
                [res.acc], [yi], s=52, color=accent, zorder=3,
                edgecolors="white", linewidths=1.0,
            )
            ax.annotate(
                f"{res.acc:.3f}  {'PASS' if passed else 'FAIL'}",
                xy=(hi, yi), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9,
                color=_INK if passed else "#B65566",
                fontweight="bold" if not passed else "normal",
            )
        ax.axvline(chance, color=_HAIR, lw=1.0, ls=(0, (4, 3)))
        ax.annotate(
            "chance", xy=(chance, y.max() + 0.5), ha="center", fontsize=9, color=_MUTED,
        )
        ax.axvline(chance + gate_pp / 100, color=_HAIR, lw=0.8, ls=(0, (1, 2)))
        ax.set_yticks(y, tasks)
        ax.set_xlabel("accuracy at full privilege (Wilson 95% CI)", color=_INK)
        ax.set_xlim(0.0, max(0.6, max(r.wilson[1] for r in results.values()) + 0.14))
        ax.grid(visible=True, axis="x", color="0.91", lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(
            "E1 · headroom gate", loc="left", fontweight="bold", color=_INK, pad=12,
        )
    return _save_or_show(fig, save_stem, {
        "what": "full-privilege accuracy with Wilson 95% CI, per task",
        "chance": chance,
        "gate_pp": gate_pp,
        "tasks": {
            t: {
                "acc": r.acc, "n": r.n, "n_correct": r.n_correct,
                "wilson_lo": r.wilson[0], "wilson_hi": r.wilson[1],
                "mean_margin": r.mean_margin,
                "passed": bool(
                    (r.acc - chance) >= gate_pp / 100 and r.wilson[0] > chance,
                ),
            }
            for t, r in results.items()
        },
    })
