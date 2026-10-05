#!/usr/bin/env python
"""Fine-tune a capability-gated (NLPN) language model end to end, then report on it.

One invocation produces a
self-contained run directory holding everything needed to write up the result:

* ``final_model/``            -- the trained weights + tokenizer (reloadable with
                                 :func:`capability_gated.load_lowrank_checkpoint`)
* ``training_loss.{png,pdf}`` -- loss progression over training: the combined
                                 multi-rank objective, plus per-rank NLL so you can
                                 see the low ranks catch up after the warmup
* ``perplexity_vs_rank.{png,pdf}`` -- the privilege-utility frontier, with the
                                 *dense* pretrained model drawn as a reference line
* ``metrics.json`` / ``config.json`` / ``log_history.json`` -- every number behind
                                 the figures, so they can be re-plotted without
                                 re-running the GPU work

The dense reference deserves a note. It is the untouched pretrained model -- no
factorization, no SVD, no fine-tuning -- evaluated once at full capacity on the
same held-out slice. It answers "what did the capability gate cost us at full
privilege?", which the rank sweep alone cannot: every point on that sweep is
already a factorized model. Pass ``--svd-baseline`` to additionally sweep an
SVD-truncated-but-untrained model, which is the comparison in App. D of the paper
(NLPN degrades gracefully; naive SVD truncation collapses).

Example::

    uv run python scripts/finetune_and_gate.py \\
        --base-id EleutherAI/pythia-160m --max-rank 512 --max-steps 300

    uv run python scripts/finetune_and_gate.py \\
        --base-id Qwen/Qwen2.5-0.5B --max-rank 896 \\
        --dataset JeanKaddour/minipile --dataset-config "" \\
        --num-train-epochs 1 --batch-size 16 --svd-baseline
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")  # headless: this script only ever writes files

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset, DatasetDict, load_dataset  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FuncFormatter,
    LogLocator,
    NullFormatter,
    NullLocator,
)
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    set_seed,
)

from capability_gated import (  # noqa: E402
    DynamicLowRankLinear,
    LowRankTrainer,
    build_lowrank_model,
    decompose_and_replace,
    load_text_causal_lm,
    strip_non_text_towers,
    evaluate_perplexity_at_rank,
    find_max_rank,
    prepare_lm_dataset,
    provenance,
    save_lowrank_checkpoint,
    set_rank_all,
)
from capability_gated.rank_control import attach_rank_control  # noqa: E402
from capability_gated.training import geometric_rank_ladder  # noqa: E402

# `evaluate_perplexity_at_rank` clamps loss at 20 before exponentiating, so a
# fully-collapsed rank reports this value rather than inf. Worth flagging on the
# figure, since a clamped point is a floor artefact and not a measurement.
PPL_CLAMP = math.exp(20.0)

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

# Paper style, matching the figures in the original repo.
PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
}
INK, MUTED, HAIR = "0.15", "0.45", "0.80"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """The full CLI. Split out so a wrapper script can extend it."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    model = p.add_argument_group("model")
    model.add_argument("--base-id", default="EleutherAI/pythia-160m", help="Base HF model id.")
    model.add_argument(
        "--max-rank",
        default="512",
        help="Maximum rank r_max, or 'auto' for the largest rank the MLPs allow.",
    )
    model.add_argument("--target-substring", default="mlp", help="Name filter for layers to factorize.")
    model.add_argument(
        "--train-ranks",
        type=int,
        nargs="+",
        default=None,
        help="Rank ladder to train/evaluate. Default: geometric [r, r/2, r/4, ..., 1].",
    )
    model.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    model.add_argument("--device-map", default=None, help="Passed to from_pretrained; default 'auto' on GPU.")
    model.add_argument("--cache-dir", default=None, help="HF cache dir (default: ~/.cache/huggingface).")

    data = p.add_argument_group("data")
    data.add_argument("--dataset", default="wikitext")
    data.add_argument(
        "--dataset-config",
        default="wikitext-2-raw-v1",
        help="Dataset config name; pass an empty string for datasets that have none.",
    )
    data.add_argument("--block-size", type=int, default=256)
    data.add_argument("--num-workers", type=int, default=4)
    data.add_argument(
        "--ppl-samples",
        type=int,
        default=256,
        help="Held-out blocks used for every perplexity measurement.",
    )
    data.add_argument(
        "--max-train-docs",
        type=int,
        default=None,
        help="Cap the raw training documents before tokenization. For corpora far "
        "larger than the token budget (e.g. fineweb-edu sample-10BT), tokenizing "
        "the full split wastes hours of CPU and tens of GB of cache.",
    )
    data.add_argument(
        "--holdout-docs",
        type=int,
        default=0,
        help="Carve this many documents off the end of the (capped) train split as "
        "a validation split, for datasets that ship without one. Overrides any "
        "existing validation split.",
    )

    train = p.add_argument_group("training")
    train.add_argument("--max-steps", type=int, default=300, help="-1 to train by epochs instead.")
    train.add_argument("--num-train-epochs", type=float, default=1.0)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--eval-batch-size", type=int, default=8)
    train.add_argument("--grad-accum", type=int, default=2)
    train.add_argument("--lr", type=float, default=2e-5)
    train.add_argument(
        "--sampling-strategy",
        choices=[
            "uniform", "rank_weighted", "curriculum", "epoch_curriculum",
            "faster_curriculum",
        ],
        default="uniform",
    )
    train.add_argument(
        "--variant-mode",
        choices=["global", "mixed", "hybrid"],
        default="global",
        help=(
            "Shape of the lower-privilege pass: the original global rank, a sparse "
            "per-coordinate configuration, or a probabilistic mixture of both."
        ),
    )
    train.add_argument(
        "--mixed-probability",
        type=float,
        default=0.5,
        help="Probability of a sparse variant when --variant-mode=hybrid.",
    )
    train.add_argument(
        "--mixed-coordinate-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Possible numbers of gated projections in a sparse variant.",
    )
    train.add_argument(
        "--mixed-coordinate-weights",
        type=float,
        nargs="+",
        default=[0.7, 0.2, 0.1],
        help="Sampling weights corresponding to --mixed-coordinate-counts.",
    )
    train.add_argument("--warmup-proportion", type=float, default=0.07)
    train.add_argument("--logging-steps", type=int, default=25)
    train.add_argument("--eval-steps", type=int, default=0, help="0 disables periodic eval.")
    train.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="Write a resumable checkpoint every N steps (0 = only the final model). "
        "Worth setting on multi-hour runs; resume with --resume-from-checkpoint.",
    )
    train.add_argument("--save-total-limit", type=int, default=2)
    train.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a checkpoint dir under <run>/checkpoints to continue from, or "
        "'latest' to pick the highest-numbered one there. Pair with --run-name "
        "<existing run dir> so the resumed run writes into the same directory.",
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--report-to", default="none", help="e.g. 'wandb' (needs --group logging).")

    base = p.add_argument_group("baselines")
    base.add_argument(
        "--skip-dense-baseline",
        action="store_true",
        help="Skip the untouched pretrained-model reference measurement.",
    )
    base.add_argument(
        "--svd-baseline",
        action="store_true",
        help="Also sweep an SVD-truncated but untrained model (paper App. D comparison).",
    )

    out = p.add_argument_group("output")
    out.add_argument("--output-root", default="runs", help="Run directories are created underneath.")
    out.add_argument("--run-name", default=None, help="Override the generated run-directory name.")

    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def reclaim() -> None:
    """Reclaim GPU memory between the models this script loads in turn.

    Callers must ``del`` their own references first -- deleting an argument inside
    a helper only drops that helper's reference, leaving the caller's binding alive
    and the memory held.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def jsonable(obj: Any) -> Any:
    """numpy/torch scalars are not JSON-serializable; unwrap them."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n")


def save_figure(fig: plt.Figure, stem: Path, data: Any = None) -> None:
    """Write both a PNG (for a quick look) and a PDF (for the paper).

    ``data`` is the exact series plotted; when given it is written beside the
    image as ``<stem>.data.json`` so the figure can be redrawn in another style
    without re-deriving it from the 50 MB raw log.
    """
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    extra = ""
    if data is not None:
        write_json(stem.with_suffix(".data.json"), data)
        extra = f" / {stem.with_suffix('.data.json').name}"
    log(f"wrote {stem.with_suffix('.png').name} / "
        f"{stem.with_suffix('.pdf').name}{extra}")


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average that keeps the input length.

    Non-finite entries are skipped rather than propagated: a single inf/NaN in a
    cumulative sum would poison every later point, wiping out the rest of the curve.
    """
    if values.size == 0:
        return values
    window = max(1, min(int(window), values.size))

    finite = np.isfinite(values)
    vals = np.where(finite, values, 0.0).astype(float)
    sums = np.cumsum(np.insert(vals, 0, 0.0))
    counts = np.cumsum(np.insert(finite.astype(float), 0, 0.0))

    out = np.full(values.size, np.nan, dtype=float)
    for i in range(values.size):
        lo = max(0, i + 1 - window)
        n = counts[i + 1] - counts[lo]
        if n > 0:
            out[i] = (sums[i + 1] - sums[lo]) / n
    return out


def fmt_ppl(v: float) -> str:
    """A decimal is informative at ppl ~15, noise at ppl ~3000."""
    if not math.isfinite(v):
        return "n/a"
    return f"{v:,.1f}" if v < 100 else f"{v:,.0f}"


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def load_tokenizer(base_id: str, cache_dir: str | None):
    tokenizer = AutoTokenizer.from_pretrained(base_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _get_text_field_from_features(stream: Any) -> str:
    """Detect the text column of a streaming dataset (mirrors prepare_lm_dataset).

    A streaming ``IterableDataset`` exposes ``.features``/``.column_names`` before
    iteration for parquet-backed corpora like fineweb-edu; fall back to peeking at
    the first row if features are absent.
    """
    names = list(getattr(stream, "column_names", None) or [])
    if not names and getattr(stream, "features", None):
        names = list(stream.features)
    for candidate in ("text", "content", "raw", "data"):
        if candidate in names:
            return candidate
    if names:
        return names[0]
    first = next(iter(stream))
    for candidate in ("text", "content", "raw", "data"):
        if candidate in first:
            return candidate
    return next(iter(first))


def max_admissible_rank(
    base_id: str, target_substring: str, cache_dir: str | None
) -> int:
    """Largest rank the target layers admit, without downloading any weights.

    Instantiating the architecture on the meta device costs milliseconds and no
    memory; :func:`find_max_rank` only reads ``in_features``/``out_features``. This
    has to be independent of the dense measurement, because ``--skip-dense-baseline``
    would otherwise disable the ceiling check entirely -- and exceeding the ceiling
    leaves the tail of every factor at random init while still calling it "full rank".
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(base_id, cache_dir=cache_dir)
    with torch.device("meta"):
        probe = AutoModelForCausalLM.from_config(config)
    # The ceiling must be computed on the same module tree adaptation will build.
    # Without this the probe includes the image/audio encoders, whose narrower
    # MLPs win the minimum: gemma-4-E2B reports 768 instead of 1536, and the
    # ceiling check then silently passes a ladder half the intended height.
    strip_non_text_towers(probe)
    return find_max_rank(probe, target_substring)


def measure_dense_baseline(
    base_id: str,
    tokenizer: Any,
    ppl_dataset: Any,
    *,
    dtype: torch.dtype,
    device_map: str | None,
    cache_dir: str | None,
    batch_size: int,
) -> dict[str, float]:
    """Perplexity of the untouched pretrained model, measured once at full capacity.

    No factorization and no SVD -- this is the model as shipped, measured once at
    full capacity. ``rank=None`` is what makes that possible: it tells
    :func:`evaluate_perplexity_at_rank` to leave the model alone rather than call
    ``_set_rank_all``, which a dense model does not have.
    """
    log(f"loading dense reference model {base_id!r}")
    # The dense reference is the denominator of every perplexity ratio, so it has
    # to be the same text-only model the gated arms are, not the multimodal
    # wrapper.
    model, dropped = load_text_causal_lm(
        base_id,
        pad_token_id=tokenizer.pad_token_id,
        dtype=dtype,
        device_map=device_map,
        cache_dir=cache_dir,
    )
    if dropped:
        log(f"dropped non-text towers: {', '.join(dropped)}")
    model.config.use_cache = False
    model.eval()

    ppl, loss = evaluate_perplexity_at_rank(
        model, tokenizer, ppl_dataset, rank=None, batch_size=batch_size
    )
    log(f"dense reference: perplexity {ppl:.3f} (loss {loss:.4f})")

    del model
    reclaim()
    return {"perplexity": float(ppl), "loss": float(loss)}


def sweep_perplexity(
    model: Any,
    tokenizer: Any,
    ppl_dataset: Any,
    ranks: list[int],
    batch_size: int,
    label: str,
) -> dict[int, dict[str, float]]:
    """Perplexity at each rank on the ladder, restoring full rank afterwards."""
    out: dict[int, dict[str, float]] = {}
    for rank in sorted(set(ranks), reverse=True):
        ppl, loss = evaluate_perplexity_at_rank(
            model, tokenizer, ppl_dataset, rank=rank, batch_size=batch_size
        )
        out[int(rank)] = {"perplexity": float(ppl), "loss": float(loss)}
        log(f"{label} rank {rank:>5d} -> perplexity {ppl:10.3f} (loss {loss:.4f})")
    set_rank_all(model, model.max_rank)
    return out


def measure_svd_baseline(
    base_id: str,
    tokenizer: Any,
    ppl_dataset: Any,
    ranks: list[int],
    *,
    max_rank: int,
    dtype: torch.dtype,
    device_map: str | None,
    cache_dir: str | None,
    batch_size: int,
    target_substring: str,
) -> dict[int, dict[str, float]]:
    """Sweep an SVD-factorized but *untrained* model -- the paper's App. D comparison."""
    log("loading SVD-only baseline (factorized, never fine-tuned)")
    model, _ = load_text_causal_lm(
        base_id,
        pad_token_id=tokenizer.pad_token_id,
        dtype=dtype,
        device_map=device_map,
        cache_dir=cache_dir,
    )
    model.config.use_cache = False
    decompose_and_replace(model, max_rank=max_rank, target_substring=target_substring)
    model.max_rank = max_rank
    attach_rank_control(model)
    model.eval()

    results = sweep_perplexity(model, tokenizer, ppl_dataset, ranks, batch_size, "svd  ")
    del model
    reclaim()
    return results


# ---------------------------------------------------------------------------
# Parsing the trainer's log history
# ---------------------------------------------------------------------------
_EVAL_RANK_RE = re.compile(r"^eval_rank_(\d+)/loss$")


def parse_log_history(
    log_history: list[dict[str, Any]], max_rank: int
) -> dict[str, Any]:
    """Split ``trainer.state.log_history`` into the series the figures need.

    ``LowRankTrainer.compute_loss`` logs on every micro-batch, so the objective and
    the per-rank NLLs are dense. Those records are told apart from the Trainer's own
    periodic optimizer logs by the keys they carry: ours have ``nll_anchor`` (warmup)
    or ``r_anchor`` (joint phase), the Trainer's have ``learning_rate``.
    """
    objective: list[tuple[int, float]] = []
    warmup_steps: list[int] = []
    per_rank: dict[int, list[tuple[int, float]]] = {}
    mixed_variant: list[tuple[int, float]] = []
    optimizer: list[tuple[int, float, float]] = []  # step, lr, grad_norm
    evals: list[dict[str, Any]] = []

    for record in log_history:
        step = int(record.get("step", 0))

        # --- our per-step objective / NLL records ---
        is_warmup = "nll_anchor" in record
        is_joint = "r_anchor" in record
        if is_warmup or is_joint:
            if "loss" in record:
                objective.append((step, float(record["loss"])))
            if is_warmup:
                warmup_steps.append(step)
                per_rank.setdefault(int(max_rank), []).append(
                    (step, float(record["nll_anchor"]))
                )
            else:
                for role in ("r_anchor", "r_variant"):
                    if role not in record:
                        continue
                    rank = int(record[role])
                    key = f"nll_rank_{rank}"
                    if key in record:
                        per_rank.setdefault(rank, []).append((step, float(record[key])))
                if record.get("variant_is_mixed") and "nll_variant" in record:
                    mixed_variant.append((step, float(record["nll_variant"])))
            continue

        # --- the Trainer's own optimizer logs ---
        if "learning_rate" in record:
            optimizer.append(
                (step, float(record["learning_rate"]), float(record.get("grad_norm", math.nan)))
            )
            continue

        # --- multi-rank evaluations ---
        rank_losses = {
            int(m.group(1)): float(v)
            for k, v in record.items()
            if (m := _EVAL_RANK_RE.match(k))
        }
        if rank_losses:
            evals.append({"step": step, "loss_by_rank": rank_losses})

    return {
        "objective": objective,
        "warmup_last_step": max(warmup_steps) if warmup_steps else None,
        "per_rank_nll": {r: sorted(v) for r, v in sorted(per_rank.items(), reverse=True)},
        "mixed_variant_nll": mixed_variant,
        "optimizer": optimizer,
        "evals": evals,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="0.91", lw=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(HAIR)
    ax.tick_params(colors=MUTED, length=3, width=0.8)


def plot_training_loss(series: dict[str, Any], model_name: str, stem: Path) -> None:
    """Two panels: the combined objective, and the NLL of each rank on the ladder."""
    objective = series["objective"]
    per_rank = series["per_rank_nll"]
    mixed_variant = series.get("mixed_variant_nll", [])
    if not objective and not per_rank and not mixed_variant:
        log("no training records to plot; skipping the loss figure")
        return

    plt.rcParams.update(PLOT_STYLE)
    fig, (ax_obj, ax_rank) = plt.subplots(
        1, 2, figsize=(12.5, 4.4), dpi=200, constrained_layout=True
    )

    accent = plt.get_cmap("coolwarm")(0.08)
    warmup_end = series["warmup_last_step"]

    # --- Panel A: the multi-rank objective ---
    if objective:
        steps = np.array([s for s, _ in objective], dtype=float)
        vals = np.array([v for _, v in objective], dtype=float)
        window = max(1, len(vals) // 40)
        ax_obj.plot(steps, vals, color=accent, lw=0.7, alpha=0.28, zorder=2)
        ax_obj.plot(
            steps, rolling_mean(vals, window), color=accent, lw=2,
            solid_capstyle="round", zorder=3,
            label=f"moving average ({window} steps)" if window > 1 else "objective",
        )
        ax_obj.legend(frameon=False, fontsize=10, labelcolor=MUTED, loc="upper right")

    if warmup_end is not None and objective:
        ax_obj.axvline(warmup_end, color=HAIR, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax_obj.annotate(
            "anchor-only warmup ends",
            xy=(warmup_end, 1.0), xycoords=("data", "axes fraction"),
            xytext=(5, -12), textcoords="offset points",
            fontsize=9.5, color=MUTED, ha="left", va="top",
        )

    ax_obj.set_title(
        "Multi-rank objective", fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=10
    )
    ax_obj.set_xlabel("Training step", fontsize=11.5, color=INK, labelpad=7)
    ax_obj.set_ylabel(r"$\sum_r (e^{-s_r}+\epsilon)\,\mathrm{nll}_r + s_r$", fontsize=11.5, color=INK, labelpad=7)
    _style_axes(ax_obj)

    # --- Panel B: NLL per rank (ordered -> one hue, light = low rank) ---
    ranks = sorted(per_rank, reverse=True)
    ramp = plt.get_cmap("Blues")
    shades = np.linspace(0.95, 0.35, num=max(len(ranks), 1))

    for shade, rank in zip(shades, ranks, strict=False):
        points = per_rank[rank]
        if not points:
            continue
        steps = np.array([s for s, _ in points], dtype=float)
        vals = np.array([v for _, v in points], dtype=float)
        colour = ramp(float(shade))
        window = max(1, len(vals) // 20)
        ax_rank.plot(steps, vals, color=colour, lw=0.6, alpha=0.22, zorder=2)
        ax_rank.plot(
            steps, rolling_mean(vals, window), color=colour, lw=1.8,
            solid_capstyle="round", zorder=3, label=f"rank {rank}",
        )

    if mixed_variant:
        steps = np.array([s for s, _ in mixed_variant], dtype=float)
        vals = np.array([v for _, v in mixed_variant], dtype=float)
        window = max(1, len(vals) // 20)
        ax_rank.plot(steps, vals, color="0.35", lw=0.6, alpha=0.18, zorder=2)
        ax_rank.plot(
            steps, rolling_mean(vals, window), color="0.25", lw=1.8,
            ls=(0, (4, 2)), solid_capstyle="round", zorder=3,
            label="sparse mixed variant",
        )

    if warmup_end is not None:
        ax_rank.axvline(warmup_end, color=HAIR, lw=0.8, ls=(0, (4, 3)), zorder=1)

    ax_rank.set_title(
        "Next-token NLL by active rank", fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=10
    )
    ax_rank.set_xlabel("Training step", fontsize=11.5, color=INK, labelpad=7)
    ax_rank.set_ylabel("NLL (nats)", fontsize=11.5, color=INK, labelpad=7)
    if ranks or mixed_variant:
        ax_rank.legend(
            frameon=False, fontsize=9.5, labelcolor=MUTED, ncol=2 if len(ranks) > 5 else 1,
            loc="upper right", handlelength=1.6, columnspacing=1.2,
        )
    _style_axes(ax_rank)

    fig.suptitle(
        f"{model_name}: training a capability-gated model",
        fontsize=14, fontweight="bold", color=INK, x=0.0, ha="left",
    )
    save_figure(fig, stem, {
        "what": "left: the multi-rank objective per step; right: per-rank NLL",
        "model": model_name,
        "warmup_last_step": series["warmup_last_step"],
        "objective": objective,
        "per_rank_nll": per_rank,
        "mixed_variant_nll": mixed_variant,
        "evals": series["evals"],
    })


def plot_perplexity_vs_rank(
    trained: dict[int, dict[str, float]],
    dense: dict[str, float] | None,
    svd: dict[int, dict[str, float]] | None,
    model_name: str,
    stem: Path,
) -> None:
    """The privilege-utility frontier, with the dense pretrained model as reference."""
    if not trained:
        log("no perplexity results to plot; skipping the frontier figure")
        return

    plt.rcParams.update(PLOT_STYLE)

    xs = np.array(sorted(trained), dtype=float)
    ys = np.array([trained[int(r)]["perplexity"] for r in xs], dtype=float)
    full_ppl, min_ppl = ys[-1], ys[0]
    ratio = min_ppl / full_ppl
    ratio_s = f"{ratio:,.1f}" if ratio < 10 else f"{ratio:,.0f}"

    accent = plt.get_cmap("coolwarm")(0.08)
    warm = plt.get_cmap("coolwarm")(0.92)

    fig, ax = plt.subplots(figsize=(7.4, 4.5), dpi=200, constrained_layout=True)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    all_y = [ys]
    if dense is not None:
        all_y.append(np.array([dense["perplexity"]]))
    if svd:
        all_y.append(np.array([v["perplexity"] for v in svd.values()]))
    lo = float(min(a.min() for a in all_y))
    hi = float(max(a.max() for a in all_y))

    ax.set_xlim(xs.max() * 1.4, xs.min() / 1.4)  # reversed: full privilege -> none
    ax.set_ylim(lo / 1.6, hi * 1.9)

    # Reference: the untouched pretrained model, one measurement at full capacity.
    if dense is not None:
        ax.axhline(dense["perplexity"], color=HAIR, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.annotate(
            f"dense pretrained model · {fmt_ppl(dense['perplexity'])} ppl",
            xy=(0.985, dense["perplexity"]), xycoords=("axes fraction", "data"),
            xytext=(0, 6), textcoords="offset points",
            ha="right", fontsize=9.5, color=MUTED,
        )

    # Optional: SVD-truncated but untrained, swept over the same ladder.
    if svd:
        sx = np.array(sorted(svd), dtype=float)
        sy = np.array([svd[int(r)]["perplexity"] for r in sx], dtype=float)
        ax.plot(
            sx, sy, color=warm, lw=1.8, ls=(0, (5, 2)), marker="s", markersize=4.5,
            markerfacecolor=warm, markeredgecolor="white", markeredgewidth=1.0,
            zorder=3, label="SVD only (untrained)",
        )

    ax.fill_between(xs, ys, ax.get_ylim()[0], color=accent, alpha=0.15, lw=0, zorder=2)
    ax.plot(
        xs, ys, color=accent, lw=2, solid_capstyle="round", solid_joinstyle="round",
        marker="o", markersize=6, markerfacecolor=accent,
        markeredgecolor="white", markeredgewidth=1.4, zorder=4,
        label="NLPN (fine-tuned)",
    )
    if svd:
        ax.legend(frameon=False, fontsize=10, labelcolor=MUTED, loc="upper left")

    ax.annotate(
        fmt_ppl(min_ppl), xy=(xs[0], min_ppl), textcoords="offset points",
        xytext=(-4, 11), ha="right", fontsize=10.5, color=INK, fontweight="bold",
    )
    ax.annotate(
        fmt_ppl(full_ppl), xy=(xs[-1], full_ppl), textcoords="offset points",
        xytext=(8, -14), ha="left", fontsize=10.5, color=MUTED,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(r):d}" for r in xs])
    ax.xaxis.set_minor_locator(NullLocator())
    # 1/2/5 per decade reads well over 2-3 decades; past that only the decades fit.
    decades = math.log10(max(hi * 1.9, 1e-12) / max(lo / 1.6, 1e-12))
    tick_subs = (1.0,) if decades > 3 else (1.0, 2.0, 5.0)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=tick_subs, numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    _style_axes(ax)

    ax.set_xlabel("Active rank   (privilege decreases →)", fontsize=12, color=INK, labelpad=8)
    ax.set_ylabel("Perplexity", fontsize=12, color=INK, labelpad=8)
    ax.set_title(
        f"{model_name}: perplexity vs. global rank",
        fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=20,
    )

    subtitle = (
        f"rank {int(xs[-1])} → {int(xs[0])}:  {fmt_ppl(full_ppl)} → {fmt_ppl(min_ppl)} ppl "
        f"({ratio_s}× worse)"
    )
    if dense is not None and dense["perplexity"] > 0:
        # Deliberately not phrased as "the gate costs X": the fine-tuned model is
        # compared against a pretrained one, so this ratio also contains the
        # in-domain fine-tuning gain. See the note in metrics.json.
        gate_ratio = full_ppl / dense["perplexity"]
        subtitle += f"   ·   full rank vs. dense: {gate_ratio:,.2f}×"
    ax.text(
        0.0, 1.02, subtitle, transform=ax.transAxes,
        fontsize=10.5, color=MUTED, va="bottom",
    )

    if float(ys.max()) >= PPL_CLAMP * 0.999:
        ax.text(
            1.0, -0.19, "note: perplexity is clamped at e²⁰; the top point is a floor artefact",
            transform=ax.transAxes, fontsize=9, color=MUTED, ha="right", va="top",
        )

    save_figure(fig, stem, {
        "what": "privilege-utility frontier: held-out perplexity vs active rank",
        "model": model_name,
        "ppl_clamp": PPL_CLAMP,
        "rank": [int(r) for r in xs],
        "nlpn_perplexity": ys.tolist(),
        "nlpn": {int(r): trained[int(r)] for r in xs},
        "svd": ({int(r): svd[int(r)] for r in sorted(svd)} if svd else None),
        "dense": dense,
    })


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
class Reporter:
    """No-op experiment-tracking hooks.

    :func:`run` calls these at the points where something worth recording exists.
    The default does nothing, so this script stays dependency-free; an
    experiment-tracking entry point can subclass it, and the pipeline itself is
    still written once, so the two entry points cannot drift.
    """

    def start(self, args: argparse.Namespace, resolved: dict, run_dir: Path) -> None:
        """Called once the configuration is settled, before the model is built."""

    def measurements(self, metrics: dict) -> None:
        """Called with the finished metrics payload (baselines + rank sweeps)."""

    def figures(self, figures: dict[str, Path]) -> None:
        """Called with ``{name: png_path}`` once the figures are on disk."""

    def finish(self, *, failed: bool = False) -> None:
        """Called once at the end. Callers may also fire it from an error path."""


def resolve_run_dir(args: argparse.Namespace, ranks: list[int]) -> Path:
    if args.run_name:
        return Path(args.output_root) / args.run_name
    model_name = args.base_id.rsplit("/", maxsplit=1)[-1]
    dataset_name = args.dataset.rsplit("/", maxsplit=1)[-1]
    ladder = "_".join(str(r) for r in ranks)
    stamp = datetime.now(tz=timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    return Path(args.output_root) / f"{model_name}_{dataset_name}_{ladder}_{stamp}"


def main() -> None:
    run(parse_args())


def run(  # noqa: PLR0915 - a linear experiment driver reads better in one piece
    args: argparse.Namespace, reporter: Reporter | None = None
) -> Path:
    """Run the whole pipeline and return the run directory."""
    reporter = reporter or Reporter()
    set_seed(args.seed)

    dtype = DTYPES[args.dtype]
    device_map = args.device_map
    if device_map is None:
        device_map = "auto" if torch.cuda.is_available() else None
    dataset_config = args.dataset_config or None

    log(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'} | dtype: {args.dtype}")

    if args.ppl_samples < 1:
        raise SystemExit("--ppl-samples must be >= 1; an empty slice reports perplexity 1.0.")

    # ---- rank ceiling: resolved from the architecture alone, so --skip-dense-baseline
    # cannot disable the check ----
    max_possible_rank = max_admissible_rank(
        args.base_id, args.target_substring, args.cache_dir
    )
    if str(args.max_rank).lower() == "auto":
        max_rank = max_possible_rank
        log(f"--max-rank auto -> {max_rank}")
    else:
        max_rank = int(args.max_rank)
    if max_rank > max_possible_rank:
        raise SystemExit(
            f"--max-rank {max_rank} exceeds {max_possible_rank}, the largest rank these MLPs "
            f"admit. SVD can only fill the first {max_possible_rank} components, so the rest "
            f"would stay at random init and 'full rank' would not reproduce the base model."
        )
    if max_rank < 1:
        raise SystemExit(f"--max-rank must be >= 1; got {max_rank}.")

    train_ranks = (
        sorted(set(args.train_ranks), reverse=True)
        if args.train_ranks
        else geometric_rank_ladder(max_rank)
    )
    if max_rank not in train_ranks:
        raise SystemExit(
            f"--train-ranks must include the anchor rank {max_rank}; got {train_ranks}."
        )
    out_of_range = [r for r in train_ranks if not 1 <= r <= max_rank]
    if out_of_range:
        # set_rank clamps silently, so these would appear on the frontier as duplicate
        # points drawn at a rank that was never actually evaluated.
        raise SystemExit(
            f"--train-ranks entries outside [1, {max_rank}]: {out_of_range}."
        )

    # ---- data (shared by every measurement below, so comparisons are like-for-like) ----
    tokenizer = load_tokenizer(args.base_id, args.cache_dir)
    dataset_source: str | DatasetDict = args.dataset
    if args.max_train_docs or args.holdout_docs:
        # Take only the documents we need by STREAMING, instead of downloading and
        # generating the whole split (e.g. fineweb-edu sample-10BT is 28.5GB / 9.67M
        # docs -- materialising all of it to use a slice is slow, and writing the
        # full arrow cache to a pod's network volume is fragile). Streaming reads
        # just the leading rows; selection stays positional, so the slice is
        # reproducible from the flags alone.
        want = (args.max_train_docs or 0) + args.holdout_docs
        if want <= 0:
            raise SystemExit("--holdout-docs/--max-train-docs must select >=1 doc.")
        if args.holdout_docs >= want:
            raise SystemExit(
                f"--holdout-docs {args.holdout_docs} leaves no training documents."
            )
        stream = load_dataset(
            args.dataset, dataset_config, split="train",
            streaming=True, cache_dir=args.cache_dir,
        )
        col = _get_text_field_from_features(stream)
        # Write streamed rows straight to sharded JSONL on local disk, then load
        # them back with load_dataset("json"). We deliberately DO NOT build one
        # Dataset.from_list: a single in-memory Arrow table whose string column
        # exceeds ~2GB overflows pyarrow's 32-bit offsets during .map fingerprint
        # (fine at smoke scale, fatal at 1.4M docs). File-backed shards are
        # memory-mapped and fingerprinted by path, sidestepping that entirely.
        slice_dir = Path(tempfile.mkdtemp(prefix="cgd_slice_"))
        holdout_start = want - args.holdout_docs
        shard_size = 100_000
        train_shards: list[str] = []
        n_train = n_hold = 0
        tf = None
        hf = open(slice_dir / "validation.jsonl", "w", encoding="utf-8") \
            if args.holdout_docs else None
        try:
            for i, row in enumerate(itertools.islice(stream, want)):
                line = json.dumps({col: row[col]}, ensure_ascii=False) + "\n"
                if hf is not None and i >= holdout_start:
                    hf.write(line); n_hold += 1
                else:
                    if n_train % shard_size == 0:
                        if tf is not None:
                            tf.close()
                        p = slice_dir / f"train-{len(train_shards):04d}.jsonl"
                        train_shards.append(str(p))
                        tf = open(p, "w", encoding="utf-8")
                    tf.write(line); n_train += 1
        finally:
            if tf is not None:
                tf.close()
            if hf is not None:
                hf.close()
        if n_train + n_hold < want:
            log(f"warning: stream yielded only {n_train + n_hold:,} of {want:,} docs")
        if args.holdout_docs and n_hold == 0:
            raise SystemExit("stream ended before the holdout slice; lower --holdout-docs.")
        data_files = {"train": train_shards}
        if args.holdout_docs:
            data_files["validation"] = [str(slice_dir / "validation.jsonl")]
        dataset_source = load_dataset(
            "json", data_files=data_files, cache_dir=args.cache_dir,
        )
        log(
            f"streamed document slice: train={n_train:,}"
            + (f" holdout={n_hold:,}" if args.holdout_docs else "")
            + f" -> {len(train_shards)} shard(s) on local disk"
        )
    lm_dataset = prepare_lm_dataset(
        dataset_source,
        tokenizer,
        block_size=args.block_size,
        num_workers=args.num_workers,
        dataset_config=dataset_config,
        cache_dir=args.cache_dir,
    )
    train_dataset = lm_dataset["train"]
    eval_dataset = None
    for split in ("validation", "test"):
        if split in lm_dataset:
            eval_dataset = lm_dataset[split]
            break
    if eval_dataset is None:
        raise SystemExit(
            f"Dataset {args.dataset!r} has no validation/test split to measure perplexity on."
        )
    ppl_dataset = eval_dataset.select(range(min(args.ppl_samples, len(eval_dataset))))
    log(f"train blocks: {len(train_dataset):,} | eval blocks: {len(eval_dataset):,} "
        f"| perplexity blocks: {len(ppl_dataset):,}")

    # ---- dense reference: the untouched pretrained model, one measurement ----
    dense_metrics: dict[str, float] | None = None
    if not args.skip_dense_baseline:
        dense_metrics = measure_dense_baseline(
            args.base_id, tokenizer, ppl_dataset,
            dtype=dtype, device_map=device_map, cache_dir=args.cache_dir,
            batch_size=args.eval_batch_size,
        )

    run_dir = resolve_run_dir(args, train_ranks)
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"run directory: {run_dir}")
    resolved = {
        "max_rank": max_rank,
        "max_possible_rank": max_possible_rank,
        "train_ranks": train_ranks,
        "dtype": args.dtype,
        "device_map": device_map,
        "dataset_config": dataset_config,
        "cuda": torch.cuda.is_available(),
        "train_blocks": len(train_dataset),
        "eval_blocks": len(eval_dataset),
        "perplexity_blocks": len(ppl_dataset),
        # Several settings reach a run only through the CLI; reconstructing the
        # command from the flags afterwards is guesswork, so keep it verbatim.
        "argv": list(sys.argv),
        "cwd": str(Path.cwd()),
        "provenance": provenance({"run_name": args.run_name}),
    }
    write_json(run_dir / "config.json", {**vars(args), "resolved": resolved})
    reporter.start(args, resolved, run_dir)

    # ---- build + train the capability-gated model ----
    model, _ = build_lowrank_model(
        args.base_id,
        max_rank=max_rank,
        target_substring=args.target_substring,
        train_ranks=train_ranks,
        dtype=dtype,
        device_map=device_map,
        cache_dir=args.cache_dir,
    )
    n_gated = sum(isinstance(m, DynamicLowRankLinear) for m in model.modules())
    log(f"{n_gated} DynamicLowRankLinear layers installed; train_ranks={model.train_ranks}")

    use_bf16 = dtype is torch.bfloat16 and torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        run_name=args.base_id.rsplit("/", maxsplit=1)[-1],
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.02,
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps" if args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps if args.save_steps > 0 else None,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        bf16=use_bf16,
        bf16_full_eval=use_bf16,
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
    )

    trainer = LowRankTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.eval_steps > 0 else None,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        sampling_strategy=args.sampling_strategy,
        variant_mode=args.variant_mode,
        mixed_probability=args.mixed_probability,
        mixed_coordinate_counts=args.mixed_coordinate_counts,
        mixed_coordinate_weights=args.mixed_coordinate_weights,
        warmup_proportion=args.warmup_proportion,
    )

    log("training ...")
    # `True` makes the Trainer pick the newest checkpoint in its own output_dir; a
    # string is taken as a literal path, so "latest" has to be translated here.
    resume: str | bool | None = args.resume_from_checkpoint
    if isinstance(resume, str) and resume.lower() in {"latest", "auto"}:
        resume = True
    if resume:
        log(f"resuming from {resume if resume is not True else 'the latest checkpoint'}")
    train_output = trainer.train(resume_from_checkpoint=resume)
    log(f"training finished: {train_output.metrics}")

    # ---- weights ----
    final_dir = run_dir / "final_model"
    # Not trainer.save_model(): that writes the "original" checkpoint layout, whose
    # key names do not always match the live module tree (see save_lowrank_checkpoint).
    save_lowrank_checkpoint(model, str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log(f"saved weights to {final_dir}")

    write_json(run_dir / "log_history.json", trainer.state.log_history)
    series = parse_log_history(trainer.state.log_history, max_rank)
    # log_history.json is tens of MB of raw HF records; the parsed series is what
    # the loss figure actually plots, and keeping it means a redraw never has to
    # re-derive (or re-agree with) that parse.
    write_json(run_dir / "training_series.json", series)

    # ---- privilege-utility frontier for the trained model ----
    model.eval()
    set_rank_all(model, max_rank)
    trained_sweep = sweep_perplexity(
        model, tokenizer, ppl_dataset, train_ranks, args.eval_batch_size, "nlpn"
    )
    del model, trainer
    reclaim()

    # ---- optional: SVD-truncated but untrained, the paper's App. D comparison ----
    svd_sweep = None
    if args.svd_baseline:
        svd_sweep = measure_svd_baseline(
            args.base_id, tokenizer, ppl_dataset, train_ranks,
            max_rank=max_rank, dtype=dtype, device_map=device_map,
            cache_dir=args.cache_dir, batch_size=args.eval_batch_size,
            target_substring=args.target_substring,
        )

    # ---- results + figures ----
    model_name = args.base_id.rsplit("/", maxsplit=1)[-1]
    metrics = {
        "base_id": args.base_id,
        "max_rank": max_rank,
        "train_ranks": train_ranks,
        "dense_baseline": dense_metrics,
        "nlpn_by_rank": trained_sweep,
        "svd_by_rank": svd_sweep,
        "train_metrics": train_output.metrics,
        "eval_history": series["evals"],
        "n_gated_modules": n_gated,
        "training_recipe": {
            "sampling_strategy": args.sampling_strategy,
            "variant_mode": args.variant_mode,
            "mixed_probability": args.mixed_probability,
            "mixed_coordinate_counts": args.mixed_coordinate_counts,
            "mixed_coordinate_weights": args.mixed_coordinate_weights,
            "warmup_proportion": args.warmup_proportion,
        },
    }
    if dense_metrics and trained_sweep.get(max_rank):
        dense_ppl = dense_metrics["perplexity"]
        full = trained_sweep[max_rank]["perplexity"]
        comparison = {
            "dense_pretrained_perplexity": dense_ppl,
            "nlpn_full_rank_perplexity": full,
            "nlpn_vs_dense_ratio": full / dense_ppl,
            "note": (
                "nlpn_vs_dense_ratio is NOT the cost of the capability gate on its own: "
                "the NLPN model was fine-tuned on this dataset's train split and the dense "
                "reference was not, so the ratio mixes the cost of factorization with the "
                "gain from in-domain fine-tuning, and can fall below 1. Run with "
                "--svd-baseline to get svd_vs_dense_ratio, which is the factorization cost "
                "alone (same weights, no fine-tuning)."
            ),
        }
        if svd_sweep and svd_sweep.get(max_rank):
            svd_full = svd_sweep[max_rank]["perplexity"]
            comparison["svd_full_rank_perplexity"] = svd_full
            comparison["svd_vs_dense_ratio"] = svd_full / dense_ppl
            comparison["nlpn_vs_svd_ratio"] = full / svd_full
        metrics["full_rank_comparison"] = comparison
    write_json(run_dir / "metrics.json", metrics)
    reporter.measurements(metrics)

    plot_training_loss(series, model_name, run_dir / "training_loss")
    plot_perplexity_vs_rank(
        trained_sweep, dense_metrics, svd_sweep, model_name, run_dir / "perplexity_vs_rank"
    )
    reporter.figures({
        "training_loss": run_dir / "training_loss.png",
        "perplexity_vs_rank": run_dir / "perplexity_vs_rank.png",
    })

    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.is_dir() and not any(checkpoint_dir.iterdir()):
        checkpoint_dir.rmdir()  # save_strategy="no" leaves it empty

    reporter.finish(failed=False)
    log(f"done -> {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
