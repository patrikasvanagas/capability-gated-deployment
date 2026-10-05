"""Significance testing and rank-drop sensitivity visualization.

Given a per-block / per-MLP rank sweep (see :mod:`capability_gated.mlp_rank_sweep`),
these helpers quantify and visualize which (task, block) cells are *significantly*
degraded by rank reduction:

* :func:`benjamini_hochberg` -- BH-FDR correction over many hypothesis tests.
* :func:`compute_delta_and_significance_masks` -- Δ-accuracy tensor + significance
  mask vs the full-rank baseline.
* :func:`plot_rank_drop_heatmaps` -- per-rank heatmaps of Δ accuracy by (task, block),
  with non-significant cells masked (paper Fig. 7 style).

The two consumers above expect a ``rank_reduced_results_dict`` keyed by ``"max_rank"``
and ``"{block}_{rank}"`` (for one fixed MLP module). The sweep script emits a richer
payload; :func:`sweep_payload_to_rank_reduced_dict` and
:func:`sweep_payload_to_single_intervention_json` adapt it to the schemas the
analysis functions and the suppression search expect.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ---------------------------------------------------------------------------
# Significance testing
# ---------------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    # Standard normal CDF via erf.
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _p_two_sided_from_z(z: float) -> float:
    # p = 2 * (1 - Phi(|z|)).
    return 2.0 * (1.0 - _norm_cdf(abs(z)))


def benjamini_hochberg(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR control.

    Returns a boolean array ``reject`` of the same shape as ``pvals`` (``True`` =
    statistically significant at FDR level ``q``). NaNs are treated as
    non-testable (``reject=False``).
    """
    p = pvals.copy()
    reject = np.zeros(p.shape, dtype=bool)

    finite_mask = np.isfinite(p)
    p_flat = p[finite_mask]
    if p_flat.size == 0:
        return reject

    order = np.argsort(p_flat)
    p_sorted = p_flat[order]
    m = p_sorted.size
    thresh = q * (np.arange(1, m + 1) / m)

    passed = p_sorted <= thresh
    if not np.any(passed):
        return reject

    kmax = np.max(np.where(passed)[0])
    cutoff = p_sorted[kmax]

    reject[finite_mask] = p_flat <= cutoff
    return reject


def compute_delta_and_significance_masks(
    rank_reduced_results_dict,
    ranks_to_try,
    num_blocks,
    tasks=None,
    q_fdr=0.05,
    min_effect=0.0,  # e.g. 0.01 for a 1pp practical-significance floor
):
    """Δ-accuracy tensor, BH significance mask, p-values and SEs vs baseline.

    Reads ``rank_reduced_results_dict["max_rank"]`` as the full-rank baseline and
    ``rank_reduced_results_dict["{block}_{rank}"]`` as each intervention. Returns
    ``(deltas, reject, pvals, se_deltas, tasks)`` with array shapes ``(R, T, B)``.
    """
    max_rank_results = rank_reduced_results_dict["max_rank"]
    if tasks is None:
        tasks = list(max_rank_results.keys())

    R, T, B = len(ranks_to_try), len(tasks), num_blocks
    deltas = np.full((R, T, B), np.nan, dtype=float)
    se_deltas = np.full((R, T, B), np.nan, dtype=float)
    pvals = np.full((R, T, B), np.nan, dtype=float)

    for ri, rank in enumerate(ranks_to_try):
        for bi in range(num_blocks):
            key = f"{bi}_{rank}"
            run = rank_reduced_results_dict.get(key)
            if run is None:
                continue
            for ti, task in enumerate(tasks):
                a = run[task]["acc,none"]
                se_a = run[task].get("acc_stderr,none", np.nan)

                b = max_rank_results[task]["acc,none"]
                se_b = max_rank_results[task].get("acc_stderr,none", np.nan)

                delta = a - b
                deltas[ri, ti, bi] = delta

                if np.isfinite(se_a) and np.isfinite(se_b):
                    se_delta = math.sqrt(se_a * se_a + se_b * se_b)
                    se_deltas[ri, ti, bi] = se_delta
                    if se_delta > 0:
                        z = delta / se_delta
                        pvals[ri, ti, bi] = _p_two_sided_from_z(z)

    # Practical-significance filter first (optional).
    if min_effect > 0:
        pvals[np.abs(deltas) < min_effect] = np.nan

    # FDR correction over all tested cells.
    reject = benjamini_hochberg(pvals, q=q_fdr)

    # reject=True means "statistically significant".
    return deltas, reject, pvals, se_deltas, tasks


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_rank_drop_heatmaps(
    rank_reduced_results_dict,
    ranks_to_try,
    num_blocks,
    tasks=None,
    save_path=None,
    fig_title=None,
    reject=None,  # bool array (R, T, B)
    nonsig_mode="zero",  # "zero" or "mask"
    cmap="RdBu_r",
):
    """Per-rank heatmaps of Δ accuracy by (task, block) vs the full-rank baseline.

    One subplot per rank in ``ranks_to_try``. If ``reject`` (from
    :func:`compute_delta_and_significance_masks`) is given, non-significant cells
    are either zeroed (``nonsig_mode="zero"``) or masked grey
    (``nonsig_mode="mask"``). Zero is pinned to the white midpoint of the diverging
    colormap.
    """
    max_rank_results = rank_reduced_results_dict["max_rank"]

    if tasks is None:
        tasks = list(max_rank_results.keys())

    R, T, B = len(ranks_to_try), len(tasks), num_blocks
    deltas = np.full((R, T, B), np.nan, dtype=float)

    # ---- Build delta tensor ----
    for ri, rank in enumerate(ranks_to_try):
        for bi in range(num_blocks):
            key = f"{bi}_{rank}"
            run = rank_reduced_results_dict.get(key)
            if run is None:
                continue
            for ti, task in enumerate(tasks):
                deltas[ri, ti, bi] = (
                    run[task]["acc,none"] - max_rank_results[task]["acc,none"]
                )

    # ---- Apply significance filter ----
    if reject is not None:
        reject = np.asarray(reject, dtype=bool)
        if reject.shape != deltas.shape:
            raise ValueError(
                f"reject has shape {reject.shape}, expected {deltas.shape}."
            )

        nonsig = np.isfinite(deltas) & (~reject)
        if nonsig_mode == "zero":
            deltas[nonsig] = 0.0
        elif nonsig_mode == "mask":
            deltas[nonsig] = np.nan
        else:
            raise ValueError("nonsig_mode must be 'zero' or 'mask'.")

    # ---- Colour scaling: asymmetric limits, but 0 fixed at the white midpoint ----
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        vmin, vmax = -0.5, 0.5
    else:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))

    # Ensure 0 lies strictly inside (vmin, vmax) so TwoSlopeNorm is valid.
    eps = 1e-6
    if vmax <= 0.0:
        vmax = +eps
    if vmin >= 0.0:
        vmin = -eps

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(
        1,
        R,
        figsize=(4.8 * R, 3.4),
        sharey=True,
        constrained_layout=False,
        dpi=300,
    )
    if R == 1:
        axes = [axes]

    # NaNs as light grey (useful for nonsig_mode="mask" or missing runs).
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color="0.9")

    im = None
    for ax, rank, data in zip(axes, ranks_to_try, deltas):
        im = ax.imshow(
            data,
            aspect="auto",
            interpolation="nearest",
            norm=norm,
            cmap=cm,
        )
        ax.set_title(f"Rank {rank}")
        ax.set_xlabel("Block index")
        xticks = np.arange(0, num_blocks, 2)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])

    axes[0].set_ylabel("Task")
    axes[0].set_yticks(np.arange(len(tasks)))

    clean_task_names = []
    for t in tasks:
        cleaned = t.replace("mmlu_", "")
        cleaned = cleaned.replace("college_", "").replace("high_school_", "")
        cleaned = cleaned.replace("_", " ").title()
        clean_task_names.append(cleaned)
    axes[0].set_yticklabels(clean_task_names)

    fig.subplots_adjust(right=0.88, wspace=0.15)

    if fig_title:
        fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=1.05)

    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label("Δ accuracy vs max-rank baseline", labelpad=10)

    # Ticks: include vmin, 0, vmax, plus midpoints (asymmetric).
    ticks = np.array([vmin, 0.5 * vmin, 0.0, 0.5 * vmax, vmax], dtype=float)
    ticks = np.unique(np.sort(ticks))
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:+.3f}" for t in ticks])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Adapters: sweep payload -> consumer schemas
# ---------------------------------------------------------------------------
def sweep_payload_to_rank_reduced_dict(
    payload: dict[str, Any],
    module_name: str,
) -> dict[str, Any]:
    """Convert an ``mlp_rank_sweep`` payload into a ``rank_reduced_results_dict``.

    Selects the sweep entries for a single MLP projection (``module_name``, e.g.
    ``"down_proj"``) and keys them as ``"{block}_{rank}"``, plus a ``"max_rank"``
    baseline -- exactly the schema consumed by
    :func:`compute_delta_and_significance_masks` and :func:`plot_rank_drop_heatmaps`.
    """
    out: dict[str, Any] = {"max_rank": payload["baseline"]}
    for entry in payload["sweeps"]:
        if entry["module_name"] != module_name:
            continue
        out[f"{entry['block_idx']}_{entry['rank']}"] = entry["results"]
    return out


def sweep_payload_to_single_intervention_json(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Convert an ``mlp_rank_sweep`` payload into a beam-search ``single_json``.

    Keys each single-MLP intervention as ``"{block}_{rank}_{module}"`` with the
    ``results`` / ``rank`` / ``rank_percentage`` fields the suppression search
    expects, plus a ``"max_rank"`` baseline mapping task -> stats.
    """
    out: dict[str, Any] = {"max_rank": payload["baseline"]}
    for entry in payload["sweeps"]:
        key = f"{entry['block_idx']}_{entry['rank']}_{entry['module_name']}"
        out[key] = {
            "results": entry["results"],
            "rank": entry["rank"],
            "rank_percentage": entry.get("rank_percentage"),
        }
    return out
