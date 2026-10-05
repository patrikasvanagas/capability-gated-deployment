#!/usr/bin/env python
"""E3: search for rank configurations that suppress a WMDP domain, split A only.

Consumes an E2 localisation run (``mech_interp_results/e2_localisation/...``):
its frozen splits and headroom gate define what is searched, and its evaluator
settings are reused so every cached number stays comparable. The search procedure is
``suppression_search.beam_search`` + ``refine_config_ranks`` -- **unchanged** --
driven by the fast split-A ``MemoisedMCQEvaluator`` (seconds per configuration
instead of lm-eval's minutes). ``--beam-width 1`` is greedy forward selection;
any other solver that consumes ``(memo_eval, objective, candidate_ranks)`` drops
in the same way, and the saved per-arm caches are the dataset for judging
whether a different procedure would have done better.

**Shortlisting is combined by default** (``--shortlist combined``), because the
two available instruments fail differently. Attribution chooses *which ranks* are
worth measuring per coordinate, which cuts a full single-intervention sweep by
~4x (``--sweep-mode attribution``); by default the sweep instead uses a *uniform*
ladder so every coordinate is measured at the same intervention strengths and the
ordering compares like with like. The measured sweep over **every** coordinate supplies the ground-truth
ordering (its accuracy signal is unbiased but quantised, and it only sees a
coordinate alone); and the top ``--n-rescue`` attribution coordinates are kept
regardless, since attribution's score is continuous, low-variance and additive
and can flag a coordinate whose solo effect hides inside the sweep's noise. The
sweep's evaluations are cached, so the beam's depth 1 over the shortlist is then
free -- the sweep is depth 1 done exhaustively and used to choose where to go
deeper. ``--shortlist attribution`` restores the pure predicted shortlists, one
arm per ``--strategies`` entry, for the paper's matched-budget ablation.

The retention constraint is pre-registered as **two pooled macros**
(``mmlu_adjacent_macro`` n=172, ``mmlu_distant_macro`` n=512) instead of the
seven per-task epsilons of Rauba et al. (2026): eps=0.01 on a 50-item member task is
half an item, so a per-task penalty fires on bf16 noise. Per-member accuracies
are still measured and reported for every final configuration; only the
objective's constraint is pooled. The deviation and its reason are recorded in
the run's ``config.json``.

**Three safeguards keep suppression from becoming model damage.** (1) A chance
floor: the suppression term reads a clamped copy of the domain, because
below-chance accuracy is systematic avoidance of the right answer -- a masking
signature -- and is unbounded, so an adaptive search chases it. (2) A fluency
price and a fluency gate, which are different instruments and both are needed.
The objective is accuracy-only, so nothing stops it trading away held-out
perplexity. The *price* steers: with ``--e2-fluency-prices``, E2's first-order
delta-NLL estimates enter the objective during the sweep, beam and refinement, so
the search prefers cheap coordinates. It is additive and first-order, so it is a
guide and never a guarantee. The *gate* decides: candidates above
``--max-ppl-ratio`` are rejected on measurement, since suppression by lobotomy is
not a capability gate. That gate is settled on the full reporting set -- the
perplexity the paper quotes -- while the search and the cheap candidate screen aim
``--screen-margin`` below it, because the 64-block screening slice under-reads
that set by up to ~2% and would otherwise certify configurations that are
reported outside their own budget. (3) A stability guard: the search maximises
a statistic measured on the very items it searches over, so the reported config
is not the raw argmax but the best of the top ``--final-top-k`` under a
bootstrapped lower quantile of the objective.

Budgets are variable and *estimated before anything runs*: the script measures
one real evaluation per domain, prints a projected eval-count / wall-clock table
per (domain, strategy) arm, and only then starts. ``--estimate-only`` prints the
table and exits without creating a run directory.

Example::

    uv run python scripts/find_suppression_configs.py \\
        --e2-run mech_interp_results/e2_localisation/<ckpt>/<stamp> \\
        --estimate-only

    uv run python scripts/find_suppression_configs.py \\
        --e2-run mech_interp_results/e2_localisation/<ckpt>/<stamp> \\
        --preset standard          # full 3x4 grid, ~5-8 h

Interrupting is safe: caches autosave atomically, partial results are marked,
and ``--resume <run dir>`` replays completed evaluations from cache (~free) and
skips arms whose results are already complete.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from capability_gated import (
    CHANCE,
    MMLU_ADJACENT,
    MMLU_DISTANT,
    FluencyPriceMap,
    MCQEvaluator,
    MemoisedMCQEvaluator,
    RunLog,
    RunPaths,
    SplitRegistry,
    evaluate_perplexity_at_rank,
    load_candidate_ranks,
    load_lowrank_checkpoint,
    load_task,
    pool_mcq_results,
    load_or_build_ppl_blocks,
    provenance,
    save_item_level,
    set_rank_all,
    write_json,
)
from capability_gated.artifacts import (
    format_table,
    jsonable,
    save_figure,
    sha256_file,
)
from capability_gated.attribution import PAPER_RC, ScoreMaps, build_shortlist
from capability_gated.suppression_search import (
    BeamSearchSettings,
    FluencyPrices,
    Config,
    Coord,
    RefinementSettings,
    TaskTradeoffObjective,
    apply_config,
    beam_search,
    measure_fluency_prices,
    compute_objective,
    rank_budget_used,
    refine_config_ranks,
    serialise_cfg,
)

WMDP_DOMAINS = ("wmdp-bio", "wmdp-chem", "wmdp-cyber")
STRATEGIES = ("selectivity", "magnitude", "random", "anti")
ADJACENT = sorted(set(MMLU_ADJACENT.values()))
DISTANT = list(MMLU_DISTANT)
MACROS: dict[str, list[str]] = {
    "mmlu_adjacent_macro": ADJACENT,
    "mmlu_distant_macro": DISTANT,
}
PRESETS: dict[str, tuple[int, int, int]] = {  # (beam_width, max_depth, refine_rounds)
    "greedy": (1, 10, 1),
    "standard": (4, 8, 2),
    "paper": (8, 9, 2),
}
# Fixed arm order and colors (validated categorical palette; random/anti are the
# control arms and keep their slots regardless of which arms are run).
ARM_COLORS = {
    "combined": "#2a78d6",
    "selectivity": "#2a78d6",
    "magnitude": "#eb6834",
    "random": "#1baf7a",
    "anti": "#eda100",
}
RESULT_SCHEMA = "capability_gated.e3/result/v1"
CONFIG_SCHEMA = "capability_gated.e3/config/v1"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = p.add_argument_group("source")
    source.add_argument(
        "--e2-run",
        default=None,
        help="E2 localisation run dir (pins the shortlist source; pre-registered). "
        "Required for everything except --replot.",
    )
    source.add_argument("--cache-dir", default=None, help="HF cache dir.")

    scope = p.add_argument_group("scope")
    scope.add_argument(
        "--domains", nargs="+", choices=list(WMDP_DOMAINS), default=None,
        help="Default: the E2 headroom gate's active domains.",
    )
    scope.add_argument("--strategies", nargs="+", choices=list(STRATEGIES),
                       default=list(STRATEGIES))

    budget = p.add_argument_group("budget")
    budget.add_argument("--preset", choices=list(PRESETS), default="standard")
    budget.add_argument("--beam-width", type=int, default=None,
                        help="Override the preset (1 = greedy forward selection).")
    budget.add_argument("--max-depth", type=int, default=None)
    budget.add_argument("--refine-rounds", type=int, default=None,
                        help="0 skips the coordinate-descent refinement.")

    shortlist = p.add_argument_group("shortlist")
    shortlist.add_argument(
        "--shortlist", choices=["combined", "attribution"], default="combined",
        help="combined (default): measure every coordinate once, rank by "
        "measurement, keep a rescue quota for attribution. attribution: use the "
        "E2 predicted shortlists as-is, one arm per --strategies entry.",
    )
    shortlist.add_argument(
        "--rebuild-shortlist", action="store_true",
        help="Attribution arms only: rebuild the shortlist from the E2 score "
        "maps at this run's --top-k/--ranks-per-coord instead of reading the "
        "frozen shortlists.json, whose top_k is whatever the E2 campaign used. "
        "Required to size a control arm's candidate pool to match a combined "
        "arm's, i.e. for the matched-budget ablation.",
    )
    shortlist.add_argument("--top-k", type=int, default=12,
                           help="Coordinates carried into the beam search.")
    shortlist.add_argument("--ranks-per-coord", type=int, default=2)
    shortlist.add_argument(
        "--sweep-mode", choices=["uniform", "attribution"], default="uniform",
        help="uniform (default): every coordinate is swept at the SAME ranks, so "
        "coordinates are compared at equal intervention strength. attribution: "
        "per-coordinate mass-target ranks (~30%% cheaper, but coordinates are "
        "then compared at different strengths).",
    )
    shortlist.add_argument(
        "--sweep-fractions", type=float, nargs="+", default=[0.5, 0.125, 0.03125],
        help="Uniform sweep ranks as fractions of max_rank (snapped to the ladder).",
    )
    shortlist.add_argument("--sweep-ranks-per-coord", type=int, default=2,
                           help="Ranks per coordinate in attribution sweep mode.")
    shortlist.add_argument("--n-rescue", type=int, default=4,
                           help="Attribution coordinates kept regardless of the sweep.")
    shortlist.add_argument("--mass-targets", type=float, nargs="+",
                           default=[0.5, 0.9])

    objective = p.add_argument_group(
        "objective (pre-registered; overrides are recorded verbatim)",
    )
    objective.add_argument("--lambda-suppress", type=float, default=2.0)
    objective.add_argument("--eps-preserve-drop", type=float, default=0.01)
    objective.add_argument("--penalty-scale", type=float, default=100.0)
    objective.add_argument(
        "--e2-fluency-prices", action="store_true",
        help="Use E2's fluency_prices.npz as an additive first-order NLL cost "
        "during the singleton sweep, beam search, and refinement.",
    )
    objective.add_argument(
        "--fluency-price-weight", type=float, default=1.0,
        help="Lagrange weight on E2's additive delta-NLL price (only active with "
        "--e2-fluency-prices). Weight 1 makes a predicted 10%% PPL increase cost "
        "about 0.095 objective units.",
    )
    objective.add_argument("--n-to-save", type=int, default=50)
    objective.add_argument(
        "--no-chance-floor", action="store_true",
        help="Reward below-chance accuracy. Off by default: sub-chance means "
        "systematic avoidance of the right answer (a masking signature), and it "
        "is unbounded, so an adaptive search will chase it.",
    )

    final = p.add_argument_group("final selection (fluency gate + stability guard)")
    final.add_argument("--final-top-k", type=int, default=20,
                       help="Cached configs re-measured and re-ranked at the end.")
    final.add_argument(
        "--max-ppl-ratio", type=float, default=1.10,
        help="Reject a config whose held-out perplexity exceeds this multiple of "
        "the full-privilege value -- the guard against suppression-by-lobotomy, "
        "which the objective itself does not see. Adjudicated on the FULL "
        "reporting set, so a config that passes is one whose reported number "
        "is inside the budget.",
    )
    final.add_argument(
        "--screen-margin", type=float, default=0.025,
        help="How much tighter the cheap screening slice aims than the real gate. "
        "The screen is 64 blocks against the reporting set's 256 and reads "
        "systematically LOW: on the two priced bio runs the reporting set came in "
        "1.43%% and 1.93%% above it, enough for a config to clear a 1.10 screen "
        "and still be quoted at 1.11. Search and pricing therefore target "
        "`--max-ppl-ratio - this`; the gate itself does not move. 0.025 is the "
        "smallest 0.005 step that clears both observed offsets (0.020 would have "
        "landed the worse one at 1.1008). Over-tightening only costs suppression; "
        "under-tightening costs the run, since nothing then survives the gate.",
    )
    final.add_argument(
        "--verify-top-k", type=int, default=8,
        help="Screen-passing candidates re-measured on the full reporting set, "
        "best-scoring first; the first that genuinely passes --max-ppl-ratio is "
        "selected. 0 trusts the screen (the pre-2026-08 behaviour, which could "
        "certify a config whose reported perplexity breached the gate).",
    )
    final.add_argument(
        "--fluency-prices", action="store_true",
        help="Measure the solo fluency cost of every shortlisted (coordinate, "
        "rank) before searching, and skip expansions predicted to breach the "
        "budget. Coordinate costs span 1.009x-1.819x on the 1.7B checkpoint, and "
        "without this the search cannot tell a cheap coordinate from one that "
        "consumes the entire budget by itself. Costs one screening evaluation "
        "per pair (~40 for a 16-coordinate shortlist).",
    )
    final.add_argument(
        "--fluency-budget", type=float, default=None,
        help="Budget for the pricing screen; defaults to --max-ppl-ratio. Set it "
        "higher to screen loosely and let the measured gate do the deciding.",
    )
    final.add_argument(
        "--fluency-interaction", type=float, default=1.0,
        help="Multiplier on the product-of-singles estimate. Default 1.0 (plain "
        "product) classified all four measured E3 configurations correctly; the "
        "true interaction term ranged 0.92-1.40 and is not monotone in the "
        "product, so it is not modelled. Raising this makes the screen stricter "
        "and risks pruning viable configurations.",
    )
    final.add_argument("--ppl-screen-blocks", type=int, default=64,
                       help="Blocks used to screen candidates (winner gets the full set).")
    final.add_argument("--bootstrap", type=int, default=1000,
                       help="Item-resampling draws for the stability guard (0 = off).")
    final.add_argument("--stability-quantile", type=float, default=0.25,
                       help="Configs are ranked by this quantile of the bootstrapped "
                       "objective, not by the point estimate.")

    control = p.add_argument_group("run control")
    control.add_argument("--estimate-only", action="store_true",
                         help="Print the projection table and exit; writes nothing.")
    control.add_argument("--resume", default=None,
                         help="Existing e3 run dir: reuse caches, skip finished arms.")
    control.add_argument("--smoke", action="store_true",
                         help="item_cap=128, W=2 D=2 R=1, '_smoke' dir suffix.")
    control.add_argument("--item-cap", type=int, default=None,
                         help="Cap items per task (implies the smoke dir suffix).")
    control.add_argument("--autosave-every", type=int, default=20)
    control.add_argument("--replot", default=None,
                         help="Regenerate figures/summary from a run dir; no GPU.")

    evalg = p.add_argument_group("eval / perplexity (defaults locked to E2)")
    evalg.add_argument("--tokens-per-batch", type=int, default=8192)
    evalg.add_argument("--max-seq-len", type=int, default=1024)
    evalg.add_argument("--ppl-samples", type=int, default=256)
    evalg.add_argument("--ppl-batch-size", type=int, default=8)
    evalg.add_argument("--seed", type=int, default=20260807)

    out = p.add_argument_group("output")
    out.add_argument("--output-root", default="mech_interp_results")
    return p


@dataclass(frozen=True)
class SearchBudget:
    preset: str
    beam_width: int
    max_depth: int
    refine_rounds: int


def resolve_budget(args: argparse.Namespace) -> SearchBudget:
    width, depth, rounds = PRESETS[args.preset]
    if args.smoke:
        width, depth, rounds = 2, 2, 1
    if args.beam_width is not None:
        width = args.beam_width
    if args.max_depth is not None:
        depth = args.max_depth
    if args.refine_rounds is not None:
        rounds = args.refine_rounds
    return SearchBudget(args.preset, int(width), int(depth), int(rounds))


@dataclass(frozen=True)
class E2Source:
    run_dir: Path
    config: dict[str, Any]
    headroom: dict[str, Any]
    shortlists_path: Path
    shortlists: dict[str, Any]

    @property
    def checkpoint_dir(self) -> Path:
        return REPO_ROOT / self.config["checkpoint"]["run_dir"]

    @property
    def max_rank(self) -> int:
        return int(self.config["checkpoint"]["max_rank"])

    @property
    def rank_ref(self) -> int:
        return int(self.shortlists["rank_ref"])

    @property
    def fluency_prices_path(self) -> Path:
        return self.run_dir / "fluency_prices.npz"

    def sha256s(self) -> dict[str, str]:
        hashes = {
            "config.json": sha256_file(self.run_dir / "config.json"),
            "headroom.json": sha256_file(self.run_dir / "headroom.json"),
            "shortlists.json": sha256_file(self.shortlists_path),
        }
        if self.fluency_prices_path.exists():
            hashes["fluency_prices.npz"] = sha256_file(self.fluency_prices_path)
        return hashes


def load_e2_source(path: str | Path) -> E2Source:
    run_dir = Path(path)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    shortlists_path = run_dir / "shortlists.json"
    for required in ("config.json", "headroom.json", "shortlists.json"):
        if not (run_dir / required).exists():
            raise SystemExit(f"--e2-run {run_dir} is missing {required}")
    return E2Source(
        run_dir=run_dir,
        config=json.loads((run_dir / "config.json").read_text()),
        headroom=json.loads((run_dir / "headroom.json").read_text()),
        shortlists_path=shortlists_path,
        shortlists=json.loads(shortlists_path.read_text()),
    )


# ---------------------------------------------------------------------------
# Estimate model
# ---------------------------------------------------------------------------
def arm_pairs(e2: E2Source, domain: str, strategy: str) -> int:
    cand = e2.shortlists["domains"][domain][strategy]["candidate_ranks"]
    return sum(len(v) for v in cand.values())


def arm_coords(e2: E2Source, domain: str, strategy: str) -> int:
    return len(e2.shortlists["domains"][domain][strategy]["candidate_ranks"])


def projected_evals(
    n_pairs: int, n_coords: int, budget: SearchBudget,
) -> dict[str, float]:
    """Upper-bound and optimistic eval counts for one arm.

    Beam: depth 1 evaluates every (coord, rank) pair once; at depth d each of
    the W states extends over the remaining coords, ~uniform over pairs. The
    beam never stops early while coords remain, so the upper bound is realistic;
    the optimistic case discounts 20% for cross-state duplicates hitting the
    memo (none at W=1). Refinement: |cfg| coords x (8 coarse + ~3 midpoints)
    x rounds.
    """
    width, depth, rounds = budget.beam_width, budget.max_depth, budget.refine_rounds
    beam_ub = float(n_pairs)
    for d in range(2, min(depth, n_coords) + 1):
        beam_ub += width * n_pairs * (n_coords - d + 1) / n_coords
    beam_opt = beam_ub if width == 1 else 0.8 * beam_ub
    k_ub = min(depth, n_coords)
    refine_ub = k_ub * 11.0 * rounds
    refine_opt = max(2, round(0.6 * k_ub)) * 11.0 * rounds * 0.8
    return {
        "beam_ub": beam_ub,
        "beam_opt": beam_opt,
        "refine_ub": refine_ub,
        "refine_opt": refine_opt,
        "total_ub": beam_ub + refine_ub + 1,
        "total_opt": beam_opt + refine_opt + 1,
    }


def projection_rows(  # noqa: PLR0913, PLR0917 - a pure projection reads best flat
    e2: E2Source,
    domains: Sequence[str],
    arms: Sequence[str],
    budget: SearchBudget,
    sec_per_eval: Mapping[str, float],
    finals_seconds: float,
    *,
    sweep_evals: int = 0,
    top_k: int = 12,
    ranks_per_coord: int = 2,
    final_top_k: int = 20,
    verify_top_k: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Projected evaluations and wall-clock per arm.

    In combined mode the sweep's evaluations are not additional work for the
    beam: every singleton it measures is cached, so the beam's depth 1 is free
    and only depths 2+ cost anything. The projection reflects that.
    """
    rows = []
    total_ub = total_opt = 0.0
    for domain in domains:
        for strategy in arms:
            if strategy == "combined":
                n_pairs = top_k * ranks_per_coord
                n_coords = top_k
                est = projected_evals(n_pairs, n_coords, budget)
                # depth 1 is already in the cache courtesy of the sweep
                est = {k: v - (n_pairs if "beam" in k or "total" in k else 0)
                       for k, v in est.items()}
                est["total_ub"] += sweep_evals
                est["total_opt"] += sweep_evals
                n_pairs = sweep_evals
            else:
                n_pairs = arm_pairs(e2, domain, strategy)
                n_coords = arm_coords(e2, domain, strategy)
                est = projected_evals(n_pairs, n_coords, budget)
            sec = sec_per_eval[domain]
            # Final selection: one MCQ re-measure + one screening perplexity per
            # candidate, then a reporting-set perplexity for each candidate that
            # gets verified -- 256 blocks against the screen's 64, so ~4x. The
            # upper bound verifies every one; the optimistic case stops at the
            # first pass.
            est["total_ub"] += 2 * final_top_k + 4 * verify_top_k
            est["total_opt"] += 2 * final_top_k + 4
            t_ub = est["total_ub"] * sec + finals_seconds
            t_opt = est["total_opt"] * sec + finals_seconds
            total_ub += t_ub
            total_opt += t_opt
            rows.append(
                {
                    "arm": f"{domain}/{strategy}",
                    "pairs": n_pairs,
                    "evals_ub": round(est["total_ub"]),
                    "evals_opt": round(est["total_opt"]),
                    "s_per_eval": sec,
                    "min_ub": t_ub / 60,
                    "min_opt": t_opt / 60,
                },
            )
    totals = {"hours_ub": total_ub / 3600, "hours_opt": total_opt / 3600}
    return rows, totals


# ---------------------------------------------------------------------------
# Search + measurement units
# ---------------------------------------------------------------------------
def item_path_for(cache_path: Path) -> Path:
    """``cache/<arm>.cache.json`` -> ``items/<arm>.items.npz``."""
    arm = cache_path.name.removesuffix(".cache.json")
    return cache_path.parent.parent / "items" / f"{arm}.items.npz"


def key_to_cfg(key: Sequence[Sequence[Any]]) -> Config:
    return {(int(b), str(m)): int(r) for b, m, r in key}


def reconstruct_trajectory(
    pre_refine_keys: set[Any],
    cache: Mapping[Any, dict[str, Any]],
    base_acc: Mapping[str, float],
    objective: TaskTradeoffObjective,
    max_depth: int,
) -> list[dict[str, Any]]:
    """The beam trajectory, read back off the pre-refinement cache slice.

    Every size-d config the beam evaluated is in the cache with |cfg| = d, and
    refinement trials are excluded by the snapshot -- so "best objective among
    cached configs of size <= d" is exactly the running best after depth d,
    without instrumenting ``beam_search``.

    The same slice gives the run's COST at each depth, which the matched-budget
    ablation needs: one distinct key is one configuration measured, so counting
    keys per depth yields spend-so-far alongside best-so-far. Recovered rather
    than instrumented for the same reason as the scores -- and it works for
    shortlists that skip the single-coordinate sweep, where the depth-1 count is
    exactly the saving being claimed.
    """
    by_depth: dict[int, tuple[float, Any]] = {}
    n_at_depth: dict[int, int] = {}
    for key in pre_refine_keys:
        depth = len(key)
        n_at_depth[depth] = n_at_depth.get(depth, 0) + 1
        score = compute_objective(
            cache[key]["acc"], dict(base_acc), objective, key_to_cfg(key),
        )
        if depth not in by_depth or score > by_depth[depth][0]:
            by_depth[depth] = (score, key)
    rows = []
    spent = 0
    running = -float("inf")
    running_key: Any = ()
    for depth in range(max_depth + 1):
        spent += n_at_depth.get(depth, 0)
        if depth in by_depth and by_depth[depth][0] > running:
            running, running_key = by_depth[depth]
        if depth in by_depth or depth == 0:
            entry = cache[running_key]
            rows.append(
                {
                    "depth": depth,
                    "best_score_at_depth": by_depth.get(depth, (None,))[0],
                    "running_best_score": running,
                    "running_best_cfg": serialise_cfg(key_to_cfg(running_key)),
                    "configs_measured_at_depth": n_at_depth.get(depth, 0),
                    "configs_measured_cumulative": spent,
                    "acc": entry["acc"],
                },
            )
    return rows


def rescore_top_configs(
    memo: MemoisedMCQEvaluator,
    base_acc: Mapping[str, float],
    objective: TaskTradeoffObjective,
    n_to_save: int,
) -> list[dict[str, Any]]:
    """Every config the search ever evaluated, re-scored and truncated.

    Port of ``optimise_task_suppression``'s tail: the frontier the paper plots
    comes from here, not just from beam survivors.
    """
    scored = []
    for key, entry in memo.cache.items():
        cfg = key_to_cfg(key)
        scored.append(
            {
                "cfg": serialise_cfg(cfg),
                "score": compute_objective(
                    entry["acc"], dict(base_acc), objective, cfg,
                ),
                "fluency_price_nll": (
                    float(objective.fluency_price(cfg))
                    if objective.fluency_price is not None else 0.0
                ),
                "num_interventions": len(cfg),
                "rank_budget_used": rank_budget_used(cfg, memo.max_rank),
                "acc": entry["acc"],
                "se": entry["se"],
            },
        )
    scored.sort(key=lambda row: row["score"], reverse=True)
    return scored[:n_to_save]


def final_measure(
    evaluator: MCQEvaluator,
    model: Any,
    tokenizer: Any,
    cfg: Config,
    ppl_ds: Any,
    *,
    max_rank: int,
    ppl_batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Full 10-task re-measure + macros + fluency guard for one configuration.

    Cross-domain WMDP accuracies are deliberately included -- what C_bio does to
    cyber/chem is the E4 composition record. ``rank=None`` on the perplexity
    call is load-bearing: an int would wipe the per-module gating.

    Returns ``(summary_payload, per_task_results)``; the second element carries
    the per-item vectors, which belong in the arm's NPZ rather than the JSON.
    """
    results = dict(evaluator.evaluate_config(cfg))
    macro_results = {
        name: pool_mcq_results(results, members, name=name)
        for name, members in MACROS.items()
    }
    for name, pooled in macro_results.items():
        members = MACROS[name]
        recount = sum(results[m].n_correct for m in members) / sum(
            results[m].n for m in members
        )
        if abs(pooled.acc - recount) > 1e-12:
            raise RuntimeError(f"macro {name} pooling mismatch: {pooled.acc} vs {recount}")
    try:
        apply_config(model, dict(cfg), max_rank)
        ppl, ppl_loss = evaluate_perplexity_at_rank(
            model, tokenizer, ppl_ds, rank=None, batch_size=ppl_batch_size,
        )
    finally:
        set_rank_all(model, max_rank)
    payload = {
        "mcq": {t: r.to_json() for t, r in results.items()},
        "macros": {t: r.to_json() for t, r in macro_results.items()},
        "minipile_ppl_gated": float(ppl),
        # The NLL is what the perplexity is an exponential of; keeping only the
        # exponential loses precision and cannot be averaged.
        "minipile_nll_gated": float(ppl_loss),
        "rank_budget_used": rank_budget_used(cfg, max_rank),
        "num_interventions": len(cfg),
    }
    # Per-item vectors go to the arm's NPZ, not into this JSON.
    return payload, {**results, **macro_results}


def beam_state_payload(
    state: Any,
    max_rank: int,
    objective: TaskTradeoffObjective | None = None,
) -> dict[str, Any]:
    payload = {
        "cfg": serialise_cfg(state.cfg),
        "score": float(state.score),
        "acc": state.acc,
        "num_interventions": len(state.cfg),
        "rank_budget_used": rank_budget_used(state.cfg, max_rank),
    }
    if objective is not None and objective.fluency_price is not None:
        payload["fluency_price_nll"] = float(objective.fluency_price(state.cfg))
    return payload


def uniform_sweep_ranks(ladder: Sequence[int], fractions: Sequence[float]) -> list[int]:
    """Snap each fraction of max_rank onto the nearest ladder rung (log distance).

    A uniform ladder means every coordinate is measured at the *same* intervention
    strengths, so the sweep's ranking compares like with like. Per-coordinate
    mass-target ranks are ~30% cheaper but compare a coordinate gated to r=224
    against one gated to r=3, which is not a fair ordering.
    """
    max_rank = max(ladder)
    picks = []
    for frac in fractions:
        target = max(1.0, frac * max_rank)
        best = min(
            (r for r in ladder if r < max_rank),
            key=lambda r: abs(np.log(r) - np.log(target)),
        )
        if best not in picks:
            picks.append(best)
    return sorted(picks, reverse=True)


def full_candidate_grid(
    maps: Any, domain: str, rank_ref: int, args: argparse.Namespace,
) -> dict[Coord, list[int]]:
    """Every coordinate, with attribution-chosen ranks: the sweep's grid.

    Attribution's job here is *rank selection* -- picking the 2-3 ranks per
    coordinate whose predicted drop reaches the mass targets, instead of
    measuring the whole ladder. That is a ~4x saving on the sweep and is the one
    thing a measured sweep cannot supply for itself.
    """
    if args.sweep_mode == "uniform":
        ranks = uniform_sweep_ranks(
            [int(r) for r in maps.ranks], args.sweep_fractions,
        )
        return {coord: list(ranks) for coord in maps.coord_list()}
    shortlist = build_shortlist(
        maps, domain,
        rank_ref=rank_ref,
        top_k=len(maps.coord_list()),
        ranks_per_coord=args.sweep_ranks_per_coord,
        strategy="all",
        mass_targets=tuple(args.mass_targets),
        seed=args.seed,
    )
    return shortlist.candidate_ranks()


def measured_singleton_sweep(  # noqa: PLR0913, PLR0917 - flat measurement loop
    memo: MemoisedMCQEvaluator,
    base_acc: Mapping[str, float],
    objective: TaskTradeoffObjective,
    grid: Mapping[Coord, list[int]],
    domain: str,
    out: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Measure every (coordinate, rank) singleton on split A.

    These evaluations land in the memo cache, so the beam's depth 1 over any
    subset of them is afterwards free -- the sweep is not an extra stage so much
    as depth 1 done exhaustively and then used to choose where to go deeper.

    ``out``, when given, is the accumulator: rows appear in it as they are
    measured, so a caller that catches a Ctrl-C still holds the partial sweep.
    """
    rows = [] if out is None else out
    total = sum(len(v) for v in grid.values())
    done = 0
    for coord, ranks in grid.items():
        for rank in ranks:
            payload = memo.evaluate({coord: rank})
            rows.append(
                {
                    "block": coord[0],
                    "module": coord[1],
                    "rank": int(rank),
                    "score": compute_objective(
                        payload["acc"], dict(base_acc), objective, {coord: rank},
                    ),
                    "fluency_price_nll": (
                        float(objective.fluency_price({coord: rank}))
                        if objective.fluency_price is not None else 0.0
                    ),
                    "suppress_drop": base_acc[domain] - payload["acc"][domain],
                    "macro_drops": {
                        m: base_acc[m] - payload["acc"][m] for m in MACROS
                    },
                },
            )
            done += 1
            if done % 25 == 0:
                log(f"    sweep {done}/{total}")
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def combine_shortlist(
    sweep_rows: Sequence[Mapping[str, Any]],
    predicted: Mapping[Coord, list[int]],
    *,
    top_k: int,
    ranks_per_coord: int,
    n_rescue: int,
) -> tuple[dict[Coord, list[int]], dict[str, list[str]]]:
    """Measured ordering, with a small rescue quota for attribution.

    The two instruments measure different things and fail differently: the sweep
    is unbiased but its accuracy signal is quantised (one item is 0.16-0.61 pp
    depending on the task) and it only sees a coordinate *alone*; attribution is
    first-order-biased but continuous, low-variance, and additive, so it can
    flag a coordinate whose solo effect sits inside the sweep's noise while its
    attributable mass matters in combination. So: rank by measurement, then give
    the top ``n_rescue`` attribution coordinates the sweep did not already pick
    a seat anyway. Provenance per coordinate is returned for the record.
    """
    best_by_coord: dict[Coord, list[Mapping[str, Any]]] = {}
    for row in sweep_rows:
        best_by_coord.setdefault((row["block"], row["module"]), []).append(row)

    measured_order = sorted(
        best_by_coord.items(), key=lambda kv: kv[1][0]["score"], reverse=True,
    )
    chosen: dict[Coord, list[int]] = {}
    provenance_by_coord: dict[str, list[str]] = {}
    for coord, rows in measured_order[:top_k]:
        ranks = [r["rank"] for r in rows[:ranks_per_coord]]
        chosen[coord] = ranks
        provenance_by_coord[f"{coord[0]}_{coord[1]}"] = ["measured"]

    rescued = 0
    for coord, ranks in predicted.items():
        if rescued >= n_rescue:
            break
        key = f"{coord[0]}_{coord[1]}"
        if coord in chosen:
            provenance_by_coord[key].append("attribution")
            continue
        chosen[coord] = list(ranks)
        provenance_by_coord[key] = ["attribution_rescue"]
        rescued += 1
    return chosen, provenance_by_coord


def bootstrap_objective(
    per_task_correct: Mapping[str, np.ndarray],
    base_acc: Mapping[str, float],
    objective: TaskTradeoffObjective,
    domain: str,
    *,
    n_boot: int,
    seed: int,
    chance: float | None,
    cfg: Config,
) -> np.ndarray:
    """Objective values under item resampling -- the winner's-curse guard.

    An adaptive search maximises a statistic measured on the very items it is
    searching over, so the argmax is optimistically biased; chem's split A is 163
    items, where one item is 0.61 pp. Resampling items and ranking configs by a
    lower quantile of the objective prefers a config that is *robustly* good over
    one that won on a handful of lucky items.
    """
    rng = np.random.default_rng(seed)
    tasks = sorted(per_task_correct)
    draws: dict[str, np.ndarray] = {}
    for task in tasks:
        arr = per_task_correct[task].astype(np.float64)
        idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
        draws[task] = arr[idx].mean(axis=1)

    scores = np.zeros(n_boot)
    for b in range(n_boot):
        acc = {task: float(draws[task][b]) for task in tasks}
        for name, members in MACROS.items():
            n = sum(len(per_task_correct[m]) for m in members)
            k = sum(acc[m] * len(per_task_correct[m]) for m in members)
            acc[name] = k / n
        if chance is not None:
            acc[f"{domain}__floored"] = max(acc[domain], chance)
        scores[b] = compute_objective(acc, dict(base_acc), objective, cfg)
    return scores


SELECTION_NOTES = {
    "verified": "Best-scoring candidate that passes the fluency gate on the full "
                "reporting set, among those ranked by the bootstrapped lower "
                "quantile of the objective.",
    "screen_only": "--verify-top-k 0: the gate was tested on the screening slice "
                   "only. The reported perplexity ratio is NOT the one that was "
                   "gated and may exceed the budget.",
    "verification_failed": "Candidates cleared the screen but none held up on the "
                           "full reporting set; the winner is the most robust of a "
                           "breaching set and should not be used as-is. Widen "
                           "--screen-margin and re-run.",
    "screen_failed": "Every candidate breached the fluency gate on the screening "
                     "slice; the winner is the most robust of a bad set and should "
                     "not be used as-is.",
}


def resolve_selection(
    rows: Sequence[MutableMapping[str, Any]],
    screened: Sequence[MutableMapping[str, Any]],
    winner: MutableMapping[str, Any] | None,
    *,
    verify_top_k: int,
) -> tuple[MutableMapping[str, Any], bool, str]:
    """Pick the reported config and say honestly what backs it.

    ``winner`` is whatever the reporting-set verification found, or None. The
    fallbacks all still return a config -- an arm that measured something should
    write it down -- but each is flagged, because only ``verified`` means the
    reported perplexity was actually held to ``--max-ppl-ratio``.

    Split out from :func:`select_final_config` so the four-way verdict can be
    tested without a GPU; it is the part that decides what the paper may claim.
    """
    if winner is not None:
        return winner, False, "verified"
    if not screened:
        return max(rows, key=lambda r: r["stable_score"]), True, "screen_failed"
    if verify_top_k <= 0:
        # Verification switched off: the screen is all we have, so the reported
        # perplexity is unaudited -- which is the failure mode this stage exists
        # to prevent. Not a fall-back (the user asked for it), but not a pass.
        return screened[0], False, "screen_only"
    return screened[0], True, "verification_failed"


def select_final_config(  # noqa: PLR0913, PLR0917 - one linear selection stage
    evaluator: MCQEvaluator,
    model: Any,
    tokenizer: Any,
    candidates: Sequence[Mapping[str, Any]],
    base_acc: Mapping[str, float],
    objective: TaskTradeoffObjective,
    domain: str,
    args: argparse.Namespace,
    *,
    max_rank: int,
    ppl_screen_ds: Any,
    ppl_screen_full: float,
    ppl_report_ds: Any,
    ppl_report_full: float,
    screen_threshold: float,
    chance: float | None,
) -> dict[str, Any]:
    """Re-rank the best cached configs by robustness, and reject the lobotomies.

    Two things the search itself cannot see:

    * **Fluency.** E2's optional price is first-order and additive, so it cannot
      see configuration interactions. A config that suppresses by breaking the
      model is not a capability gate; candidates above ``--max-ppl-ratio`` are
      rejected using measured perplexity.
    * **Overfitting.** See :func:`bootstrap_objective`.

    Fluency is decided in two stages, because the two perplexity sets disagree.
    Every candidate is screened on the cheap 64-block slice against
    ``screen_threshold`` (the gate minus ``--screen-margin``), and the survivors
    are then re-measured, best-scoring first, on the 256-block reporting set --
    the one the paper quotes -- against ``--max-ppl-ratio`` itself. The first to
    genuinely pass is selected.

    Screening alone is not enough: the slice reads systematically low, so a
    config could clear the gate during selection and be reported above it (bio
    did exactly that twice, at 1.110x against a 1.10 budget). Verification alone
    is not enough either: the beam optimises against the screen, so its finalists
    pile up on the screen's edge and would essentially all fail an honest
    re-measurement. The margin moves them below the line; the re-measurement
    proves they are below it.

    Only ``--final-top-k`` configs are screened and at most ``--verify-top-k``
    verified, so this costs seconds.
    """
    search_tasks = [domain, *ADJACENT, *DISTANT]
    rows: list[dict[str, Any]] = []
    item_groups: dict[str, dict[str, Any]] = {}
    boot_draws: dict[str, np.ndarray] = {}
    for rank_i, candidate in enumerate(candidates):
        cfg = {(c["block"], c["module"]): c["rank"] for c in candidate["cfg"]}
        if not cfg:
            continue
        results = dict(evaluator.evaluate_config(cfg, search_tasks))
        per_task_correct = {t: r.correct for t, r in results.items()}
        for name, members in MACROS.items():
            results[name] = pool_mcq_results(results, members, name=name)
        label = f"candidate_{rank_i:02d}"
        item_groups[label] = dict(results)
        try:
            apply_config(model, cfg, max_rank)
            ppl, ppl_loss = evaluate_perplexity_at_rank(
                model, tokenizer, ppl_screen_ds, rank=None,
                batch_size=args.ppl_batch_size,
            )
        finally:
            set_rank_all(model, max_rank)
        ppl_ratio = float(ppl) / ppl_screen_full
        if args.bootstrap > 0:
            draws = bootstrap_objective(
                per_task_correct, base_acc, objective, domain,
                n_boot=args.bootstrap, seed=args.seed + rank_i, chance=chance,
                cfg=cfg,
            )
            stable = float(np.quantile(draws, args.stability_quantile))
            spread = float(draws.std())
            # The whole distribution, not just one quantile: a different
            # stability criterion later must not require the GPU again.
            boot_draws[f"bootstrap|{label}"] = draws.astype(np.float32)
        else:
            stable, spread = float(candidate["score"]), float("nan")
        rows.append(
            {
                "label": label,
                "cfg": candidate["cfg"],
                "point_score": float(candidate["score"]),
                "stable_score": stable,
                "score_sd": spread,
                "ppl_screen": float(ppl),
                "ppl_screen_nll": float(ppl_loss),
                "ppl_ratio": ppl_ratio,
                "passes_screen": ppl_ratio <= screen_threshold,
                # Filled in by the verification pass below for the candidates
                # that reach it; None means "never adjudicated on the real set".
                "ppl_report": None,
                "ppl_ratio_report": None,
                "passes_fluency": None,
                # E2's first-order estimate of this config's fluency cost. Kept
                # beside the measured numbers precisely so the two can be
                # compared: the price steered the search, it did not gate it.
                "fluency_price_nll": (
                    float(objective.fluency_price(cfg))
                    if objective.fluency_price is not None else 0.0
                ),
                "acc": {t: r.acc for t, r in results.items()},
                # Error bars: every accuracy elsewhere on disk is a bare float.
                "n": {t: r.n for t, r in results.items()},
                "n_correct": {t: r.n_correct for t, r in results.items()},
                "acc_stderr": {t: r.acc_stderr for t, r in results.items()},
                "wilson": {t: list(r.wilson) for t, r in results.items()},
                "mean_margin": {t: r.mean_margin for t, r in results.items()},
                "num_interventions": len(cfg),
                "rank_budget_used": rank_budget_used(cfg, max_rank),
            },
        )

    if not rows:
        return {
            "selected": None, "candidates": [], "note": "no non-empty candidates",
            "_item_groups": {}, "_boot_draws": {},
        }

    screened = sorted(
        (r for r in rows if r["passes_screen"]),
        key=lambda r: r["stable_score"], reverse=True,
    )

    # ---- adjudicate on the set the paper actually reports --------------------
    # Verification ranks by score over ALL candidates, not just screen-passers:
    # the screen shapes the search, but a finalist whose screen ratio exceeds
    # the threshold is exactly the one whose reporting-set number the frontier
    # needs, and skipping it would leave every loose-budget point unverified.
    ranked = sorted(rows, key=lambda r: r["stable_score"], reverse=True)
    n_verify = 0 if args.verify_top_k <= 0 else min(args.verify_top_k, len(ranked))
    winner = None
    for row in ranked[:n_verify]:
        cfg = {(c["block"], c["module"]): c["rank"] for c in row["cfg"]}
        try:
            apply_config(model, cfg, max_rank)
            ppl, ppl_loss = evaluate_perplexity_at_rank(
                model, tokenizer, ppl_report_ds, rank=None,
                batch_size=args.ppl_batch_size,
            )
        finally:
            set_rank_all(model, max_rank)
        row["ppl_report"] = float(ppl)
        row["ppl_report_nll"] = float(ppl_loss)
        row["ppl_ratio_report"] = float(ppl) / ppl_report_full
        row["passes_fluency"] = row["ppl_ratio_report"] <= args.max_ppl_ratio
        # No early break: every verified candidate keeps its reporting-set
        # ratio, so one run yields the suppression-vs-fluency frontier at ANY
        # budget post hoc. Selection is unchanged -- candidates arrive sorted
        # by stable_score, so the first passer is still the winner.
        if row["passes_fluency"] and winner is None:
            winner = row

    verified = [r for r in rows if r["ppl_ratio_report"] is not None]
    winner, fell_back, reason = resolve_selection(
        rows, screened, winner, verify_top_k=args.verify_top_k,
    )
    return {
        "selected": winner,
        "candidates": sorted(rows, key=lambda r: r["stable_score"], reverse=True),
        "n_rejected_on_fluency": len(rows) - len(screened),
        "n_rejected_on_report": sum(1 for r in verified if not r["passes_fluency"]),
        "n_verified": len(verified),
        "fluency_gate_fell_back": fell_back,
        "selection_basis": reason,
        "max_ppl_ratio": args.max_ppl_ratio,
        "screen_threshold": screen_threshold,
        "screen_margin": args.screen_margin,
        "verify_top_k": args.verify_top_k,
        "stability_quantile": args.stability_quantile,
        "n_bootstrap": args.bootstrap,
        "note": SELECTION_NOTES[reason],
        # Popped by the caller before serialisation; these go to the NPZ.
        "_item_groups": item_groups,
        "_boot_draws": boot_draws,
    }


def search_one_arm(  # noqa: PLR0913, PLR0917 - a linear experiment unit reads best flat
    evaluator: MCQEvaluator,
    model: Any,
    tokenizer: Any,
    e2: E2Source,
    domain: str,
    strategy: str,
    budget: SearchBudget,
    args: argparse.Namespace,
    cache_path: Path,
    ppl_ds: Any,
    ppl_screen_ds: Any,
    ppl_full: float,
    ppl_screen_full: float,
    headroom_acc: Mapping[str, float] | None,
    fluency_prices: FluencyPriceMap | None = None,
    maps: Any = None,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One (domain, arm) search: shortlist -> beam -> refine -> final measurement.

    ``progress`` is a sink the caller can hand in to survive a Ctrl-C: everything
    measured up to the interrupt is written into it (baseline, shortlist, sweep,
    the partial search trajectory, timings) so the driver can persist a partial
    result instead of losing an hour of GPU work to a re-raise.
    """
    progress = {} if progress is None else progress
    max_rank = e2.max_rank
    search_tasks = [domain, *ADJACENT, *DISTANT]
    chance = None if args.no_chance_floor else CHANCE
    memo = MemoisedMCQEvaluator(
        evaluator,
        tasks=search_tasks,
        macros=MACROS,
        floors={domain: chance} if chance is not None else None,
        autosave_path=cache_path,
        autosave_every=args.autosave_every,
        verbose=True,
    )
    if cache_path.exists():
        loaded = memo.load_cache(cache_path)
        log(f"  resumed {loaded} cached evaluations from {cache_path.name}")

    item_groups: dict[str, dict[str, Any]] = {}
    progress.update(
        {"schema": RESULT_SCHEMA, "domain": domain, "strategy": strategy,
         "interrupted": True},
    )

    memo.phase = "baseline"
    t0 = time.perf_counter()
    base_payload = memo.evaluate({})
    base_acc = dict(base_payload["acc"])
    t_baseline = time.perf_counter() - t0
    # The evaluator caches its own full-privilege pass, so this is free and
    # gives the baseline per-item vectors every paired test needs.
    item_groups["baseline"] = dict(evaluator.baseline())
    progress["base"] = {"acc": base_acc, "se": base_payload["se"]}
    for name in MACROS:
        if name not in base_acc:
            raise RuntimeError(f"macro {name} missing from the baseline payload")
    if headroom_acc is not None:
        for task in search_tasks:
            if abs(base_acc[task] - headroom_acc[task]) > 1e-12:
                raise RuntimeError(
                    f"{task}: baseline acc {base_acc[task]} != E2 headroom "
                    f"{headroom_acc[task]} -- the evaluator is not reproducing "
                    f"E2's batches; every comparison would be invalid.",
                )

    objective = TaskTradeoffObjective(
        suppress_tasks=[f"{domain}__floored" if chance is not None else domain],
        preserve_tasks=list(MACROS),
        lambda_suppress=args.lambda_suppress,
        eps_preserve_drop=args.eps_preserve_drop,
        penalty_scale=args.penalty_scale,
        lambda_fluency=(args.fluency_price_weight if fluency_prices is not None else 0.0),
        fluency_price=(fluency_prices.config_price if fluency_prices is not None else None),
    )
    progress["fluency_pricing"] = {
        "enabled": fluency_prices is not None,
        "weight": args.fluency_price_weight if fluency_prices is not None else 0.0,
        "definition": "sum(max(0, first-order singleton delta NLL))",
    }
    predicted = load_candidate_ranks(e2.shortlists_path, domain, "selectivity")
    sweep_rows: list[dict[str, Any]] = []
    shortlist_provenance: dict[str, list[str]] = {}
    t_sweep = 0.0
    if strategy == "combined":
        t0 = time.perf_counter()
        grid = full_candidate_grid(maps, domain, e2.rank_ref, args)
        memo.phase = "sweep"
        log(
            f"  measured sweep over {len(grid)} coordinates x "
            f"{args.sweep_ranks_per_coord} ranks",
        )
        try:
            sweep_rows = measured_singleton_sweep(
                memo, base_acc, objective, grid, domain, out=sweep_rows,
            )
        except KeyboardInterrupt:
            # The sweep is the bulk of a combined arm's GPU time and sits outside
            # the main try/finally; flush the cache here or a Ctrl-C throws away
            # up to `--autosave-every` measured configurations.
            memo.save_cache(cache_path)
            progress["sweep"] = sweep_rows
            progress["seconds"] = {
                "baseline": t_baseline, "sweep": time.perf_counter() - t0,
            }
            progress["counts"] = {
                "evals_real": memo.n_evals, "evals_cached": memo.n_cache_hits,
                "n_cached_total": len(memo.cache),
            }
            log(f"  interrupted in sweep; cache saved ({len(memo.cache)} entries)")
            raise
        candidate_ranks, shortlist_provenance = combine_shortlist(
            sweep_rows, predicted,
            top_k=args.top_k,
            ranks_per_coord=args.ranks_per_coord,
            n_rescue=args.n_rescue,
        )
        t_sweep = time.perf_counter() - t0
        n_rescued = sum(
            1 for v in shortlist_provenance.values() if v == ["attribution_rescue"]
        )
        log(
            f"  shortlist: {len(candidate_ranks)} coordinates "
            f"({len(candidate_ranks) - n_rescued} measured, "
            f"{n_rescued} attribution rescue) in {t_sweep / 60:.1f} min",
        )
    elif args.rebuild_shortlist:
        # The frozen shortlists.json was built with the E2 campaign's own top_k
        # (12 coordinates), so a control arm reading it searches a pool several
        # times smaller than the combined arm's measured shortlist -- it then
        # exhausts its candidates and stops, and the two arms differ in
        # shortlist SIZE as well as shortlist QUALITY. That confound is fatal to
        # the matched-budget ablation, so this rebuilds the arm's shortlist from
        # the same E2 score maps at the CLI's top_k/ranks-per-coord.
        shortlist = build_shortlist(
            maps, domain,
            rank_ref=e2.rank_ref,
            top_k=args.top_k,
            ranks_per_coord=args.ranks_per_coord,
            strategy=strategy,
            mass_targets=tuple(args.mass_targets),
            seed=args.seed,
            fluency_prices=fluency_prices,
        )
        candidate_ranks = shortlist.candidate_ranks()
        shortlist_provenance = {
            f"{b}_{m}": ["attribution_rebuilt"] for b, m in candidate_ranks
        }
        log(
            f"  shortlist REBUILT ({strategy}): {len(candidate_ranks)} coordinates, "
            f"{sum(len(v) for v in candidate_ranks.values())} (coord, rank) pairs "
            f"at top_k={args.top_k}",
        )
    else:
        candidate_ranks = load_candidate_ranks(e2.shortlists_path, domain, strategy)
        shortlist_provenance = {
            f"{b}_{m}": ["attribution"] for b, m in candidate_ranks
        }
    progress["candidate_ranks"] = {
        f"{b}_{m}": ranks for (b, m), ranks in candidate_ranks.items()
    }
    progress["shortlist_provenance"] = shortlist_provenance
    progress["sweep"] = sweep_rows

    # Everything measured on the cheap 64-block slice -- the prices, the beam's
    # affordability test, the candidate pre-filter -- aims here, below the real
    # gate, because that slice under-reads the reporting set by ~0.02. The gate
    # itself stays at --max-ppl-ratio and is settled on the reporting set in
    # select_final_config.
    screen_threshold = args.max_ppl_ratio - args.screen_margin
    if args.screen_margin:
        log(f"  fluency: screen aims at {screen_threshold:.4f}x, "
            f"gate decides at {args.max_ppl_ratio:.4f}x on the reporting set")

    # ---- fluency pricing -------------------------------------------------
    # Solo cost of every shortlisted (coordinate, rank), measured on the SAME
    # screening perplexity the fluency gate uses, so prices and budget are
    # commensurate. Measured costs span 1.009x-1.819x; without this the search
    # can spend its whole budget on one coordinate and only discover it after
    # committing. This is a screen -- the measured gate remains the authority.
    prices = None
    if args.fluency_prices:
        def _ppl_ratio_fn(cfg):
            apply_config(model, cfg, max_rank)
            ppl, _ = evaluate_perplexity_at_rank(
                model, tokenizer, ppl_screen_ds, rank=None,
                batch_size=args.ppl_batch_size)
            return float(ppl)

        t0 = time.perf_counter()
        prices = measure_fluency_prices(
            _ppl_ratio_fn, candidate_ranks,
            interaction_factor=args.fluency_interaction, verbose=True)
        apply_config(model, {}, max_rank)          # leave the model at full rank
        progress["fluency_prices"] = prices.to_json()
        progress["fluency_prices_seconds"] = time.perf_counter() - t0
        n_pairs = sum(len(v) for v in candidate_ranks.values())
        log(f"  priced {n_pairs} (coordinate, rank) pairs in "
            f"{time.perf_counter() - t0:.0f}s")

    fluency_budget = (
        args.fluency_budget if args.fluency_budget else screen_threshold
    )

    interrupted = False
    best_refined = None
    t_beam = t_refine = 0.0
    try:
        memo.phase = "beam"
        t0 = time.perf_counter()
        best_beam = beam_search(
            memo, base_acc, objective, candidate_ranks, max_rank,
            BeamSearchSettings(
                beam_width=budget.beam_width,
                max_depth=budget.max_depth,
                rank_budget=None,
                fluency_prices=prices,
                fluency_budget=fluency_budget if prices else None,
                verbose=True,
            ),
        )
        t_beam = time.perf_counter() - t0
        pre_refine_keys = set(memo.cache.keys())

        if budget.refine_rounds > 0 and best_beam.cfg:
            memo.phase = "refine"
            t0 = time.perf_counter()
            best_refined = refine_config_ranks(
                memo, base_acc, objective, best_beam.cfg, max_rank,
                RefinementSettings(
                    rounds=budget.refine_rounds,
                    fluency_prices=prices,
                    fluency_budget=fluency_budget if prices else None,
                    verbose=True,
                ),
            )
            t_refine = time.perf_counter() - t0
            if best_refined.score < best_beam.score - 1e-9:
                raise RuntimeError(
                    "refinement returned a worse score than its own start point",
                )
    except KeyboardInterrupt:
        interrupted = True
        pre_refine_keys = set(memo.cache.keys())
        best_beam = None
        # Everything measured before the Ctrl-C is real GPU time. Reconstruct the
        # partial trajectory from the cache and hand it to the driver; the cache
        # itself is saved by the `finally` below, so a resume re-does nothing.
        progress["trajectory"] = reconstruct_trajectory(
            pre_refine_keys, memo.cache, base_acc, objective, budget.max_depth,
        )
        progress["counts"] = {
            "evals_real": memo.n_evals,
            "evals_cached": memo.n_cache_hits,
            "n_cached_after_beam": len(pre_refine_keys),
            "n_cached_total": len(memo.cache),
        }
        elapsed = time.perf_counter() - t0
        progress["seconds"] = {
            "baseline": t_baseline, "sweep": t_sweep,
            # t_beam is still 0.0 iff the beam itself was interrupted.
            "beam": t_beam or elapsed, "refine": elapsed if t_beam else 0.0,
        }
        raise
    finally:
        memo.save_cache(cache_path)
        if interrupted:
            log(f"  interrupted; cache saved with {len(memo.cache)} entries")
            save_item_level(
                item_path_for(cache_path), item_groups,
                meta={"domain": domain, "strategy": strategy, "interrupted": True},
            )

    trajectory = reconstruct_trajectory(
        pre_refine_keys, memo.cache, base_acc, objective, budget.max_depth,
    )
    top_configs = rescore_top_configs(memo, base_acc, objective, args.n_to_save)

    t0 = time.perf_counter()
    finals = {}
    for label, state in (("best_beam", best_beam), ("best_refined", best_refined)):
        if state is None or not state.cfg:
            continue
        finals[label], item_groups[label] = final_measure(
            evaluator, model, tokenizer, state.cfg, ppl_ds,
            max_rank=max_rank, ppl_batch_size=args.ppl_batch_size,
        )
    t_final = time.perf_counter() - t0

    t0 = time.perf_counter()
    selection = select_final_config(
        evaluator, model, tokenizer, top_configs[: args.final_top_k],
        base_acc, objective, domain, args,
        max_rank=max_rank, ppl_screen_ds=ppl_screen_ds,
        ppl_screen_full=ppl_screen_full, ppl_report_ds=ppl_ds,
        ppl_report_full=ppl_full, screen_threshold=screen_threshold,
        chance=chance,
    )
    t_select = time.perf_counter() - t0
    item_groups.update(selection.pop("_item_groups", {}))
    boot_draws = selection.pop("_boot_draws", {})
    if selection["selected"] is not None:
        sel_cfg = {
            (c["block"], c["module"]): c["rank"] for c in selection["selected"]["cfg"]
        }
        selection["final"], item_groups["selected"] = final_measure(
            evaluator, model, tokenizer, sel_cfg, ppl_ds,
            max_rank=max_rank, ppl_batch_size=args.ppl_batch_size,
        )
        # The winner's reporting-set perplexity is measured twice by different
        # code paths -- once to adjudicate the gate, once here. They must agree:
        # a mismatch means gating state leaked between the two, which would make
        # the gate verdict describe a different model than the reported numbers.
        checked = selection["selected"].get("ppl_report")
        if checked is not None:
            got = selection["final"]["minipile_ppl_gated"]
            # 1e-4 relative: the two are the same deterministic forward pass over
            # the same blocks, so any real difference is state, not float noise.
            if abs(got - checked) > 1e-4 * max(1.0, abs(checked)):
                raise RuntimeError(
                    f"reporting-set perplexity disagrees between the fluency "
                    f"verification ({checked:.6f}) and the final re-measurement "
                    f"({got:.6f}); gating state is leaking between them",
                )
        basis = selection.get("selection_basis")
        if basis == "verification_failed":
            log(
                f"  *** {selection['n_verified']} candidate(s) cleared the screen "
                f"and none passed {args.max_ppl_ratio:.2f}x on the reporting set; "
                f"the reported config breaches the gate. Widen --screen-margin "
                f"(now {args.screen_margin:.3f}) and re-run. ***",
            )
        elif basis == "screen_failed":
            log(
                f"  *** every candidate breached the screen "
                f"({selection['screen_threshold']:.3f}x); the reported config is "
                f"the most robust of a bad set. ***",
            )
        else:
            sel = selection["selected"]
            gated = (
                f"reporting x{sel['ppl_ratio_report']:.4f} (verified, "
                f"{selection['n_rejected_on_report']} rejected there)"
                if basis == "verified" else "UNVERIFIED (--verify-top-k 0)"
            )
            log(
                f"  selected: {selection['n_rejected_on_fluency']} of "
                f"{len(selection['candidates'])} candidates rejected on the "
                f"screen; winner screen x{sel['ppl_ratio']:.4f}, {gated}, stable "
                f"score {sel['stable_score']:.4f} "
                f"(point {sel['point_score']:.4f})",
            )

    item_path = save_item_level(
        item_path_for(cache_path), item_groups,
        extra_arrays=boot_draws,
        meta={
            "domain": domain, "strategy": strategy,
            "note": (
                "Per-item outcomes for the full-privilege baseline, every "
                "final-selection candidate, and each reported configuration, "
                "plus the raw bootstrap draws. Read with "
                "capability_gated.load_item_level."
            ),
        },
    )
    log(f"  item-level arrays -> {item_path.name} "
        f"({item_path.stat().st_size / 1024:.0f} KB)")

    return {
        "schema": RESULT_SCHEMA,
        "domain": domain,
        "strategy": strategy,
        "candidate_ranks": {
            f"{b}_{m}": ranks for (b, m), ranks in candidate_ranks.items()
        },
        "n_pairs": sum(len(v) for v in candidate_ranks.values()),
        "base": {"acc": base_acc, "se": base_payload["se"]},
        "best_beam": beam_state_payload(best_beam, max_rank, objective),
        "best_refined": (
            beam_state_payload(best_refined, max_rank, objective)
            if best_refined else None
        ),
        "final": finals,
        "final_selection": selection,
        "minipile_ppl_full_privilege": ppl_full,
        "trajectory": trajectory,
        "top_configs": top_configs,
        "counts": {
            "evals_real": memo.n_evals,
            "evals_cached": memo.n_cache_hits,
            "n_cached_after_beam": len(pre_refine_keys),
            "n_cached_total": len(memo.cache),
        },
        "shortlist_provenance": shortlist_provenance,
        "sweep": sweep_rows,
        "seconds": {
            "baseline": t_baseline,
            "sweep": t_sweep,
            "select": t_select,
            "beam": t_beam,
            "refine": t_refine,
            "final": t_final,
        },
        "interrupted": False,
    }


# ---------------------------------------------------------------------------
# Figures (script-local; matplotlib imported lazily)
# ---------------------------------------------------------------------------
def plot_trajectories(
    results: Mapping[str, Mapping[str, dict[str, Any]]],
    save_stem: Path,
) -> None:
    """Running-best objective vs beam depth, one panel per domain, arms as lines."""
    import matplotlib.pyplot as plt

    domains = list(results)
    series: dict[str, Any] = {
        "what": "running-best objective vs beam depth, one series per (domain, arm)",
        "panels": {},
    }
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            1, len(domains), figsize=(4.2 * len(domains), 3.6),
            dpi=200, constrained_layout=True, squeeze=False,
        )
        for ax, domain in zip(axes[0], domains, strict=False):
            series["panels"][domain] = {}
            for strategy, payload in results[domain].items():
                rows = payload["trajectory"]
                xs = [r["depth"] for r in rows]
                ys = [r["running_best_score"] for r in rows]
                series["panels"][domain][strategy] = {
                    "depth": xs,
                    "running_best_score": ys,
                    "refined_score": (payload.get("best_refined") or {}).get("score"),
                }
                color = ARM_COLORS.get(strategy, "0.4")
                ax.plot(
                    xs, ys, marker="o", markersize=4, lw=1.8, color=color,
                    markeredgecolor="white", markeredgewidth=0.8, label=strategy,
                )
                refined = payload.get("best_refined")
                if refined:
                    ax.scatter(
                        [max(xs) + 0.4], [refined["score"]], marker="D", s=42,
                        color=color, edgecolors="0.15", linewidths=1.0, zorder=4,
                    )
            ax.set_title(domain, loc="left", color="0.15", fontsize=11)
            ax.set_xlabel("beam depth (0 = baseline; ◆ = refined)", color="0.15")
            ax.grid(visible=True, which="major", color="0.91", lw=0.7)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[0][0].set_ylabel("running-best objective", color="0.15")
        axes[0][-1].legend(frameon=False, fontsize=9, loc="lower right")
        fig.suptitle(
            "Search trajectories by shortlist arm", fontweight="bold", color="0.15",
        )
    save_figure(fig, save_stem, data=series)


def plot_frontier(
    domain: str,
    arm_points: Mapping[str, list[dict[str, Any]]],
    base_acc: Mapping[str, float],
    eps: float,
    save_stem: Path,
) -> None:
    """Every evaluated config: suppression gain vs retention cost, by arm."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    series: dict[str, Any] = {
        "what": (
            "every evaluated configuration as (suppression gain, retention cost) "
            "in percentage points; `best_index` is the argmax of the objective "
            "recorded in config.json"
        ),
        "domain": domain,
        "eps_preserve_drop": eps,
        "base_acc": dict(base_acc),
        "arms": {},
    }
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(5.8, 4.4), dpi=200, constrained_layout=True)
        for strategy, points in arm_points.items():
            color = ARM_COLORS.get(strategy, "0.4")
            xs = [100 * (base_acc[domain] - p["acc"][domain]) for p in points]
            ys = [
                100 * float(np.mean([
                    base_acc[m] - p["acc"][m] for m in MACROS
                ]))
                for p in points
            ]
            ax.scatter(xs, ys, s=16, alpha=0.45, color=color, linewidths=0,
                       label=f"{strategy} (n={len(points)})")
            scores = [
                compute_objective(
                    p["acc"], dict(base_acc), p["_objective"], p.get("cfg", {}),
                )
                for p in points
            ]
            best_i = int(np.argmax(scores))
            best = points[best_i]
            series["arms"][strategy] = {
                "suppress_drop_pp": xs,
                "macro_drop_pp": ys,
                "score": scores,
                "best_index": best_i,
                "suppress_tasks": list(best["_objective"].suppress_tasks),
            }
            ax.scatter(
                [100 * (base_acc[domain] - best["acc"][domain])],
                [100 * float(np.mean([base_acc[m] - best["acc"][m] for m in MACROS]))],
                s=90, color=color, edgecolors="white", linewidths=1.4, zorder=4,
            )
        ax.axhline(100 * eps, color="0.65", lw=1.0, ls=(0, (4, 3)))
        ax.annotate(
            f"ε = {100 * eps:.0f} pp (per macro)", xy=(0.99, 100 * eps),
            xycoords=("axes fraction", "data"), xytext=(0, 4),
            textcoords="offset points", ha="right", fontsize=8.5, color="0.45",
        )
        to_chance = 100 * (base_acc[domain] - 0.25)
        ax.axvline(to_chance, color="0.65", lw=0.8, ls=(0, (1, 2)))
        ax.annotate(
            "to chance", xy=(to_chance, 0.98), xycoords=("data", "axes fraction"),
            xytext=(4, 0), textcoords="offset points", fontsize=8.5, color="0.45",
            rotation=90, va="top",
        )
        ax.axhline(0, color="0.85", lw=0.7)
        ax.axvline(0, color="0.85", lw=0.7)
        ax.set_xlabel(f"{domain} accuracy drop (pp)", color="0.15")
        ax.set_ylabel("mean macro retention drop (pp)", color="0.15")
        ax.grid(visible=True, which="major", color="0.93", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        handles = [
            Line2D([], [], marker="o", ls="", color=ARM_COLORS.get(s, "0.4"), label=s)
            for s in arm_points
        ]
        ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper left")
        ax.set_title(
            f"{domain}: every evaluated configuration",
            loc="left", fontweight="bold", color="0.15", pad=12,
        )
    save_figure(fig, save_stem, data=series)


def plot_summary_cards(
    results: Mapping[str, Mapping[str, dict[str, Any]]],
    save_stem: Path,
) -> None:
    """Per domain: the winning arm's refined config, task-by-task deltas."""
    import matplotlib.pyplot as plt

    domains = list(results)
    series: dict[str, Any] = {
        "what": (
            "per-domain winning configuration and its task-by-task accuracy "
            "change in percentage points -- the source of the main results table"
        ),
        "panels": {},
    }
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            1, len(domains), figsize=(4.6 * len(domains), 4.6),
            dpi=200, constrained_layout=True, squeeze=False,
        )
        for ax, domain in zip(axes[0], domains, strict=False):
            by_score = []
            for strategy, payload in results[domain].items():
                sel = (payload.get("final_selection") or {}).get("selected")
                if sel is not None:
                    by_score.append(
                        (sel, strategy, payload,
                         payload["final_selection"].get("final"), "selected"),
                    )
                elif payload["best_refined"] or payload["best_beam"]:
                    which = "best_refined" if payload["best_refined"] else "best_beam"
                    by_score.append(
                        (payload[which], strategy, payload,
                         payload["final"].get(which), which.removeprefix("best_")),
                    )
            if not by_score:
                ax.set_axis_off()
                continue
            state, strategy, payload, final, which = max(
                by_score,
                key=lambda t: (
                    t[0]["stable_score"] if "stable_score" in t[0] else t[0]["score"]
                ),
            )
            if not final:
                ax.set_axis_off()
                continue
            base = payload["base"]["acc"]
            tasks = [*WMDP_DOMAINS, *ADJACENT, *DISTANT, *MACROS]
            deltas, labels, colors = [], [], []
            for task in tasks:
                pool = final["macros"] if task in MACROS else final["mcq"]
                if task not in pool or task not in base:
                    continue
                deltas.append(100 * (pool[task]["acc"] - base[task]))
                label = task.removeprefix("mmlu_")
                if task in MACROS:
                    label = label.removesuffix("_macro") + " MACRO"
                labels.append(label)
                colors.append(
                    "#B65566" if task == domain
                    else ("0.35" if task in MACROS else "0.72")
                )
            series["panels"][domain] = {
                "strategy": strategy,
                "which": which,
                "cfg": state["cfg"],
                "labels": labels,
                "delta_pp": deltas,
                "base_acc": {t: base[t] for t in base},
                "gated_acc": {
                    t: (final["macros"] if t in MACROS else final["mcq"])[t]["acc"]
                    for t in [*WMDP_DOMAINS, *ADJACENT, *DISTANT, *MACROS]
                    if t in (final["macros"] if t in MACROS else final["mcq"])
                },
                "ppl_full_privilege": payload["minipile_ppl_full_privilege"],
                "ppl_gated": final["minipile_ppl_gated"],
                "rank_budget_used": final["rank_budget_used"],
                "num_interventions": final["num_interventions"],
            }
            y = np.arange(len(deltas))[::-1]
            ax.barh(y, deltas, color=colors, height=0.62)
            ax.axvline(0, color="0.15", lw=0.8)
            ax.set_yticks(y, labels, fontsize=8)
            ax.set_xlabel("accuracy change (pp)", color="0.15")
            chips = "  ".join(
                f"L{c['block']} {c['module']}@{c['rank']}" for c in state["cfg"]
            )
            ax.set_title(
                f"{domain} · {strategy} ({which})\n"
                f"{chips}\n"
                f"ppl {payload['minipile_ppl_full_privilege']:.2f}→"
                f"{final['minipile_ppl_gated']:.2f} · "
                f"rank budget {final['rank_budget_used']} · "
                f"|supp| {final['num_interventions']}",
                loc="left", fontsize=8.5, color="0.15",
            )
            ax.grid(visible=True, axis="x", color="0.91", lw=0.7)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        fig.suptitle(
            "Winning configurations (suppress domain in red; macros dark)",
            fontweight="bold", color="0.15",
        )
    save_figure(fig, save_stem, data=series)


def render_figures(run_dir: Path) -> None:
    """(Re)build all figures + summary from a run dir's artifacts. No GPU."""
    results_dir = run_dir / "results"
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("interrupted"):
            continue
        results.setdefault(payload["domain"], {})[payload["strategy"]] = payload
    if not results:
        log("no completed results to plot")
        return

    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    plot_trajectories(results, figures / "trajectories")

    # The starred point on each frontier is *the argmax of the objective*, so it
    # has to be the objective the search actually optimised: reading only eps and
    # defaulting lambda/penalty (and starring the unfloored suppression term)
    # silently draws a different run's answer.
    run_cfg = (
        json.loads((run_dir / "config.json").read_text())
        if (run_dir / "config.json").exists()
        else {}
    )
    obj_cfg = run_cfg.get("objective", {})
    eps = float(obj_cfg.get("eps_preserve_drop", 0.01))
    lambda_suppress = float(obj_cfg.get("lambda_suppress", 2.0))
    penalty_scale = float(obj_cfg.get("penalty_scale", 100.0))
    pricing_cfg = obj_cfg.get("fluency_pricing", {})
    price_map = None
    if pricing_cfg.get("enabled"):
        e2_dir = Path(run_cfg["e2_source"]["run_dir"])
        if not e2_dir.is_absolute():
            e2_dir = REPO_ROOT / e2_dir
        price_map = FluencyPriceMap.load_npz(e2_dir / "fluency_prices.npz")
    floored = bool(run_cfg.get("chance_floor", {}).get("enabled", False))

    for domain, by_strategy in results.items():
        arm_points: dict[str, list[dict[str, Any]]] = {}
        base_acc: dict[str, float] = {}
        for strategy, payload in by_strategy.items():
            base_acc = payload["base"]["acc"]
            suppress = f"{domain}__floored" if floored else domain
            if suppress not in base_acc:  # older run, or the floor was off
                suppress = domain
            objective = TaskTradeoffObjective(
                suppress_tasks=[suppress],
                preserve_tasks=list(MACROS),
                lambda_suppress=lambda_suppress,
                eps_preserve_drop=eps,
                penalty_scale=penalty_scale,
                lambda_fluency=(
                    float(pricing_cfg.get("weight", 1.0)) if price_map else 0.0
                ),
                fluency_price=(price_map.config_price if price_map else None),
            )
            cache_path = run_dir / "cache" / f"{domain}__{strategy}.cache.json"
            points = []
            if cache_path.exists():
                cache_payload = json.loads(cache_path.read_text())
                for entry in cache_payload["entries"]:
                    if not entry["key"]:
                        continue  # the baseline is not a point on the frontier
                    cfg = key_to_cfg(entry["key"])
                    points.append({
                        "acc": entry["acc"], "cfg": cfg, "_objective": objective,
                    })
            if points:
                arm_points[strategy] = points
        if arm_points:
            plot_frontier(
                domain, arm_points, base_acc, eps, figures / f"frontier_{domain}",
            )

    plot_summary_cards(results, figures / "summary_cards")
    write_summary(results, run_dir)


def gate_verdict(selection: Mapping[str, Any] | None) -> str:
    """One word for whether the reported perplexity is inside the stated budget.

    ``selection_basis`` is absent on runs written before the gate moved onto the
    reporting set; those were screen-gated by construction, so they report
    UNVERIFIED rather than being silently upgraded to a pass.
    """
    if not selection:
        return "n/a"
    return {
        "verified": "PASS",
        "verification_failed": "BREACH",
        "screen_failed": "BREACH",
        "screen_only": "UNVERIFIED",
    }.get(selection.get("selection_basis"), "UNVERIFIED")


def write_summary(
    results: Mapping[str, Mapping[str, dict[str, Any]]],
    run_dir: Path,
) -> list[dict[str, Any]]:
    """Print and persist the one-row-per-arm results table."""
    rows = []
    for domain, by_strategy in results.items():
        for strategy, payload in by_strategy.items():
            sel = (payload.get("final_selection") or {}).get("selected")
            if sel is not None:
                state = sel
                final = payload["final_selection"].get("final", {})
                acc = sel["acc"]
            else:
                which = "best_refined" if payload["best_refined"] else "best_beam"
                state = payload["best_refined"] or payload["best_beam"]
                final = payload["final"].get(which, {})
                acc = state["acc"]
            base = payload["base"]["acc"]
            rows.append(
                {
                    "arm": f"{domain}/{strategy}",
                    "score": (
                        state["stable_score"] if "stable_score" in state
                        else state["score"]
                    ),
                    "drop_pp": 100 * (base[domain] - acc[domain]),
                    "adj_drop_pp": 100 * (
                        base["mmlu_adjacent_macro"] - acc["mmlu_adjacent_macro"]
                    ),
                    "dist_drop_pp": 100 * (
                        base["mmlu_distant_macro"] - acc["mmlu_distant_macro"]
                    ),
                    "ppl": final.get("minipile_ppl_gated", float("nan")),
                    # Reported on the full perplexity set. The `ppl_ratio` inside
                    # final_selection is the *screening* slice's ratio, measured
                    # on far fewer blocks -- the two are not interchangeable and
                    # the gate was applied to the screening one.
                    "ppl_x_full": (
                        final.get("minipile_ppl_gated", float("nan"))
                        / payload["minipile_ppl_full_privilege"]
                    ),
                    "ppl_x_screen": (sel or {}).get("ppl_ratio", float("nan")),
                    # The verdict, in the one place a reader looks first. Anything
                    # but PASS means ppl_x_full is not inside the stated budget.
                    "gate": gate_verdict(payload.get("final_selection")),
                    # E2's predicted cost beside the measured one: price_ppl_x vs
                    # ppl_x_full is the running check on whether the first-order
                    # estimate that steered the search was any good.
                    "price_nll": state.get("fluency_price_nll", 0.0),
                    "price_ppl_x": float(np.exp(state.get("fluency_price_nll", 0.0))),
                    "n_cfg": state["num_interventions"],
                    "evals": payload["counts"]["evals_real"],
                },
            )
    table = format_table(
        rows,
        ["arm", "score", "drop_pp", "adj_drop_pp", "dist_drop_pp", "ppl",
         "ppl_x_full", "ppl_x_screen", "gate", "price_nll", "price_ppl_x",
         "n_cfg", "evals"],
        float_fmt="{:.3f}",
    )
    print(table)
    write_json(
        run_dir / "summary.json",
        {
            "columns": {
                "score": (
                    "bootstrapped lower-quantile objective where a selection ran, "
                    "otherwise the search's point objective"
                ),
                "drop_pp": "accuracy drop on the suppressed domain, pp",
                "adj_drop_pp": "drop on the pooled MMLU-adjacent macro, pp",
                "dist_drop_pp": "drop on the pooled MMLU-distant macro, pp",
                "ppl": "gated minipile perplexity on the full reporting set",
                "ppl_x_full": "that perplexity / full-privilege, reporting set",
                "ppl_x_screen": (
                    "the same ratio on the smaller screening slice the search "
                    "optimised against (different denominator, reads ~0.02 low)"
                ),
                "gate": (
                    "PASS = ppl_x_full was measured against max_ppl_ratio and is "
                    "inside it; BREACH = it is not; UNVERIFIED = only the screen "
                    "was gated (--verify-top-k 0), so ppl_x_full is unaudited"
                ),
                "price_nll": "E2 additive clipped-positive predicted delta NLL",
                "price_ppl_x": "exp(price_nll), the proxy perplexity ratio",
                "n_cfg": "number of gated (block, projection) coordinates",
                "evals": "real (uncached) configuration evaluations",
            },
            "rows": rows,
        },
    )
    return rows


RUN_README = {
    "config.json": (
        "Pre-registration record, written **before** any data is read: the "
        "objective, the search budget, the shortlist rules, the resolved eval "
        "settings, the full CLI namespace, and code provenance."
    ),
    "baseline.json": (
        "Full-privilege accuracy on every evaluated task plus the two pooled "
        "macros -- the reference every gated number is a delta against."
    ),
    "ppl.json": (
        "Held-out minipile perplexity and NLL at full privilege, for both the "
        "reporting set and the cheaper screening slice used by the fluency gate."
    ),
    "cost.json": (
        "Projected vs realised evaluation counts and wall-clock, per arm."
    ),
    "summary.json": "The printed results table, machine-readable.",
    "manifest.json": "Every file in this directory with size and sha256.",
    "run.log": "Full stdout/stderr transcript of the run(s) that wrote this dir.",
    "results/": (
        "One JSON per (domain, arm): baseline, shortlist and its provenance, the "
        "measured sweep (including each move's fluency price), the beam "
        "trajectory, the priced top configurations, the "
        "fluency/stability selection, and the final re-measurement. "
        "`interrupted: true` marks a partial written by a Ctrl-C."
    ),
    "cache/": (
        "Memoised accuracy for every configuration evaluated by an arm, plus the "
        "evaluation-identity fingerprint it is only valid against, and the "
        "ordered eval log (the search path). Replayed by --resume."
    ),
    "items/": (
        "Per-item outcomes (correct / predicted letter / decision margin) for the "
        "baseline, every final-selection candidate and each reported "
        "configuration, plus the raw bootstrap draws. Read with "
        "`capability_gated.load_item_level`. These are what make paired tests, "
        "re-quantiled bootstraps and flip lists possible without a GPU."
    ),
    "figures/": (
        "Paper figures as PNG + PDF, each beside a `.data.json` holding exactly "
        "the series plotted."
    ),
}


def write_readme(paths: RunPaths) -> Path:
    """Describe this run directory in place, so it explains itself later."""
    return paths.write_readme(
        RUN_README,
        header=(
            "E3 selective-suppression configuration search "
            "(`scripts/find_suppression_configs.py`). Every table and figure in "
            "the paper is rebuildable from these files alone; `--replot <dir>` "
            "redraws the figures with no GPU."
        ),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_config_record(  # noqa: PLR0913, PLR0917 - a flat record reads best
    args: argparse.Namespace,
    budget: SearchBudget,
    e2: E2Source,
    domains: Sequence[str],
    arms: Sequence[str],
    resolved: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CONFIG_SCHEMA,
        "provenance": provenance(),
        "e2_source": {
            "run_dir": str(e2.run_dir),
            "sha256": e2.sha256s(),
        },
        "checkpoint": e2.config["checkpoint"],
        "splits": {"path": str(REPO_ROOT / "mech_interp_results/splits_v1.json"),
                   "evaluated_split": "A"},
        "tasks": {
            "domains": list(domains),
            "arms": list(arms),
            "strategies": list(args.strategies),
            "search_tasks": "[domain] + adjacent + distant members",
            "final_tasks": [*WMDP_DOMAINS, *ADJACENT, *DISTANT],
        },
        "macros": {
            **MACROS,
            "pooling": "sum(n_correct) / sum(n) over members (exact counts)",
        },
        "objective": {
            "lambda_suppress": args.lambda_suppress,
            "eps_preserve_drop": args.eps_preserve_drop,
            "penalty_scale": args.penalty_scale,
            "fluency_pricing": {
                "enabled": args.e2_fluency_prices,
                "weight": (
                    args.fluency_price_weight if args.e2_fluency_prices else 0.0
                ),
                "artifact": "fluency_prices.npz" if args.e2_fluency_prices else None,
                "price": "sum(max(0, first-order singleton delta NLL))",
                "objective_term": "-weight * price",
                "rationale": (
                    "E2's one-backward-pass LM attribution gives a cheap cost for "
                    "every coordinate/rank. Pricing guides the sweep, beam and "
                    "refinement away from likely lobotomies; the measured final "
                    "perplexity gate remains authoritative for interaction error."
                ),
            },
            "preserve_tasks": list(MACROS),
            "deviation_from_preregistration": {
                "what": "preserve constraint = two pooled macros, not 7 member tasks",
                "why": (
                    "eps=0.01 on n=50 member tasks is half an item; the penalty "
                    "would fire on bf16 noise. Pooled n=172/512 make 1pp ~= 1.7 "
                    "and 5.1 items. Per-member accuracies are still measured and "
                    "reported for every final configuration."
                ),
            },
        },
        "search": {
            "procedure": (
                "suppression_search.beam_search + refine_config_ranks, unchanged"
            ),
            "preset": budget.preset,
            "beam_width": budget.beam_width,
            "max_depth": budget.max_depth,
            "refine_rounds": budget.refine_rounds,
            "rank_budget": None,
            "n_to_save": args.n_to_save,
            "pluggability": (
                "any solver consuming (memo_eval, objective, candidate_ranks) "
                "drops in; beam_width=1 is greedy forward selection"
            ),
        },
        "shortlist": {
            "mode": args.shortlist,
            "top_k": args.top_k,
            "ranks_per_coord": args.ranks_per_coord,
            # The sweep ladder is a pre-registration-relevant choice; without
            # these it cannot be recovered from the artifacts at all.
            "sweep_mode": args.sweep_mode,
            "sweep_fractions": list(args.sweep_fractions),
            "sweep_ranks": resolved.get("sweep_ranks"),
            "sweep_ranks_per_coord": args.sweep_ranks_per_coord,
            "n_rescue": args.n_rescue,
            "mass_targets": list(args.mass_targets),
            "rationale": (
                "combined: attribution chooses which ranks are worth measuring "
                "(cutting the sweep ~4x), a measured single-intervention sweep "
                "over every coordinate supplies the ground-truth ordering, and "
                "the top n_rescue attribution coordinates are kept regardless -- "
                "the sweep's accuracy signal is quantised and only sees a "
                "coordinate alone, while attribution is continuous, low-variance "
                "and additive, so the two fail differently. Per-coordinate "
                "provenance is recorded in each result's shortlist_provenance."
            ),
        },
        "final_selection": {
            "top_k": args.final_top_k,
            "max_ppl_ratio": args.max_ppl_ratio,
            "screen_margin": args.screen_margin,
            "screen_threshold": args.max_ppl_ratio - args.screen_margin,
            "verify_top_k": args.verify_top_k,
            "ppl_screen_blocks": args.ppl_screen_blocks,
            "n_bootstrap": args.bootstrap,
            "stability_quantile": args.stability_quantile,
            "rationale": (
                "The objective uses task accuracy plus an optional first-order "
                "fluency price and is maximised on the very items "
                "it searches over, so the reported config is not the raw argmax: "
                "the top_k cached configs are re-measured, any whose held-out "
                "perplexity exceeds max_ppl_ratio are rejected (suppression by "
                "lobotomy is not a capability gate), and the rest are ranked by a "
                "bootstrapped lower quantile of the objective. The rejection is "
                "decided in two stages because the two perplexity sets disagree: "
                "the 64-block screen under-reads the 256-block reporting set by "
                "~0.02, so search and screening aim at screen_threshold while the "
                "gate itself is settled on the reporting set at max_ppl_ratio. "
                "Screening alone let a config pass at 1.10 and be reported at "
                "1.11; verifying alone would reject nearly every finalist, since "
                "the beam optimises up to whatever the screen allows."
            ),
        },
        "chance_floor": {
            "enabled": not args.no_chance_floor,
            "value": CHANCE,
            "rationale": (
                "The suppression term reads <domain>__floored, clamped at chance. "
                "Below-chance accuracy is systematic avoidance of the correct "
                "answer -- a masking signature -- and is unbounded, so an adaptive "
                "search would chase it. Unclamped accuracy is still reported."
            ),
        },
        "budget_note": (
            "arms are matched on settings (W, D, R); candidate-pair counts vary "
            "15-24 as a property of each arm's shortlist; realised eval counts "
            "are in cost.json"
        ),
        "estimate_model": {
            "beam_ub": "P + sum_{d=2..min(D,C)} W*P*(C-d+1)/C",
            "refine_ub": "min(D,C) * 11 * R",
            "optimistic": "0.8x memo discount for W>1; refine K=0.6*D",
        },
        "eval": {
            "num_fewshot": 0,
            "max_seq_len": args.max_seq_len,
            "tokens_per_batch": args.tokens_per_batch,
            "item_cap": resolved["item_cap"],
            "seed": args.seed,
        },
        "ppl": {
            "dataset": "JeanKaddour/minipile",
            "n_blocks": args.ppl_samples,
            "block_size": resolved["block_size"],
            "num_workers": resolved["num_workers"],
        },
        "smoke": args.smoke,
        # Every flag the CLI could set, verbatim, so none is recoverable only
        # from shell history.
        "cli": {k: jsonable(v) for k, v in vars(args).items()},
        "resumed": [],
        "split_b_touched": False,
    }


def check_resume_compatibility(existing: dict[str, Any], fresh: dict[str, Any]) -> None:
    """Pre-registered fields must not drift across resumes."""
    for section in ("e2_source", "objective", "search", "eval"):
        old = {k: v for k, v in existing.get(section, {}).items() if k != "sha256"}
        new = {k: v for k, v in fresh.get(section, {}).items() if k != "sha256"}
        if old != new:
            raise SystemExit(
                f"--resume: section {section!r} differs from the run's frozen "
                f"config.json. A pre-registered run cannot change {section} "
                f"mid-flight; start a new run dir instead.\n  frozen: {old}\n  "
                f"now:    {new}",
            )


def run(args: argparse.Namespace) -> None:
    with ExitStack() as stack:
        _run(args, stack)


def _run(args: argparse.Namespace, stack: ExitStack) -> None:  # noqa: PLR0915
    if args.replot:
        render_figures(Path(args.replot))
        return
    if not args.e2_run:
        raise SystemExit("--e2-run is required (only --replot works without it).")

    budget = resolve_budget(args)
    if args.fluency_price_weight < 0:
        raise SystemExit("--fluency-price-weight must be non-negative")
    e2 = load_e2_source(args.e2_run)
    fluency_prices = None
    if args.e2_fluency_prices:
        if not e2.fluency_prices_path.exists():
            raise SystemExit(
                "--e2-fluency-prices requires E2 artifact fluency_prices.npz; "
                f"{e2.run_dir} predates fluency pricing and must be rerun.",
            )
        fluency_prices = FluencyPriceMap.load_npz(e2.fluency_prices_path)
        if fluency_prices.max_rank != e2.max_rank:
            raise SystemExit(
                f"fluency price max_rank={fluency_prices.max_rank} does not match "
                f"E2 checkpoint max_rank={e2.max_rank}",
            )
        log(
            f"fluency pricing enabled: {fluency_prices.n_blocks} blocks, "
            f"weight={args.fluency_price_weight:g}, "
            f"artifact={e2.fluency_prices_path.name}",
        )
    ckpt_cfg_full = json.loads((e2.checkpoint_dir / "config.json").read_text())
    active = e2.headroom["active_domains"]
    domains = [d for d in (args.domains or active) if d in active]
    if not domains:
        raise SystemExit(
            f"No requested domain passed the E2 headroom gate (active: {active}).",
        )
    smoke = args.smoke or args.item_cap is not None
    item_cap = 128 if (args.smoke and args.item_cap is None) else args.item_cap

    # The realised arms and the measured-sweep ladder are pre-registration
    # material, so both are resolved *before* the config record is written --
    # score_maps.npz is a small CPU-side load and needs no model.
    arms = ["combined"] if args.shortlist == "combined" else list(args.strategies)
    # --rebuild-shortlist needs the maps too: it is the artifact the rebuilt
    # control shortlist is drawn from.
    maps = (
        ScoreMaps.load_npz(e2.run_dir / "score_maps.npz")
        if args.shortlist == "combined" or args.rebuild_shortlist
        else None
    )
    if maps is None or args.shortlist != "combined":
        sweep_evals = 0
        sweep_ranks: list[int] | None = None
    else:
        sweep_ranks = (
            uniform_sweep_ranks([int(r) for r in maps.ranks], args.sweep_fractions)
            if args.sweep_mode == "uniform"
            else None
        )
        n_sweep_ranks = (
            len(sweep_ranks) if sweep_ranks is not None
            else args.sweep_ranks_per_coord
        )
        sweep_evals = len(maps.coord_list()) * n_sweep_ranks
        log(
            f"sweep grid: {len(maps.coord_list())} coordinates x {n_sweep_ranks} "
            f"ranks ({args.sweep_mode}) = {sweep_evals} evaluations per domain",
        )
    log(
        f"budget: preset={budget.preset} W={budget.beam_width} D={budget.max_depth} "
        f"R={budget.refine_rounds} | domains={domains} | "
        f"shortlist={args.shortlist} | arms={arms}",
    )

    # ---- run dir + pre-registration (skipped entirely by --estimate-only) ----
    paths = None
    resolved = {
        "item_cap": item_cap,
        "block_size": ckpt_cfg_full["block_size"],
        "num_workers": ckpt_cfg_full["num_workers"],
        "sweep_ranks": sweep_ranks,
    }
    config_record = build_config_record(args, budget, e2, domains, arms, resolved)
    if not args.estimate_only:
        if args.resume:
            root = Path(args.resume)
            if not root.is_absolute():
                root = REPO_ROOT / root
            existing = json.loads((root / "config.json").read_text())
            check_resume_compatibility(existing, config_record)
            existing.setdefault("resumed", []).append(provenance())
            paths = RunPaths(root=root)
            write_json(root / "config.json", existing)
            log(f"resuming into {root}")
        else:
            paths = RunPaths.create(
                REPO_ROOT / args.output_root,
                checkpoint_name=e2.checkpoint_dir.name,
                experiment="e3_search",
                smoke=smoke,
            )
            write_json(paths.root / "config.json", config_record)
            log(f"run dir: {paths.root}")
        # Installed before the first warning is printed. Every artifact schema
        # below is a deliberate choice about what to keep, and any such choice
        # can be wrong; the transcript is the net under them.
        stack.enter_context(RunLog(paths.root / "run.log"))
        if config_record["provenance"]["git"]["dirty"] is None:
            log("*** WARNING: git could not report tree state; provenance is incomplete.")
        elif config_record["provenance"]["git"]["dirty"]:
            log("*** WARNING: git tree is dirty; commit before results feed the paper.")
        (paths.root / "results").mkdir(exist_ok=True)
        (paths.root / "cache").mkdir(exist_ok=True)
        (paths.root / "items").mkdir(exist_ok=True)
        write_readme(paths)

    # ---- model + data + frozen splits ----
    ckpt = e2.config["checkpoint"]
    model, tokenizer = load_lowrank_checkpoint(
        ckpt["base_id"],
        str(e2.checkpoint_dir / "final_model"),
        max_rank=e2.max_rank,
        target_substring=ckpt["target_substring"],
        train_ranks=ckpt["train_ranks"],
        dtype=getattr(torch, ckpt["dtype"]),
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=args.cache_dir,
    )
    model.eval()

    all_tasks = [*WMDP_DOMAINS, *ADJACENT, *DISTANT]
    items_by_task = {t: load_task(t, cache_dir=args.cache_dir) for t in all_tasks}
    registry = SplitRegistry.load(REPO_ROOT / "mech_interp_results/splits_v1.json")
    registry.verify(items_by_task)
    split_a = {t: registry.take(items, t, "A") for t, items in items_by_task.items()}
    evaluator = MCQEvaluator(
        model, tokenizer, split_a,
        max_rank=e2.max_rank,
        max_seq_len=args.max_seq_len,
        max_tokens_per_batch=args.tokens_per_batch,
        item_cap=item_cap,
        seed=args.seed,
    )
    log(f"evaluator ready: {len(all_tasks)} tasks, split A"
        + (f", item_cap={item_cap}" if item_cap else ""))

    # Shared, fingerprinted perplexity blocks: downloaded in seconds when a
    # matching artifact exists, built from MiniPile validation otherwise (and
    # uploaded so the next pod skips even that). The fingerprint pins tokenizer,
    # split and block size, so every run scores byte-identical blocks.
    ppl_split = load_or_build_ppl_blocks(
        tokenizer,
        block_size=ckpt_cfg_full["block_size"],
        num_workers=ckpt_cfg_full["num_workers"],
        cache_dir=args.cache_dir,
    )
    ppl_ds = ppl_split.select(range(min(args.ppl_samples, len(ppl_split))))
    ppl_full, nll_full = evaluate_perplexity_at_rank(
        model, tokenizer, ppl_ds, rank=None, batch_size=args.ppl_batch_size,
    )
    # A smaller slice screens the final candidates cheaply; the winner is then
    # re-measured on the full set. Screening ratios use the screening baseline,
    # so the two block counts are never mixed.
    ppl_screen_ds = ppl_ds.select(range(min(args.ppl_screen_blocks, len(ppl_ds))))
    ppl_screen_full, nll_screen_full = evaluate_perplexity_at_rank(
        model, tokenizer, ppl_screen_ds, rank=None, batch_size=args.ppl_batch_size,
    )
    log(
        f"full-privilege minipile ppl: {ppl_full:.4f} ({len(ppl_ds)} blocks) | "
        f"screen {ppl_screen_full:.4f} ({len(ppl_screen_ds)} blocks)",
    )

    # ---- live per-domain eval timing -> projection table ----
    headroom_acc = (
        {t: float(r["acc"]) for t, r in e2.headroom["results"].items()}
        if item_cap is None
        else None
    )
    sec_per_eval: dict[str, float] = {}
    for domain in domains:
        probe = MemoisedMCQEvaluator(
            evaluator, tasks=[domain, *ADJACENT, *DISTANT], macros=MACROS,
            verbose=False,
        )
        t0 = time.perf_counter()
        payload = probe.evaluate({})
        sec_per_eval[domain] = time.perf_counter() - t0
        if headroom_acc is not None:
            for task in (domain, *ADJACENT, *DISTANT):
                if abs(payload["acc"][task] - headroom_acc[task]) > 1e-12:
                    raise SystemExit(
                        f"{task}: baseline {payload['acc'][task]} != E2 headroom "
                        f"{headroom_acc[task]}; batches are not reproducing E2.",
                    )
    finals_seconds = 2 * (max(sec_per_eval.values()) + 10.0)
    rows, totals = projection_rows(
        e2, domains, arms, budget, sec_per_eval, finals_seconds,
        sweep_evals=sweep_evals, top_k=args.top_k,
        ranks_per_coord=args.ranks_per_coord, final_top_k=args.final_top_k,
        verify_top_k=args.verify_top_k,
    )
    print()
    print(format_table(
        rows,
        ["arm", "pairs", "evals_ub", "evals_opt", "s_per_eval", "min_ub", "min_opt"],
        float_fmt="{:.1f}",
    ))
    print(
        f"\nprojected total: {totals['hours_ub']:.1f} h (upper bound) / "
        f"{totals['hours_opt']:.1f} h (optimistic, memoisation)\n",
    )
    if args.estimate_only:
        log("--estimate-only: exiting without creating a run directory.")
        return

    if paths is None:  # pragma: no cover - guarded above
        raise RuntimeError("run dir was not created")
    write_json(
        paths.root / "cost.json",
        {"projection": {"rows": rows, "totals": totals}, "actual": {}},
    )

    # ---- run-level references, written before any search starts ----
    # Each arm's own baseline covers only its searched tasks, so the *other*
    # WMDP domains have a gated number and no reference -- exactly the cells the
    # cross-domain composition table needs. The evaluator caches this pass, so
    # taking it here only moves work the first arm would do anyway.
    base_all = evaluator.baseline()
    write_json(
        paths.root / "baseline.json",
        {
            "note": (
                "Full-privilege accuracy on all evaluated tasks (split A), the "
                "reference every gated measurement is a delta against. Per-item "
                "vectors for the same pass are in items/<arm>.items.npz under "
                "the label 'baseline'."
            ),
            "tasks": {t: r.to_json() for t, r in base_all.items()},
            "macros": {
                name: pool_mcq_results(base_all, members, name=name).to_json()
                for name, members in MACROS.items()
            },
        },
    )
    save_item_level(
        paths.root / "items" / "baseline.items.npz",
        {"baseline": dict(base_all)},
        meta={"note": "run-level full-privilege pass over every evaluated task"},
    )
    write_json(
        paths.root / "ppl.json",
        {
            "note": (
                "Held-out language-modelling reference. 'full' is the set the "
                "reported configurations are re-measured on; 'screen' is the "
                "cheaper slice the fluency gate screens candidates against. "
                "Ratios must never mix the two -- different block counts."
            ),
            "dataset": "JeanKaddour/minipile",
            # load_or_build_ppl_blocks pins the split in its fingerprint and
            # always serves validation; the old `minipile` DatasetDict it used
            # to be probed against no longer exists in this scope.
            "split": "validation",
            "block_size": ckpt_cfg_full["block_size"],
            "num_workers": ckpt_cfg_full["num_workers"],
            "batch_size": args.ppl_batch_size,
            "full": {
                "n_blocks_requested": args.ppl_samples,
                "n_blocks": len(ppl_ds),
                "ppl": float(ppl_full),
                "nll": float(nll_full),
            },
            "screen": {
                "n_blocks_requested": args.ppl_screen_blocks,
                "n_blocks": len(ppl_screen_ds),
                "ppl": float(ppl_screen_full),
                "nll": float(nll_screen_full),
            },
            "max_ppl_ratio": args.max_ppl_ratio,
        },
    )

    # ---- the grid ----
    cost_actual: dict[str, Any] = {}
    completed = 0
    arm = ""
    t_arm = time.perf_counter()
    progress: dict[str, Any] = {}

    def flush_cost(*, interrupted: bool = False) -> None:
        """Fold the actuals measured so far into cost.json.

        Called after every arm rather than only at the end: a hard crash halfway
        through a multi-day grid used to lose the realised eval counts and
        timings for every arm that had already finished.
        """
        cost = json.loads((paths.root / "cost.json").read_text())
        cost["actual"] = cost_actual
        cost["interrupted"] = interrupted
        write_json(paths.root / "cost.json", cost)

    try:
        for domain in domains:
            for strategy in arms:
                arm = f"{domain}__{strategy}"
                result_path = paths.root / "results" / f"{arm}.json"
                if result_path.exists():
                    existing = json.loads(result_path.read_text())
                    if not existing.get("interrupted"):
                        log(f"[{arm}] already complete; skipping")
                        completed += 1
                        continue
                pairs = (
                    sweep_evals if strategy == "combined"
                    else arm_pairs(e2, domain, strategy)
                )
                log(f"[{arm}] starting (candidate evaluations: {pairs})")
                t_arm = time.perf_counter()
                progress = {}
                result = search_one_arm(
                    evaluator, model, tokenizer, e2, domain, strategy, budget,
                    args, paths.root / "cache" / f"{arm}.cache.json",
                    ppl_ds, ppl_screen_ds, ppl_full, ppl_screen_full,
                    headroom_acc, fluency_prices, maps, progress,
                )
                result["seconds"]["total"] = time.perf_counter() - t_arm
                write_json(result_path, result)
                # The arm is complete and on disk; drop the progress sink so an
                # interrupt landing between arms (e.g. during the manifest's
                # sha256 pass) cannot overwrite this result with a stale partial.
                progress = {}
                cost_actual[arm] = {
                    "evals_real": result["counts"]["evals_real"],
                    "evals_cached": result["counts"]["evals_cached"],
                    "seconds": result["seconds"],
                    "sec_per_eval_measured": (
                        result["seconds"]["total"]
                        / max(1, result["counts"]["evals_real"])
                    ),
                }
                flush_cost()
                completed += 1
                state = result["best_refined"] or result["best_beam"]
                log(
                    f"[{arm}] done in {result['seconds']['total'] / 60:.1f} min: "
                    f"score {state['score']:.4f}, "
                    f"{result['base']['acc'][domain]:.4f} -> "
                    f"{state['acc'][domain]:.4f} on {domain}, "
                    f"|supp|={state['num_interventions']}",
                )
                write_json(paths.root / "manifest.json", paths.manifest())
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except KeyboardInterrupt:
        # A second Ctrl-C (or a signal delivered to both uv and python, as
        # `timeout --signal=INT` does) must not kill the cleanup: ignore further
        # SIGINTs while flushing state. The caches were already saved atomically
        # by search_one_arm's finally.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        log("interrupted -- caches saved; resume with --resume " + str(paths.root))
        if progress:
            # Whatever the in-flight arm managed to measure. `interrupted: true`
            # keeps it out of the figures and makes --resume redo the arm; the
            # partial is here to be read, not to be treated as a result.
            progress["seconds"] = {
                **progress.get("seconds", {}), "total": time.perf_counter() - t_arm,
            }
            write_json(paths.root / "results" / f"{arm}.json", progress)
            log(f"partial result for {arm} written")
        flush_cost(interrupted=True)
        write_readme(paths)
        write_json(paths.root / "manifest.json", paths.manifest())
        sys.exit(130)

    flush_cost()

    log(f"{completed} arms complete; rendering figures + summary")
    render_figures(paths.root)
    write_readme(paths)
    write_json(paths.root / "manifest.json", paths.manifest())
    log(f"done -> {paths.root}")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
