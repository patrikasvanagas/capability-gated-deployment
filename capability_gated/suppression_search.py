"""Beam search for selective capability suppression.

Given per-single-MLP intervention results (from :mod:`capability_gated.mlp_rank_sweep`),
search for a *combination* of interventions ``(block_idx, mlp_name) -> rank`` that
suppresses a chosen set of tasks while preserving the others. The algorithm:

1. shortlist candidate ``(block, module)`` coordinates from the single-intervention JSON,
2. beam search over combinations,
3. local rank refinement (coordinate descent).

The model only needs the rank-gating methods ``_set_rank_all`` and
``_set_rank_for_named_module_at_block`` -- i.e. anything prepared by
:func:`capability_gated.rank_control.attach_rank_control` (so
:func:`~capability_gated.training.load_lowrank_checkpoint` works directly).

Evaluation uses the ``lm-eval`` harness, imported lazily so this module (and the
rest of the package) import fine without it. Install the optional extra to run:
``pip install lm-eval peft``.

Example
-------
>>> from capability_gated import load_lowrank_checkpoint, optimise_task_suppression
>>> model, tok = load_lowrank_checkpoint("Qwen/Qwen2.5-0.5B", "runs/qwen/final_model", 896)
>>> tasks = [f"mmlu_college_{s}" for s in
...          ("biology", "chemistry", "computer_science", "mathematics", "physics")]
>>> out = optimise_task_suppression(
...     model, tok, max_rank=896,
...     single_intervention_json_path="sweeps/college_sweep.single.json",
...     tasks=tasks, suppress_tasks="mmlu_college_chemistry")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# -----------------------------
# Types
# -----------------------------

Coord = Tuple[int, str]  # (block_idx, module_name), e.g. (12, "gate_proj")
Config = Dict[Coord, int]  # mapping from coord -> rank


# -----------------------------
# Small helpers
# -----------------------------

GLOBAL_EVAL_COUNTER = 0


def _clean_task_name(t: str) -> str:
    s = t.replace("mmlu_", "")
    s = s.replace("high_school_", "").replace("college_", "")
    return s.replace("_", " ").title()


def parse_intervention_key(key: str) -> Tuple[int, int, str]:
    """Parse keys like ``"0_851_gate_proj"`` -> ``(block_idx, rank, module_name)``."""
    parts = key.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse key={key}")
    block_idx = int(parts[0])
    rank = int(parts[1])
    module_name = "_".join(parts[2:])
    return block_idx, rank, module_name


def config_key(cfg: Config) -> Tuple[Tuple[int, str, int], ...]:
    """Canonical, hashable config representation."""
    return tuple(sorted((int(b), str(m), int(r)) for (b, m), r in cfg.items()))


def serialise_cfg(cfg: Config) -> List[Dict[str, Any]]:
    """Convert ``{(block, module): rank, ...}`` into a JSON-friendly list."""
    return [
        {"block": int(b), "module": str(m), "rank": int(r)}
        for (b, m), r in sorted(cfg.items())
    ]


def apply_config(model, cfg: Config, max_rank: int) -> None:
    """Reset the whole model to ``max_rank``, then apply per-(block,module) overrides."""
    model._set_rank_all(int(max_rank))
    for (b, m), r in cfg.items():
        model._set_rank_for_named_module_at_block(
            block_idx=int(b),
            module_name=str(m),
            rank=int(r),
        )


def extract_acc_and_stderr(
    results_dict: Dict[str, Any], tasks: List[str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Pull ``acc,none`` / ``acc_stderr,none`` for ``tasks`` from lm-eval results."""
    acc: Dict[str, float] = {}
    se: Dict[str, float] = {}
    for t in tasks:
        acc[t] = float(results_dict[t]["acc,none"])
        se[t] = float(results_dict[t].get("acc_stderr,none", np.nan))
    return acc, se


# -----------------------------
# Memoised evaluator (lm-eval backed)
# -----------------------------


@dataclass
class EvalSettings:
    tasks: List[str]
    num_fewshot: int = 5
    batch_size: int = 8
    device: str = "cuda:0"
    verbosity: str = "NOTSET"


class MemoisedEvaluator:
    """Wrap lm-eval-harness evaluation, memoised by intervention config.

    ``lm-eval`` is imported lazily in ``__init__`` so importing this module does
    not require it; construct an instance only when you actually want to evaluate.
    """

    def __init__(self, model, tokenizer, max_rank: int, eval_settings: EvalSettings):
        from lm_eval.models.huggingface import HFLM  # lazy: optional dependency

        self.model = model
        self.tokenizer = tokenizer
        self.max_rank = int(max_rank)
        self.eval_settings = eval_settings

        self.lm = HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            batch_size=eval_settings.batch_size,
        )

        self.cache: Dict[Tuple[Tuple[int, str, int], ...], Dict[str, Any]] = {}

        # ---- progress bookkeeping ----
        self.n_evals = 0  # number of *actual* harness runs (cache misses)
        self.n_cache_hits = 0
        self.t0 = time.time()

        self.autosave_path = "trained_results/_autosave.json"
        self.cache_path = "trained_results/_cache.json"
        os.makedirs(os.path.dirname(self.autosave_path), exist_ok=True)
        self.autosave_every = 25

    def evaluate(self, cfg: Config) -> Dict[str, Any]:
        from lm_eval import evaluator  # lazy: optional dependency

        global GLOBAL_EVAL_COUNTER

        k = config_key(cfg)

        # Cache hit: no new expensive eval.
        if k in self.cache:
            self.n_cache_hits += 1
            return self.cache[k]

        # Cache miss: run lm-eval (expensive).
        self.n_evals += 1
        GLOBAL_EVAL_COUNTER += 1

        if self.n_evals % 10 == 0:
            elapsed = time.time() - self.t0
            avg = elapsed / max(1, self.n_evals)
            print(
                f"[EVAL] runs={self.n_evals} cache_hits={self.n_cache_hits} "
                f"elapsed={elapsed / 60:.1f} min avg={avg:.2f} s/eval"
            )

        apply_config(self.model, cfg, self.max_rank)

        self.model.eval()
        with torch.inference_mode():
            res = evaluator.simple_evaluate(
                model=self.lm,
                tasks=self.eval_settings.tasks,
                num_fewshot=self.eval_settings.num_fewshot,
                batch_size="auto",
                device=self.eval_settings.device,
                verbosity=self.eval_settings.verbosity,
                random_seed=42,
                numpy_random_seed=42,
                torch_random_seed=42,
            )
        acc, se = extract_acc_and_stderr(res["results"], self.eval_settings.tasks)

        out = {"acc": acc, "se": se, "raw": res["results"]}
        self.cache[k] = out

        if (self.n_evals % self.autosave_every) == 0:
            tmp = {
                "n_evals": self.n_evals,
                "n_cache_hits": self.n_cache_hits,
                "num_cached": len(self.cache),
            }
            with open(self.autosave_path, "w") as f:
                json.dump(tmp, f, indent=2)
            self._save_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return out

    def _save_cache(self):
        """Save cache with config keys serialized as strings."""
        serialized_cache = {}
        for k, v in self.cache.items():
            key_str = ";".join(f"{b}_{m}_{r}" for b, m, r in k)
            serialized_cache[key_str] = v

        with open(self.cache_path, "w") as f:
            json.dump(serialized_cache, f, indent=2)

        print(f"[Cache saved] {len(self.cache)} configs to {self.cache_path}")

    def load_cache(self):
        """Load cache from disk if it exists."""
        if not os.path.exists(self.cache_path):
            return

        with open(self.cache_path, "r") as f:
            serialized = json.load(f)

        for key_str, v in serialized.items():
            parts = key_str.split(";")
            k = tuple(
                (int(p.split("_")[0]), p.split("_")[1], int(p.split("_")[2]))
                for p in parts
            )
            self.cache[k] = v

        print(f"[Cache loaded] {len(self.cache)} configs from {self.cache_path}")


# -----------------------------
# Objective specification
# -----------------------------


@dataclass
class TaskTradeoffObjective:
    """Higher score is better.

    ``score = mean(preserve_acc) - lambda_suppress * mean(suppress_acc)``
    ``- preserve_penalty - lambda_fluency * fluency_price(cfg)``,
    where ``penalty`` triggers if any preserve task drops more than
    ``eps_preserve_drop`` vs baseline. ``fluency_price`` is optional so legacy
    callers retain the exact accuracy-only objective.
    """

    suppress_tasks: List[str]
    preserve_tasks: List[str]

    lambda_suppress: float = 2.0
    eps_preserve_drop: float = 0.01
    penalty_scale: float = 100.0
    lambda_fluency: float = 0.0
    fluency_price: Callable[[Config], float] | None = None


def compute_objective(
    acc: Dict[str, float],
    base_acc: Dict[str, float],
    spec: TaskTradeoffObjective,
    cfg: Config | None = None,
) -> float:
    preserve_vals = [float(acc[t]) for t in spec.preserve_tasks]
    suppress_vals = [float(acc[t]) for t in spec.suppress_tasks]

    preserve_mean = float(np.mean(preserve_vals)) if preserve_vals else 0.0
    suppress_mean = float(np.mean(suppress_vals)) if suppress_vals else 0.0

    penalty = 0.0
    for t in spec.preserve_tasks:
        drop = float(base_acc[t] - acc[t])
        if drop > spec.eps_preserve_drop:
            penalty += spec.penalty_scale * (drop - spec.eps_preserve_drop)

    fluency_penalty = 0.0
    if spec.fluency_price is not None and spec.lambda_fluency != 0:
        if cfg is None:
            raise ValueError("cfg is required when fluency pricing is enabled")
        fluency_penalty = spec.lambda_fluency * float(spec.fluency_price(cfg))

    return (
        preserve_mean
        - spec.lambda_suppress * suppress_mean
        - penalty
        - fluency_penalty
    )


# -----------------------------
# Shortlisting candidates from single-intervention JSON
# -----------------------------


@dataclass
class CandidateMove:
    coord: Coord
    rank: int
    rank_pct: int
    suppress_drop_mean: float
    preserve_drop_mean: float
    score: float


def build_candidates_from_single_interventions(
    single_json: Dict[str, Any],
    max_rank: int,
    suppress_tasks: List[str],
    preserve_tasks: List[str],
    min_suppress_drop: float = 0.01,
    max_preserve_drop: float = 0.01,
    alpha_preserve: float = 2.0,
) -> List[CandidateMove]:
    """Build candidate ``(coord -> rank)`` moves from a single-intervention JSON.

    Keeps only interventions that reduce the suppressed tasks sufficiently and do
    not harm the preserved tasks too much.
    """
    if "max_rank" not in single_json:
        raise ValueError("Expected 'max_rank' key in JSON")

    base_res = single_json["max_rank"]
    base_acc = {t: float(base_res[t]["acc,none"]) for t in base_res.keys()}

    candidates: List[CandidateMove] = []

    for key, val in single_json.items():
        if key == "max_rank":
            continue
        if not isinstance(val, dict) or "results" not in val:
            continue

        block_idx, rank, module_name = parse_intervention_key(key)
        rank = int(val.get("rank", rank))
        rank_pct = int(val.get("rank_percentage", round(100 * rank / max_rank)))

        res = val["results"]
        acc = {t: float(res[t]["acc,none"]) for t in res.keys()}

        suppress_drops = [
            base_acc[t] - acc[t]
            for t in suppress_tasks
            if (t in base_acc) and (t in acc)
        ]
        preserve_drops = [
            base_acc[t] - acc[t]
            for t in preserve_tasks
            if (t in base_acc) and (t in acc)
        ]

        # Require all tasks present (otherwise skip this intervention).
        if len(suppress_drops) != len(suppress_tasks):
            continue
        if len(preserve_drops) != len(preserve_tasks):
            continue

        suppress_drop_mean = float(np.mean(suppress_drops)) if suppress_drops else 0.0
        preserve_drop_mean = float(np.mean(preserve_drops)) if preserve_drops else 0.0

        if suppress_drop_mean < min_suppress_drop:
            continue
        if preserve_drop_mean > max_preserve_drop:
            continue

        score = suppress_drop_mean - alpha_preserve * max(0.0, preserve_drop_mean)

        candidates.append(
            CandidateMove(
                coord=(int(block_idx), str(module_name)),
                rank=int(rank),
                rank_pct=int(rank_pct),
                suppress_drop_mean=float(suppress_drop_mean),
                preserve_drop_mean=float(preserve_drop_mean),
                score=float(score),
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def shortlist_coords_and_ranks(
    candidates: List[CandidateMove],
    top_coords: int = 30,
    per_coord_ranks: int = 2,
) -> Dict[Coord, List[int]]:
    """Convert flat candidate moves into ``coord -> [rank1, rank2, ...]``.

    Keeps up to ``top_coords`` coordinates and ``per_coord_ranks`` ranks each.
    """
    coord_to_moves: Dict[Coord, List[CandidateMove]] = {}
    for c in candidates:
        coord_to_moves.setdefault(c.coord, []).append(c)

    coords_sorted = sorted(
        coord_to_moves.keys(),
        key=lambda coord: max(m.score for m in coord_to_moves[coord]),
        reverse=True,
    )[:top_coords]

    out: Dict[Coord, List[int]] = {}
    for coord in coords_sorted:
        moves = sorted(coord_to_moves[coord], key=lambda m: m.score, reverse=True)

        ranks: List[int] = []
        seen = set()
        for m in moves:
            if m.rank in seen:
                continue
            ranks.append(int(m.rank))
            seen.add(int(m.rank))
            if len(ranks) >= per_coord_ranks:
                break

        out[coord] = ranks

    return out


# -----------------------------
# Beam search over combinations
# -----------------------------


# -----------------------------
# Fluency pricing
# -----------------------------
# Measured on qwen3-17b-fineweb-r2048: the held-out perplexity cost of gating a
# SINGLE coordinate, with every other coordinate at full rank, spans 1.009x to
# 1.819x -- an 80-fold range in excess-over-1. Cut depth is not the driver:
# b21/up_proj survives rank 2 at 1.139x while b27/down_proj costs 1.819x at rank
# 64. The search previously had no way to see this and could spend its whole
# fluency budget on one coordinate; `rank_budget` (sum of rank reductions) is a
# poor proxy for exactly that reason, since it prices rank 2 as the expensive one.
#
# Joint cost is well approximated by the product of the solo costs times a modest
# interaction term (measured 0.92-1.42 over four real configurations), so the
# search can treat the fluency gate as a budget and reject unaffordable
# configurations without evaluating them.


@dataclass
class FluencyPrices:
    """Measured solo fluency cost per (coordinate, rank), as a perplexity ratio.

    This is a **screen, not the gate.** The authority remains the measured
    perplexity ratio applied to final candidates; this only prunes expansions
    cheaply enough to be worth doing before spending an evaluation.

    On ``interaction_factor``. Gating several coordinates at once does not cost
    exactly the product of the solo costs, but the correction is not a constant:
    measured over four real E3 configurations it ranged 0.92-1.40 and was not
    monotone in the product (1.09 -> 1.00, 1.57 -> 0.92, 2.25 -> 1.40,
    2.98 -> 1.38). Four points do not support fitting it, so the default is
    **1.0** -- plain product of singles, which classified all four correctly
    against a 1.10 budget.

    That leaves the estimate *under*-predicting the expensive configurations
    (2.98 predicted vs 4.10 measured), which is the safe direction for a screen:
    it errs towards admitting a config that the real gate will then reject,
    rather than silently pruning one that would have passed. Raising this above
    1.0 makes the screen stricter and risks discarding viable configurations --
    an early draft defaulted to 1.45 and rejected the only configuration that
    has ever passed the gate.
    """

    prices: Dict[Tuple[int, str, int], float]
    full_privilege_ppl: float
    interaction_factor: float = 1.0

    def price(self, coord: Coord, rank: int) -> Optional[float]:
        return self.prices.get((int(coord[0]), str(coord[1]), int(rank)))

    def predict(self, cfg: Config) -> Optional[float]:
        """Predicted joint perplexity ratio, or None if any price is missing."""
        product = 1.0
        for coord, rank in cfg.items():
            p = self.price(coord, rank)
            if p is None:
                return None
            product *= p
        if not cfg:
            return 1.0
        return product * self.interaction_factor

    def affordable(self, cfg: Config, budget: float) -> bool:
        """True when the config is predicted to fit the budget (unknown => True).

        Unknown prices pass rather than fail: a missing measurement must not
        silently prune part of the search space. Measure every candidate rank if
        you want the budget enforced everywhere.
        """
        predicted = self.predict(cfg)
        return True if predicted is None else predicted <= float(budget)

    def to_json(self) -> Dict[str, Any]:
        return {
            "full_privilege_ppl": self.full_privilege_ppl,
            "interaction_factor": self.interaction_factor,
            "prices": [
                {"block": b, "module": m, "rank": r, "x_full": v}
                for (b, m, r), v in sorted(self.prices.items())
            ],
        }

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "FluencyPrices":
        prices = {
            (int(p["block"]), str(p["module"]), int(p["rank"])): float(p["x_full"])
            for p in payload["prices"]
        }
        return cls(
            prices=prices,
            full_privilege_ppl=float(payload["full_privilege_ppl"]),
            interaction_factor=float(payload.get("interaction_factor", 1.0)),
        )


def measure_fluency_prices(
    ppl_fn,
    candidate_ranks: Dict[Coord, List[int]],
    *,
    interaction_factor: float = 1.0,
    verbose: bool = True,
) -> FluencyPrices:
    """Measure the solo fluency cost of every (coordinate, rank) candidate.

    Args:
        ppl_fn: callable mapping a :data:`Config` to held-out perplexity. Pass
            ``{}`` for full privilege. Keeping this a callable leaves the choice
            of corpus and sample size to the caller -- it should be the *same*
            perplexity the fluency gate uses, or the prices will not commensurate
            with the budget.
        candidate_ranks: the shortlist the search will explore.

    Costs one evaluation per (coordinate, rank), plus one for the baseline.
    """
    base = float(ppl_fn({}))
    prices: Dict[Tuple[int, str, int], float] = {}
    total = sum(len(rs) for rs in candidate_ranks.values())
    if verbose:
        print(f"[Prices] measuring {total} (coordinate, rank) pairs; "
              f"full-privilege ppl {base:.4f}")

    i = 0
    for coord, ranks in candidate_ranks.items():
        for r in ranks:
            i += 1
            ratio = float(ppl_fn({coord: int(r)})) / base
            prices[(int(coord[0]), str(coord[1]), int(r))] = ratio
            if verbose:
                print(f"[Prices] {i}/{total} b{coord[0]}/{coord[1]}@{r} = {ratio:.3f}x")

    if verbose and prices:
        ordered = sorted(prices.items(), key=lambda kv: -kv[1])
        print("[Prices] most expensive: " + ", ".join(
            f"b{b}/{m}@{r}={v:.2f}x" for (b, m, r), v in ordered[:3]))
        print("[Prices] cheapest:       " + ", ".join(
            f"b{b}/{m}@{r}={v:.2f}x" for (b, m, r), v in ordered[-3:]))

    return FluencyPrices(prices=prices, full_privilege_ppl=base,
                         interaction_factor=interaction_factor)


@dataclass
class BeamSearchSettings:
    beam_width: int = 8
    max_depth: int = 10
    rank_budget: Optional[int] = None  # sum(max_rank - r) <= budget
    # Treat the fluency gate as a knapsack budget: skip any expansion predicted
    # to breach it. Both must be set for the constraint to apply.
    fluency_prices: Optional[FluencyPrices] = None
    fluency_budget: Optional[float] = None
    verbose: bool = True


@dataclass
class BeamState:
    cfg: Config
    score: float
    acc: Dict[str, float]


def rank_budget_used(cfg: Config, max_rank: int) -> int:
    return int(sum(int(max_rank) - int(r) for r in cfg.values()))


def beam_search(
    memo_eval: MemoisedEvaluator,
    base_acc: Dict[str, float],
    objective_spec: TaskTradeoffObjective,
    candidate_ranks: Dict[Coord, List[int]],
    max_rank: int,
    settings: BeamSearchSettings,
) -> BeamState:
    """Beam search that builds configs by adding one coordinate at a time."""
    max_rank = int(max_rank)

    baseline_cfg: Config = {}
    base_eval = memo_eval.evaluate(baseline_cfg)
    base_score = compute_objective(
        base_eval["acc"], base_acc, objective_spec, baseline_cfg
    )

    beam: List[BeamState] = [
        BeamState(cfg=baseline_cfg, score=float(base_score), acc=base_eval["acc"])
    ]
    best = beam[0]

    coords = list(candidate_ranks.keys())

    for depth in range(1, settings.max_depth + 1):
        proposals: List[BeamState] = []
        n_skipped = 0

        for state in beam:
            used = set(state.cfg.keys())
            remaining = [c for c in coords if c not in used]

            for coord in remaining:
                for r in candidate_ranks[coord]:
                    new_cfg = dict(state.cfg)
                    new_cfg[coord] = int(r)

                    if settings.rank_budget is not None:
                        if rank_budget_used(new_cfg, max_rank) > settings.rank_budget:
                            continue

                    # Knapsack constraint: reject before spending an evaluation.
                    if settings.fluency_prices is not None and settings.fluency_budget:
                        if not settings.fluency_prices.affordable(
                            new_cfg, settings.fluency_budget
                        ):
                            n_skipped += 1
                            continue

                    ev = memo_eval.evaluate(new_cfg)
                    sc = compute_objective(
                        ev["acc"], base_acc, objective_spec, new_cfg
                    )
                    proposals.append(
                        BeamState(cfg=new_cfg, score=float(sc), acc=ev["acc"])
                    )

        if not proposals:
            if settings.verbose:
                extra = (f" ({n_skipped} skipped: predicted to breach the "
                         f"fluency budget)" if n_skipped else "")
                print(f"[Beam] depth={depth}: no proposals{extra}; stopping.")
            break

        proposals.sort(key=lambda s: s.score, reverse=True)

        dedup: List[BeamState] = []
        seen = set()
        for s in proposals:
            k = config_key(s.cfg)
            if k in seen:
                continue
            seen.add(k)
            dedup.append(s)
            if len(dedup) >= settings.beam_width:
                break

        beam = dedup

        if beam[0].score > best.score:
            best = beam[0]

        if settings.verbose:
            sup_mean = float(
                np.mean([best.acc[t] for t in objective_spec.suppress_tasks])
            )
            pre_mean = float(
                np.mean([best.acc[t] for t in objective_spec.preserve_tasks])
            )
            priced = ""
            if settings.fluency_prices is not None:
                est = settings.fluency_prices.predict(best.cfg)
                if est is not None:
                    priced = f" | est_ppl_x={est:.3f}"
            skipped = f" | skipped={n_skipped}" if n_skipped else ""
            print(
                f"[Beam] depth={depth} best_score={best.score:.4f} "
                f"suppress_mean={sup_mean:.4f} preserve_mean={pre_mean:.4f} "
                f"| interventions={len(best.cfg)}{priced}{skipped}"
            )

    return best


# -----------------------------
# Local refinement (coordinate descent over ranks)
# -----------------------------


@dataclass
class RefinementSettings:
    rounds: int = 2
    coarse_fracs: Tuple[float, ...] = (0.95, 0.90, 0.80, 0.65, 0.50, 0.25, 0.15, 0.05)
    refine_steps: int = 2
    # Refinement moves ranks, so it must respect the same knapsack budget the
    # beam honoured -- otherwise it can walk a compliant config back out of it.
    fluency_prices: Optional["FluencyPrices"] = None
    fluency_budget: Optional[float] = None
    verbose: bool = True


def _unique_sorted_ints(xs: List[int]) -> List[int]:
    return sorted(set(int(x) for x in xs), reverse=True)


def refine_config_ranks(
    memo_eval: MemoisedEvaluator,
    base_acc: Dict[str, float],
    objective_spec: TaskTradeoffObjective,
    cfg_init: Config,
    max_rank: int,
    refine: RefinementSettings,
) -> BeamState:
    """Coordinate-wise rank refinement for the coordinates present in ``cfg_init``."""
    max_rank = int(max_rank)
    cfg = dict(cfg_init)

    def affordable(c: Config) -> bool:
        if refine.fluency_prices is None or not refine.fluency_budget:
            return True
        return refine.fluency_prices.affordable(c, refine.fluency_budget)

    def score_cfg(c: Config) -> Tuple[float, Dict[str, float]]:
        ev = memo_eval.evaluate(c)
        sc = compute_objective(ev["acc"], base_acc, objective_spec, c)
        return float(sc), ev["acc"]

    best_score, best_acc = score_cfg(cfg)
    coords = list(cfg.keys())

    for rr in range(refine.rounds):
        if refine.verbose:
            print(f"[Refine] round {rr + 1}/{refine.rounds}")

        for coord in coords:
            current_r = int(cfg[coord])

            coarse = [
                max(1, min(max_rank, int(round(max_rank * f))))
                for f in refine.coarse_fracs
            ]
            coarse.append(current_r)
            coarse = _unique_sorted_ints(coarse)

            best_local_r = current_r
            best_local_score = best_score
            best_local_acc = best_acc

            # Coarse scan.
            for r in coarse:
                trial = dict(cfg)
                trial[coord] = int(r)
                if not affordable(trial):
                    continue
                sc, acc = score_cfg(trial)
                if sc > best_local_score:
                    best_local_score, best_local_r, best_local_acc = sc, int(r), acc

            # Midpoint refinement around best_local_r.
            ranks = list(coarse)
            ranks.sort(reverse=True)

            for _ in range(refine.refine_steps):
                if best_local_r not in ranks:
                    ranks.append(best_local_r)
                    ranks.sort(reverse=True)

                idx = ranks.index(best_local_r)
                neighbours = []
                if idx > 0:
                    neighbours.append(ranks[idx - 1])
                if idx < len(ranks) - 1:
                    neighbours.append(ranks[idx + 1])

                mids = []
                for nb in neighbours:
                    mid = int(round((best_local_r + nb) / 2))
                    if mid != best_local_r and mid != nb:
                        mids.append(mid)

                mids = _unique_sorted_ints(mids)

                improved = False
                for r in mids:
                    trial = dict(cfg)
                    trial[coord] = int(r)
                    if not affordable(trial):
                        continue
                    sc, acc = score_cfg(trial)
                    if sc > best_local_score:
                        best_local_score, best_local_r, best_local_acc = sc, int(r), acc
                        improved = True

                ranks = _unique_sorted_ints(ranks + mids)
                if not improved:
                    break

            cfg[coord] = int(best_local_r)
            best_score, best_acc = best_local_score, best_local_acc

            if refine.verbose:
                sup_mean = float(
                    np.mean([best_acc[t] for t in objective_spec.suppress_tasks])
                )
                pre_mean = float(
                    np.mean([best_acc[t] for t in objective_spec.preserve_tasks])
                )
                print(
                    f"  coord={coord} -> rank={best_local_r} | score={best_score:.4f} "
                    f"suppress_mean={sup_mean:.4f} preserve_mean={pre_mean:.4f}"
                )

    return BeamState(cfg=cfg, score=float(best_score), acc=best_acc)


# -----------------------------
# End-to-end driver
# -----------------------------


def optimise_task_suppression(
    model,
    tokenizer,
    max_rank: int,
    single_intervention_json_path: str,
    tasks: List[str],
    suppress_tasks: Union[str, List[str]],
    preserve_tasks: Optional[List[str]] = None,
    # eval
    num_fewshot: int = 5,
    batch_size: int = 8,
    device: str = "cuda:0",
    # shortlist filtering
    shortlist_top_coords: int = 30,
    shortlist_per_coord_ranks: int = 2,
    min_suppress_drop: float = 0.01,
    max_preserve_drop: float = 0.01,
    # objective weights
    lambda_suppress: float = 2.0,
    eps_preserve_drop: float = 0.01,
    # beam
    beam_width: int = 8,
    max_depth: int = 10,
    rank_budget: Optional[int] = None,
    # refinement
    refine_rounds: int = 2,
    # saving
    n_to_save: int = 50,
) -> Dict[str, Any]:
    """General task-suppression optimiser.

    Returns a dict with the best beam config, the refined config, and the top-n
    configs observed during search.
    """
    max_rank = int(max_rank)
    n_to_save = int(n_to_save)

    if isinstance(suppress_tasks, str):
        suppress_tasks = [suppress_tasks]
    suppress_tasks = list(suppress_tasks)

    if preserve_tasks is None:
        preserve_tasks = [t for t in tasks if t not in suppress_tasks]
    else:
        preserve_tasks = list(preserve_tasks)

    if not suppress_tasks:
        raise ValueError("suppress_tasks must be non-empty.")
    if any(t not in tasks for t in suppress_tasks):
        raise ValueError(
            f"suppress_tasks contains tasks not in tasks list: {suppress_tasks}"
        )
    if any(t not in tasks for t in preserve_tasks):
        raise ValueError(
            f"preserve_tasks contains tasks not in tasks list: {preserve_tasks}"
        )

    with open(single_intervention_json_path, "r") as f:
        single_json = json.load(f)

    if "max_rank" in single_json and isinstance(single_json["max_rank"], dict):
        base_acc = {t: float(single_json["max_rank"][t]["acc,none"]) for t in tasks}
    else:
        eval_settings = EvalSettings(
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
        )
        memo_eval_tmp = MemoisedEvaluator(
            model=model,
            tokenizer=tokenizer,
            max_rank=max_rank,
            eval_settings=eval_settings,
        )
        base_acc = memo_eval_tmp.evaluate({})["acc"]

    # 1) Build candidates from single interventions.
    candidates = build_candidates_from_single_interventions(
        single_json=single_json,
        max_rank=max_rank,
        suppress_tasks=suppress_tasks,
        preserve_tasks=preserve_tasks,
        min_suppress_drop=min_suppress_drop,
        max_preserve_drop=max_preserve_drop,
        alpha_preserve=2.0,
    )

    if len(candidates) == 0:
        raise RuntimeError(
            "No candidate single interventions passed filters.\n"
            "Try relaxing min_suppress_drop and/or max_preserve_drop."
        )

    candidate_ranks = shortlist_coords_and_ranks(
        candidates,
        top_coords=shortlist_top_coords,
        per_coord_ranks=shortlist_per_coord_ranks,
    )

    print(f"[Shortlist] kept coords: {len(candidate_ranks)}")
    for i, (coord, ranks) in enumerate(candidate_ranks.items()):
        if i >= 10:
            break
        print(f"  {coord}: ranks={ranks}")

    # 2) Set up memo evaluator for the expensive combined search.
    eval_settings = EvalSettings(
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
    )
    memo_eval = MemoisedEvaluator(
        model=model,
        tokenizer=tokenizer,
        max_rank=max_rank,
        eval_settings=eval_settings,
    )

    objective_spec = TaskTradeoffObjective(
        suppress_tasks=suppress_tasks,
        preserve_tasks=preserve_tasks,
        lambda_suppress=float(lambda_suppress),
        eps_preserve_drop=float(eps_preserve_drop),
        penalty_scale=100.0,
    )

    # 3) Beam search.
    bs_settings = BeamSearchSettings(
        beam_width=int(beam_width),
        max_depth=int(max_depth),
        rank_budget=rank_budget,
        verbose=True,
    )

    best_beam = beam_search(
        memo_eval=memo_eval,
        base_acc=base_acc,
        objective_spec=objective_spec,
        candidate_ranks=candidate_ranks,
        max_rank=max_rank,
        settings=bs_settings,
    )

    print("\n[Beam best] interventions:")
    for (b, m), r in sorted(best_beam.cfg.items()):
        print(f"  block={b:02d}  {m:9s}  rank={r}")

    # 4) Local refinement.
    ref_settings = RefinementSettings(rounds=int(refine_rounds), verbose=True)
    best_refined = refine_config_ranks(
        memo_eval=memo_eval,
        base_acc=base_acc,
        objective_spec=objective_spec,
        cfg_init=best_beam.cfg,
        max_rank=max_rank,
        refine=ref_settings,
    )

    print("\n[Refined best] interventions:")
    for (b, m), r in sorted(best_refined.cfg.items()):
        print(f"  block={b:02d}  {m:9s}  rank={r}")

    # Final report.
    def report(acc: Dict[str, float]) -> None:
        print("\n[Final accuracies]")
        for t in tasks:
            print(
                f"  {_clean_task_name(t):28s}: {acc[t]:.4f} "
                f"(baseline {base_acc[t]:.4f}, Δ={acc[t] - base_acc[t]:+.4f})"
            )
        sup_mean = float(np.mean([acc[t] for t in suppress_tasks]))
        pre_mean = (
            float(np.mean([acc[t] for t in preserve_tasks])) if preserve_tasks else 0.0
        )
        print(f"\n  {'Suppress mean':28s}: {sup_mean:.4f}")
        print(f"  {'Preserve mean':28s}: {pre_mean:.4f}")

    report(best_refined.acc)

    # Save the top-n configurations observed during the run.
    scored_cfgs: List[Dict[str, Any]] = []
    for k, ev in memo_eval.cache.items():
        cfg_from_key: Config = {(int(b), str(m)): int(r) for (b, m, r) in k}
        sc = float(
            compute_objective(ev["acc"], base_acc, objective_spec, cfg_from_key)
        )
        scored_cfgs.append(
            {
                "cfg": serialise_cfg(cfg_from_key),
                "score": sc,
                "num_interventions": int(len(cfg_from_key)),
                "acc": {t: float(ev["acc"][t]) for t in tasks},
                "se": {t: float(ev["se"][t]) for t in tasks},
            }
        )

    scored_cfgs.sort(key=lambda x: x["score"], reverse=True)
    top_n_cfgs = scored_cfgs[: max(0, n_to_save)]

    shortlist_json_safe = {
        f"{int(b)}_{str(m)}": [int(r) for r in ranks]
        for (b, m), ranks in candidate_ranks.items()
    }

    return {
        "tasks": list(tasks),
        "suppress_tasks": list(suppress_tasks),
        "preserve_tasks": list(preserve_tasks),
        "base_acc": {t: float(base_acc[t]) for t in tasks},
        "best_beam": {
            "cfg": serialise_cfg(best_beam.cfg),
            "score": float(best_beam.score),
            "acc": {t: float(best_beam.acc[t]) for t in tasks},
        },
        "best_refined": {
            "cfg": serialise_cfg(best_refined.cfg),
            "score": float(best_refined.score),
            "acc": {t: float(best_refined.acc[t]) for t in tasks},
        },
        "shortlist_candidate_ranks": shortlist_json_safe,
        "num_cached_evals": int(len(memo_eval.cache)),
        "n_to_save": int(n_to_save),
        "top_configs": top_n_cfgs,
    }
