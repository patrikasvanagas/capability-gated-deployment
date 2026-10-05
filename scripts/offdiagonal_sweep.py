#!/usr/bin/env python
"""E0: how much does gating *k* coordinates cost, and does the adaptation recipe
change that?

The suppression search deploys sparse, uneven configurations -- two to seven
coordinates at assorted ranks. Global adaptation never rehearses one: every
variant it trains on sets all coordinates to the same rank, the diagonal of the
lattice. The mixed recipe rehearses sparse states directly. So the question this
measures is whether rehearsing them matters:

    degradation vs k, one curve per adaptation recipe

and the gap between the curves is the whole claim. If they overlap, off-diagonal
adaptation buys nothing and the cheaper recipe suffices -- a reportable negative.

**Random draws, not searched ones.** E3's beam also walks up in k, so its
trajectory looks superficially usable here. It is not: those configurations were
chosen to maximise suppression, so reading typical damage off them would
overstate it badly. Coordinates here are drawn uniformly and ranks are drawn the
way the trainer draws them, so the measurement matches what adaptation actually
rehearsed.

**The draws are shared across checkpoints.** Coordinate and rank draws depend
only on ``--seed`` and the ladder, never on the model, so two checkpoints see
byte-identical configurations. The comparison is then paired: the enormous
variance between one random configuration and another cancels, and what is left
is the recipe difference. Running with different seeds per checkpoint would
mostly measure which model got the kinder draws.

Perplexity is the primary metric. The MMLU macros are reported too, but the
adjacent macro's standard error is 3.64pp, wide enough to hide anything this
experiment could produce.

Example::

    python scripts/offdiagonal_sweep.py --model-id Qwen/Qwen3-1.7B-Base \\
        --checkpoint-path runs/ckpt/final_model --max-rank 2048 \\
        --coordinate-counts 1 2 3 4 6 8 12 --draws 12 --out offdiag.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from capability_gated import (  # noqa: E402
    MMLU_ADJACENT,
    MMLU_DISTANT,
    WMDP_TASKS,
    MCQEvaluator,
    SplitRegistry,
    evaluate_perplexity_at_rank,
    load_lowrank_checkpoint,
    load_task,
    pool_mcq_results,
    load_or_build_ppl_blocks,
)
from capability_gated.suppression_search import apply_config  # noqa: E402

MODULES = ("gate_proj", "up_proj", "down_proj")
MACROS = {"mmlu_adjacent_macro": sorted(set(MMLU_ADJACENT.values())),
          "mmlu_distant_macro": list(MMLU_DISTANT)}


def log(msg: str) -> None:
    print(msg, flush=True)


def draw_configs(
    coords: list[tuple[int, str]],
    variants: list[int],
    counts: list[int],
    draws: int,
    *,
    seed: int,
    strategy: str,
) -> dict[int, list[dict[tuple[int, str], int]]]:
    """``{k: [config, ...]}``, mirroring ``LowRankTrainer._apply_mixed_variant``.

    Coordinates uniformly without replacement; ranks by the trainer's own
    strategy. ``rank_weighted`` weights each candidate rank *by its own value*,
    so gentle gating is drawn far more often than deep gating -- which is what
    adaptation saw, and therefore the distribution E0 must ask about.

    Seeded from ``seed`` alone, so every checkpoint receives identical draws.
    """
    out: dict[int, list[dict[tuple[int, str], int]]] = {}
    for k in counts:
        rng = random.Random(seed * 1000 + k)   # per-k stream: adding a k later
        configs = []                           # cannot shift the earlier ones
        for _ in range(draws):
            picked = rng.sample(coords, k=k)
            if strategy == "rank_weighted":
                ranks = rng.choices(variants, weights=variants, k=k)
            else:
                ranks = [rng.choice(variants) for _ in range(k)]
            configs.append(dict(zip(picked, ranks, strict=True)))
        out[k] = configs
    return out


def main() -> None:  # noqa: PLR0915 - one linear measurement reads best flat
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--max-rank", type=int, required=True)
    p.add_argument("--train-ranks", type=int, nargs="+", default=None,
                   help="The adaptation ladder. Default: read from the "
                        "checkpoint's config.json, which is what makes the "
                        "draws match what this model actually rehearsed.")
    p.add_argument("--sampling-strategy", default=None,
                   choices=["rank_weighted", "uniform"],
                   help="Default: the checkpoint's own. Two checkpoints being "
                        "compared MUST use the same one, or the draws differ.")
    p.add_argument("--coordinate-counts", type=int, nargs="+",
                   default=[1, 2, 3, 4, 6, 8, 12],
                   help="Values of k. Adaptation only ever rehearsed 1-3, so "
                        "the larger values test generalisation beyond the "
                        "envelope -- where the recipes should differ most.")
    p.add_argument("--draws", type=int, default=12,
                   help="Random configurations per k. Variance across draws is "
                        "large, which is why the draws are shared and paired.")
    p.add_argument("--splits", default="mech_interp_results/splits_v1.json")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--tokens-per-batch", type=int, default=8192)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--ppl-samples", type=int, default=256)
    p.add_argument("--ppl-batch-size", type=int, default=4)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--skip-mcq", action="store_true",
                   help="Perplexity only. The MCQ macros cost most of the "
                        "wall-clock and are too noisy to bind (adjacent SE "
                        "3.64pp); drop them for a quick pass.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="offdiagonal_sweep.json")
    args = p.parse_args()

    ckpt_cfg_path = Path(args.checkpoint_path).parent / "config.json"
    ckpt_cfg: dict[str, Any] = (
        json.loads(ckpt_cfg_path.read_text()) if ckpt_cfg_path.exists() else {}
    )
    resolved = ckpt_cfg.get("resolved", {}) or {}

    ladder = args.train_ranks or ckpt_cfg.get("train_ranks") or resolved.get("train_ranks")
    if not ladder:
        raise SystemExit(
            "no rank ladder: pass --train-ranks, or point --checkpoint-path at a "
            "final_model/ whose parent config.json records train_ranks.")
    ladder = sorted({int(r) for r in ladder}, reverse=True)
    variants = [r for r in ladder if r < args.max_rank]
    strategy = (args.sampling_strategy or ckpt_cfg.get("sampling_strategy")
                or resolved.get("sampling_strategy") or "rank_weighted")
    block_size = args.block_size or int(ckpt_cfg.get("block_size") or resolved.get("block_size", 256))

    log(f"ladder {ladder}")
    log(f"variant ranks {variants}  |  sampling {strategy}  |  block_size {block_size}")

    model, tokenizer = load_lowrank_checkpoint(
        args.model_id, args.checkpoint_path, max_rank=args.max_rank,
        dtype=getattr(torch, args.dtype),
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=args.cache_dir)
    model.eval()

    n_blocks = len(model.model.layers) if hasattr(model, "model") else None
    coords = [(b, m) for b in range(n_blocks) for m in MODULES]
    log(f"{len(coords)} gated coordinates ({n_blocks} blocks x {len(MODULES)} modules)")

    configs = draw_configs(coords, variants, args.coordinate_counts, args.draws,
                           seed=args.seed, strategy=strategy)
    total = sum(len(v) for v in configs.values())
    log(f"{total} configurations ({args.draws} draws x {len(args.coordinate_counts)} values of k)")

    ppl_split = load_or_build_ppl_blocks(
        tokenizer, block_size=block_size,
        num_workers=int(ckpt_cfg.get("num_workers") or resolved.get("num_workers", 8)),
        cache_dir=args.cache_dir)
    ppl_ds = ppl_split.select(range(min(args.ppl_samples, len(ppl_split))))

    evaluator = None
    if not args.skip_mcq:
        all_tasks = [*WMDP_TASKS, *sorted(set(MMLU_ADJACENT.values())), *MMLU_DISTANT]
        items = {t: load_task(t, cache_dir=args.cache_dir) for t in all_tasks}
        registry = SplitRegistry.load(REPO_ROOT / args.splits)
        registry.verify(items)
        split = {t: registry.take(v, t, "A") for t, v in items.items()}
        evaluator = MCQEvaluator(model, tokenizer, split, max_rank=args.max_rank,
                                 max_seq_len=args.max_seq_len,
                                 max_tokens_per_batch=args.tokens_per_batch)

    def measure(cfg: dict) -> dict[str, float]:
        try:
            apply_config(model, cfg, args.max_rank)
            ppl, nll = evaluate_perplexity_at_rank(
                model, tokenizer, ppl_ds, rank=None, batch_size=args.ppl_batch_size)
            row = {"ppl": float(ppl), "nll": float(nll)}
            if evaluator is not None:
                res = dict(evaluator.evaluate_current())
                for name, members in MACROS.items():
                    row[name] = pool_mcq_results(res, members, name=name).acc
        finally:
            model._set_rank_all(args.max_rank)
        return row

    t0 = time.perf_counter()
    base = measure({})
    log(f"full privilege: ppl {base['ppl']:.4f}  ({time.perf_counter() - t0:.0f}s)")

    rows: list[dict[str, Any]] = []
    for k in args.coordinate_counts:
        t_k = time.perf_counter()
        for i, cfg in enumerate(configs[k]):
            m = measure(cfg)
            row = {"k": k, "draw": i,
                   "cfg": [{"block": b, "module": mod, "rank": r}
                           for (b, mod), r in sorted(cfg.items())],
                   "ppl": m["ppl"], "ppl_ratio": m["ppl"] / base["ppl"]}
            for name in MACROS:
                if name in m:
                    row[f"{name}_drop_pp"] = 100 * (base[name] - m[name])
            rows.append(row)
        ratios = [r["ppl_ratio"] for r in rows if r["k"] == k]
        log(f"  k={k:<3} ppl x{np.mean(ratios):.4f} "
            f"[{np.min(ratios):.4f}, {np.max(ratios):.4f}]  "
            f"({time.perf_counter() - t_k:.0f}s)")

    summary = []
    for k in args.coordinate_counts:
        sel = [r for r in rows if r["k"] == k]
        ratios = np.array([r["ppl_ratio"] for r in sel])
        # Percentile band over draws, not a standard error: the spread across
        # random configurations is the quantity of interest, not noise to be
        # averaged away -- two recipes can share a mean and differ in tail.
        entry = {"k": k, "n_draws": len(sel),
                 "ppl_ratio_mean": float(ratios.mean()),
                 "ppl_ratio_p10": float(np.quantile(ratios, 0.10)),
                 "ppl_ratio_p90": float(np.quantile(ratios, 0.90)),
                 "ppl_ratio_max": float(ratios.max())}
        for name in MACROS:
            vals = [r[f"{name}_drop_pp"] for r in sel if f"{name}_drop_pp" in r]
            if vals:
                entry[f"{name}_drop_pp_mean"] = float(np.mean(vals))
        summary.append(entry)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "capability_gated.offdiagonal/v1",
        "note": "E0 off-diagonal usability: degradation vs number of gated "
                "coordinates, random draws mirroring the adaptation sampler.",
        "checkpoint_path": args.checkpoint_path,
        "recipe": {"variant_mode": ckpt_cfg.get("variant_mode"),
                   "sampling_strategy": strategy, "ladder": ladder,
                   "max_steps": ckpt_cfg.get("max_steps")},
        "draw_provenance": {
            "seed": args.seed, "coordinate_counts": args.coordinate_counts,
            "draws": args.draws, "n_coordinates": len(coords),
            "note": "Draws depend only on seed, ladder and strategy -- never on "
                    "the model -- so checkpoints compared at the same seed and "
                    "ladder receive identical configurations and pair exactly.",
        },
        "baseline": base, "summary": summary, "rows": rows,
        "elapsed_seconds": time.perf_counter() - t0,
        "cli": vars(args),
    }, indent=2, default=float))
    log(f"\nwrote {out}")


if __name__ == "__main__":
    main()
