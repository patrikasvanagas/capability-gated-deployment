#!/usr/bin/env python
"""Score WMDP/MMLU accuracy for a base or capability-gated model.

Uses ``capability_gated.mcq`` -- the *same* evaluator that produced the E1
headroom table -- so numbers here are directly comparable to
``e2_localisation/<stamp>/headroom.json`` rather than merely nominally the same
metric. lm-eval would measure the same quantity through a different prompt
assembly, batching and dataset release, which turns a model comparison into a
model-plus-harness comparison.

Two model kinds:

* **Base / any HF causal LM** -- pass ``--model-id`` alone. The scorer only
  needs ``get_decoder()`` and ``get_output_embeddings()``, so no rank machinery
  is involved.
* **Capability-gated checkpoint** -- add ``--checkpoint-path`` and
  ``--max-rank``; ``--rank`` optionally evaluates at a reduced global rank
  (default: full privilege).

Scoring uses **split A** of the frozen registry, the same half every adaptive
step of the protocol runs on. Split B stays locked unless ``--unlock-reason``
is given, which is recorded in the output as the audit trail.

Examples::

    # base model (the capability-erosion reference)
    uv run python scripts/eval_mcq.py --model-id Qwen/Qwen3-1.7B-Base \\
        --out runs/mcq_qwen3-1.7b-base.json

    # our fine-tuned checkpoint, full privilege
    uv run python scripts/eval_mcq.py --model-id Qwen/Qwen3-1.7B-Base \\
        --checkpoint-path runs/qwen3-17b-fineweb-r2048/final_model \\
        --max-rank 2048 --baseline-json runs/mcq_qwen3-1.7b-base.json \\
        --out runs/mcq_qwen3-1.7b-gated.json

    # a bigger base model, to inform a scale decision
    uv run python scripts/eval_mcq.py --model-id Qwen/Qwen3-4B-Base \\
        --baseline-json runs/mcq_qwen3-1.7b-base.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from capability_gated.mcq import (  # noqa: E402
    CHANCE,
    MMLU_ADJACENT,
    MMLU_DISTANT,
    WMDP_TASKS,
    MCQResult,
    SplitRegistry,
    build_batches,
    evaluate_batches,
    letter_token_ids,
    load_task,
    subject_for,
)
from capability_gated.training import load_lowrank_checkpoint  # noqa: E402

DEFAULT_TASKS = (*WMDP_TASKS, *sorted(set(MMLU_ADJACENT.values())), *MMLU_DISTANT)
SPLITS_PATH = REPO_ROOT / "mech_interp_results/splits_v1.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-id", required=True, help="HF id of the base model.")
    p.add_argument(
        "--checkpoint-path",
        default=None,
        help="Gated checkpoint dir/file; omit to score the base model itself.",
    )
    p.add_argument(
        "--max-rank",
        type=int,
        default=None,
        help="r_max of the checkpoint (required with --checkpoint-path).",
    )
    p.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Evaluate the gated model at this global rank (default: full).",
    )
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--split", choices=["A", "B"], default="A")
    p.add_argument(
        "--unlock-reason",
        default=None,
        help="Required to read split B; recorded in the output.",
    )
    p.add_argument("--item-cap", type=int, default=None, help="Cap items per task.")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--tokens-per-batch", type=int, default=8192)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device-map", default=None, help="Default: 'auto' on CUDA.")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--out", default=None, help="Write results JSON here.")
    p.add_argument(
        "--baseline-json",
        default=None,
        help="Previous output to diff against (prints a delta column).",
    )
    return p


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    dtype = getattr(torch, args.dtype)
    device_map = args.device_map
    if device_map is None:
        device_map = "auto" if torch.cuda.is_available() else None

    if args.checkpoint_path:
        if args.max_rank is None:
            raise SystemExit("--max-rank is required with --checkpoint-path.")
        log(f"loading gated checkpoint {args.checkpoint_path} (r_max={args.max_rank})")
        model, tokenizer = load_lowrank_checkpoint(
            args.model_id,
            args.checkpoint_path,
            max_rank=args.max_rank,
            dtype=dtype,
            device_map=device_map,
            cache_dir=args.cache_dir,
        )
        if args.rank is not None:
            model._set_rank_all(args.rank)
            log(f"gated to global rank {args.rank}")
    else:
        log(f"loading base model {args.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            device_map=device_map,
            cache_dir=args.cache_dir,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # build_batches refuses left padding: decoder position_ids default to
    # arange(T), so left padding silently corrupts RoPE positions.
    tokenizer.padding_side = "right"
    model.eval()
    return model, tokenizer


def run(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer = load_model(args)
    device = next(model.parameters()).device
    letter_ids = letter_token_ids(tokenizer)

    items_by_task = {
        task: load_task(task, cache_dir=args.cache_dir) for task in args.tasks
    }
    registry = SplitRegistry.load(SPLITS_PATH)
    # Guards against a changed dataset release masquerading as the frozen one.
    registry.verify(items_by_task)
    scored = {
        task: registry.take(
            items, task, args.split, unlock_reason=args.unlock_reason
        )
        for task, items in items_by_task.items()
    }
    if args.item_cap:
        scored = {t: items[: args.item_cap] for t, items in scored.items()}

    results: dict[str, MCQResult] = {}
    for task, items in scored.items():
        batches, _ = build_batches(
            items,
            tokenizer,
            subject=subject_for(task),
            max_seq_len=args.max_seq_len,
            max_tokens_per_batch=args.tokens_per_batch,
            device=device,
        )
        results[task] = evaluate_batches(
            model, batches, letter_ids, task=task, n_items=len(items)
        )
        r = results[task]
        log(f"{task:32s} acc {r.acc:.4f}  n={r.n:<5} ({r.seconds:.0f}s)")

    return {
        "model": {
            "model_id": args.model_id,
            "checkpoint_path": args.checkpoint_path,
            "max_rank": args.max_rank,
            "rank": args.rank,
            "dtype": args.dtype,
            "kind": "gated" if args.checkpoint_path else "base",
        },
        "split": args.split,
        "unlock_reason": args.unlock_reason,
        "item_cap": args.item_cap,
        "chance": CHANCE,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "results": {t: r.to_json() for t, r in results.items()},
    }


def report(payload: dict[str, Any], baseline: dict[str, Any] | None) -> None:
    base_results = (baseline or {}).get("results", {})
    head = f"{'task':32s} {'acc':>8} {'wilson':>18} {'margin':>9} {'n':>6}"
    if base_results:
        head += f" {'Δ vs base':>11}"
    print("\n" + head)
    print("-" * len(head))
    for task, r in payload["results"].items():
        wilson = f"[{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}]"
        line = (
            f"{task:32s} {r['acc']:>8.4f} {wilson:>18} "
            f"{r['mean_margin']:>+9.3f} {r['n']:>6}"
        )
        if base_results and task in base_results:
            delta_pp = 100 * (r["acc"] - base_results[task]["acc"])
            line += f" {delta_pp:>+10.1f}pp"
        print(line)
    print(f"\nchance = {payload['chance']:.2f}; split {payload['split']}")


def main() -> None:
    args = build_parser().parse_args()
    payload = run(args)
    baseline = (
        json.loads(Path(args.baseline_json).read_text()) if args.baseline_json else None
    )
    report(payload, baseline)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        log(f"wrote {out}")


if __name__ == "__main__":
    main()
