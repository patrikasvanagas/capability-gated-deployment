"""Per-block / per-MLP rank sweep that produces the JSON for downstream analysis.

For a trained capability-gated checkpoint, this reduces the rank of one MLP
projection in one transformer block at a time and evaluates MMLU subject accuracy
with the ``lm-eval`` harness. The resulting JSON feeds:

* :func:`capability_gated.analysis.plot_rank_drop_heatmaps` (via
  :func:`capability_gated.analysis.sweep_payload_to_rank_reduced_dict`), and
* :func:`capability_gated.suppression_search.optimise_task_suppression` (via
  :func:`capability_gated.analysis.sweep_payload_to_single_intervention_json`).

``lm-eval`` is imported lazily, so importing this module does not require it.
Install the optional extra to run: ``pip install lm-eval peft``.

Output payload schema (one file per task group)::

    {
      "metadata": {... base_id, max_rank, tasks, ranks_to_try, ...},
      "baseline": {task -> {"acc,none": float, "acc_stderr,none": float, ...}},
      "sweeps":   [{"block_idx", "module_name", "rank", "rank_percentage", "results"}]
    }

CLI example::

    python -m capability_gated.mlp_rank_sweep \\
        --base-id Qwen/Qwen2.5-0.5B \\
        --checkpoint-path runs/qwen/final_model \\
        --max-rank 896 --task-groups college
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .training import load_lowrank_checkpoint

TASK_GROUPS = {
    "high_school": [
        "mmlu_high_school_biology",
        "mmlu_high_school_chemistry",
        "mmlu_high_school_computer_science",
        "mmlu_high_school_mathematics",
        "mmlu_high_school_physics",
    ],
    "college": [
        "mmlu_college_biology",
        "mmlu_college_chemistry",
        "mmlu_college_computer_science",
        "mmlu_college_mathematics",
        "mmlu_college_physics",
    ],
}

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def make_ranks_to_try(max_rank: int, rank_percentages: list[float]) -> list[int]:
    """Turn rank fractions into a sorted, de-duplicated list of integer ranks."""
    ranks = {
        max(1, min(max_rank, int(max_rank * percentage)))
        for percentage in rank_percentages
    }
    return sorted(ranks, reverse=True)


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _run_lm_eval(lm, tasks, num_fewshot, eval_batch_size, device) -> dict:
    from lm_eval import evaluator  # lazy: optional dependency

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=eval_batch_size,
        device=device,
        verbosity="NOTSET",
    )
    return results["results"]


def run_mlp_rank_sweep(
    model,
    tokenizer,
    max_rank: int,
    tasks: list[str],
    output_path: str | Path,
    *,
    base_id: str = "",
    checkpoint_path: str = "",
    target_substring: str = "mlp",
    rank_percentages: list[float] | None = None,
    mlp_module_names: list[str] | None = None,
    num_fewshot: int = 5,
    lm_batch_size: int = 16,
    eval_batch_size: int | str = "auto",
    device: str = "cuda:0",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the full-rank baseline + every single-MLP rank ablation, saving JSON.

    ``model`` must expose ``_set_rank_all``, ``_get_transformer_blocks`` and
    ``_set_rank_for_named_module_at_block`` (as attached by
    :func:`~capability_gated.rank_control.attach_rank_control`). Results are saved
    incrementally to ``output_path`` after each evaluation.

    Returns the payload dict.
    """
    from lm_eval.models.huggingface import HFLM  # lazy: optional dependency

    if rank_percentages is None:
        rank_percentages = [0.95, 0.90, 0.80, 0.65, 0.50, 0.25, 0.15, 0.05]
    if mlp_module_names is None:
        mlp_module_names = ["gate_proj", "up_proj", "down_proj"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass overwrite=True to replace it."
        )

    ranks_to_try = make_ranks_to_try(max_rank, rank_percentages)
    num_blocks = len(model._get_transformer_blocks())

    payload: dict[str, Any] = {
        "metadata": {
            "base_id": base_id,
            "checkpoint_path": checkpoint_path,
            "max_rank": max_rank,
            "target_substring": target_substring,
            "tasks": tasks,
            "num_fewshot": num_fewshot,
            "lm_batch_size": lm_batch_size,
            "eval_batch_size": eval_batch_size,
            "device": device,
            "rank_percentages": rank_percentages,
            "ranks_to_try": ranks_to_try,
            "mlp_module_names": mlp_module_names,
            "num_transformer_blocks": num_blocks,
        },
        "baseline": None,
        "sweeps": [],
    }

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=lm_batch_size)

    logging.info("Running full-rank baseline")
    model._set_rank_all(max_rank)
    payload["baseline"] = _run_lm_eval(
        lm, tasks, num_fewshot, eval_batch_size, device
    )
    _save_json(output_path, payload)

    total = num_blocks * len(mlp_module_names) * len(ranks_to_try)
    progress = tqdm(total=total, desc="rank sweep")

    for block_idx in range(num_blocks):
        for module_name in mlp_module_names:
            for attempted_rank in ranks_to_try:
                try:
                    model._set_rank_for_named_module_at_block(
                        block_idx=block_idx,
                        module_name=module_name,
                        rank=attempted_rank,
                    )
                    results = _run_lm_eval(
                        lm, tasks, num_fewshot, eval_batch_size, device
                    )
                    payload["sweeps"].append(
                        {
                            "block_idx": block_idx,
                            "module_name": module_name,
                            "rank": attempted_rank,
                            "rank_percentage": round(attempted_rank / max_rank, 6),
                            "results": results,
                        }
                    )
                    _save_json(output_path, payload)
                finally:
                    model._set_rank_all(max_rank)
                    progress.update(1)

    progress.close()
    logging.info("Saved results to %s", output_path)
    return payload


def _build_output_path(output_dir: str, group_name: str, base_id: str, num_fewshot: int) -> Path:
    model_name = base_id.split("/")[-1]
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{group_name}_mlp_rank_sweep_{model_name}_{num_fewshot}shot.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-block/per-MLP rank sweep for a trained capability-gated LM."
    )
    parser.add_argument("--base-id", type=str, required=True, help="Base HF model id.")
    parser.add_argument(
        "--checkpoint-path", type=str, required=True, help="Trained checkpoint dir/file."
    )
    parser.add_argument("--max-rank", type=int, required=True, help="Model max rank.")
    parser.add_argument("--target-substring", type=str, default="mlp")
    parser.add_argument(
        "--task-groups",
        nargs="+",
        choices=sorted(TASK_GROUPS.keys()),
        default=["high_school", "college"],
    )
    parser.add_argument(
        "--rank-percentages",
        type=float,
        nargs="+",
        default=[0.95, 0.90, 0.80, 0.65, 0.50, 0.25, 0.15, 0.05],
    )
    parser.add_argument(
        "--mlp-module-names", nargs="+", default=["gate_proj", "up_proj", "down_proj"]
    )
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--lm-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=str, default="auto")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--output-dir", type=str, default="sweep_results")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    eval_batch_size: int | str = (
        args.eval_batch_size
        if args.eval_batch_size == "auto"
        else int(args.eval_batch_size)
    )

    model, tokenizer = load_lowrank_checkpoint(
        base_model_id=args.base_id,
        checkpoint_path=args.checkpoint_path,
        max_rank=args.max_rank,
        target_substring=args.target_substring,
        dtype=_DTYPES[args.dtype],
        device_map=args.device,
    )
    model.eval()

    for group_name in args.task_groups:
        output_path = _build_output_path(
            args.output_dir, group_name, args.base_id, args.num_fewshot
        )
        run_mlp_rank_sweep(
            model=model,
            tokenizer=tokenizer,
            max_rank=args.max_rank,
            tasks=TASK_GROUPS[group_name],
            output_path=output_path,
            base_id=args.base_id,
            checkpoint_path=args.checkpoint_path,
            target_substring=args.target_substring,
            rank_percentages=args.rank_percentages,
            mlp_module_names=args.mlp_module_names,
            num_fewshot=args.num_fewshot,
            lm_batch_size=args.lm_batch_size,
            eval_batch_size=eval_batch_size,
            device=args.device,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
