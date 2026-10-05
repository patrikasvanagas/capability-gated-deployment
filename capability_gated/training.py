"""Next-token-prediction fine-tuning of capability-gated (NLPN) language models.

The pipeline:

1. :func:`build_lowrank_model` -- load a HuggingFace causal LM, replace its MLP
   linears with SVD-initialized :class:`DynamicLowRankLinear` layers, and attach
   the rank-gating API plus the training bookkeeping (``max_rank``,
   ``train_ranks``, ``log_vars_ce``).
2. :func:`prepare_lm_dataset` -- tokenize and pack a text dataset into fixed-size
   next-token-prediction blocks.
3. :class:`LowRankTrainer` -- a ``transformers.Trainer`` that, at each step,
   evaluates an *anchor* (full) rank and a sampled variant.  The variant can be
   a global rank (the original recipe), a sparse per-coordinate configuration,
   or a mixture of the two.  Their losses are combined with a learned
   uncertainty weighting, so the model stays usable away from full rank.
4. :func:`finetune` -- a convenience wrapper that runs the whole thing.

Reload a trained model with :func:`load_lowrank_checkpoint`.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from safetensors.torch import load_file
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .lowrank import (
    DynamicLowRankLinear,
    decompose_and_replace,
    load_text_causal_lm,
    replace_linear_with_lowrank,
)
from .rank_control import attach_rank_control


def geometric_rank_ladder(max_rank: int) -> list[int]:
    """Return ``[max_rank, max_rank//2, max_rank//4, ..., 1]`` (strictly positive)."""
    return [
        max_rank // (2**k)
        for k in range(int(math.log2(max_rank)) + 1)
        if max_rank // (2**k) > 0
    ]


def build_lowrank_model(
    base_model_id: str,
    max_rank: int,
    target_substring: str = "mlp",
    train_ranks: list[int] | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = "auto",
    cache_dir: str | None = None,
) -> tuple[nn.Module, Any]:
    """Load a causal LM and convert its MLP linears into gated low-rank layers.

    Attaches the attributes/methods the trainer and gating API rely on:
    ``max_rank``, ``train_ranks``, ``log_vars_ce`` (a learnable log-variance per
    rank), and the ``_set_rank_*`` methods from
    :func:`~capability_gated.rank_control.attach_rank_control`.

    Returns:
        ``(model, tokenizer)``.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Loads the wrapper, prunes any image/audio towers, and applies pad_token_id
    # via the config rather than a from_pretrained kwarg that multimodal classes
    # reject. No-op beyond a plain load for text-only models like Qwen3.
    model, dropped = load_text_causal_lm(
        base_model_id,
        pad_token_id=tokenizer.pad_token_id,
        dtype=dtype,
        device_map=device_map,
        cache_dir=cache_dir,
    )
    if dropped:
        print(f"dropped non-text towers: {', '.join(dropped)}")

    # Caching (past_key_values) is for generation, not training.
    model.config.use_cache = False

    # Target by name substring across the whole model, so this works regardless
    # of whether the backbone lives under `.model`, `.transformer` or `.gpt_neox`.
    decompose_and_replace(model, max_rank=max_rank, target_substring=target_substring)

    model.max_rank = max_rank
    model.train_ranks = train_ranks or geometric_rank_ladder(max_rank)
    model.log_vars_ce = nn.ParameterDict(
        {str(r): nn.Parameter(torch.zeros(())) for r in model.train_ranks},
    ).to(model.device)

    attach_rank_control(model)
    print(
        f"Built low-rank model from {base_model_id!r}: "
        f"max_rank={max_rank}, train_ranks={model.train_ranks}",
    )
    return model, tokenizer


def _get_text_field(dataset: Dataset) -> str:
    """Best-effort detection of the text column in a HuggingFace dataset."""
    for candidate in ("text", "content", "raw", "data"):
        if candidate in dataset.column_names:
            return candidate
    for col, feat in dataset.features.items():
        if getattr(feat, "dtype", None) == "string":
            return col
    raise ValueError(
        f"No obvious text column in dataset columns={dataset.column_names}.",
    )


def prepare_lm_dataset(
    dataset_or_name: str | Dataset | DatasetDict,
    tokenizer: Any,
    block_size: int = 256,
    num_workers: int = 4,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
    text_field: str | None = None,
    cache_dir: str | None = None,
) -> Dataset | DatasetDict:
    """Tokenize and pack text into fixed-size next-token-prediction blocks.

    Accepts a loaded ``Dataset``/``DatasetDict`` or a dataset name to load.
    ``dataset_split`` can restrict a named dataset before preprocessing, which
    avoids tokenizing unused splits during evaluation. Text is tokenized without
    truncation, concatenated, and split into ``block_size`` chunks; ``labels``
    is a copy of ``input_ids`` (HF shifts internally).
    """
    if isinstance(dataset_or_name, str):
        dataset = load_dataset(
            dataset_or_name, dataset_config, split=dataset_split,
            cache_dir=cache_dir,
        )
    else:
        dataset = dataset_or_name

    # Pick a representative split to detect the text column.
    probe = dataset[next(iter(dataset.keys()))] if isinstance(
        dataset, DatasetDict,
    ) else dataset
    col = text_field or _get_text_field(probe)

    def tokenize_function(examples: dict) -> dict:
        return tokenizer(
            examples[col],
            return_attention_mask=False,
            truncation=False,  # we pack/chunk ourselves
        )

    def group_texts(examples: dict) -> dict:
        concatenated = {k: sum(examples[k], []) for k in examples}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    remove_columns = probe.column_names
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=remove_columns,
    )
    return tokenized.map(group_texts, batched=True, num_proc=num_workers)


class LowRankTrainer(Trainer):
    """Trainer implementing the multi-rank uncertainty-weighted NTP objective.

    At each step it evaluates the anchor (full) rank and one sampled variant,
    then minimizes ``sum_r exp(-s_r) * nll_r + s_r`` where ``s_r`` is the
    learned log-variance for rank ``r`` (Kendall et al., 2018).  A variant is
    either the original global-rank configuration or a sparse set of low-rank
    module overrides.  For an independently ranked sparse configuration its
    uncertainty scalar is the mean of the selected ranks' scalars.  A short
    warmup phase trains the anchor rank only. ``log_vars_ce`` parameters are
    optimized with a separate, higher learning rate.

    Args:
        sampling_strategy: how variant ranks are drawn -- one of ``"uniform"``,
            ``"rank_weighted"``, ``"curriculum"``, ``"epoch_curriculum"``, or
            ``"faster_curriculum"``.
        variant_mode: ``"global"`` (original recipe), ``"mixed"`` (always
            sparse per-coordinate), or ``"hybrid"`` (sample between them).
        mixed_probability: probability of choosing a sparse variant in hybrid
            mode.
        mixed_coordinate_counts / mixed_coordinate_weights: categorical
            distribution for the number of independently gated modules.
        warmup_proportion: fraction of training spent in the anchor-only warmup.
        log_var_lr: learning rate for the ``log_vars_ce`` parameters.

    All other arguments are forwarded to ``transformers.Trainer``. The legacy
    ``tokenizer=`` kwarg is accepted and mapped onto ``processing_class`` (see
    :meth:`__init__`).
    """

    def __init__(
        self,
        *args: Any,
        sampling_strategy: str = "uniform",
        variant_mode: str = "global",
        mixed_probability: float = 0.5,
        mixed_coordinate_counts: list[int] | None = None,
        mixed_coordinate_weights: list[float] | None = None,
        warmup_proportion: float = 0.15,
        log_var_lr: float = 1e-2,
        **kwargs: Any,
    ) -> None:
        # transformers renamed Trainer's tokenizer slot to `processing_class` in
        # 4.46 and dropped `tokenizer=` entirely in 5.0. Accept the old spelling
        # so call sites written against transformers 4.x keep working.
        if "tokenizer" in kwargs:
            kwargs.setdefault("processing_class", kwargs.pop("tokenizer"))

        super().__init__(*args, **kwargs)
        self.sampling_strategy = sampling_strategy
        self.variant_mode = variant_mode
        self.mixed_probability = float(mixed_probability)
        self.mixed_coordinate_counts = mixed_coordinate_counts or [1, 2, 3]
        self.mixed_coordinate_weights = mixed_coordinate_weights or [0.7, 0.2, 0.1]
        self.warmup_proportion = warmup_proportion
        self.log_var_lr = log_var_lr

        assert hasattr(self.model, "train_ranks"), "Model must have train_ranks."
        assert hasattr(self.model, "log_vars_ce"), "Model must have log_vars_ce."
        assert hasattr(self.model, "_set_rank_all"), "Model must have _set_rank_all."
        assert hasattr(self.model, "max_rank"), "Model must have max_rank."

        if self.variant_mode not in {"global", "mixed", "hybrid"}:
            raise ValueError(f"Unknown variant_mode={self.variant_mode!r}.")
        if not 0.0 <= self.mixed_probability <= 1.0:
            raise ValueError("mixed_probability must be in [0, 1].")
        if len(self.mixed_coordinate_counts) != len(self.mixed_coordinate_weights):
            raise ValueError(
                "mixed_coordinate_counts and mixed_coordinate_weights must have "
                "the same length.",
            )
        if not self.mixed_coordinate_counts or any(
            int(count) < 1 for count in self.mixed_coordinate_counts
        ):
            raise ValueError("mixed coordinate counts must all be positive integers.")
        if any(float(weight) < 0 for weight in self.mixed_coordinate_weights) or not any(
            float(weight) > 0 for weight in self.mixed_coordinate_weights
        ):
            raise ValueError("mixed coordinate weights must be non-negative and not all zero.")

        self._mixed_modules = [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, DynamicLowRankLinear)
        ]
        if self.variant_mode != "global" and not self._mixed_modules:
            raise ValueError("mixed training requested, but the model has no gated modules.")
        if self._mixed_modules and max(self.mixed_coordinate_counts) > len(self._mixed_modules):
            raise ValueError(
                "mixed coordinate count exceeds the number of gated modules: "
                f"max count={max(self.mixed_coordinate_counts)}, "
                f"modules={len(self._mixed_modules)}.",
            )

    def create_optimizer(self):  # noqa: ANN201
        """Optimize model params at the base LR and log-variances at ``log_var_lr``."""
        log_var_params = list(self.model.log_vars_ce.parameters())
        log_var_ids = {id(p) for p in log_var_params}
        model_params = [
            p for p in self.model.parameters() if id(p) not in log_var_ids
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": model_params, "lr": self.args.learning_rate},
                {"params": log_var_params, "lr": self.log_var_lr},
            ],
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer

    def _sample_variant_rank(self, variants: list[int]) -> int:
        """Pick a variant (lower) rank according to the sampling strategy."""
        progress = self.state.global_step / max(1, self.state.max_steps)

        if self.sampling_strategy == "rank_weighted":
            return random.choices(variants, weights=variants, k=1)[0]

        if self.sampling_strategy == "curriculum":
            num_stages = len(variants)
            stage_idx = min(int(progress * num_stages), num_stages - 1)
            available = variants[: stage_idx + 1] or [variants[0]]
            return random.choice(available)

        if self.sampling_strategy == "epoch_curriculum":
            if self.state.epoch is not None and self.state.epoch < 1.0:
                num_stages = len(variants)
                stage_idx = min(int(progress * num_stages), num_stages - 1)
                available = variants[: stage_idx + 1] or [variants[0]]
                return random.choice(available)
            return random.choice(variants)

        if self.sampling_strategy == "faster_curriculum":
            num_variants = len(variants)
            if progress < 0.5:
                phase1 = progress / 0.5
                num_unlocked = min(int(phase1 * num_variants) + 1, num_variants)
                return random.choice(variants[:num_unlocked])
            return random.choice(variants)

        # Default: uniform over all lower ranks.
        return random.choice(variants)

    def _use_mixed_variant(self) -> bool:
        """Return whether this micro-batch should use a sparse configuration."""
        if self.variant_mode == "mixed":
            return True
        if self.variant_mode == "global":
            return False
        return random.random() < self.mixed_probability

    def _apply_mixed_variant(
        self,
        model: nn.Module,
        variants: list[int],
    ) -> list[tuple[str, int]]:
        """Reset to full rank, then apply independently sampled sparse overrides."""
        model._set_rank_all(model.max_rank)
        count = int(random.choices(
            self.mixed_coordinate_counts,
            weights=self.mixed_coordinate_weights,
            k=1,
        )[0])
        selected = random.sample(self._mixed_modules, k=count)
        applied: list[tuple[str, int]] = []
        for name, module in selected:
            rank = int(self._sample_variant_rank(variants))
            module.set_rank(rank)
            applied.append((name, rank))
        return applied

    def compute_loss(  # noqa: ANN201
        self,
        model: nn.Module,
        inputs: dict,
        return_outputs: bool = False,
        **kwargs: Any,
    ):
        if not model.training:
            # Evaluation: rank is set by `evaluate`; just run a standard forward.
            outputs = model(**inputs)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        r_anchor = model.max_rank
        variants = sorted(
            [r for r in model.train_ranks if r < r_anchor], reverse=True,
        )
        progress = self.state.global_step / max(1, self.state.max_steps)

        # --- Warmup: anchor rank only ---
        if progress < self.warmup_proportion or not variants:
            model._set_rank_all(r_anchor)
            outputs = model(**inputs)
            nll_anchor = outputs.loss
            log_var_a = torch.clamp(
                model.log_vars_ce[str(r_anchor)].to(nll_anchor.device), -10, 10,
            )
            loss = torch.exp(-log_var_a) * nll_anchor + log_var_a
            self.log(
                {
                    "loss": loss.item(),
                    "nll_anchor": nll_anchor.item(),
                    "phase": 0.0,
                },
            )
            return (loss, outputs) if return_outputs else loss

        # --- Joint anchor + sampled global or sparse variant ---
        model._set_rank_all(r_anchor)
        out_anchor = model(**inputs)
        nll_anchor = out_anchor.loss

        is_mixed = self._use_mixed_variant()
        if is_mixed:
            mixed_config = self._apply_mixed_variant(model, variants)
            variant_ranks = [rank for _, rank in mixed_config]
        else:
            r_variant = self._sample_variant_rank(variants)
            model._set_rank_all(r_variant)
            mixed_config = []
            variant_ranks = [r_variant]
        try:
            out_variant = model(**inputs)
            nll_variant = out_variant.loss
        finally:
            # Never leak a sampled deployment configuration into evaluation or
            # the next batch if the variant forward raises.
            model._set_rank_all(model.max_rank)

        log_var_a = torch.clamp(
            model.log_vars_ce[str(r_anchor)].to(nll_anchor.device), -10, 10,
        )
        # A mixed configuration can assign distinct ranks to its coordinates.
        # Averaging their learned log-variances gives the configuration a single
        # uncertainty weight without introducing an impossible parameter per
        # point in the exponentially large configuration lattice.
        log_var_v = torch.stack([
            model.log_vars_ce[str(rank)].to(nll_anchor.device)
            for rank in variant_ranks
        ]).mean()
        log_var_v = torch.clamp(log_var_v, -10, 10)

        gradient_floor = 1e-2
        loss_a = (torch.exp(-log_var_a) + gradient_floor) * nll_anchor + log_var_a
        loss_b = (torch.exp(-log_var_v) + gradient_floor) * nll_variant + log_var_v
        loss = loss_a + loss_b

        rank_min = min(variant_ranks)
        rank_mean = sum(variant_ranks) / len(variant_ranks)
        record = {
            "loss": loss.item(),
            f"nll_rank_{r_anchor}": nll_anchor.item(),
            "nll_variant": nll_variant.item(),
            "r_anchor": float(r_anchor),
            "r_variant_min": float(rank_min),
            "r_variant_mean": float(rank_mean),
            "variant_is_mixed": float(is_mixed),
            "variant_coordinate_count": float(len(mixed_config) if is_mixed else 0),
            f"s_rank_{r_anchor}": log_var_a.item(),
            "s_variant": log_var_v.item(),
        }
        if not is_mixed:
            record[f"nll_rank_{variant_ranks[0]}"] = nll_variant.item()
            record["r_variant"] = float(variant_ranks[0])
            record[f"s_rank_{variant_ranks[0]}"] = log_var_v.item()
        self.log(record)
        return (loss, out_anchor) if return_outputs else loss

    def evaluate(  # noqa: ANN201
        self,
        eval_dataset: torch.utils.data.Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ):
        """Evaluate loss/perplexity at every rank in ``model.train_ranks``."""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        original_rank = self.model.max_rank
        all_metrics: dict[str, float] = {}

        print("\n--- Running multi-rank evaluation ---")
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        for rank in sorted(self.model.train_ranks, reverse=True):
            print(f"Evaluating at rank {rank}")
            self.model._set_rank_all(rank)
            output = self.evaluation_loop(
                eval_dataloader,
                description=f"Evaluating rank {rank}",
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            metrics = dict(output.metrics)
            if "eval_loss" in output.metrics:
                metrics["eval_perplexity"] = float(
                    np.exp(min(20.0, output.metrics["eval_loss"])),
                )
            for k, v in metrics.items():
                key = k.replace(metric_key_prefix + "_", "")
                all_metrics[f"{metric_key_prefix}_rank_{rank}/{key}"] = v

        self.model._set_rank_all(original_rank)

        # Primary metric for load_best_model_at_end: loss at the full rank.
        max_rank_key = f"{metric_key_prefix}_rank_{original_rank}/loss"
        if max_rank_key in all_metrics:
            all_metrics[f"{metric_key_prefix}_loss"] = all_metrics[max_rank_key]

        self.log(all_metrics)
        return all_metrics


def finetune(
    base_model_id: str,
    max_rank: int,
    output_dir: str,
    dataset_or_name: str | Dataset | DatasetDict = "wikitext",
    dataset_config: str | None = "wikitext-2-raw-v1",
    *,
    seed: int = 42,
    max_steps: int = -1,
    num_train_epochs: float = 1.0,
    batch_size: int = 8,
    eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-5,
    block_size: int = 256,
    num_workers: int = 4,
    sampling_strategy: str = "uniform",
    variant_mode: str = "global",
    mixed_probability: float = 0.5,
    mixed_coordinate_counts: list[int] | None = None,
    mixed_coordinate_weights: list[float] | None = None,
    warmup_proportion: float = 0.15,
    bf16: bool = True,
    report_to: str = "none",
    resume_from_checkpoint: str | bool | None = None,
    logging_steps: int = 15,
    eval_steps: int = 500,
    save_steps: int = 500,
    device_map: str | None = "auto",
    cache_dir: str | None = None,
) -> tuple[LowRankTrainer, nn.Module, Any]:
    """Fine-tune a causal LM as a capability-gated (NLPN) model end to end.

    Builds the low-rank model, prepares the dataset, trains with
    :class:`LowRankTrainer`, and saves the final model to
    ``<output_dir>/final_model``. ``report_to`` defaults to ``"none"``; pass
    ``"wandb"`` to enable Weights & Biases logging.

    Returns:
        ``(trainer, model, tokenizer)``.
    """
    set_seed(seed)

    dtype = torch.bfloat16 if bf16 else torch.float32
    model, tokenizer = build_lowrank_model(
        base_model_id,
        max_rank=max_rank,
        dtype=dtype,
        device_map=device_map,
        cache_dir=cache_dir,
    )

    lm_dataset = prepare_lm_dataset(
        dataset_or_name,
        tokenizer,
        block_size=block_size,
        num_workers=num_workers,
        dataset_config=dataset_config,
        cache_dir=cache_dir,
    )
    train_dataset = lm_dataset["train"]
    if "validation" in lm_dataset:
        eval_dataset = lm_dataset["validation"]
    elif "test" in lm_dataset:
        eval_dataset = lm_dataset["test"]
    else:
        eval_dataset = None

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=base_model_id.rsplit("/", maxsplit=1)[-1],
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.02,
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        report_to=report_to,
        bf16=bf16,
        bf16_full_eval=bf16,
        dataloader_num_workers=num_workers,
    )

    trainer = LowRankTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        sampling_strategy=sampling_strategy,
        variant_mode=variant_mode,
        mixed_probability=mixed_probability,
        mixed_coordinate_counts=mixed_coordinate_counts,
        mixed_coordinate_weights=mixed_coordinate_weights,
        warmup_proportion=warmup_proportion,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_dir = f"{output_dir}/final_model"
    print(f"\n--- Saving final model to {final_dir} ---")
    save_lowrank_checkpoint(model, final_dir)
    tokenizer.save_pretrained(final_dir)

    return trainer, model, tokenizer


def save_lowrank_checkpoint(model: nn.Module, output_dir: str) -> None:
    """Save a gated model in the layout :func:`load_lowrank_checkpoint` reads back.

    ``save_pretrained`` defaults to writing the *original* checkpoint layout, which
    for some architectures is not the live module tree: GPT-NeoX / Pythia hold the
    LM head at ``lm_head`` in memory but write it as ``embed_out`` on disk
    (transformers >= 5). Our loader installs the low-rank structure by hand and then
    does a plain ``load_state_dict``, which cannot see through that rename -- it
    would leave the *pretrained* head in place and silently discard the fine-tuned
    one. Writing the live layout keeps both ends in one namespace.
    """
    import inspect

    kwargs: dict[str, Any] = {}
    if "save_original_format" in inspect.signature(model.save_pretrained).parameters:
        # transformers >= 5; on 4.x the two layouts already agree.
        kwargs["save_original_format"] = False
    model.save_pretrained(output_dir, **kwargs)


def _reconcile_checkpoint_keys(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    missing: list[str],
    unexpected: list[str],
) -> None:
    """Make sure every trained weight actually reached the model.

    ``strict=False`` is unavoidable here -- ``log_vars_ce`` is training-only
    bookkeeping, and a tied LM head is legitimately absent from the file -- but it
    also hides genuine mismatches. The one that bites in practice is an architecture
    whose on-disk names differ from its live names (see
    :func:`save_lowrank_checkpoint`): the head lands in ``unexpected``, the model's
    own head sits in ``missing``, and the run quietly evaluates a half-stale model.

    Anything left over is paired by shape and re-loaded; ambiguous pairings are
    refused rather than guessed, and anything unresolved raises.
    """
    tied: set[str] = set()
    if getattr(model.config, "tie_word_embeddings", False):
        tied = set(getattr(type(model), "_tied_weights_keys", None) or {})

    real_missing = [
        k for k in missing if not k.startswith("log_vars_ce") and k not in tied
    ]
    if not real_missing:
        return

    live = model.state_dict()
    spare = {k: state_dict[k] for k in unexpected if not k.startswith("log_vars_ce")}

    rescued: dict[str, torch.Tensor] = {}
    for key in real_missing:
        wanted = tuple(live[key].shape)
        candidates = [k for k, v in spare.items() if tuple(v.shape) == wanted]
        if len(candidates) == 1:
            source = candidates[0]
            rescued[key] = spare.pop(source)
            print(
                f"[load_lowrank_checkpoint] remapped {source!r} -> {key!r} "
                f"(renamed between save and load).",
            )

    if rescued:
        model.load_state_dict(rescued, strict=False)

    unresolved = [k for k in real_missing if k not in rescued]
    if unresolved:
        raise RuntimeError(
            "Checkpoint is missing weights the model needs, and they could not be "
            f"matched by shape: {unresolved}. Unused checkpoint keys: "
            f"{sorted(spare)}. This usually means the checkpoint was written for a "
            "different architecture or max_rank than the one being rebuilt.",
        )


def load_lowrank_checkpoint(
    base_model_id: str,
    checkpoint_path: str,
    max_rank: int,
    target_substring: str = "mlp",
    train_ranks: list[int] | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = "auto",
    cache_dir: str | None = None,
) -> tuple[nn.Module, Any]:
    """Rebuild a trained capability-gated model from a saved checkpoint.

    Loads the base model, installs the low-rank *structure* (no SVD), loads the
    trained factors from ``checkpoint_path`` (a ``model.safetensors`` file or a
    directory containing one), attaches the gating API, and returns the model
    set to full rank.

    Returns:
        ``(model, tokenizer)``.
    """
    import os

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Must match the load done at adaptation time, or the module tree the
    # checkpoint was saved from and the one being rebuilt here disagree.
    model, _ = load_text_causal_lm(
        base_model_id,
        pad_token_id=tokenizer.pad_token_id,
        dtype=dtype,
        device_map=device_map,
        cache_dir=cache_dir,
    )
    model.config.use_cache = False

    replace_linear_with_lowrank(
        model, max_rank=max_rank, target_substring=target_substring,
    )

    model.max_rank = max_rank
    model.train_ranks = train_ranks or geometric_rank_ladder(max_rank)
    model.log_vars_ce = nn.ParameterDict(
        {str(r): nn.Parameter(torch.zeros(())) for r in model.train_ranks},
    ).to(model.device)

    ckpt_file = checkpoint_path
    if os.path.isdir(checkpoint_path):
        ckpt_file = os.path.join(checkpoint_path, "model.safetensors")
    state_dict = load_file(ckpt_file)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    _reconcile_checkpoint_keys(model, state_dict, missing, unexpected)

    attach_rank_control(model)
    model._set_rank_all(max_rank)
    return model, tokenizer
