#!/usr/bin/env python
"""E5/E8: did the gate REMOVE the capability, or just MASK the answer?

A profile that suppresses a domain behaviourally can be doing one of two very
different things. It can delete the computation -- the model no longer knows the
answer. Or it can cancel the readout -- the model still computes the answer and
the gate merely stops it reaching the output. Behaviour alone cannot tell them
apart; both look like "accuracy fell".

A linear probe can. Fit one on the answer-position residual and ask what the
INTERNALS still know:

    masking gap   mu = probe_recovery - behavioural_accuracy

``mu ~ 0`` is a gate (the knowledge is gone, so nothing can read it out).
``mu >> 0`` is a filter (the knowledge is there and readable; only the output
was suppressed). This is the paper's E5, and its result is explicitly allowed to
retitle the work.

**Why this matters now (E8).** The 24-run parent grid found chem suppression
*evaporating* when chem was met with cyber -- ~10pp of suppression lost across
most parent pairs. Masking predicts exactly that: a cancellation is a fragile
balance, and a second gate landing on other coordinates perturbs it, so the
suppressed behaviour returns. Removal predicts the opposite -- deleted
computation cannot come back. So the grid is already a masking detector under
pure query access, and the probes are the confirmation:

    prediction: profiles whose suppression EVAPORATED under composition have a
    LARGE mu; profiles whose suppression SURVIVED have a SMALL one.

That is a real, falsifiable, pre-registered correlation, and this script
computes both sides of it.

**Controls, both mandatory.** A probe that reads answers off *any* activation
proves nothing, so every probe is paired with a random-label probe -- identical
architecture and data, labels shuffled within the task -- which measures how
much apparent recovery is memorisation. And a foil -- a filter by construction --
is included as a config, to calibrate what a large mu actually looks like on this
substrate. There are two, and only the second one works:

* ``"preamble"``: the model instructed to refuse the domain. Measured on the
  122k 1.7B this moved behaviour by +4.3/+5.5/-0.1 pp (bio/chem/cyber), i.e.
  barely at all, and landed BELOW most real profiles instead of above all of
  them. Two reasons, and the second is not fixable by changing checkpoint: the
  substrate is a base model that does not follow instructions, and the evaluator
  scores restricted-logit argmax over " A".." D", so there is no channel through
  which a model can refuse -- to score at chance it would have to flatten its own
  letter logits. Kept because "prompting does not produce a filter here" is
  itself worth recording, but it is not the reference.
* ``"readout_cancel"``: the letter logits are permuted per item after the
  forward. Same weights, same evaluator, residual stream bit-identical, so the
  probe reads full-privilege knowledge while behaviour sits at chance BY
  CONSTRUCTION. This is the paper's own definition of masking applied directly,
  and it gives an analytic ceiling, mu_abs = probe_full - alpha_D, that the
  measured run should reproduce.

**Splits.** ``--split`` defaults to A. Split B is the one-shot confirmatory set
and requires an explicit ``--unlock-reason``, which is recorded in the output as
``split_b_unlock_reason``; there is deliberately no default, because an unlock a
script writes for itself is not an audit trail. Whatever split is extracted here
must be passed to the fitters too -- they take their labels from the registry
independently, and a mismatch is silent rather than an error. The declared probe class for the paper is
linear on the final-layer residual, which is the default; ``--all-depths`` adds
the per-block sweep that E8 names as a secondary analysis (where in the network
does the knowledge die?).

Example::

    python scripts/probe_gate_or_filter.py \\
        --model-id Qwen/Qwen3-1.7B-Base --checkpoint-path runs/ckpt/final_model \\
        --max-rank 2048 --configs configs.json --out probe_results
"""

from __future__ import annotations

import argparse
import json
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
    load_lowrank_checkpoint,
    load_task,
)
from capability_gated.lattice import deserialise_config  # noqa: E402
from capability_gated.mcq import PromptStyle  # noqa: E402
from capability_gated.training import geometric_rank_ladder  # noqa: E402

N_CLASSES = 4
CHANCE = 1.0 / N_CLASSES


def log(msg: str) -> None:
    print(msg, flush=True)


def fit_linear_probe(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden: int = 0,
    pca: int = 64,
    folds: int = 5,
    epochs: int = 800,
    lr: float = 3e-2,
    weight_decay: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> float:
    """Cross-validated accuracy of a probe. Returns mean held-out accuracy.

    ``hidden=0`` is the declared probe class -- multinomial linear on the
    answer-position residual. ``hidden>0`` inserts one ReLU layer, which is the
    stronger MLP variant; it is a SECONDARY reading, because the paper declares
    the linear class and because a stronger probe can memorise, which is exactly
    what the random-label control is there to expose.

    **``pca`` is not optional in practice.** These activations are 2048-dim and
    the panels are small -- chem is 163 items. Fitting 2048 weights on 130
    training rows does not find a sparse signal, it fits noise: on a synthetic
    chem-sized problem (n=163, d=2048, answer in 8 directions) the raw probe
    scores 27% against a 25% chance floor, while a 32-64 component projection
    scores ~80%. Reported as-is, that failure would look exactly like "the
    knowledge is gone" and would have inverted the E5 conclusion for the very
    domain the masking hypothesis is about. Stronger weight decay does not fix
    it (tested to wd=3.0); only reducing dimension or adding rows does, and the
    split is fixed.

    Everything that touches feature scaling -- the PCA basis and the
    standardisation -- is fitted on the TRAIN fold only. Fitting either on all
    rows leaks the held-out fold into the projection and inflates every number
    here.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    if n < folds * N_CLASSES:
        folds = max(2, n // N_CLASSES)
    order = rng.permutation(n)
    accs = []
    for f in range(folds):
        te = order[f::folds]
        tr = np.setdiff1d(order, te, assume_unique=False)
        if len(tr) == 0 or len(te) == 0:
            continue
        xtr_r, xte_r = x[tr], x[te]
        if pca and pca < x.shape[1]:
            centre = xtr_r.mean(0)
            _, _, vt = np.linalg.svd(xtr_r - centre, full_matrices=False)
            basis = vt[: min(pca, vt.shape[0])].T
            xtr_r = (xtr_r - centre) @ basis
            xte_r = (xte_r - centre) @ basis
        mu, sd = xtr_r.mean(0), xtr_r.std(0) + 1e-6
        xtr = torch.tensor((xtr_r - mu) / sd, dtype=torch.float32, device=device)
        xte = torch.tensor((xte_r - mu) / sd, dtype=torch.float32, device=device)
        ytr = torch.tensor(y[tr], dtype=torch.long, device=device)
        yte = torch.tensor(y[te], dtype=torch.long, device=device)
        torch.manual_seed(seed + f)
        d_in = xtr.shape[1]  # post-PCA width, not x.shape[1]
        if hidden:
            head = torch.nn.Sequential(
                torch.nn.Linear(d_in, hidden), torch.nn.ReLU(),
                torch.nn.Linear(hidden, N_CLASSES),
            ).to(device)
        else:
            head = torch.nn.Linear(d_in, N_CLASSES).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(xtr), ytr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            accs.append((head(xte).argmax(1) == yte).float().mean().item())
    return float(np.mean(accs)) if accs else float("nan")


def fit_probe_across_splits(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    hidden: int = 0,
    pca: int = 64,
    epochs: int = 800,
    lr: float = 3e-2,
    weight_decay: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> float:
    """Fit on ALL of one split, score once on the other. The declared protocol.

    :func:`fit_linear_probe` cross-validates *within* one split, which is the
    right estimator while split B is locked. It is not what the paper registers
    for the confirmatory pass: "trained on split-A activations and evaluated on
    split B". That is a different estimator and this is it -- no folds, one fit,
    one score.

    Two consequences worth stating wherever this number is reported. There is no
    fold-to-fold variance, so a seed here perturbs only the initialisation and
    the spread will be much smaller than the cross-validated one; the two spreads
    are not comparable. And the evaluation set is read once, so this must not be
    called in a loop over hyperparameters -- everything (``pca``, ``hidden``, the
    optimiser settings) is frozen on split A beforehand.

    Feature scaling is fitted on ``x_fit`` only, exactly as the per-fold version
    fits it on the train fold only.
    """
    if len(x_fit) != len(y_fit) or len(x_eval) != len(y_eval):
        raise ValueError(
            f"row mismatch: fit {len(x_fit)}x{len(y_fit)}, "
            f"eval {len(x_eval)}x{len(y_eval)}")
    xf, xe = x_fit, x_eval
    if pca and pca < xf.shape[1]:
        centre = xf.mean(0)
        _, _, vt = np.linalg.svd(xf - centre, full_matrices=False)
        basis = vt[: min(pca, vt.shape[0])].T
        xf = (xf - centre) @ basis
        xe = (xe - centre) @ basis
    mu, sd = xf.mean(0), xf.std(0) + 1e-6
    xtr = torch.tensor((xf - mu) / sd, dtype=torch.float32, device=device)
    xte = torch.tensor((xe - mu) / sd, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_fit, dtype=torch.long, device=device)
    yte = torch.tensor(y_eval, dtype=torch.long, device=device)
    torch.manual_seed(seed)
    d_in = xtr.shape[1]
    if hidden:
        head = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, N_CLASSES),
        ).to(device)
    else:
        head = torch.nn.Linear(d_in, N_CLASSES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((head(xte).argmax(1) == yte).float().mean().item())


def probe_task(
    acts: np.ndarray,
    answers: np.ndarray,
    *,
    seed: int,
    device: str,
    mlp_hidden: int = 0,
    pca: int = 64,
) -> dict[str, float]:
    """Declared linear probe + random-label control, optionally an MLP variant.

    Both probes get their own control. A gap that survives only under the MLP is
    still evidence of masking -- the knowledge is present but not linearly
    readable -- but it is a weaker claim, so the two are reported separately
    rather than pooled.
    """
    out = {
        "probe_acc": fit_linear_probe(acts, answers, pca=pca, seed=seed, device=device),
        "control_acc": fit_linear_probe(
            acts, np.random.default_rng(seed + 1).permutation(answers),
            pca=pca, seed=seed + 2, device=device),
    }
    if mlp_hidden:
        out["mlp_probe_acc"] = fit_linear_probe(
            acts, answers, hidden=mlp_hidden, pca=pca, seed=seed, device=device)
        out["mlp_control_acc"] = fit_linear_probe(
            acts, np.random.default_rng(seed + 3).permutation(answers),
            hidden=mlp_hidden, pca=pca, seed=seed + 4, device=device)
    return out


def main() -> None:  # noqa: PLR0915 - one linear measurement reads best flat
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--max-rank", type=int, required=True)
    p.add_argument("--configs", required=True,
                   help="JSON: [{name, domain, cfg|null, kind}]. cfg is a "
                        "serialised profile; null means full privilege. kind is "
                        "free text recorded in the output (parent/meet/"
                        "baseline/foil).")
    p.add_argument("--splits", default="mech_interp_results/splits_v1.json")
    p.add_argument("--split", default="A", choices=["A", "B"],
                   help="B is the one-shot confirmatory set and requires an "
                        "explicit --unlock-reason; leave at A.")
    p.add_argument("--unlock-reason", default=None,
                   help="Required with --split B, and recorded in the output as "
                        "split_b_unlock_reason. There is deliberately no "
                        "default: an unlock the script writes for itself is not "
                        "an audit trail, and this script is the one that would "
                        "spend the confirmatory set. Mirrors eval_mcq.py.")
    p.add_argument("--all-depths", action="store_true",
                   help="Also probe every block's residual (E8 secondary). One "
                        "extra forward pass' worth of memory, no extra passes.")
    p.add_argument("--probe-device", default="cpu")
    p.add_argument("--extract-only", action="store_true",
                   help="Run the forward passes, save activations and "
                        "behavioural accuracy, and skip all probe fitting. This "
                        "is the right split of work: the forwards need the GPU "
                        "and cannot be redone cheaply, while probe fitting is "
                        "CPU-bound, needs tuning against the data, and is free "
                        "to repeat locally from the saved activations. Fitting "
                        "on the pod cost 91 minutes per configuration, almost "
                        "all of it in the per-depth sweep.")
    p.add_argument("--pca", type=int, default=32,
                   help="PCA components before probing, fitted per train fold. "
                        "Required in practice: 2048-dim activations against a "
                        "163-item chem panel score at chance without it. 0 "
                        "disables (do not, without checking the capacity report).")
    p.add_argument("--mlp-hidden", type=int, default=256,
                   help="Hidden width of the secondary MLP probe (0 disables). "
                        "The linear probe stays the declared, primary reading.")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--tokens-per-batch", type=int, default=8192)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--no-save-activations", action="store_true",
                   help="Skip dumping raw activations. Default is to SAVE them: "
                        "re-extracting costs a GPU pod, while any further probe "
                        "variant (different class, capacity, layer) is free once "
                        "the activations are on disk.")
    p.add_argument("--out", default="probe_gate_or_filter")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    specs = json.loads(Path(args.configs).read_text())
    log(f"{len(specs)} configurations to probe")

    if args.split == "B" and not args.unlock_reason:
        raise SystemExit(
            "--split B requires --unlock-reason. Split B is read once, at "
            "write-up, and the reason is the audit trail.")

    registry = SplitRegistry.load(Path(args.splits))
    tasks = [*WMDP_TASKS, *sorted(set(MMLU_ADJACENT.values())), *MMLU_DISTANT]
    items, full_items = {}, {}
    for t in tasks:
        full = load_task(t, cache_dir=args.cache_dir)
        full_items[t] = full
        items[t] = registry.take(
            full, t, args.split, unlock_reason=args.unlock_reason,
        )
    # Check the content fingerprint: a silently re-released dataset would
    # invalidate the frozen indices without any error.
    registry.verify(full_items)
    log(f"tasks {len(items)}; items " + ", ".join(f"{t}={len(v)}" for t, v in items.items()))

    # Same loading path as scripts/evaluate_lattice_composition.py -- the
    # checkpoint's own config.json carries base id, target substring, dtype and
    # rank ladder, and the 122k checkpoint records train_ranks as null, so the
    # trainer's geometric default has to be reconstructed rather than indexed.
    ckpt_cfg_path = Path(args.checkpoint_path).parent / "config.json"
    ckpt = json.loads(ckpt_cfg_path.read_text()) if ckpt_cfg_path.exists() else {}
    ckpt = {**(ckpt.get("resolved") or {}), **ckpt}
    ladder = ckpt.get("train_ranks") or geometric_rank_ladder(args.max_rank)
    model, tok = load_lowrank_checkpoint(
        ckpt.get("base_id", args.model_id),
        args.checkpoint_path,
        max_rank=args.max_rank,
        target_substring=ckpt.get("target_substring", "mlp"),
        train_ranks=[int(r) for r in ladder],
        dtype=getattr(torch, ckpt.get("dtype", args.dtype)),
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=args.cache_dir,
    )
    model.eval()
    ev = MCQEvaluator(
        model, tok, items, max_rank=args.max_rank,
        max_seq_len=args.max_seq_len,
        max_tokens_per_batch=args.tokens_per_batch,
    )
    answers = {t: np.array([it.answer for it in items[t]], dtype=np.int64) for t in items}

    # The refusal foil is the SAME weights under a refusal instruction: a filter
    # by construction, so it calibrates what a large mu looks like on this
    # substrate. Built lazily because it needs its own tokenised batches.
    foil_ev: MCQEvaluator | None = None

    def evaluator_for(spec: dict[str, Any]) -> MCQEvaluator:
        nonlocal foil_ev
        if not spec.get("preamble"):
            return ev
        if foil_ev is None:
            style = PromptStyle(preamble=spec["preamble"])
            log(f"  building foil batches with preamble: {spec['preamble']!r}")
            foil_ev = MCQEvaluator(
                model, tok, items, max_rank=args.max_rank,
                max_seq_len=args.max_seq_len,
                max_tokens_per_batch=args.tokens_per_batch, prompt_style=style,
            )
        return foil_ev

    rows: list[dict[str, Any]] = []
    for i, spec in enumerate(specs, 1):
        name, kind = spec["name"], spec.get("kind", "")
        cfg = spec.get("cfg")
        started = time.perf_counter()
        sink: dict[str, np.ndarray] = {}
        this_ev = evaluator_for(spec)
        # "readout_cancel": permute the letter logits after the forward. The
        # foil that the refusal preamble was meant to be: behaviour at chance
        # with the residual stream untouched, so it is the measured ceiling for
        # mu rather than another point in the middle of the range.
        readout_seed = args.seed if spec.get("readout_cancel") else None
        if cfg:
            res = this_ev.evaluate_config(
                deserialise_config(cfg), activation_sink=sink,
                all_depths=args.all_depths, readout_seed=readout_seed,
            )
        else:
            model._set_rank_all(args.max_rank)
            res = this_ev.evaluate_current(
                activation_sink=sink, all_depths=args.all_depths,
                readout_seed=readout_seed,
            )
        for task, acts in sink.items():
            flat = acts.reshape(len(acts), -1) if acts.ndim == 2 else acts[:, -1, :]
            pr = ({} if args.extract_only else
                  probe_task(flat, answers[task], seed=args.seed,
                             device=args.probe_device, mlp_hidden=args.mlp_hidden,
                             pca=args.pca))
            row = {
                "config": name, "kind": kind, "task": task,
                "behavioural_acc": res[task].acc,
                **pr,
                **({"masking_gap_pp": (pr["probe_acc"] - res[task].acc) * 100}
                   if "probe_acc" in pr else {}),
                **({"mlp_masking_gap_pp": (pr["mlp_probe_acc"] - res[task].acc) * 100}
                   if "mlp_probe_acc" in pr else {}),
                **({"control_margin_pp": (pr["probe_acc"] - pr["control_acc"]) * 100}
                   if "probe_acc" in pr and "control_acc" in pr else {}),
                "n": int(res[task].n),
            }
            if args.all_depths and acts.ndim == 3 and not args.extract_only:
                depths = []
                for d in range(acts.shape[1]):
                    dp = fit_linear_probe(acts[:, d, :], answers[task], pca=args.pca,
                                          seed=args.seed, device=args.probe_device)
                    depths.append(dp)
                row["per_depth_probe_acc"] = depths
            rows.append(row)
        if not args.no_save_activations:
            np.savez_compressed(out / f"acts__{name}.npz", **sink)
        def _summary(task: str) -> str:
            cell = f"{task.split('-')[-1]}: acc {res[task].acc * 100:.1f}"
            hit = next((r for r in rows
                        if r["config"] == name and r["task"] == task), None)
            if hit and "probe_acc" in hit:      # absent in --extract-only
                cell += f" probe {hit['probe_acc'] * 100:.1f}"
            return cell

        log(f"[{i}/{len(specs)}] {name:<28} {time.perf_counter()-started:6.1f}s  "
            + "  ".join(_summary(t) for t in WMDP_TASKS if t in sink))

    (out / "probe_results.json").write_text(json.dumps(
        {"chance": CHANCE, "split": args.split, "seed": args.seed,
         "split_b_unlock_reason": args.unlock_reason if args.split == "B" else None,
         "n_items": {t: len(v) for t, v in items.items()}, "rows": rows},
        indent=2, default=float))
    log(f"\nwrote {out/'probe_results.json'} ({len(rows)} rows)")

    if args.extract_only:
        log("\nextract-only: activations saved, no probes fitted. "
            "Fit locally with scripts/probe_masking_gap.py")
        return
    log("\n=== masking gap on each config's own hazard domain ===")
    log(f"{'config':<30}{'domain':<12}{'behav':>8}{'probe':>8}{'ctrl':>8}{'mu (pp)':>10}")
    for r in rows:
        dom = r["task"].replace("wmdp-", "")
        if r["kind"] in ("baseline", "foil") or dom in r["config"].lower():
            log(f"{r['config'][:29]:<30}{dom:<12}{r['behavioural_acc']*100:>7.1f}%"
                f"{r['probe_acc']*100:>7.1f}%{r['control_acc']*100:>7.1f}%"
                f"{r['masking_gap_pp']:>+10.2f}")


if __name__ == "__main__":
    main()
