#!/usr/bin/env python
"""Appendix: how probe recovery varies with readout DEPTH, at full privilege.

The paper's declared probe is linear on the **final-layer** residual, and it
stays the confirmatory instrument. This is the secondary analysis E8 names, and
it exists to answer one objection: chem's probe sits below chem's behavioural
accuracy at the declared readout (35.8% against 52.8%), which taken alone looks
exactly like "the chem knowledge is gone" and would invert E5's reading for the
very domain the masking hypothesis is about.

It is not gone. It is upstream. Every domain peaks well before the final block,
and chem peaks ~11pp above where the declared probe reads it.

**No layer is selected.** Re-running across seeds -- each seed changes the 5-fold
split and the probe init -- moves the argmax around inside a plateau (chem: 22,
22, 25, 23, 23) while leaving the plateau itself stable. An argmax over 29
layers x 2 projections on a 163-item panel is mostly fitting CV noise, so the
reportable object is the curve with its seed spread, not a winning layer. That
is also why nothing downstream consumes this: no mu_rel is recomputed at a
chosen depth, because choosing a depth is the step that would not survive.

Selection here would in any case be confined to full privilege -- whether the
instrument can read the knowledge at all is a question about the instrument, and
never touches a gated configuration.

    python scripts/probe_depth_sweep.py --dir tmp/probe/merged \
        --out tmp/probe/depth_stability.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from capability_gated import WMDP_TASKS, SplitRegistry, load_task  # noqa: E402
from probe_gate_or_filter import fit_linear_probe  # noqa: E402
from probe_masking_gap import labels_for  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True,
                   help="Directory holding acts__full-privilege.npz (needs the "
                        "--all-depths extraction: (n, layers, d_model)).")
    p.add_argument("--pcas", type=int, nargs="+", default=[8, 32])
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[20260807, 11, 22, 33, 44],
                   help="Each seed reshuffles the CV folds and the probe init, "
                        "which is the variance an argmax would exploit.")
    p.add_argument("--configs", nargs="+", default=["full-privilege"],
                   help="Which acts__*.npz to sweep. E8's panel (b) needs the "
                        "deployed profiles as well as full privilege, so that "
                        "'where does the knowledge die' can be read against the "
                        "blocks each profile actually gates.")
    p.add_argument("--splits", default="mech_interp_results/splits_v1.json")
    p.add_argument("--split", default="A", choices=["A", "B"],
                   help="Which frozen split the activations were extracted on. Must match the extraction: the labels come from here and a mismatch is silent, not an error. B additionally requires --unlock-reason.")
    p.add_argument("--unlock-reason", default=None,
                   help="Required with --split B; recorded in the "
                        "output. No default on purpose.")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    d = Path(args.dir)
    res = json.loads((d / "probe_results.json").read_text())
    behav = {(r["config"], r["task"]): r["behavioural_acc"] for r in res["rows"]}
    if args.split == "B" and not args.unlock_reason:
        raise SystemExit("--split B requires --unlock-reason.")
    reg = SplitRegistry.load(Path(args.splits))
    full = {t: load_task(t) for t in WMDP_TASKS}
    reg.verify(full)
    answers = {t: np.array([it.answer for it in reg.take(
        full[t], t, args.split, unlock_reason=args.unlock_reason)])
        for t in WMDP_TASKS}
    out = {}
    for cfg in args.configs:
        z = np.load(d / f"acts__{cfg}.npz")
        # A profile is only interesting on its OWN hazard domain: sweeping
        # B1-bio against chem and cyber costs three times the compute and is
        # never plotted. full-privilege is the exception -- it is the reference
        # for all three.
        want = ([t for t in WMDP_TASKS if cfg.lower().endswith(t.split("-")[-1])]
                or list(WMDP_TASKS))
        for task in want:
            if task not in z.files:
                continue
            a = z[task]
            if a.ndim != 3:
                raise SystemExit(f"{task}: need --all-depths activations, got "
                                 f"shape {a.shape}")
            y = labels_for(answers, task, len(a))
            b = behav[(cfg, task)]
            n_layers = a.shape[1]
            probe = np.zeros((len(args.seeds), n_layers, len(args.pcas)))
            ctrl = np.zeros_like(probe)
            print(f"\n=== {cfg} {task}  n={len(a)}  behavioural={b*100:.1f}%  "
                  f"layers={n_layers}  seeds={len(args.seeds)}", flush=True)
            for si, seed in enumerate(args.seeds):
                perm = np.random.default_rng(seed + 999).permutation(y)
                for li in range(n_layers):
                    for pi, pca in enumerate(args.pcas):
                        probe[si, li, pi] = fit_linear_probe(
                            a[:, li, :], y, pca=pca, seed=seed, device="cpu")
                        ctrl[si, li, pi] = fit_linear_probe(
                            a[:, li, :], perm, pca=pca, seed=seed, device="cpu")
                peak = int(np.unravel_index(probe[si].argmax(),
                                            probe[si].shape)[0])
                print(f"  seed {seed:>9}: argmax layer {peak:>2}, "
                      f"best {probe[si].max()*100:.1f}%", flush=True)
            out[f"{cfg}|{task}"] = {
                "config": cfg, "task": task, "behavioural": b,
                "pcas": args.pcas, "seeds": args.seeds,
                "n_items": int(len(a)), "probe": probe.tolist(),
                "control": ctrl.tolist()}
            m = probe.mean(0)
            print(f"  final block {m[-1].max()*100:.1f}%   "
                  f"best depth ~{int(m.max(axis=1).argmax())} "
                  f"{m.max()*100:.1f}%", flush=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
