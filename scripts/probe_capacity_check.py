#!/usr/bin/env python
"""Was the probe big enough to trust its answer?

The masking gap only means something if the probe is capable of reading the
information when it IS there. A probe that is too small reports a small gap for
every configuration -- not because nothing is hidden, but because it cannot see.
That failure mode looks exactly like the "clean gate" result we are hoping for,
which is why it has to be ruled out rather than assumed away.

Two checks, both run against the FULL-PRIVILEGE activations:

1. **Floor.** At full privilege the model answers correctly, so the answer is
   present and linearly available -- the LM head is itself a linear readout of
   this very vector. A capable probe must therefore recover AT LEAST the
   behavioural accuracy. If it recovers less, it is under-powered and every gap
   it reports elsewhere is uninterpretable.

2. **Plateau.** Sweep the hidden width. If accuracy is still climbing at the
   width used for the headline numbers, that width was too small and the run's
   MLP numbers understate what the model knows. If it has flattened, the width
   is adequate and a bigger probe would only buy memorisation -- which the
   random-label control at each width is there to show.

Costs nothing on a GPU: it refits from the saved activation dumps, which is why
``probe_gate_or_filter.py`` saves them by default.

Example::

    python scripts/probe_capacity_check.py \\
        --acts probes/<stamp>/acts__full-privilege.npz \\
        --results probes/<stamp>/probe_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from probe_gate_or_filter import fit_linear_probe  # noqa: E402

from capability_gated import WMDP_TASKS, SplitRegistry, load_task  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acts", required=True,
                   help="acts__full-privilege.npz from the probe run.")
    p.add_argument("--results", required=True, help="probe_results.json.")
    p.add_argument("--widths", type=int, nargs="+", default=[0, 64, 256, 1024, 4096],
                   help="Hidden widths to sweep; 0 is the declared linear probe.")
    p.add_argument("--headline-width", type=int, default=256,
                   help="The width the run's reported MLP numbers used.")
    p.add_argument("--splits", default="mech_interp_results/splits_v1.json")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    z = np.load(args.acts, allow_pickle=True)
    res = json.loads(Path(args.results).read_text())
    behav = {
        r["task"]: r["behavioural_acc"]
        for r in res["rows"] if r["config"] == "full-privilege"
    }
    registry = SplitRegistry.load(Path(args.splits))

    verdicts, out_rows = [], []
    for task in WMDP_TASKS:
        if task not in z.files:
            continue
        acts = z[task]
        if acts.ndim == 3:          # per-depth dump: the declared probe is the last block
            acts = acts[:, -1, :]
        items = registry.take(load_task(task, cache_dir=args.cache_dir), task, "A")
        y = np.array([it.answer for it in items], dtype=np.int64)
        if len(y) != len(acts):
            raise SystemExit(
                f"{task}: {len(acts)} activation rows but {len(y)} labels -- "
                f"the activations were extracted on a different split.")
        n = len(y)
        acts, y = acts[:n], y[:n]
        b = behav.get(task, float("nan"))
        print(f"\n=== {task}  (n={n}, behavioural accuracy {b*100:.1f}%) ===")
        print(f"{'width':>8}{'probe':>9}{'control':>10}{'vs behav':>10}")
        accs = {}
        for w in args.widths:
            a = fit_linear_probe(acts, y, hidden=w, seed=args.seed, device=args.device)
            c = fit_linear_probe(
                acts, np.random.default_rng(args.seed + 1).permutation(y),
                hidden=w, seed=args.seed + 2, device=args.device)
            accs[w] = a
            label = "linear" if w == 0 else str(w)
            print(f"{label:>8}{a*100:>8.1f}%{c*100:>9.1f}%{(a-b)*100:>+9.1f}pp")
            out_rows.append({"task": task, "width": w, "probe_acc": a,
                             "control_acc": c, "behavioural_acc": b})
        best = max(accs.values())
        head = accs.get(args.headline_width, float("nan"))
        floor_ok = head >= b - 0.02
        plateau_ok = head >= best - 0.02
        verdicts.append((task, floor_ok, plateau_ok))
        print(f"  floor  ({args.headline_width} reaches behavioural): "
              f"{'PASS' if floor_ok else 'FAIL -- probe under-powered'}")
        print(f"  plateau ({args.headline_width} within 2pp of best): "
              f"{'PASS' if plateau_ok else f'FAIL -- still climbing, best {best*100:.1f}%'}")

    print("\n=== verdict ===")
    for task, f, pl in verdicts:
        print(f"  {task:<12} floor {'ok' if f else 'FAIL':<5} plateau {'ok' if pl else 'FAIL'}")
    if all(f and pl for _, f, pl in verdicts):
        print(f"\n  Width {args.headline_width} is adequate: it recovers behavioural "
              f"accuracy and larger probes add nothing. Masking gaps stand.")
    else:
        print(f"\n  Width {args.headline_width} is NOT adequate on at least one task. "
              f"Refit the reported gaps at the largest swept width -- the "
              f"activations are on disk, so this costs no GPU time.")
    if args.out:
        Path(args.out).write_text(json.dumps(out_rows, indent=2, default=float))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
