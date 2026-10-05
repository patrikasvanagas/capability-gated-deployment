#!/usr/bin/env python
"""Family-wide multiplicity correction for lattice composition verdicts.

One lattice run reports each composition check with its own 95% CI, and a lone
CI reads as a finding. Across a grid of runs that logic breaks: 5 runs x 12
checks is 60 chances for a spurious star, and per-run stars are exactly what a
reviewer strikes. This script pools every check from every run given to it into
ONE family, applies Benjamini-Hochberg at q, and reports which verdicts survive
-- so "significant" always means "significant given everything we looked at".

Two families, corrected separately (they answer different questions and mixing
them would let the easy one subsidise the hard one):

* ``suppression``  -- node vs full privilege, from the per-item NPZs (paired).
  "Did this configuration suppress at all?" Expected to pass; not news.
* ``composition``  -- joint vs individual profile, from the stored CIs.
  "Did composing change suppression?" The family the paper's claims live on.

Offline, stdlib + numpy only.

Example (the 5-run pair grid plus the campaign-A run)::

    python scripts/lattice_family_bh.py --q 0.05 \\
        --run tierA=path/to/campaignA/split_A.json[:items.npz] \\
        --run b110c125=path/to/pair1/split_A.json \\
        ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def config_key(cfg: list[dict] | None) -> str:
    """Canonical hash of a gating configuration, order-independent."""
    if not cfg:
        return "full"
    items = sorted((int(c["block"]), str(c["module"]), int(c["rank"])) for c in cfg)
    return hashlib.sha256(repr(items).encode()).hexdigest()[:16]


def dedup(tests: list[dict]) -> tuple[list[dict], list[dict]]:
    """Collapse tests that compare byte-identical configurations on one domain.

    A grid built from tiered profiles reuses the same profile in several runs, so
    the same physical comparison recurs under different run labels -- e.g. a
    chem+cyber meet is identical in t110 and b125c110 when both take chem and
    cyber from the 1.10 search. Counting it twice inflates the family and
    over-tightens Benjamini-Hochberg, which silently *raises* the bar for every
    real effect. Each distinct comparison is kept once; duplicates are returned
    separately so the report can name what was folded.
    """
    seen: dict[tuple, dict] = {}
    dropped: list[dict] = []
    for t in tests:
        key = (t["family"], t.get("cfg_key"), t.get("ref_key"), t["domain"])
        if key in seen:
            seen[key].setdefault("also_in", []).append(t["run"])
            dropped.append(t)
        else:
            seen[key] = t
    return list(seen.values()), dropped


def bh(tests: list[dict], q: float) -> list[dict]:
    """Benjamini-Hochberg: mark each test's survival within THIS family."""
    ordered = sorted(tests, key=lambda t: t["p"])
    n = len(ordered)
    kmax = 0
    for i, t in enumerate(ordered, 1):
        if t["p"] <= i / n * q:
            kmax = i
    for i, t in enumerate(ordered, 1):
        t["bh_significant"] = i <= kmax
        t["bh_threshold"] = i / n * q
    return ordered


def composition_tests(run: str, payload: dict) -> list[dict]:
    cfgs = {n: v.get("cfg") for n, v in (payload.get("results") or {}).items()}
    out = []
    for label, check in (payload.get("composition_checks") or {}).items():
        for dom, t in check["targets"].items():
            if "ci_lo_pp" not in t:
                continue  # pre-CI schema; recompute from items or exclude
            se = (t["ci_hi_pp"] - t["ci_lo_pp"]) / (2 * 1.96)
            delta = t["joint_minus_individual_pp"]
            z = abs(delta) / se if se else 0.0
            out.append({
                "family": "composition", "run": run, "node": label,
                "cfg_key": config_key(cfgs.get(label)),
                "ref_key": config_key(cfgs.get(dom)),
                "domain": dom, "delta_pp": delta,
                "direction": "weaker" if delta > 0 else "stronger",
                "p": math.erfc(z / math.sqrt(2)),
            })
    return out


def suppression_tests(run: str, items_path: Path,
                      cfgs: dict[str, list] | None = None) -> list[dict]:
    z = np.load(items_path, allow_pickle=True)
    nodes = sorted({k.split("|")[0] for k in z.files if k.endswith("|correct")})
    tasks = sorted({k.split("|")[1] for k in z.files
                    if k.endswith("|correct") and k.split("|")[1].startswith("wmdp")})
    out = []
    for node in nodes:
        if node == "full":
            continue
        for task in tasks:
            key = f"{node}|{task}|correct"
            if key not in z.files:
                continue
            diff = z[f"full|{task}|correct"].astype(float) - z[key].astype(float)
            se = diff.std(ddof=1) / math.sqrt(len(diff))
            zs = diff.mean() / se if se else 0.0
            out.append({
                "family": "suppression", "run": run, "node": node,
                "cfg_key": config_key((cfgs or {}).get(node)),
                "ref_key": "full",
                "domain": task, "delta_pp": 100 * diff.mean(),
                "direction": "suppressed" if diff.mean() > 0 else "raised",
                "p": math.erfc(abs(zs) / math.sqrt(2)),
            })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True,
                   help="NAME=split.json or NAME=split.json:items.npz. Repeat "
                        "for every run in the grid; the correction is only as "
                        "honest as the family is complete.")
    p.add_argument("--q", type=float, default=0.05)
    p.add_argument("--no-dedup", action="store_true",
                   help="Keep duplicate comparisons. Default is to collapse "
                        "tests on byte-identical configurations, because a "
                        "tiered grid reuses profiles and double-counting "
                        "inflates the family, over-tightening the correction.")
    p.add_argument("--out", default=None, help="Optional JSON output path.")
    args = p.parse_args()

    comp, supp = [], []
    corrected: dict[str, list] = {}
    for spec in args.run:
        name, _, rhs = spec.partition("=")
        json_path, _, npz_path = rhs.partition(":")
        payload = json.loads(Path(json_path).read_text())
        comp += composition_tests(name, payload)
        if npz_path:
            supp += suppression_tests(
                name, Path(npz_path),
                {n: v.get("cfg") for n, v in (payload.get("results") or {}).items()})

    for family, tests in (("composition", comp), ("suppression", supp)):
        if not tests:
            continue
        n_raw = len(tests)
        if not args.no_dedup:
            tests, folded = dedup(tests)
            if folded:
                print(f"\n[{family}] deduped {n_raw} -> {len(tests)} tests "
                      f"({len(folded)} duplicate comparisons folded)")
        ordered = bh(tests, args.q)
        corrected[family] = ordered      # what --out must record: the family
        n_sig = sum(t["bh_significant"] for t in ordered)   # actually corrected
        print(f"\n=== {family}: {len(ordered)} tests across "
              f"{len({t['run'] for t in ordered})} runs -> BH keeps {n_sig} ===")
        for t in ordered:
            mark = "*" if t["bh_significant"] else " "
            print(f" {mark} {t['run']:<10} {t['node']:<16} {t['domain']:<11}"
                  f"{t['delta_pp']:+7.2f}pp  p={t['p']:.4f}  ({t['direction']})")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"q": args.q, "deduped": not args.no_dedup,
             "composition": corrected.get("composition", []),
             "suppression": corrected.get("suppression", [])},
            indent=2, default=float))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
