#!/usr/bin/env python3
"""Offline check of the two-stage fluency gate's verdict logic. No GPU, no torch.

``find_suppression_configs`` imports torch and matplotlib, so it cannot be
imported on a laptop. The two functions that decide *what the paper may claim*
are pure, so this pulls their source out of the script with ``ast`` and
exercises them directly -- which is worth doing, because the failure this gate
exists to catch (a config certified as passing while its reported perplexity
breaches the budget) is silent and ends up in a table.

Run:  python3 scripts/test_fluency_gate.py
"""
from __future__ import annotations

import ast
import sys
from collections.abc import Mapping, MutableMapping, Sequence  # noqa: F401
from pathlib import Path
from typing import Any  # noqa: F401

SCRIPT = Path(__file__).with_name("find_suppression_configs.py")
WANTED = ("resolve_selection", "gate_verdict", "SELECTION_NOTES")


def load_pure_parts() -> dict[str, Any]:
    """exec only the named top-level defs/assignments, without the imports."""
    tree = ast.parse(SCRIPT.read_text())
    picked = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in WANTED)
        or (isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in WANTED for t in node.targets))
    ]
    missing = set(WANTED) - {
        getattr(n, "name", None) or n.targets[0].id for n in picked  # type: ignore[union-attr]
    }
    if missing:
        raise SystemExit(f"not found in {SCRIPT.name}: {sorted(missing)}")
    ns: dict[str, Any] = {"Sequence": Sequence, "MutableMapping": MutableMapping,
                          "Mapping": Mapping, "Any": Any}
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(SCRIPT), "exec"), ns)
    return ns


def row(label: str, score: float, *, report: float | None = None) -> dict[str, Any]:
    return {"label": label, "stable_score": score, "ppl_ratio_report": report}


def main() -> int:
    ns = load_pure_parts()
    resolve, verdict, notes = (
        ns["resolve_selection"], ns["gate_verdict"], ns["SELECTION_NOTES"]
    )
    failures: list[str] = []

    def check(name: str, got: Any, want: Any) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    best, mid, worst = row("best", 0.9), row("mid", 0.5), row("worst", 0.1)

    # A verified pass reports the verified config, not the highest-scoring one:
    # the top of the beam is exactly what tends to breach.
    w, fb, why = resolve([best, mid, worst], [best, mid], mid, verify_top_k=8)
    check("verified/winner", w["label"], "mid")
    check("verified/fell_back", fb, False)
    check("verified/reason", why, "verified")

    # Screen passed, reporting set refused everyone -> flagged, not silently kept.
    w, fb, why = resolve([best, mid], [best, mid], None, verify_top_k=8)
    check("breach/winner", w["label"], "best")
    check("breach/fell_back", fb, True)
    check("breach/reason", why, "verification_failed")

    # Nothing cleared even the screen.
    w, fb, why = resolve([best, mid], [], None, verify_top_k=8)
    check("screen_failed/winner", w["label"], "best")
    check("screen_failed/fell_back", fb, True)
    check("screen_failed/reason", why, "screen_failed")

    # Verification off: keep the old behaviour but never call it a pass.
    w, fb, why = resolve([best, mid], [best, mid], None, verify_top_k=0)
    check("screen_only/winner", w["label"], "best")
    check("screen_only/fell_back", fb, False)
    check("screen_only/reason", why, "screen_only")

    # An empty screen beats verify_top_k=0 to the punch: with nothing to report,
    # "unverified" would understate a set that failed outright.
    check("screen_only/empty", resolve([best], [], None, verify_top_k=0)[2],
          "screen_failed")

    # The summary verdict, including runs written before the gate moved.
    check("verdict/pass", verdict({"selection_basis": "verified"}), "PASS")
    check("verdict/breach", verdict({"selection_basis": "verification_failed"}),
          "BREACH")
    check("verdict/screen_failed", verdict({"selection_basis": "screen_failed"}),
          "BREACH")
    check("verdict/screen_only", verdict({"selection_basis": "screen_only"}),
          "UNVERIFIED")
    check("verdict/legacy", verdict({"max_ppl_ratio": 1.1}), "UNVERIFIED")
    check("verdict/absent", verdict(None), "n/a")

    # Every branch resolve_selection can return must have a note to print.
    for reason in ("verified", "screen_only", "verification_failed", "screen_failed"):
        if reason not in notes:
            failures.append(f"SELECTION_NOTES missing {reason!r}")

    if failures:
        print("FAIL")
        for f in failures:
            print("  " + f)
        return 1
    print("ok: fluency-gate verdict logic (11 checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
