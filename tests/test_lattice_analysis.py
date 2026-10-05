# ruff: noqa: ANN001, ANN201, CPY001, INP001, S101
"""The analysis layer of evaluate_lattice_composition: the parts that decide
what the run is allowed to claim.

These live in the script rather than the package, so they are loaded by path.
They are pure functions over already-measured rows, which is exactly the layer
worth testing offline -- a wrong sign or a mislabelled bound here turns a null
result into a finding.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_lattice_composition.py"


@pytest.fixture(scope="module")
def elc():
    spec = importlib.util.spec_from_file_location("elc", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(**overrides):
    """Baseline 60% everywhere, each atom suppressing its own domain to 50%."""
    rows = {}
    for node in ("full", "bio", "chem", "cyber", "bio+chem", "bio+cyber",
                 "chem+cyber", "bio+chem+cyber", "bio|chem", "bio|cyber",
                 "chem|cyber", "bio|chem|cyber"):
        row = {"node": node, "n_coordinates": 0, "ppl_ratio": 1.0,
               "fluency_pass": True}
        for d in ("bio", "chem", "cyber"):
            row[f"wmdp-{d}_acc"] = 0.60
        rows[node] = row
    for d in ("bio", "chem", "cyber"):
        rows[d][f"wmdp-{d}_acc"] = 0.50
    for node, patch in overrides.items():
        rows[node].update(patch)
    return rows


def test_utility_flags_only_when_parts_pass_and_meet_fails(elc):
    rows = _rows(bio={"ppl_ratio": 1.09}, chem={"ppl_ratio": 1.09})
    rows["bio+chem"]["ppl_ratio"] = 1.1881          # exactly the product

    out = elc.utility_composition(rows, 1.10)["bio+chem"]
    assert out["all_singles_within_budget"] is True
    assert out["meet_within_budget"] is False
    assert out["utility_fails_to_compose"] is True
    assert out["interaction"] == pytest.approx(1.0, abs=1e-6)


def test_utility_does_not_flag_when_a_part_already_breached(elc):
    # If an operand was already over budget, the composition breaching says
    # nothing about composition -- the claim must not fire here.
    rows = _rows(bio={"ppl_ratio": 1.20}, chem={"ppl_ratio": 1.05})
    rows["bio+chem"]["ppl_ratio"] = 1.26

    assert elc.utility_composition(rows, 1.10)["bio+chem"][
        "utility_fails_to_compose"] is False


def _groups(default, overrides=()):
    """Per-item outcome vectors for every node composition_checks walks.

    ``overrides`` is an iterable of ``((node, task), vector)`` -- tuple keys, so
    it cannot be passed as kwargs.
    """
    labels = ("full", "bio", "chem", "cyber", "bio+chem", "bio+cyber",
              "chem+cyber", "bio+chem+cyber")
    tasks = ("wmdp-bio", "wmdp-chem", "wmdp-cyber")
    result = {
        lbl: {t: type("R", (), {"correct": default.copy()})() for t in tasks}
        for lbl in labels
    }
    for (lbl, task), vec in overrides:
        result[lbl][task] = type("R", (), {"correct": vec})()
    return result


def test_composition_check_reports_weaker_only_beyond_noise(elc):
    # The meet leaves bio at 58% where the bio profile alone reached 50%: the
    # joint policy suppressed 8pp LESS, which is a real composition failure.
    rows = _rows()
    rows["bio+chem"]["wmdp-bio_acc"] = 0.58
    n = 600
    groups = _groups(
        np.zeros(n, bool),
        [(("bio+chem", "wmdp-bio"), np.ones(n, bool))],
    )
    out = elc.composition_checks(rows, groups, n_boot=200, seed=0)["bio+chem"]

    assert out["targets"]["bio"]["non_weaker"] is False
    assert out["targets"]["bio"]["weaker_beyond_noise"] is True
    assert out["any_target_weaker_beyond_noise"] is True
    # ... and the untouched pair must stay quiet.
    assert elc.composition_checks(rows, groups, n_boot=200, seed=0)[
        "chem+cyber"]["any_target_weaker_beyond_noise"] is False


def test_composition_check_stays_silent_on_a_tie(elc):
    # Identical per-item outcomes: zero difference, CI must straddle zero and
    # the finding must not fire.
    rows = _rows()
    groups = _groups(np.array([1, 0] * 300, bool))
    out = elc.composition_checks(rows, groups, n_boot=200, seed=0)["bio+chem"]

    for t in out["targets"].values():
        assert t["joint_minus_individual_pp"] == pytest.approx(0.0)
        assert t["weaker_beyond_noise"] is False


def test_join_recovery_measures_against_the_least_suppressed_operand(elc):
    rows = _rows()
    rows["bio|chem"]["wmdp-bio_acc"] = 0.55        # baseline .60, bio atom .50
    out = elc.join_recovery(rows)["bio|chem"]["targets"]["bio"]

    # chem leaves bio at .60, so the least-suppressed operand IS the baseline:
    # no gap to recover, and the fraction must be undefined rather than infinite.
    assert out["least_suppressed_operand_acc"] == pytest.approx(0.60)
    assert out["recovered_fraction"] is None


def test_join_recovery_fraction_when_both_operands_suppress(elc):
    rows = _rows()
    for node in ("bio", "chem"):
        rows[node]["wmdp-bio_acc"] = 0.50          # both suppress bio to .50
    rows["bio|chem"]["wmdp-bio_acc"] = 0.55        # join gives half of it back

    out = elc.join_recovery(rows)["bio|chem"]["targets"]["bio"]
    assert out["recovered_fraction"] == pytest.approx(0.5)
