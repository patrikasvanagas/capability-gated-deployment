from __future__ import annotations

import numpy as np

from capability_gated.attribution import FluencyPriceMap
from capability_gated.suppression_search import (
    BeamSearchSettings,
    TaskTradeoffObjective,
    beam_search,
    compute_objective,
)


def test_fluency_price_map_sign_clipping_and_roundtrip(tmp_path):
    prices = FluencyPriceMap(
        blocks=np.array([0], dtype=np.int32),
        modules=np.array(["gate_proj"]),
        component_nll=np.array([[[-0.1, -0.2, 0.3, -0.4]]], dtype=np.float32),
        baseline_nll=2.0,
        n_blocks=8,
        n_tokens=100,
        seconds=1.5,
        dataset="test-corpus",
        split="validation",
    )

    coord = (0, "gate_proj")
    assert np.isclose(prices.predicted_delta_nll(coord, 2), 0.1)
    assert np.isclose(prices.price(coord, 2), 0.1)
    assert prices.predicted_delta_nll(coord, 0) > 0
    assert prices.predicted_delta_nll(coord, 4) == 0

    path = prices.save_npz(tmp_path / "prices.npz")
    loaded = FluencyPriceMap.load_npz(path)
    assert loaded.dataset == "test-corpus"
    assert loaded.max_rank == 4
    assert np.isclose(loaded.config_price({coord: 2}), 0.1)


class _FakeMemo:
    def __init__(self):
        self.payloads = {
            (): {"acc": {"suppress": 1.0, "preserve": 1.0}},
            ((0, "gate_proj", 1),): {
                "acc": {"suppress": 0.2, "preserve": 1.0},
            },
            ((1, "gate_proj", 1),): {
                "acc": {"suppress": 0.3, "preserve": 1.0},
            },
        }

    def evaluate(self, cfg):
        key = tuple(sorted((b, m, r) for (b, m), r in cfg.items()))
        return self.payloads[key]


def test_pricing_changes_beam_choice_without_changing_accuracy_term():
    base = {"suppress": 1.0, "preserve": 1.0}
    candidates = {(0, "gate_proj"): [1], (1, "gate_proj"): [1]}
    settings = BeamSearchSettings(beam_width=1, max_depth=1, verbose=False)
    unpriced = TaskTradeoffObjective(
        suppress_tasks=["suppress"],
        preserve_tasks=["preserve"],
    )
    prices = {
        (0, "gate_proj", 1): 0.5,
        (1, "gate_proj", 1): 0.0,
    }
    priced = TaskTradeoffObjective(
        suppress_tasks=["suppress"],
        preserve_tasks=["preserve"],
        lambda_fluency=1.0,
        fluency_price=lambda cfg: sum(prices[(b, m, r)] for (b, m), r in cfg.items()),
    )

    best_unpriced = beam_search(_FakeMemo(), base, unpriced, candidates, 4, settings)
    best_priced = beam_search(_FakeMemo(), base, priced, candidates, 4, settings)

    assert best_unpriced.cfg == {(0, "gate_proj"): 1}
    assert best_priced.cfg == {(1, "gate_proj"): 1}
    acc = {"suppress": 0.2, "preserve": 1.0}
    assert np.isclose(
        compute_objective(acc, base, unpriced, {(0, "gate_proj"): 1})
        - compute_objective(acc, base, priced, {(0, "gate_proj"): 1}),
        0.5,
    )
