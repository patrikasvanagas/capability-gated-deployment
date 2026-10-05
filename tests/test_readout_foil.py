# ruff: noqa: ANN001, ANN201, CPY001, INP001, S101
"""The readout-cancellation foil: chance behaviour, untouched activations.

The foil is only a valid reference for the masking gap if BOTH halves hold --
behaviour falls to chance and the probe still sees full-privilege knowledge. The
second half is the one that is easy to break silently: move the permutation one
line earlier, above ``answer_logits``, and the activations change with it, at
which point the foil measures a gate rather than a filter and nothing in the
output looks wrong.
"""
import numpy as np
import pytest
import torch

from capability_gated.mcq import (
    CHANCE,
    LETTERS,
    evaluate_batches,
    readout_permutations,
)


class _FakeBatch:
    """One batch, indexed like TokenisedBatch but without a tokenizer."""

    def __init__(self, rows, answers):
        self.rows = torch.tensor(rows)
        self.answers = torch.tensor(answers)
        self.attention_mask = None
        self.input_ids = None
        self.last_index = None


def _oracle_logits(hidden, answers):
    """Logits that always pick the right letter, plus a fixed residual."""
    z = torch.full((len(answers), len(LETTERS)), -1.0)
    z[torch.arange(len(answers)), torch.tensor(answers)] = 5.0
    return z, hidden


@pytest.fixture
def patched(monkeypatch):
    """Stub answer_logits: deterministic, and fills hidden_sink like the real one."""
    from capability_gated import mcq

    hidden = {}

    def fake_answer_logits(model, batch, letter_ids, *, hidden_sink=None,
                           all_depths=False, inputs_embeds=None):
        n = len(batch.rows)
        h = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)
        if hidden_sink is not None:
            hidden_sink.append(h)
        z, _ = _oracle_logits(h, batch.answers.tolist())
        return z

    monkeypatch.setattr(mcq, "answer_logits", fake_answer_logits)
    return hidden


def test_permutations_are_permutations_and_deterministic():
    p = readout_permutations(50, seed=7)

    assert p.shape == (50, len(LETTERS))
    for row in p:
        assert sorted(row.tolist()) == list(range(len(LETTERS)))
    assert np.array_equal(p, readout_permutations(50, seed=7))
    assert not np.array_equal(p, readout_permutations(50, seed=8))


def test_unpermuted_oracle_scores_perfectly(patched):
    answers = [0, 1, 2, 3, 2, 1]
    batch = _FakeBatch(rows=list(range(6)), answers=answers)

    res = evaluate_batches(None, [batch], torch.arange(4), task="t", n_items=6)

    assert res.acc == 1.0


def test_readout_cancellation_drops_the_oracle_to_chance(patched):
    # 4000 items: the binomial se at chance is ~0.7pp, so a 5pp window is loose
    # enough never to flake and tight enough to catch a permutation that is not
    # uniform (e.g. one that leaves the correct letter in place too often).
    n = 4000
    rng = np.random.default_rng(0)
    answers = rng.integers(0, 4, size=n).tolist()
    batch = _FakeBatch(rows=list(range(n)), answers=answers)

    res = evaluate_batches(
        None, [batch], torch.arange(4), task="t", n_items=n,
        readout_perm=readout_permutations(n, seed=11),
    )

    assert abs(res.acc - CHANCE) < 0.05


def test_activations_are_bit_identical_under_cancellation(patched):
    """The half that makes the foil a FILTER and not a gate."""
    n = 64
    answers = (np.arange(n) % 4).tolist()
    batch = _FakeBatch(rows=list(range(n)), answers=answers)

    plain: dict[str, np.ndarray] = {}
    evaluate_batches(None, [batch], torch.arange(4), task="t", n_items=n,
                     activation_sink=plain)

    foiled: dict[str, np.ndarray] = {}
    res = evaluate_batches(
        None, [batch], torch.arange(4), task="t", n_items=n,
        activation_sink=foiled, readout_perm=readout_permutations(n, seed=3),
    )

    assert np.array_equal(plain["t"], foiled["t"])
    assert res.acc < 1.0          # the behaviour really did change
