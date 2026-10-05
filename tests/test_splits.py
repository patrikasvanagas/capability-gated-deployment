# ruff: noqa: ANN001, ANN201, CPY001, INP001, S101
"""The split-B lock, and the label/activation alignment that protects it.

Split B is read once, at write-up. Two things can spend it or corrupt it
silently, and neither had a test:

* an unlock that the calling script writes for itself, so the lock never fires;
* activations extracted on one split fitted against labels from the other.
  chem is 163 rows on A and 245 on B, so this does not even raise an
  IndexError -- the old code truncated to the shorter one and returned a probe
  near chance, which reads as "the knowledge is gone".
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from capability_gated import SplitRegistry, WMDP_TASKS  # noqa: E402
from capability_gated.mcq import SplitBLockedError  # noqa: E402

SPLITS = REPO_ROOT / "mech_interp_results" / "splits_v1.json"


@pytest.fixture(scope="module")
def registry():
    return SplitRegistry.load(SPLITS)


def _items(n):
    """Stand-ins: take() only indexes, so identity is all we need."""
    return [object() for _ in range(n)]


def test_split_b_is_locked_without_a_reason(registry):
    task = "wmdp-chem"
    items = _items(registry.tasks[task].n)

    with pytest.raises(SplitBLockedError, match="locked until write-up"):
        registry.take(items, task, "B")


def test_split_b_opens_with_a_reason(registry):
    task = "wmdp-chem"
    items = _items(registry.tasks[task].n)

    got = registry.take(items, task, "B", unlock_reason="confirmatory write-up")

    assert len(got) == len(registry.tasks[task].b)


def test_empty_reason_does_not_count_as_an_unlock(registry):
    """A falsy reason must not satisfy the guard -- it is the audit trail."""
    task = "wmdp-chem"
    items = _items(registry.tasks[task].n)

    with pytest.raises(SplitBLockedError):
        registry.take(items, task, "B", unlock_reason="")


def test_split_a_needs_no_reason(registry):
    task = "wmdp-bio"
    items = _items(registry.tasks[task].n)

    assert len(registry.take(items, task, "A")) == len(registry.tasks[task].a)


@pytest.mark.parametrize("task", sorted(WMDP_TASKS))
def test_a_b_and_shots_are_disjoint_and_cover_n(registry, task):
    s = registry.tasks[task]
    a, b, shots = set(s.a), set(s.b), set(s.shots)

    assert len(a) == len(s.a), "duplicate index within A"
    assert len(b) == len(s.b), "duplicate index within B"
    assert not (a & b) and not (a & shots) and not (b & shots)
    assert len(a | b | shots) == s.n


def test_chem_split_is_not_fifty_fifty(registry):
    """Guards the assumption that broke the fitters: B is bigger than A here.

    If a future re-split makes these equal, the alignment bug stops being
    detectable by row count and this test should be revisited, not deleted.
    """
    s = registry.tasks["wmdp-chem"]

    assert len(s.a) == 163
    assert len(s.b) == 245


def test_labels_for_rejects_a_split_mismatch():
    """The whole point: B activations against A labels must raise, not truncate."""
    from probe_masking_gap import labels_for

    a_labels = {"wmdp-chem": np.zeros(163, dtype=np.int64)}

    with pytest.raises(SystemExit, match="different split"):
        labels_for(a_labels, "wmdp-chem", 245)


def test_labels_for_passes_when_aligned():
    from probe_masking_gap import labels_for

    y = np.zeros(163, dtype=np.int64)

    assert len(labels_for({"wmdp-chem": y}, "wmdp-chem", 163)) == 163


def test_registry_file_is_the_committed_one():
    """The frozen indices are only meaningful against the recorded fingerprints."""
    payload = json.loads(SPLITS.read_text())

    assert payload["schema"] == "capability_gated.mcq/splits/v1"
    for task in WMDP_TASKS:
        assert payload["tasks"][task]["fingerprint"]
