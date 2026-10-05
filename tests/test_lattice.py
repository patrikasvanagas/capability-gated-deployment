# ruff: noqa: ANN201, CPY001, INP001, S101
import pytest

from capability_gated.lattice import (
    deserialise_config,
    join_configs,
    leq_config,
    meet_configs,
)


def test_meet_unions_coordinates_and_takes_lower_overlap():
    left = {(1, "up_proj"): 512, (2, "gate_proj"): 768}
    right = {(2, "gate_proj"): 256, (3, "down_proj"): 1024}

    assert meet_configs(left, right, max_rank=1024) == {
        (1, "up_proj"): 512,
        (2, "gate_proj"): 256,
    }


def test_meet_rejects_invalid_rank():
    with pytest.raises(ValueError, match="outside"):
        meet_configs({(1, "up_proj"): 0}, max_rank=1024)


def test_join_keeps_only_shared_coordinates_at_the_looser_rank():
    left = {(1, "up_proj"): 512, (2, "gate_proj"): 768}
    right = {(2, "gate_proj"): 256, (3, "down_proj"): 1024}

    # (1, up) and (3, down) are each restricted by only one side, so they join
    # to max_rank and drop out. Only the shared coordinate survives, at 768.
    assert join_configs(left, right, max_rank=1024) == {(2, "gate_proj"): 768}


def test_join_of_disjoint_profiles_is_full_privilege():
    left = {(1, "up_proj"): 512}
    right = {(9, "down_proj"): 256}

    assert join_configs(left, right, max_rank=1024) == {}


def test_meet_and_join_bracket_their_inputs():
    left = {(1, "up_proj"): 512, (2, "gate_proj"): 768}
    right = {(2, "gate_proj"): 256, (3, "down_proj"): 1024}
    meet = meet_configs(left, right, max_rank=1024)
    join = join_configs(left, right, max_rank=1024)

    for side in (left, right):
        assert leq_config(meet, side, max_rank=1024)
        assert leq_config(side, join, max_rank=1024)
    assert not leq_config(left, right, max_rank=1024)   # genuinely incomparable
    assert not leq_config(right, left, max_rank=1024)


def test_absorption_holds_on_the_real_profile_shape():
    # meet(a, join(a, b)) == a -- a lattice law, and a cheap guard against the
    # max-rank omission rule silently dropping a coordinate it should keep.
    left = {(0, "down_proj"): 1024, (17, "up_proj"): 1024}
    right = {(1, "down_proj"): 1024, (17, "up_proj"): 256}
    absorbed = meet_configs(left, join_configs(left, right, max_rank=2048),
                            max_rank=2048)

    assert absorbed == left


def test_join_rejects_invalid_rank():
    with pytest.raises(ValueError, match="outside"):
        join_configs({(1, "up_proj"): 0}, max_rank=1024)


def test_deserialise_config_rejects_duplicates():
    rows = [
        {"block": 1, "module": "up_proj", "rank": 512},
        {"block": 1, "module": "up_proj", "rank": 256},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        deserialise_config(rows)
