# ruff: noqa: CPY001, EM101, EM102, TC001, TC003, TRY003
"""Lattice operations for sparse per-coordinate rank configurations.

A configuration omits coordinates that run at ``max_rank``.  Lower rank means
fewer available components, so the meet of two policies is their composition:
take the lower rank at every coordinate mentioned by either policy.

The join is the other half of the same order and behaves very differently.  It
takes the *higher* rank, so a coordinate that only one policy restricts joins to
``max_rank`` -- ungated -- and drops out of the configuration entirely.  That is
why a join can be strictly weaker than either operand: it is the policy a
principal holding *either* grant runs under, not both.  When two profiles share
no coordinates the join is the empty configuration, i.e. full privilege, and the
lattice is degenerate; callers should detect that rather than measure the
baseline several times over.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .suppression_search import Config, Coord


def deserialise_config(rows: Iterable[Mapping[str, object]]) -> Config:
    """Convert E3's JSON config rows to the canonical dictionary form."""
    config: Config = {}
    for row in rows:
        coord = (int(row["block"]), str(row["module"]))
        if coord in config:
            raise ValueError(f"duplicate coordinate in configuration: {coord}")
        config[coord] = int(row["rank"])
    return config


def meet_configs(*configs: Mapping[Coord, int], max_rank: int) -> Config:
    """Compose policies by taking the minimum active rank per coordinate.

    An absent coordinate means ``max_rank``.  Max-rank entries are omitted from
    the result so semantically equivalent configurations serialize identically.
    """
    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    coordinates = set().union(*(config.keys() for config in configs))
    result: Config = {}
    for coord in coordinates:
        ranks = [int(config.get(coord, max_rank)) for config in configs]
        if any(rank <= 0 or rank > max_rank for rank in ranks):
            raise ValueError(
                f"coordinate {coord} has rank outside [1, {max_rank}]: {ranks}",
            )
        rank = min(ranks)
        if rank < max_rank:
            result[coord] = rank
    return result


def join_configs(*configs: Mapping[Coord, int], max_rank: int) -> Config:
    """The least permissive policy that is at least as permissive as each input.

    Mirror of :func:`meet_configs` with ``max`` in place of ``min``.  Because an
    absent coordinate counts as ``max_rank``, any coordinate that some input
    leaves alone ends up ungated here and is omitted -- the same max-rank
    omission rule, doing more visible work.
    """
    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    coordinates = set().union(*(config.keys() for config in configs))
    result: Config = {}
    for coord in coordinates:
        ranks = [int(config.get(coord, max_rank)) for config in configs]
        if any(rank <= 0 or rank > max_rank for rank in ranks):
            raise ValueError(
                f"coordinate {coord} has rank outside [1, {max_rank}]: {ranks}",
            )
        rank = max(ranks)
        if rank < max_rank:
            result[coord] = rank
    return result


def leq_config(
    lower: Mapping[Coord, int], upper: Mapping[Coord, int], *, max_rank: int,
) -> bool:
    """Is ``lower`` pointwise no more permissive than ``upper``?

    The order the whole composition experiment is stated in, so it is worth
    being able to assert it: ``meet <= each input <= join`` must hold by
    construction, and a failure means the nodes were built wrong.
    """
    return all(
        int(lower.get(coord, max_rank)) <= int(upper.get(coord, max_rank))
        for coord in set(lower) | set(upper)
    )
