from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

from capability_gated.lowrank import DynamicLowRankLinear
from capability_gated.training import LowRankTrainer


class _ToyModel:
    max_rank = 16

    def __init__(self, modules: list[DynamicLowRankLinear]) -> None:
        self.modules = modules

    def _set_rank_all(self, rank: int) -> None:
        for module in self.modules:
            module.set_rank(rank)


def _trainer_without_hf_init() -> LowRankTrainer:
    """Construct just enough trainer state to exercise the sampling helpers."""
    trainer = object.__new__(LowRankTrainer)
    trainer.state = SimpleNamespace(global_step=50, max_steps=100, epoch=0.5)
    trainer.sampling_strategy = "rank_weighted"
    trainer.variant_mode = "hybrid"
    trainer.mixed_probability = 0.5
    trainer.mixed_coordinate_counts = [2]
    trainer.mixed_coordinate_weights = [1.0]
    modules = [DynamicLowRankLinear(4, 4, 16, bias=False) for _ in range(4)]
    trainer._mixed_modules = [(f"mlp.proj_{i}", module) for i, module in enumerate(modules)]
    return trainer


class MixedTrainingSamplingTest(unittest.TestCase):
    def test_variant_mode_switch(self) -> None:
        trainer = _trainer_without_hf_init()
        trainer.variant_mode = "global"
        self.assertFalse(trainer._use_mixed_variant())

        trainer.variant_mode = "mixed"
        self.assertTrue(trainer._use_mixed_variant())

        trainer.variant_mode = "hybrid"
        trainer.mixed_probability = 0.0
        self.assertFalse(trainer._use_mixed_variant())
        trainer.mixed_probability = 1.0
        self.assertTrue(trainer._use_mixed_variant())

    def test_sparse_variant_resets_then_overrides_selected_modules(self) -> None:
        trainer = _trainer_without_hf_init()
        model = _ToyModel([module for _, module in trainer._mixed_modules])
        ranks = iter([8, 4])
        trainer._sample_variant_rank = lambda _variants: next(ranks)  # type: ignore[method-assign]

        random.seed(7)
        applied = trainer._apply_mixed_variant(model, [8, 4])

        self.assertEqual(len(applied), 2)
        self.assertEqual(sorted(rank for _, rank in applied), [4, 8])
        active = [module.active_rank for module in model.modules]
        self.assertEqual(sum(rank < model.max_rank for rank in active), 2)
        self.assertEqual(sum(rank == model.max_rank for rank in active), 2)

    def test_rank_weighted_sampling_prefers_larger_ranks(self) -> None:
        trainer = _trainer_without_hf_init()
        random.seed(42)
        draws = [trainer._sample_variant_rank([8, 4, 2]) for _ in range(4_000)]
        self.assertGreater(draws.count(8), draws.count(4))
        self.assertGreater(draws.count(4), draws.count(2))


if __name__ == "__main__":
    unittest.main()
