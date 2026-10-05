"""The rank-gating API: set the active rank at three granularities.

Once a model's MLP linears have been replaced with
:class:`~capability_gated.lowrank.DynamicLowRankLinear` (see
:func:`~capability_gated.training.build_lowrank_model`), these helpers let a
deployer choose *how much* internal computation the model is allowed to use:

* :func:`set_rank_all` -- every low-rank layer in the model.
* :func:`set_rank_for_block` -- every low-rank layer inside one transformer block.
* :func:`set_rank_for_named_module_at_block` -- a single projection (e.g.
  ``down_proj`` / ``dense_4h_to_h``) inside one block.

Each function takes the ``model`` as its first argument. :func:`attach_rank_control`
binds them onto a model instance as ``_set_rank_all`` etc. (matching the method
names the trainer and saved checkpoints expect).
"""

from __future__ import annotations

from types import MethodType

from torch import nn

from .lowrank import DynamicLowRankLinear


def set_rank_all(model: nn.Module, rank: int) -> None:
    """Set the active rank for every ``DynamicLowRankLinear`` in the model."""
    for module in model.modules():
        if isinstance(module, DynamicLowRankLinear):
            module.set_rank(rank)


def get_transformer_blocks(model: nn.Module) -> nn.ModuleList:
    """Return the ``ModuleList`` of transformer blocks for common HF decoder LMs.

    Supports the usual backbone layouts:

    * Llama / Qwen / Mistral: ``model.model.layers``
    * GPT-2 / GPT-J: ``model.transformer.h``
    * GPT-NeoX / Pythia: ``model.gpt_neox.layers``

    Raises:
        AttributeError: if no known block container can be located.
    """
    # Most HF causal LMs expose the backbone under `.model`.
    base = getattr(model, "model", model)

    # Llama/Qwen-style: base.layers
    if hasattr(base, "layers"):
        return base.layers

    # Some wrappers nest one more level: model.model.model.layers
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base.model.layers

    # GPT-2 / GPT-J style: transformer.h
    if hasattr(base, "transformer") and hasattr(base.transformer, "h"):
        return base.transformer.h

    # GPT-NeoX / Pythia style: gpt_neox.layers (wrapper is the root)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers

    raise AttributeError(
        "Could not locate transformer blocks (layers/h) on this model."
    )


def set_rank_for_block(model: nn.Module, block_idx: int, rank: int) -> None:
    """Set the active rank only for low-rank layers inside one block.

    Raises:
        IndexError: if ``block_idx`` is out of range.
    """
    blocks = get_transformer_blocks(model)
    if not (0 <= block_idx < len(blocks)):
        raise IndexError(
            f"block_idx={block_idx} out of range (0..{len(blocks) - 1})."
        )

    r = int(rank)
    for module in blocks[block_idx].modules():
        if isinstance(module, DynamicLowRankLinear):
            module.set_rank(r)


def set_rank_for_named_module_at_block(
    model: nn.Module,
    block_idx: int,
    module_name: str,
    rank: int,
) -> None:
    """Set the active rank for one named projection inside one block.

    ``module_name`` matches either the full dotted name relative to the block
    (e.g. ``mlp.down_proj``) or just its last component (e.g. ``down_proj``).

    Raises:
        IndexError: if ``block_idx`` is out of range.
        LookupError: if no matching ``DynamicLowRankLinear`` is found.
    """
    blocks = get_transformer_blocks(model)
    if not (0 <= block_idx < len(blocks)):
        raise IndexError(
            f"block_idx={block_idx} out of range (0..{len(blocks) - 1})."
        )

    r = int(rank)
    found = False
    for name, module in blocks[block_idx].named_modules():
        if isinstance(module, DynamicLowRankLinear) and (
            module_name == name or module_name == name.split(".")[-1]
        ):
            module.set_rank(r)
            found = True

    if not found:
        raise LookupError(
            f"Did not find a low-rank module named {module_name!r} in block {block_idx}."
        )


def set_component_mask_for_named_module_at_block(
    model: nn.Module,
    block_idx: int,
    module_name: str,
    mask,
) -> None:
    """Apply an arbitrary keep-mask to one projection inside one block.

    Analysis instrument only -- see
    :meth:`~capability_gated.lowrank.DynamicLowRankLinear.set_component_mask`.
    Pass ``mask=None`` to clear it.

    Raises:
        IndexError: if ``block_idx`` is out of range.
        LookupError: if no matching ``DynamicLowRankLinear`` is found.
    """
    blocks = get_transformer_blocks(model)
    if not (0 <= block_idx < len(blocks)):
        raise IndexError(f"block_idx={block_idx} out of range (0..{len(blocks) - 1}).")

    found = False
    for name, module in blocks[block_idx].named_modules():
        if isinstance(module, DynamicLowRankLinear) and (
            module_name == name or module_name == name.split(".")[-1]
        ):
            module.set_component_mask(mask)
            found = True

    if not found:
        raise LookupError(
            f"Did not find a low-rank module named {module_name!r} in block {block_idx}.",
        )


def clear_component_masks(model: nn.Module) -> None:
    """Remove every keep-mask, returning the model to pure rank gating."""
    for module in model.modules():
        if isinstance(module, DynamicLowRankLinear):
            module.set_component_mask(None)


def attach_rank_control(model: nn.Module) -> nn.Module:
    """Bind the rank-gating helpers onto a model instance as methods.

    After calling this, the model exposes ``_set_rank_all``,
    ``_get_transformer_blocks``, ``_set_rank_for_block`` and
    ``_set_rank_for_named_module_at_block`` -- the method names used by the
    trainer and by checkpoints saved from this package. Returns the model for
    convenient chaining.
    """
    model._set_rank_all = MethodType(
        lambda self, rank: set_rank_all(self, rank), model
    )
    model._get_transformer_blocks = MethodType(
        lambda self: get_transformer_blocks(self), model
    )
    model._set_rank_for_block = MethodType(
        lambda self, block_idx, rank: set_rank_for_block(self, block_idx, rank),
        model,
    )
    model._set_rank_for_named_module_at_block = MethodType(
        lambda self, block_idx, module_name, rank: (
            set_rank_for_named_module_at_block(self, block_idx, module_name, rank)
        ),
        model,
    )
    model._set_component_mask_for_named_module_at_block = MethodType(
        lambda self, block_idx, module_name, mask: (
            set_component_mask_for_named_module_at_block(
                self, block_idx, module_name, mask
            )
        ),
        model,
    )
    model._clear_component_masks = MethodType(
        lambda self: clear_component_masks(self), model
    )
    return model
