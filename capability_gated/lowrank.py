"""Rank-indexed low-rank linear layers and utilities for capability gating.

The core building block is :class:`DynamicLowRankLinear`, a drop-in replacement
for ``nn.Linear`` whose weight is factorized as ``W ≈ B @ A`` and whose *active
rank* can be changed at inference time. Lowering the active rank shrinks the
reachable function class of the layer -- this is the enforcement mechanism
behind capability-gated (least-privilege) deployment.

Two ways to install these layers into a pretrained model:

* :func:`decompose_and_replace` -- SVD-initializes ``A``/``B`` from the existing
  ``nn.Linear`` weights. Use this when preparing a fresh model for fine-tuning.
* :func:`replace_linear_with_lowrank` -- builds the factorized *structure* only
  (random init). Use this when you are about to load a trained checkpoint.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class DynamicLowRankLinear(nn.Module):
    """A linear layer with a dynamically adjustable low-rank factorization.

    The weight matrix ``W`` is approximated as ``W = B @ A`` where

    * ``A`` has shape ``(max_rank, in_features)`` -- projects the input into a
      low-rank space, and
    * ``B`` has shape ``(out_features, max_rank)`` -- projects back out.

    At inference time :meth:`set_rank` selects how many of the ``max_rank``
    components are actually used. Because the components are nested (a prefix of
    the columns of ``B`` / rows of ``A``), increasing the rank weakly enlarges
    the set of reachable computations and decreasing it restricts them.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_rank: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.max_rank = max_rank
        self.active_rank = max_rank
        # Optional per-component keep-mask, for *analysis only* -- see
        # :meth:`set_component_mask`. ``None`` means the layer behaves exactly as
        # the rank knob alone dictates.
        self.component_mask: torch.Tensor | None = None

        # PyTorch stores nn.Linear weights as (out, in), so:
        #   A.weight: (max_rank, in_features)
        #   B.weight: (out_features, max_rank)
        self.A = nn.Linear(in_features, max_rank, bias=False)
        self.B = nn.Linear(max_rank, out_features, bias=bias)

    def set_rank(self, rank: int) -> None:
        """Set the active rank used by subsequent forward passes.

        The value is clamped to ``[1, max_rank]``.
        """
        self.active_rank = max(1, min(int(rank), self.max_rank))

    def set_component_mask(self, mask: torch.Tensor | None) -> None:
        """Keep an *arbitrary* subset of components, not just a prefix.

        **This is a measurement instrument, not a deployment control.** The
        privilege mechanism removes a suffix, and that is what makes reachability
        monotone in the rank knob; an arbitrary mask breaks the total order on
        each coordinate's levels (though not monotonicity itself -- keep-sets are
        still ordered by inclusion). It exists so we can measure what the
        suffix-only constraint costs in selectivity, which cannot be answered
        with the deployment interface alone.

        Args:
            mask: boolean tensor of shape ``(max_rank,)``, ``True`` = keep, or
                ``None`` to clear and return to pure rank gating.
        """
        if mask is None:
            self.component_mask = None
            return
        mask = mask.to(dtype=torch.bool)
        if mask.shape != (self.max_rank,):
            raise ValueError(
                f"mask must have shape ({self.max_rank},), got {tuple(mask.shape)}",
            )
        self.component_mask = mask.to(self.B.weight.device)

    def _hidden(self, x: torch.Tensor, r: int) -> torch.Tensor:
        hidden = F.linear(x, self.A.weight[:r, :])
        if self.component_mask is not None:
            # Zeroing component j is exactly dropping the rank-1 term B[:,j]A[j,:].
            hidden = hidden * self.component_mask[:r].to(hidden.dtype)
        return hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using the currently active rank (and mask, if set)."""
        r = self.active_rank
        hidden = self._hidden(x, r)
        return F.linear(hidden, self.B.weight[:, :r], self.B.bias)

    def forward_with_hidden(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that also returns the intermediate low-rank hidden state."""
        r = self.active_rank
        hidden = self._hidden(x, r)  # (..., r)
        out = F.linear(hidden, self.B.weight[:, :r], self.B.bias)
        return out, hidden

    def extra_repr(self) -> str:
        return (
            f"in_features={self.A.in_features}, "
            f"out_features={self.B.out_features}, "
            f"max_rank={self.max_rank}, active_rank={self.active_rank}"
        )


# Submodule names that belong to a non-text tower or its projection. Matched as
# substrings against direct children, so this covers gemma4's
# vision_tower/audio_tower/embed_vision/embed_audio, gemma3/llava's
# vision_tower/multi_modal_projector, and qwen-vl's visual, without naming any
# architecture. It is a name heuristic, not a reachability analysis: a model that
# hides a tower under an unrelated name will not be caught, which is why
# strip_non_text_towers returns what it removed rather than working silently.
NON_TEXT_PATTERNS: tuple[str, ...] = (
    "vision", "visual", "image", "audio", "speech", "multi_modal", "multimodal",
)


def strip_non_text_towers(model: nn.Module) -> list[str]:
    """Delete non-text towers from a multimodal wrapper. No-op on text-only models.

    Two distinct problems, one fix. First, :func:`find_max_rank` takes a *minimum*
    over every ``nn.Linear`` whose name contains the target substring, so a vision
    encoder with narrower MLPs silently caps the ladder: on gemma-4-E2B it returns
    768 where the language model admits 1536, and nothing errors -- you simply get
    a rank ladder at half the intended ceiling. Second, ``decompose_and_replace``
    would SVD-factorize and then train those encoders, which no text batch ever
    reaches.

    Loading the full class first and pruning afterwards is deliberate. The
    checkpoint stores ``model.language_model.*`` while ``<Model>ForCausalLM``
    expects ``model.*``, and no ``_checkpoint_conversion_mapping`` bridges them,
    so constructing the causal-LM class directly and calling ``from_pretrained``
    matches zero keys and yields a randomly initialised model with only a
    warning. Load, then prune.

    Returns:
        Qualified names of the removed children, empty for a text-only model.
    """
    removed: list[str] = []
    # Towers sit either directly on the wrapper or one level down beside the
    # language model, depending on the architecture; check both.
    parents = [("", model)]
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module):
        parents.append(("model.", inner))

    for prefix, parent in parents:
        for name, _ in list(parent.named_children()):
            lowered = name.lower()
            if any(p in lowered for p in NON_TEXT_PATTERNS):
                delattr(parent, name)
                removed.append(f"{prefix}{name}")
    return removed


def load_text_causal_lm(
    base_id: str,
    *,
    pad_token_id: int | None = None,
    dtype: Any = None,
    device_map: Any = None,
    cache_dir: str | None = None,
) -> tuple[nn.Module, list[str]]:
    """Load a base model as a text-only causal LM, whatever wrapper it ships in.

    One function because multimodal checkpoints break the naive load in two
    independent ways, and every call site needs both fixes:

    * ``from_pretrained(..., pad_token_id=...)`` works for a plain
      ``<X>ForCausalLM``, which forwards unknown kwargs into its config, but
      ``Gemma4ForConditionalGeneration.__init__`` rejects them outright with a
      ``TypeError``. The kwarg is therefore dropped here and the value assigned
      to the config afterwards, which is what the kwarg did anyway -- on the
      nested ``text_config`` too, since that is the one the decoder reads.
    * The image/audio towers must go before any shape is read, or
      :func:`find_max_rank` returns the encoder's narrower minimum (768 rather
      than 1536 on gemma-4-E2B) and silently halves the rank ladder.

    Loading the wrapper and pruning is deliberate rather than constructing
    ``<X>ForCausalLM(config.text_config)`` directly: the checkpoint stores
    ``model.language_model.*`` while that class expects ``model.*``, and with no
    ``_checkpoint_conversion_mapping`` to bridge them ``from_pretrained`` matches
    zero keys and returns a randomly initialised model with only a warning.

    Returns:
        ``(model, dropped)`` where ``dropped`` names the removed towers and is
        empty for a text-only model such as Qwen3.
    """
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"cache_dir": cache_dir}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)

    dropped = strip_non_text_towers(model)

    if pad_token_id is not None:
        model.config.pad_token_id = pad_token_id
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            text_config.pad_token_id = pad_token_id
    return model, dropped


def find_max_rank(model: nn.Module, target_substring: str = "mlp") -> int:
    """Return the largest rank that is valid for every targeted linear layer.

    This is the minimum of ``min(in_features, out_features)`` across all
    ``nn.Linear`` modules whose qualified name contains ``target_substring``.
    """
    min_dim = float("inf")
    for name, module in model.named_modules():
        if target_substring in name and isinstance(module, nn.Linear):
            min_dim = min(min_dim, min(module.in_features, module.out_features))
    if min_dim == float("inf"):
        raise ValueError(
            f"No nn.Linear modules matched target_substring={target_substring!r}."
        )
    return int(min_dim)


def replace_linear_with_lowrank(
    model: nn.Module,
    max_rank: int,
    target_substring: str = "mlp",
) -> nn.Module:
    """Replace matching ``nn.Linear`` layers with untrained low-rank structure.

    Only the module structure is created (random init); weights are expected to
    be loaded afterwards from a trained checkpoint. Use
    :func:`decompose_and_replace` instead when starting from pretrained weights.

    The replacement inherits the device and dtype of the layer it replaces. That
    matters: ``load_state_dict`` casts incoming tensors to the *destination*
    dtype, so a factor left at the torch default would stay float32 inside an
    otherwise-bf16 model and the first MLP matmul would raise a dtype mismatch.
    """
    for full_name, module in list(model.named_modules()):
        if target_substring in full_name and isinstance(module, nn.Linear):
            name_parts = full_name.split(".")
            parent_name = ".".join(name_parts[:-1])
            local_name = name_parts[-1]
            parent_module = model.get_submodule(parent_name)

            new_layer = (
                DynamicLowRankLinear(
                    module.in_features,
                    module.out_features,
                    max_rank,
                    bias=(module.bias is not None),
                )
                .to(module.weight.device)
                .to(module.weight.dtype)
            )
            setattr(parent_module, local_name, new_layer)
    return model


def decompose_and_replace(
    model: nn.Module,
    max_rank: int,
    target_substring: str = "mlp",
) -> nn.Module:
    """SVD-decompose matching ``nn.Linear`` weights into low-rank factors.

    For every ``nn.Linear`` whose qualified name contains ``target_substring``,
    the weight ``W`` is decomposed via SVD and replaced with a
    :class:`DynamicLowRankLinear`. The singular values are split evenly between
    the two factors (``B = U @ sqrt(S)``, ``A = sqrt(S) @ Vh``) for numerical
    stability, and the original device, dtype and bias are preserved.

    Args:
        model: The (pretrained) module to modify in place.
        max_rank: Maximum rank ``r`` kept in the factorization.
        target_substring: Substring identifying target layers by name.
    """
    print("Decomposing and replacing linear layers...")
    for full_name, module in list(model.named_modules()):
        if target_substring in full_name and isinstance(module, nn.Linear):
            # 1. Decompose the pretrained weight W (float32 for a stable SVD).
            weight = module.weight.data.to(torch.float32)
            u, s, vh = torch.linalg.svd(weight, full_matrices=False)

            # 2. Truncate to the requested rank.
            r = min(max_rank, s.size(0))
            u = u[:, :r]
            s = s[:r]
            vh = vh[:r, :]

            # 3. Split the singular values between the two factors.
            s_sqrt = torch.sqrt(s)
            b_weight = u @ torch.diag(s_sqrt)  # (out_features, r)
            a_weight = torch.diag(s_sqrt) @ vh  # (r, in_features)

            # 4. Build the replacement layer on the original device/dtype.
            new_layer = (
                DynamicLowRankLinear(
                    module.in_features,
                    module.out_features,
                    max_rank,
                    bias=(module.bias is not None),
                )
                .to(module.weight.device)
                .to(module.weight.dtype)
            )
            new_layer.A.weight.data[:r, :] = a_weight.to(new_layer.A.weight.dtype)
            new_layer.B.weight.data[:, :r] = b_weight.to(new_layer.B.weight.dtype)
            if module.bias is not None:
                new_layer.B.bias.data = module.bias.data.to(new_layer.B.bias.dtype)

            # 5. Swap it into the parent module.
            name_parts = full_name.split(".")
            parent_name = ".".join(name_parts[:-1])
            local_name = name_parts[-1]
            parent_module = model.get_submodule(parent_name)
            setattr(parent_module, local_name, new_layer)

    return model
