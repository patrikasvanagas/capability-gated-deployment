"""The gemma-4 substrate path, exercised on CPU with a shrunken config.

Every failure this covers cost a rented H100 to discover: a missing package
export, a ``pad_token_id`` kwarg the multimodal wrapper rejects, and a rank
ceiling read off the vision encoder. None of them needed a GPU to find -- only a
model of the right *shape*, which a two-layer config provides in milliseconds.
"""

from __future__ import annotations

import pytest
import torch

from capability_gated import (
    decompose_and_replace,
    find_max_rank,
    load_text_causal_lm,
    strip_non_text_towers,
)


def _tiny_gemma4():
    """A gemma-4 wrapper with the real module topology at toy dimensions."""
    transformers = pytest.importorskip("transformers")
    try:
        cfg_cls = transformers.Gemma4Config
        model_cls = transformers.Gemma4ForConditionalGeneration
    except AttributeError:  # transformers too old to know this architecture
        pytest.skip("transformers build has no Gemma4")

    full = cfg_cls.from_pretrained("google/gemma-4-E2B")
    # Shrink in place: topology (towers, PLE, per-block MLP triples) is what is
    # under test, not width or depth.
    full.text_config.num_hidden_layers = 2
    full.text_config.hidden_size = 64
    full.text_config.intermediate_size = 128
    full.text_config.num_attention_heads = 4
    full.text_config.num_key_value_heads = 2
    full.text_config.head_dim = 16
    with torch.device("meta"):
        model = model_cls(full)
    return model


@pytest.mark.parametrize("attr", ["vision_tower", "audio_tower"])
def test_towers_present_before_prune(attr):
    model = _tiny_gemma4()
    assert hasattr(model.model, attr), f"fixture no longer has {attr}"


def test_prune_removes_towers_and_leaves_only_language_model_mlps():
    model = _tiny_gemma4()
    dropped = strip_non_text_towers(model)

    assert "model.vision_tower" in dropped
    assert "model.audio_tower" in dropped

    linears = [
        (n, m) for n, m in model.named_modules()
        if "mlp" in n and isinstance(m, torch.nn.Linear)
    ]
    assert len(linears) == 2 * 3, "expected 3 MLP projections per block"
    assert all("language_model" in n for n, _ in linears)
    # The ceiling is now the language model's own minimum, with no tower able to
    # drag it down. Stated as an identity rather than "it went up", because this
    # fixture is deliberately narrower than the real vision encoder and would not
    # move -- see test_real_config_rank_ceiling for the at-scale version.
    assert find_max_rank(model, "mlp") == min(
        min(m.in_features, m.out_features) for _, m in linears
    )


def test_real_config_rank_ceiling_and_coordinate_count():
    """At true scale, on meta device: the numbers the run config depends on.

    Without the prune ``find_max_rank`` reports the vision encoder's 768 and the
    ladder is silently built at half its intended height -- no error, just wrong
    results. A Gemma-4-E2B run config needs exactly these two numbers:
    MAX_RANK=1536 and 105 coordinates.
    """
    transformers = pytest.importorskip("transformers")
    cfg = transformers.AutoConfig.from_pretrained("google/gemma-4-E2B")
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(cfg)

    assert find_max_rank(model, "mlp") == 768, "fixture assumption changed"
    strip_non_text_towers(model)
    assert find_max_rank(model, "mlp") == 1536

    coords = [
        n for n, m in model.named_modules()
        if "mlp" in n and isinstance(m, torch.nn.Linear)
    ]
    assert len(coords) == 105, "35 blocks x 3 projections"


def test_prune_is_a_noop_on_a_text_only_model():
    """Qwen3 must be untouched -- existing checkpoints depend on it."""
    transformers = pytest.importorskip("transformers")
    cfg = transformers.AutoConfig.from_pretrained("Qwen/Qwen3-1.7B-Base")
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(cfg)

    before = find_max_rank(model, "mlp")
    assert strip_non_text_towers(model) == []
    assert find_max_rank(model, "mlp") == before == 2048


def test_loader_does_not_pass_pad_token_id_as_a_kwarg():
    """``Gemma4ForConditionalGeneration.__init__`` raises TypeError on it.

    Asserted against the signature rather than a live load, so it fails in CI
    rather than on a rented GPU.
    """
    import inspect

    from capability_gated import lowrank

    src = inspect.getsource(lowrank.load_text_causal_lm)
    body = src.split('"""')[-1]  # skip the docstring, which discusses the kwarg
    assert "pad_token_id=pad_token_id" not in body
    assert "model.config.pad_token_id = pad_token_id" in body


def test_factorized_multimodal_model_still_computes_a_loss():
    """Prune -> factorize -> forward, the sequence the training step performs."""
    model = _tiny_gemma4()
    strip_non_text_towers(model)
    model = model.to_empty(device="cpu")
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)

    decompose_and_replace(model, max_rank=8, target_substring="mlp")
    model.config.use_cache = False

    ids = torch.randint(0, 128, (2, 16))
    out = model(input_ids=ids, labels=ids)
    assert torch.isfinite(out.loss), "loss must be finite after factorization"
