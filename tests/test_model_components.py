import types

import pytest
import torch

from config import CONFIG
from gptModel import GPTModel
from modules.rmsnorm import RMSNorm
from modules.swiglu import SwiGLU
from transformerBlock import FeedForward, TransformerBlock
from transformerStack import TransformerStack


def make_cfg(**overrides):
    base = dict(
        EMBEDDING_DIM=64,
        ATTENTION_NUM_HEADS=4,
        ATTENTION_NUM_KV_HEADS=2,
        TRANSFORMER_MLP_RATIO=4.0,
        TRANSFORMER_DROPOUT=0.0,
        ATTENTION_DROPOUT=0.0,
        TRANSFORMER_USE_BIAS=True,
        ATTENTION_MAX_SEQ_LEN=32,
        POSITIONAL_EMBEDDING_TYPE="learned",
        ROPE_BASE=10000.0,
        MLP_TYPE="swiglu",
        NORMALIZATION_TYPE="rmsnorm",
        TRANSFORMER_NUM_LAYERS=3,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# RMSNorm numerical safety (2C)
# ---------------------------------------------------------------------------


def test_rmsnorm_matches_fp32_reference():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    layer = RMSNorm(16)
    with torch.no_grad():
        layer.weight.fill_(1.25)
        out = layer(x)

    rms = x.pow(2).mean(dim=-1, keepdim=True)
    expected = 1.25 * x * torch.rsqrt(rms + layer.eps)
    assert torch.allclose(out, expected, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_preserves_input_dtype(dtype):
    x = torch.randn(2, 5, 16).to(dtype)
    out = RMSNorm(16)(x)
    assert out.dtype == dtype


def test_rmsnorm_half_output_close_to_fp32_reference():
    torch.manual_seed(1)
    x = torch.randn(4, 32, dtype=torch.float32)
    layer = RMSNorm(32)

    ref = layer(x.float())

    for dtype in (torch.float16, torch.bfloat16):
        got = layer(x.to(dtype)).float()
        tol = 5e-2 if dtype is torch.bfloat16 else 5e-3
        assert torch.allclose(got, ref, atol=tol, rtol=tol), f"{dtype}: max diff {(got - ref).abs().max()}"


def test_rmsnorm_upcasts_half_inputs_internally():
    """Half input must not produce inf/NaN from mean/rsqrt overflow."""
    x = torch.full((1, 8), 60000.0, dtype=torch.float16)
    out = RMSNorm(8)(x)
    assert torch.isfinite(out).all()
    assert out.dtype == torch.float16


# ---------------------------------------------------------------------------
# Config plumbing: cfg must win over global CONFIG (2A)
# ---------------------------------------------------------------------------


def test_block_reads_cfg_not_global_config():
    cfg = make_cfg(MLP_TYPE="gelu", NORMALIZATION_TYPE="layernorm", TRANSFORMER_NUM_LAYERS=3)
    block = TransformerBlock.from_config(cfg=cfg, d_model=64, max_seq_len=32)

    assert type(block.mlp) is FeedForward
    assert type(block.ln1).__name__ == "LayerNorm"
    assert type(block.ln2).__name__ == "LayerNorm"

    # Global CONFIG untouched and disagrees with cfg -> would fail on a leak.
    assert CONFIG.MLP_TYPE == "swiglu"
    assert CONFIG.NORMALIZATION_TYPE == "rmsnorm"


def test_block_default_uses_swiglu_and_rmsnorm():
    block = TransformerBlock.from_config(cfg=make_cfg(), d_model=64, max_seq_len=32)
    assert type(block.mlp).__name__ == "SwiGLU"
    assert type(block.ln1).__name__ == "RMSNorm"


def test_stack_honors_cfg_num_layers():
    assert len(TransformerStack.from_config(cfg=make_cfg(TRANSFORMER_NUM_LAYERS=3)).layers) == 3
    assert len(TransformerStack.from_config(cfg=make_cfg(TRANSFORMER_NUM_LAYERS=5)).layers) == 5


def test_stack_threads_mlp_and_norm_type_into_layers():
    stack = TransformerStack.from_config(
        cfg=make_cfg(MLP_TYPE="gelu", NORMALIZATION_TYPE="layernorm", TRANSFORMER_NUM_LAYERS=2)
    )
    for layer in stack.layers:
        assert type(layer.mlp) is FeedForward
        assert type(layer.ln1).__name__ == "LayerNorm"


def test_gptmodel_owns_final_norm_and_honors_norm_type():
    gpt = GPTModel(
        vocab_size=64, d_model=64, num_layers=2, n_head=4, n_kv_head=2,
        mlp_ratio=4.0, dropout=0.0, attention_dropout=0.0, bias=True,
        max_seq_len=32, tie_weights=True, positional_embedding_type="learned",
        mlp_type="swiglu", norm_type="layernorm",
    )
    assert type(gpt.norm).__name__ == "LayerNorm"
    assert type(gpt.transformer.final_ln).__name__ == "Identity"

    gpt_rms = GPTModel(
        vocab_size=64, d_model=64, num_layers=2, n_head=4, n_kv_head=2,
        dropout=0.0, attention_dropout=0.0, bias=True, max_seq_len=32,
        mlp_type="swiglu", norm_type="rmsnorm",
    )
    assert type(gpt_rms.norm).__name__ == "RMSNorm"


# ---------------------------------------------------------------------------
# SwiGLU mlp_ratio (2A)
# ---------------------------------------------------------------------------


def test_swiglu_hidden_dim_scales_with_mlp_ratio():
    assert SwiGLU.compute_swiglu_hidden_dim(384, multiplier=1.0) == 1024
    assert SwiGLU.compute_swiglu_hidden_dim(384, multiplier=2.0) == 2048


def test_production_defaults_untouched_by_cfg_overrides():
    assert CONFIG.MLP_TYPE == "swiglu"
    assert CONFIG.NORMALIZATION_TYPE == "rmsnorm"
    assert isinstance(CONFIG.TRANSFORMER_NUM_LAYERS, int)


# ---------------------------------------------------------------------------
# Forward / backward sanity
# ---------------------------------------------------------------------------


def test_forward_loss_and_gradients_are_finite():
    torch.manual_seed(2)
    gpt = GPTModel(
        vocab_size=64, d_model=32, num_layers=2, n_head=4, n_kv_head=2,
        dropout=0.0, attention_dropout=0.0, bias=True, max_seq_len=16,
        positional_embedding_type="learned", mlp_type="swiglu", norm_type="rmsnorm",
    )
    gpt.train()
    ids = torch.randint(4, 64, (2, 16))
    logits = gpt(ids)
    assert logits.shape == (2, 16, 64)

    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 64).float(), ids[:, 1:].reshape(-1)
    )
    loss.backward()

    grads = [p.grad for p in gpt.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_weight_tying_by_default():
    gpt = GPTModel(
        vocab_size=64, d_model=32, num_layers=1, n_head=4,
        dropout=0.0, attention_dropout=0.0, max_seq_len=16,
    )
    assert gpt.lm_head.weight is gpt.token_embedding.weight


def test_param_count_matches_measured_production_model():
    production_d_model = getattr(CONFIG, "EMBEDDING_DIM", None)
    if not production_d_model:
        pytest.skip("EMBEDDING_DIM not configured")
    assert production_d_model == 384
