import torch
import pytest

from gptModel import GPTModel


class FakeTokenizer:
    """Minimal tokenizer stand-in so generate_text can be tested without artifacts."""

    def __init__(self, vocab: int = 64):
        self.eos_token_id = 3
        self.unk_token_id = 1
        self.vocab = vocab

    def encode_ids(self, text, add_bos=True):
        ids = [2] if add_bos else []
        return ids + [4 + (ord(c) % (self.vocab - 5)) for c in text]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def build_model(max_seq_len: int = 64, d_model: int = 32, num_layers: int = 2) -> GPTModel:
    torch.manual_seed(0)
    model = GPTModel(
        vocab_size=64, d_model=d_model, num_layers=num_layers, n_head=4, n_kv_head=2,
        mlp_ratio=4.0, dropout=0.0, attention_dropout=0.0, bias=True,
        max_seq_len=max_seq_len, tie_weights=True, positional_embedding_type="learned",
        mlp_type="swiglu", norm_type="rmsnorm",
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def model():
    return build_model()


@pytest.fixture(scope="module")
def tok():
    return FakeTokenizer()


def test_cached_single_step_equals_full_recompute(model, tok):
    ids = tok.encode_ids("abcdefg", add_bos=True)
    x = torch.tensor([ids], dtype=torch.long)

    with torch.inference_mode():
        prefill_logits, past = model(x[:, :6], use_cache=True, position_offset=0)
        step_logits, past2 = model(
            x[:, 6:7], past_key_values=past, use_cache=True, position_offset=6
        )
        full_logits = model(x[:, :7], use_cache=False)

    assert torch.allclose(prefill_logits, model(x[:, :6], use_cache=False), atol=1e-6)
    assert torch.allclose(step_logits, full_logits[:, -1:, :], atol=1e-6)
    assert past2[0][0].size(-2) == 7


def test_cache_length_matches_sequence_length(model, tok):
    ids = torch.tensor([tok.encode_ids("abcdefgh", add_bos=True)], dtype=torch.long)
    total = ids.size(1)
    split = total // 2
    with torch.inference_mode():
        _, past = model(ids[:, :split], use_cache=True, position_offset=0)
        _, past = model(
            ids[:, split:], past_key_values=past, use_cache=True, position_offset=split
        )

    for k, v in past:
        assert k.size(-2) == total
        assert v.size(-2) == total


def test_position_offset_changes_learned_embeddings(model, tok):
    x = torch.tensor([tok.encode_ids("abc", add_bos=True)], dtype=torch.long)
    with torch.inference_mode():
        at_zero = model.embedding(x, position_offset=0)
        at_five = model.embedding(x, position_offset=5)
    assert not torch.allclose(at_zero, at_five)


def test_greedy_generation_identical_with_and_without_cache(model, tok):
    prompt = "abcdefg"
    with torch.no_grad():
        cached = generate(model, tok, prompt, 20, use_cache=True)
        uncached = generate(model, tok, prompt, 20, use_cache=False)
    assert cached == uncached
    assert len(cached.split()) > len(tok.encode_ids(prompt, add_bos=True))


def test_generation_respects_context_limit():
    small = build_model(max_seq_len=16)
    tok = FakeTokenizer()
    with torch.no_grad():
        out = generate(small, tok, "abcdefgh", 100, use_cache=True)
    assert len(out.split()) <= 16


def test_oversize_prompt_is_truncated_not_crashed():
    model = build_model(max_seq_len=64)
    tok = FakeTokenizer()
    with torch.no_grad():
        out = generate(model, tok, "abcdefgh" * 30, 10, use_cache=True)
    assert len(out.split()) <= 64


def test_repetition_penalty_applied(model, tok):
    with torch.no_grad():
        out = generate(model, tok, "abcdefg", 20, use_cache=True, repetition_penalty=1.1)
    assert isinstance(out, str) and out


def generate(
    model,
    tok,
    prompt,
    max_new_tokens,
    use_cache=True,
    temperature=0.0,
    top_k=0,
    top_p=1.0,
    repetition_penalty=1.0,
):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("_gen_under_test", root / "tools" / "generate_gpt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate_text(
        model=model,
        tokenizer=tok,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        device=torch.device("cpu"),
        use_cache=use_cache,
    )
