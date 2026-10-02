import pytest

INDONESIAN_SENTENCES = [
    "Indonesia adalah negara kepulauan di Asia Tenggara.",
    "Presiden membuka rapat kerja nasional di Jakarta pagi ini.",
    "Pemerintah menargetkan pertumbuhan ekonomi tahun depan lebih tinggi.",
    "Para peneliti mengembangkan model bahasa untuk teks berbahasa Indonesia.",
    "Hujan deras mengguyur wilayah Jawa Barat sejak sore kemarin.",
]

ASCII_LATIN_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "El rapido zorro marron salta sobre el perro perezoso.",
    "Ceci est une phrase en francais sans accents speciaux.",
]


def test_roundtrip_is_exact_on_indonesian(production_tokenizer):
    for text in INDONESIAN_SENTENCES:
        ids = production_tokenizer.encode_ids(text, add_bos=True)
        decoded = production_tokenizer.decode(ids)
        assert decoded == text, f"roundtrip mismatch:\n  {text!r}\n  {decoded!r}"


def test_roundtrip_is_exact_on_ascii_latin_text(production_tokenizer):
    for text in ASCII_LATIN_SENTENCES:
        ids = production_tokenizer.encode_ids(text, add_bos=True)
        assert production_tokenizer.decode(ids) == text


def test_non_ascii_characters_fall_back_to_unknown(production_tokenizer):
    """Known limitation: chars outside the trained vocab decode as <UNK>."""
    unk_id = getattr(production_tokenizer, "unk_token_id", None)
    if unk_id is None:
        pytest.skip("tokenizer has no unk_token_id attribute")

    ids = production_tokenizer.encode_ids("Der Hund läuft im Garten.", add_bos=True)
    assert unk_id in ids
    assert "<UNK>" in production_tokenizer.decode(ids)


def test_bos_token_is_prepended(production_tokenizer, production_cfg):
    ids = production_tokenizer.encode_ids("tes singkat", add_bos=True)
    assert ids[0] == production_tokenizer.bos_token_id


def test_add_bos_false_omits_bos(production_tokenizer):
    with_bos = production_tokenizer.encode_ids("tes singkat", add_bos=True)
    without = production_tokenizer.encode_ids("tes singkat", add_bos=False)
    assert with_bos[1:] == without


def test_decode_drops_special_tokens(production_tokenizer):
    ids = production_tokenizer.encode_ids("tes", add_bos=True)
    decoded = production_tokenizer.decode(ids)
    assert str(production_tokenizer.bos_token_id) not in decoded.split()


def test_unknown_token_rate_is_low_on_indonesian(production_tokenizer):
    unk_id = getattr(production_tokenizer, "unk_token_id", None)
    if unk_id is None:
        pytest.skip("tokenizer has no unk_token_id attribute")

    total = unknown = 0
    for text in INDONESIAN_SENTENCES:
        ids = production_tokenizer.encode_ids(text, add_bos=True)
        total += len(ids)
        unknown += sum(1 for i in ids if i == unk_id)

    assert total > 0
    assert unknown / total < 0.01


def test_tokenize_produces_nonempty_ids(production_tokenizer):
    for text in INDONESIAN_SENTENCES:
        ids = production_tokenizer.encode_ids(text, add_bos=True)
        assert len(ids) >= 2
        assert all(isinstance(i, int) for i in ids)


def test_production_vocab_size_is_32768(production_cfg, production_tokenizer):
    import json
    from pathlib import Path

    path = Path(str(production_cfg.TOKEN2ID_PATH))
    if not path.exists():
        pytest.skip("token2id artifact missing")
    with path.open("r", encoding="utf-8") as fh:
        token2id = json.load(fh)
    assert len(token2id) == 32768
