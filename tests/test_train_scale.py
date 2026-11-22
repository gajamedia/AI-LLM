"""Tahap A dan B: saluran data skala besar (tools/build_scale_corpus.py) dan kode latih untuk model besar (gradient checkpointing, bench)."""
import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import load_module

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps"))
SC = load_module("build_scale_corpus_under_test", "tools/build_scale_corpus.py")
BT = load_module("bench_train_under_test", "tools/bench_train.py")


# ---------- B: gradient checkpointing tidak boleh mengubah hasil ----------
def _tiny_model(checkpoint):
    from gptModel import GPTModel
    cfg = BT.build_cfg("tiny", 32, dropout=0.0)
    torch.manual_seed(0)
    model, tok = GPTModel.from_config_with_tokenizer(cfg=cfg, load_tokenizer=True)
    model.train()
    model.transformer.gradient_checkpointing = checkpoint
    return model, len(tok.token2id)


def _batch(vocab):
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(4, vocab, (3, 32), generator=g)
    return {"input_ids": ids, "labels": ids, "document_ids": torch.zeros_like(ids)}


def test_gradient_checkpointing_gives_identical_loss_and_gradients():
    m0, vocab = _tiny_model(False)
    m1, _ = _tiny_model(True)
    b = _batch(vocab)
    l0 = m0.forward_batch(b)[0]
    l1 = m1.forward_batch(b)[0]
    assert torch.allclose(l0, l1, atol=1e-6)
    l0.backward()
    l1.backward()
    worst = max((p0.grad - p1.grad).abs().max().item() for p0, p1 in zip(m0.parameters(), m1.parameters()) if p0.grad is not None)
    assert worst < 1e-5, worst                                                  # gradien semua parameter sama (dalam toleransi numerik)


def test_checkpointing_is_inactive_in_eval_and_with_cache():
    m, vocab = _tiny_model(True)
    m.eval()
    ids = _batch(vocab)["input_ids"]
    with torch.no_grad():
        full = m(ids)
        step, past = m(ids[:, :16], use_cache=True)[:2]
    assert torch.isfinite(full).all() and past is not None                     # jalur inferensi dengan cache tetap bekerja


def test_training_script_exposes_the_new_flags_and_is_valid_python():
    src = (ROOT / "tools" / "train_gptV2.py").read_text(encoding="utf-8")
    ast.parse(src)
    for flag in ("--grad-checkpoint", "--compile", "--fused-adamw"):
        assert flag in src
    assert "forward_batch = torch.compile(model.forward_batch) if args.compile else model.forward_batch" in src   # model asli yang disimpan, bukan yang dikompilasi


def test_bench_runs_on_cpu_and_reports_the_planned_fields():
    res = BT.main(["--preset", "tiny", "--device", "cpu", "--batch", "2", "--ctx", "32", "--steps", "2", "--warmup", "1"])
    assert res["parameter_juta"] > 1 and res["token_per_detik"] > 0 and res["jam_untuk_10_miliar_token"] > 0
    assert BT.PRESETS["100m"] == (768, 12, 4, 12) and BT.PRESETS["1b"][3] == 22


def test_presets_match_the_parameter_counts_reported_in_the_plan():
    from gptModel import GPTModel
    for name, expected in (("100m", 100.9), ("300m", 305.2), ("540m", 543.4)):
        cfg = BT.build_cfg(name, 1024 if name != "100m" else 256)
        with torch.device("meta"):
            m, _ = GPTModel.from_config_with_tokenizer(cfg=cfg, load_tokenizer=True)
        assert abs(BT.count_params(m) / 1e6 - expected) < 0.2, name


# ---------- A: saluran data ----------
GOOD = ("Pemerintah daerah telah menyiapkan program baru untuk membantu para petani yang ada di desa karena harga pupuk naik, sehingga mereka bisa tetap menanam padi "
        "dan jagung dengan hasil yang lebih baik untuk keluarga mereka. Program ini akan dijalankan oleh dinas pertanian pada tahun ini di seluruh kecamatan.")


def test_keep_doc_filters():
    assert SC.keep_doc(GOOD, "https://contoh.id/berita", 0.99) == (True, "")
    assert SC.keep_doc("pendek saja", None, 0.99)[1] == "terlalu_pendek"
    assert SC.keep_doc(GOOD, None, 0.5)[1] == "skor_bahasa"
    assert SC.keep_doc(GOOD, "https://situs-judi-online.com/slot", 0.99)[1] == "situs_spam"
    assert SC.keep_doc("The quick brown fox jumps over the lazy dog and keeps running far away from here. " * 6, None, 0.99)[1] == "bukan_indonesia"
    assert SC.keep_doc(GOOD + "�" * 20, None, 0.99)[1] == "karakter_pengganti"
    assert SC.keep_doc("x" * 200_001, None, 0.99)[1] == "terlalu_panjang"


def test_normalize_and_hash_ignore_whitespace_and_case():
    assert SC.normalize_text("  a\n\n b\t c ") == "a b c"
    assert SC.doc_hash(SC.normalize_text("Halo  Dunia")) == SC.doc_hash(SC.normalize_text("halo\ndunia"))
    assert SC.doc_hash("a") != SC.doc_hash("b")


def test_process_row_group_dedupes_and_tokenizes_to_uint16_with_bos_eos():
    SC._init_worker()
    args = type("A", (), {"min_lang_score": 0.9, "min_chars": 200, "max_chars": 100_000, "min_stopwords": 5})()
    rows = [(GOOD, "https://a.id", 0.99), (GOOD.upper(), "https://b.id", 0.99), ("pendek", None, 0.99), (GOOD + " Tambahan berita lain untuk membedakan dokumen ini dari yang pertama.", None, 0.99)]
    seen = set()
    arr, lens, st = SC.process_row_group(rows, None, seen, args)
    assert st["masuk"] == 4 and st["diterima"] == 2 and st["duplikat"] == 1 and st["ditolak"] == {"terlalu_pendek": 1}
    assert arr.dtype == np.uint16 and len(arr) == sum(lens) == st["token"] and arr.max() < 32768
    assert arr[0] == SC._TOK["bos"] and arr[lens[0] - 1] == SC._TOK["eos"]
    arr2, _, st2 = SC.process_row_group(rows[:1], None, seen, args)           # sudah terlihat: duplikat
    assert st2["diterima"] == 0 and len(arr2) == 0


def test_writer_resumes_after_a_crash_and_finalizes_a_contiguous_index(tmp_path):
    w = SC.Writer(tmp_path, "contoh")
    w.append(0, np.arange(10, dtype=np.uint16), [4, 6], {"masuk": 2})
    w.append(1, np.arange(5, dtype=np.uint16), [5], {"masuk": 1})
    with open(w.tok_path, "ab") as f:                                          # tulisan setengah jadi setelah gagal di tengah row group 2
        f.write(np.arange(7, dtype=np.uint16).tobytes())
    with open(w.len_path, "ab") as f:
        f.write(np.array([7], dtype=np.int32).tobytes())
    w2 = SC.Writer(tmp_path, "contoh")
    assert sorted(w2.done) == [0, 1] and w2.tokens == 15 and w2.docs == 3
    assert w2.tok_path.stat().st_size == 30 and w2.len_path.stat().st_size == 12   # sisa dipotong ke yang tercatat
    w2.append(2, np.arange(3, dtype=np.uint16), [3], {"masuk": 1})
    n = w2.finalize()
    idx = np.load(tmp_path / "contoh.u16.docs.npy")
    assert n == 4 and idx.shape == (4, 2) and idx[:, 0].tolist() == [0, 4, 10, 15] and idx[-1, 1] == 18
    assert (idx[1:, 0] == idx[:-1, 1]).all()


def test_output_is_readable_by_the_training_dataset_with_uint16(tmp_path):
    from dataset2 import PackedDataset
    SC._init_worker()
    args = type("A", (), {"min_lang_score": 0.9, "min_chars": 200, "max_chars": 100_000, "min_stopwords": 5})()
    rows = [(GOOD + f" Dokumen nomor {i} berisi kabar daerah yang berbeda-beda setiap harinya." * 3, None, 0.99) for i in range(40)]
    w = SC.Writer(tmp_path, "uji")
    arr, lens, st = SC.process_row_group(rows, None, set(), args)
    w.append(0, arr, lens, st)
    w.finalize()
    ds = PackedDataset(str(w.tok_path), block_size=64, dtype="uint16", bos_token_id=SC._TOK["bos"], eos_token_id=SC._TOK["eos"])
    item = ds[0]
    assert len(ds) > 5 and tuple(item["input_ids"].shape) == (64,) and int(item["input_ids"].max()) < 32768


# ---------- paket dan notebook pengukuran efisiensi (Kaggle) ----------
def test_bench_kit_is_self_contained_and_notebook_is_valid(tmp_path):
    import json
    import subprocess
    import zipfile
    kit = load_module("package_bench_kit_under_test", "tools/package_bench_kit.py")
    z = kit.build(tmp_path / "kit.zip")
    root = tmp_path / "kit"
    zipfile.ZipFile(z).extractall(root)
    names = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert {"tools/bench_train.py", "apps/gptModel.py", "apps/attention.py", "apps/config.py", "data/prod_v2/token2id_prod.json"} <= names
    assert not any(n.endswith((".pt", ".bin")) for n in names)
    r = subprocess.run([sys.executable, "-I", str(root / "tools" / "bench_train.py"), "--preset", "tiny", "--device", "cpu", "--batch", "2", "--ctx", "32", "--steps", "1", "--warmup", "1"],
                       cwd=str(root), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and '"token_per_detik"' in r.stdout, r.stderr[-600:]
    nb = json.loads((ROOT / "kaggle" / "hai07_bench_train.ipynb").read_text(encoding="utf-8"))
    for c in [c for c in nb["cells"] if c["cell_type"] == "code"]:
        ast.parse("".join(c["source"]))
    assert "--grad-checkpoint" in "".join("".join(c["source"]) for c in nb["cells"])
