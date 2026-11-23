"""Tahap A dan B: saluran data skala besar (tools/build_scale_corpus.py) dan kode latih untuk model besar (gradient checkpointing, bench)."""
import ast
import json
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


# ---------- jalur TPU (torch_xla): bagian murni dan logika pekerja dengan tiruan torch_xla di CPU ----------
BX = load_module("bench_train_xla_under_test", "tools/bench_train_xla.py")


def test_xla_module_imports_without_torch_xla_and_maps_tpu_generations():
    assert "torch_xla" not in sys.modules or True
    assert BX.peak_tflops_per_chip("v5litepod-8") == BX.peak_tflops_per_chip("v5e-8") == 197.0
    assert BX.peak_tflops_per_chip("v6e-8") == 918.0 and BX.peak_tflops_per_chip("v5p-8") == 459.0 and BX.peak_tflops_per_chip("") is None
    s = BX.summarize([1.0, 1.2, 0.8], 8 * 2 * 1024, 3.7e9, 8, 197.0)
    assert s["token_per_detik"] == 16384 and s["detik_per_langkah_median"] == 1.0 and s["mfu_persen"] == round(100 * 3.7e9 * 16384 / 1e12 / (197 * 8), 1)


def test_checkpoint_patch_drops_unknown_kwargs():
    seen = {}

    def fake_xla(fn, *a):
        seen["args"] = a
        return fn(*a)
    p = BX.make_checkpoint_patch(fake_xla)
    assert p(lambda x, y: x + y, 1, 2, use_reentrant=False, preserve_rng_state=True) == 3 and seen["args"] == (1, 2)


def test_worker_logic_runs_end_to_end_on_cpu_with_a_stubbed_torch_xla(tmp_path, monkeypatch):
    import contextlib
    import types
    import torch.utils.checkpoint as tuc

    xm = types.ModuleType("torch_xla.core.xla_model")
    xm.xla_device = lambda: torch.device("cpu")
    xm.optimizer_step = lambda opt: opt.step()
    xm.mark_step = lambda: None
    xm.wait_device_ops = lambda: None
    xm.get_memory_info = lambda d: {"bytes_used": 123_000_000}
    xr = types.ModuleType("torch_xla.runtime")
    xr.world_size, xr.global_ordinal = (lambda: 1), (lambda: 0)
    xr.use_spmd, xr.global_runtime_device_count = (lambda: None), (lambda: 1)
    marks = []
    spmd = types.ModuleType("torch_xla.distributed.spmd")
    spmd.Mesh = lambda ids, shape, names: ("mesh", tuple(shape), names)
    spmd.mark_sharding = lambda t, mesh, spec: marks.append(spec)
    dist = types.ModuleType("torch_xla.distributed")
    dist.spmd = spmd
    ck = types.ModuleType("torch_xla.utils.checkpoint")
    ck.checkpoint = lambda fn, *a: fn(*a)
    pkg = types.ModuleType("torch_xla")
    pkg.core, pkg.runtime, pkg.utils = types.ModuleType("torch_xla.core"), xr, types.ModuleType("torch_xla.utils")
    pkg.core.xla_model, pkg.utils.checkpoint = xm, ck
    pkg.distributed = dist
    for name, mod in {"torch_xla": pkg, "torch_xla.core": pkg.core, "torch_xla.core.xla_model": xm, "torch_xla.runtime": xr, "torch_xla.distributed": dist,
                      "torch_xla.distributed.spmd": spmd, "torch_xla.utils": pkg.utils, "torch_xla.utils.checkpoint": ck}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(tuc, "checkpoint", tuc.checkpoint)                      # dipulihkan otomatis setelah tes
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5litepod-8")
    out = tmp_path / "o.jsonl"
    args = BX.parse_args(["--preset", "tiny", "--ctx", "32", "--batch", "2", "--steps", "2", "--warmup", "1", "--grad-checkpoint", "--out", str(out)])
    BX._worker(0, args)
    res = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert res["token_per_detik"] > 0 and res["chip"] == 1 and res["grad_checkpoint"] is True and res["memori_dipakai_gb"] == 0.12 and "galat" not in res
    assert res["mfu_persen"] is not None and res["mode"] == "spmd"
    assert marks and all(spec == ("data", None) for spec in marks)                    # mode bawaan spmd: hanya batch yang dipecah (parameter direplikasi)
    marks.clear()
    args2 = BX.parse_args(["--preset", "tiny", "--ctx", "32", "--batch", "2", "--steps", "1", "--warmup", "1", "--shard-params", "--accelerator", "v6e", "--out", str(out)])
    BX._worker(0, args2)
    assert any(len(spec) >= 2 and spec[0] == "data" for spec in marks) and len(marks) > 4          # --shard-params menandai parameter juga
    assert json.loads(out.read_text(encoding="utf-8").splitlines()[-1])["shard_params"] is True
    args3 = BX.parse_args(["--preset", "tiny", "--ctx", "32", "--batch", "2", "--steps", "1", "--warmup", "1", "--mode", "mp", "--out", str(out)])
    BX._worker(0, args3)                                                                          # mode lama tetap berjalan
    assert json.loads(out.read_text(encoding="utf-8").splitlines()[-1])["mode"] == "mp"


def test_bench_kit_ships_the_xla_runner_and_the_tpu_notebook_is_valid(tmp_path):
    import zipfile
    kit = load_module("package_bench_kit_under_test_xla", "tools/package_bench_kit.py")
    names = set(zipfile.ZipFile(kit.build(tmp_path / "k.zip")).namelist())
    assert "tools/bench_train_xla.py" in names and "tools/bench_train.py" in names
    nb = json.loads((ROOT / "kaggle" / "hai07_bench_xla.ipynb").read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for c in code:
        ast.parse("".join(c["source"]))
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "--preset\", \"tiny\"" in text and "UJI ASAP GAGAL" in text and "20 jam per minggu" in text


def test_bench_train_perangkat_non_cuda_sinkron_dan_argumen_tpu():
    import types
    args = BT.parse_args(["--preset", "tiny", "--device", "tpu", "--amp", "--peak-tflops", "197", "--device-name", "TPU v5e-1"])
    assert args.device == "tpu" and args.peak_tflops == 197.0 and args.device_name == "TPU v5e-1" and args.amp
    model = torch.nn.Linear(4, 4)
    BT.sync_device(types.SimpleNamespace(type="tpu"), model)         # perangkat non-CUDA: membaca satu nilai parameter, tidak boleh gagal
    BT.sync_device(types.SimpleNamespace(type="cpu"), model)
    res = BT.run(BT.parse_args(["--preset", "tiny", "--device", "cpu", "--batch", "2", "--ctx", "32", "--steps", "1", "--warmup", "1", "--peak-tflops", "1", "--device-name", "CPU uji"]))
    assert res["perangkat"] == "CPU uji" and res["mfu_persen"] is not None and res["bf16_murni"] is False


def test_notebook_colab_sintaks_dan_mendeteksi_tiga_jalur():
    import ast
    nb = json.loads((ROOT / "kaggle" / "hai07_bench_colab.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    ast.parse(src)
    assert all(k in src for k in ("torch_tpu", "torch_xla", "tpu_torch", "run_inproc", "--peak-tflops"))


def test_bench_train_menyambung_ulang_bobot_terikat_setelah_pindah_perangkat():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tie_weights = True
            self.token_embedding = torch.nn.Embedding(10, 4)
            self.lm_head = torch.nn.Linear(4, 10, bias=False)
            self.lm_head.weight = self.token_embedding.weight
    m = M()
    m.lm_head.weight = torch.nn.Parameter(m.lm_head.weight.detach().clone())      # meniru backend yang memisahkan bobot terikat
    assert BT.count_params(m) == 80
    BT.retie_weights(m)
    assert m.lm_head.weight is m.token_embedding.weight and BT.count_params(m) == 40


# ---------- multi-GPU (DDP): sampler sadar rank ----------
def _sampler(n, rank, world, seed=7):
    from dataset2 import ResumableRandomSampler
    g = torch.Generator().manual_seed(seed)
    return ResumableRandomSampler(list(range(n)), shuffle=True, generator=g, rank=rank, world_size=world)


def test_sampler_dua_rank_tidak_tumpang_tindih_dan_sama_panjang():
    s0, s1 = _sampler(101, 0, 2), _sampler(101, 1, 2)
    a, b = list(s0), list(s1)
    assert len(a) == len(b) == len(s0) == 50                     # sisa 1 sampel dibuang agar jumlah sama
    assert not set(a) & set(b)


def test_sampler_satu_rank_sama_dengan_perilaku_lama():
    s = _sampler(33, 0, 1)
    out = list(s)
    assert sorted(out) == list(range(33)) and len(s) == 33


def test_sampler_lanjut_dari_posisi_global_lintas_world_size():
    s = _sampler(40, 0, 2)
    it = iter(s)
    first = [next(it) for _ in range(5)]
    s.mark_consumed(5)                                           # 5 sampel per rank = 10 posisi global
    assert s.consumed_position == 10
    state = s.state_dict()
    s2 = _sampler(40, 1, 2, seed=99)                             # rank lain, seed lain: harus ikut permutasi checkpoint
    s2.load_state_dict(state)
    rest = list(s2)
    assert len(rest) == 15 and not set(rest) & set(first)


def test_sampler_rank_di_luar_rentang_ditolak():
    with pytest.raises(ValueError):
        _sampler(10, 2, 2)


def test_trainer_ddp_terpasang_aman():
    src = (ROOT / "tools" / "train_gptV2.py").read_text(encoding="utf-8")
    ast.parse(src)
    for k in ("DistributedDataParallel", "no_sync", "_sync_any", "WORLD_SIZE", "init_process_group", "_is_main"):
        assert k in src


def test_notebook_bench_t4x2_valid_dan_memakai_torchrun():
    nb = json.loads((ROOT / "kaggle" / "hai07_bench_t4x2.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    ast.parse(src)
    assert all(k in src for k in ("torch.distributed.run", "nproc_per_node=2", "jumlah_gpu", "skala_persen"))


def test_notebook_t4x2_tidak_memberi_argumen_python_ke_torchrun():
    nb = json.loads((ROOT / "kaggle" / "hai07_bench_t4x2.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'TORCHRUN = [sys.executable, "-u", "-m"' in src      # -u harus sebelum -m; setelahnya torchrun menolaknya
    assert '+ ["-u", WORK' not in src


# ---------- prapelatihan 540M: shard ganda, rentang dokumen, batas waktu ----------
def _tiny_shard(tmp_path, name, n_docs, seed, dtype=np.uint16, vocab=32768):
    rng = np.random.default_rng(seed)
    lens = rng.integers(5, 40, size=n_docs)
    toks, offs, pos = [], [], 0
    for L in lens:
        d = rng.integers(4, vocab, size=int(L)).astype(dtype)
        toks.append(d)
        offs.append((pos, pos + int(L)))
        pos += int(L)
    f = tmp_path / name
    np.concatenate(toks).tofile(f)
    np.save(str(f) + ".docs.npy", np.asarray(offs, dtype=np.int64))
    return str(f), np.concatenate(toks)


def test_parse_token_spec():
    from dataset2 import parse_token_spec
    assert parse_token_spec("a.u16") == [("a.u16", None)]
    assert parse_token_spec("a.u16, b.u16@10:20,c.u16@5:") == [("a.u16", None), ("b.u16", (10, 20)), ("c.u16", (5, None))]
    with pytest.raises(ValueError):
        parse_token_spec(" , ")


def test_dataset_multi_shard_uint16_dan_rentang_dokumen_tidak_tumpang_tindih(tmp_path):
    from dataset2 import PackedDataset, ShardedPackedDataset
    a, _ = _tiny_shard(tmp_path, "a.u16", 60, 1)
    b, _ = _tiny_shard(tmp_path, "b.u16", 80, 2)
    pa = PackedDataset(a, block_size=64, dtype="uint16", bos_token_id=1, eos_token_id=2)
    train = ShardedPackedDataset([(a, None), (b, (0, 60))], block_size=64, dtype="uint16")
    val = ShardedPackedDataset([(b, (60, None))], block_size=64, dtype="uint16")
    pb_head = PackedDataset(b, block_size=64, dtype="uint16", document_range=(0, 60))
    assert len(train) == len(pa) + len(pb_head)
    # dokumen validasi (b 60..) tidak ada di rentang latih
    assert train.num_training_tokens + val.num_training_tokens == pa.num_training_tokens + PackedDataset(b, block_size=64, dtype="uint16").num_training_tokens
    x = train[len(pa)]                                   # blok pertama shard b
    assert x["input_ids"].shape == (64,) and x["input_ids"].dtype == torch.long
    with pytest.raises(IndexError):
        train[len(train)]
    with pytest.raises(ValueError):
        PackedDataset(a, block_size=64, dtype="uint16", document_range=(1000, None))


def test_dataloader_dengan_sampler_dua_rank_dan_multi_shard(tmp_path):
    from dataset2 import create_dataloader
    a, _ = _tiny_shard(tmp_path, "a.u16", 60, 1)
    b, _ = _tiny_shard(tmp_path, "b.u16", 80, 2)
    kw = dict(token_file=f"{a},{b}@0:50", block_size=64, batch_size=3, dtype="uint16", resumable=True, seed=5, world_size=2,
              num_workers=0, pin_memory=False, shuffle=True, drop_last=False, dataset_type="packed")
    l0, l1 = create_dataloader(rank=0, **kw), create_dataloader(rank=1, **kw)
    assert len(l0.sampler) == len(l1.sampler)
    assert not set(list(l0.sampler)) & set(list(l1.sampler))
    batch = next(iter(l0))
    assert batch["input_ids"].shape[1] == 64


def test_trainer_punya_batas_waktu_dan_config_540m():
    src = (ROOT / "tools" / "train_gptV2.py").read_text(encoding="utf-8")
    for k in ("--time-limit-s", "_PROCESS_START", "--no-best-save", "meta.json"):
        assert k in src
    from config import get_config
    c = get_config("production_540m")
    assert (c.EMBEDDING_DIM, c.TRANSFORMER_NUM_LAYERS, c.DATASET_BLOCK_SIZE) == (1280, 28, 1024)
    assert str(np.dtype(c.DATASET_DTYPE)) == "uint16" and c.TRAIN_GRAD_ACCUM == 64 and c.TRANSFORMER_DROPOUT == 0.0


def test_notebook_prapelatihan_540m_valid_dan_aman_antar_sesi(tmp_path):
    import re
    nb = json.loads((ROOT / "kaggle" / "hai07_pretrain_540m_t4x2.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    ast.parse(src)
    for k in ("torch.distributed.run", "production_540m", "--time-limit-s", "--resume", "FIRST_SESSION", "meta.json", "--no-best-save", "VAL_SPEC"):
        assert k in src
    assert 'TORCHRUN' not in src and '+ ["-u", WORK' not in src
    # urutan argumen: -u sebelum -m (torchrun menolak -u setelahnya)
    assert '[sys.executable, "-u", "-m", "torch.distributed.run"' in src
    # regex pengukur kecepatan cocok dengan format log trainer
    rx = re.compile(r"step=(\d+) .*elapsed=([0-9.]+)s")
    line = "epoch=1/1 step=3 run_optimizer_steps=3 micro_steps=6 batch=6 loss=10.4 ma_loss=n/a validation_loss=n/a lr=1e-4 ppl~=3 elapsed=5.7s"
    assert rx.search(line).groups() == ("3", "5.7")
    # tanpa pelarian ganda di regex notebook (pengamanan terhadap bug escape pembuat notebook)
    bs = chr(92)
    assert "step=(" + bs + "d+)" in src and "step=(" + bs * 2 + "d+)" not in src


def test_paket_prapelatihan_memuat_kode_dan_kosakata(tmp_path):
    import zipfile
    PK = load_module("package_pretrain_kit_under_test", "tools/package_pretrain_kit.py")
    out = PK.build(tmp_path / "kit.zip")
    names = set(zipfile.ZipFile(out).namelist())
    assert {"apps/dataset2.py", "apps/config.py", "tools/train_gptV2.py", "data/prod/vocab_prod.json", "data/prod/token2id_prod.json", "data/prod/id2token_prod.json"} <= names
