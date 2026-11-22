"""
Saluran data skala besar (tahap A): FineWeb-2 bahasa Indonesia -> saring -> deduplikasi -> tokenisasi paralel -> token uint16.
Alasan dan target: docs/RENCANA_TAHAP1_PEMILIH_ALAT.md bukan untuk ini; lihat percakapan rencana model 500M/1B (butuh >= 11 miliar token).

Sifat:
  - STREAMING per row group (1.000 dokumen) lewat HfFileSystem: tidak mengunduh berkas 4,5 GB utuh (11 miliar token ~ 6.000 row group).
  - Tokenizer: kosakata 32.768 di data/prod_v2 dengan encoder trie yang sama dengan pra-tokenisasi lama (identik dengan encode_ids).
  - Keluaran uint16 (kosakata < 65.536): separuh ukuran int32 lama. Pemuat data mendukung dtype lewat konfigurasi.
  - Dapat DILANJUTKAN: manifest.jsonl per row group; berkas dipotong ke ukuran terakhir yang tercatat.
  - Penyaringan: skor bahasa, panjang, kata henti Indonesia, situs spam, karakter pengganti; duplikat tepat (hash) di dalam keluaran
    dan terhadap korpus lama (opsional, --dedupe-against).

  python tools/build_scale_corpus.py --shard 0 --max-row-groups 20          # uji awal (±40 juta token)
  python tools/build_scale_corpus.py --shard 0 --target-tokens 5e9          # lanjut sampai target; ulangi untuk shard berikutnya
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "apps"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

REPO = "HuggingFaceFW/fineweb-2"
SUBSET = "data/ind_Latn/train"
VOCAB = ("data/prod_v2/vocab_prod.json", "data/prod_v2/token2id_prod.json", "data/prod_v2/id2token_prod.json")
ID_STOPWORDS = {"ada", "adalah", "agar", "akan", "atau", "bahwa", "bagi", "banyak", "belum", "bisa", "dalam", "dan", "dari", "dengan", "di", "ia", "ini", "itu", "juga",
                "karena", "ke", "kita", "lebih", "masih", "mereka", "menjadi", "oleh", "pada", "para", "saat", "sebagai", "sehingga", "seperti", "sudah", "telah", "tidak",
                "untuk", "yang"}
SPAM_HOST = re.compile(r"(^|[./_-])(casino|judi|judol|slot|togel|taruhan|sbobet|poker|bandar|gacor|maxwin)([./_-]|$)", re.I)
_WORD = re.compile(r"[a-z]+")
_WS = re.compile(r"\s+")


# ---------- penyaringan ----------
def normalize_text(text):
    return _WS.sub(" ", text).strip()


def doc_hash(text):
    """Hash 64-bit dari teks ternormalisasi huruf kecil (duplikat tepat)."""
    return int.from_bytes(hashlib.blake2b(text.lower().encode("utf-8"), digest_size=8).digest(), "little")


def keep_doc(text, url, score, min_score=0.9, min_chars=200, max_chars=100_000, min_stopwords=5):
    """(diterima, alasan_ditolak). Penyaring ringan; FineWeb-2 sudah disaring bahasa dan diduplikasi per bahasa."""
    if score is not None and score < min_score:
        return False, "skor_bahasa"
    n = len(text)
    if n < min_chars:
        return False, "terlalu_pendek"
    if n > max_chars:
        return False, "terlalu_panjang"
    if "�" in text and text.count("�") / n > 0.001:
        return False, "karakter_pengganti"
    if url and SPAM_HOST.search(url):
        return False, "situs_spam"
    words = _WORD.findall(text[:5000].lower())
    if len(set(words) & ID_STOPWORDS) < min_stopwords:
        return False, "bukan_indonesia"
    return True, ""


# ---------- tokenisasi paralel ----------
_TOK = {}


def _init_worker():
    from config import get_config
    from tokenizer2 import load_tokenizer_from_config
    import dataset_pretokenizer_fast as F
    cfg = get_config("production_100m")
    cfg.VOCAB_PATH, cfg.TOKEN2ID_PATH, cfg.ID2TOKEN_PATH = (str(PROJECT_ROOT / p) for p in VOCAB)
    tok = load_tokenizer_from_config(cfg)
    _TOK.update(tok=tok, trie=F.build_trie(tok), enc=F.encode_tokens_fast, unk=tok.token2id[tok.unk_token], bos=tok.bos_token_id, eos=tok.eos_token_id)


def _tokenize_chunk(texts):
    tok, trie, enc, unk, t2i = _TOK["tok"], _TOK["trie"], _TOK["enc"], _TOK["unk"], _TOK["tok"].token2id
    out, lens, n_unk = [], [], 0
    for t in texts:
        ids = [_TOK["bos"]] + [t2i.get(x, unk) for x in enc(t, tok, trie)] + [_TOK["eos"]]
        n_unk += sum(i == unk for i in ids)
        out.append(np.asarray(ids, dtype=np.uint16))
        lens.append(len(ids))
    return np.concatenate(out), lens, n_unk


def tokenize_docs(pool, texts, chunk=64):
    chunks = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    parts = pool.map(_tokenize_chunk, chunks) if pool else [_tokenize_chunk(c) for c in chunks]
    arr = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, dtype=np.uint16)
    return arr, [x for p in parts for x in p[1]], sum(p[2] for p in parts)


# ---------- berkas keluaran dan manifest (dapat dilanjutkan) ----------
class Writer:
    """Menambah token ke <nama>.u16 dan panjang dokumen ke <nama>.lens (int32); manifest.jsonl mencatat kemajuan per row group."""

    def __init__(self, out_dir, name):
        self.dir, self.name = Path(out_dir), name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tok_path, self.len_path = self.dir / f"{name}.u16", self.dir / f"{name}.lens"
        self.manifest = self.dir / f"{name}.manifest.jsonl"
        self.done = {}
        self.tokens = self.docs = 0
        if self.manifest.exists():
            for line in self.manifest.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.done[rec["rg"]] = rec
                    self.tokens, self.docs = rec["cum_tokens"], rec["cum_docs"]
        for path, unit, n in ((self.tok_path, 2, self.tokens), (self.len_path, 4, self.docs)):   # buang sisa tulisan yang tidak tercatat
            if path.exists() and path.stat().st_size != unit * n:
                with open(path, "r+b") as f:
                    f.truncate(unit * n)
            elif not path.exists():
                path.touch()

    def append(self, rg, arr, lens, stats):
        with open(self.tok_path, "ab") as f:
            f.write(arr.tobytes())
        with open(self.len_path, "ab") as f:
            f.write(np.asarray(lens, dtype=np.int32).tobytes())
        self.tokens += int(len(arr))
        self.docs += len(lens)
        rec = {"rg": rg, "cum_tokens": self.tokens, "cum_docs": self.docs, **stats}
        with open(self.manifest, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self.done[rg] = rec

    def finalize(self):
        """Tulis <nama>.u16.docs.npy (n,2) bertipe int64 berisi (awal, akhir), format yang sama dengan pra-tokenisasi lama."""
        lens = np.fromfile(self.len_path, dtype=np.int32).astype(np.int64)
        ends = np.cumsum(lens)
        np.save(self.dir / f"{self.name}.u16.docs.npy", np.stack([ends - lens, ends], axis=1))
        return len(lens)


def load_old_hashes(path, cache):
    cache = Path(cache)
    if cache.exists():
        return set(np.load(cache).tolist())
    hs = []
    t0 = time.time()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                t = json.loads(line).get("text") or ""
            except ValueError:
                continue
            hs.append(doc_hash(normalize_text(t)))
            if i % 1_000_000 == 0 and i:
                print(f"  hash korpus lama: {i:,} dokumen ({time.time() - t0:.0f} dtk)", flush=True)
    arr = np.array(hs, dtype=np.uint64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return set(arr.tolist())


def process_row_group(table_rows, pool, seen, args):
    """table_rows: daftar (text, url, score). Kembalikan (token, panjang, statistik)."""
    stats = {"masuk": len(table_rows), "ditolak": {}, "duplikat": 0, "byte_teks": 0}
    texts = []
    for text, url, score in table_rows:
        text = normalize_text(text or "")
        ok, why = keep_doc(text, url, score, args.min_lang_score, args.min_chars, args.max_chars, args.min_stopwords)
        if not ok:
            stats["ditolak"][why] = stats["ditolak"].get(why, 0) + 1
            continue
        h = doc_hash(text)
        if h in seen:
            stats["duplikat"] += 1
            continue
        seen.add(h)
        texts.append(text)
        stats["byte_teks"] += len(text.encode("utf-8"))
    arr, lens, n_unk = tokenize_docs(pool, texts)
    stats.update(diterima=len(texts), tak_dikenal=n_unk, token=int(len(arr)))
    return arr, lens, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0, help="indeks berkas latih FineWeb-2 (0..31)")
    ap.add_argument("--max-row-groups", type=int, default=None)
    ap.add_argument("--target-tokens", type=float, default=None, help="berhenti bila total token berkas ini mencapai nilai ini")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "scale"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-lang-score", type=float, default=0.9)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-chars", type=int, default=100_000)
    ap.add_argument("--min-stopwords", type=int, default=5)
    ap.add_argument("--dedupe-against", default=None, help="korpus lama (jsonl) untuk deduplikasi lintas korpus; hash disimpan sebagai cache")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem
    files = sorted(f.path for f in HfApi().list_repo_tree(REPO, path_in_repo=SUBSET, repo_type="dataset") if f.path.endswith(".parquet"))
    path = files[args.shard]
    name = Path(path).stem
    w = Writer(args.out_dir, name)
    seen = set()
    if args.dedupe_against:
        print("memuat hash korpus lama ...", flush=True)
        seen = load_old_hashes(args.dedupe_against, Path(args.out_dir) / "hash_korpus_lama.npy")
        print(f"  {len(seen):,} hash", flush=True)
    pool = mp.Pool(args.workers, initializer=_init_worker) if args.workers > 1 else None
    if pool is None:
        _init_worker()
    fs = HfFileSystem()
    t0 = time.time()
    with fs.open(f"datasets/{REPO}/{path}", "rb") as fh:
        pf = pq.ParquetFile(fh)
        n_rg = pf.num_row_groups
        todo = [i for i in range(n_rg) if i not in w.done]
        if args.max_row_groups:
            todo = todo[:args.max_row_groups]
        print(f"{name}: {n_rg} row group; sudah {len(w.done)}; diproses sekarang {len(todo)}; token tercatat {w.tokens:,}", flush=True)
        agg = {"diterima": 0, "masuk": 0, "tak_dikenal": 0, "token": 0, "byte_teks": 0, "duplikat": 0}
        for k, rg in enumerate(todo, 1):
            t = pf.read_row_group(rg, columns=["text", "url", "language_score"])
            rows = list(zip(t.column("text").to_pylist(), t.column("url").to_pylist(), t.column("language_score").to_pylist()))
            arr, lens, st = process_row_group(rows, pool, seen, args)
            w.append(rg, arr, lens, st)
            for key in agg:
                agg[key] += st[key]
            if k % 10 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"  rg {rg:5d} ({k}/{len(todo)}) | token {w.tokens/1e6:8.1f} jt | {agg['token']/el:9,.0f} token/dtk | diterima {100*agg['diterima']/max(1, agg['masuk']):4.1f}% | "
                      f"tak dikenal {100*agg['tak_dikenal']/max(1, agg['token']):.3f}% | {agg['byte_teks']/max(1, agg['token']):.2f} byte/token", flush=True)
            if args.target_tokens and w.tokens >= args.target_tokens:
                print("target token tercapai", flush=True)
                break
    n = w.finalize()
    print(f"selesai: {w.docs:,} dokumen, {w.tokens:,} token ({w.tokens*2/1e9:.2f} GB uint16) -> {w.tok_path.name}, {w.tok_path.name}.docs.npy ({n:,} dokumen)")
    if pool:
        pool.close()


if __name__ == "__main__":
    main()
