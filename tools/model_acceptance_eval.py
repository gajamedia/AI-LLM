"""
Uji kelulusan model untuk aplikasi Memori: apakah sebuah model BISA menggantikan pembaca gpt_sft_best.pt dan
(untuk alur AI modern) menjadi pemilih alat? Dijalankan pada model apa pun lewat antarmuka Backend; hasilnya angka
dan keputusan lulus/gagal per kriteria, bukan perkiraan.

Empat ujian:
  router     : memilih satu dari 6 alat (urut, hitung, tanggal, cari, sapaan, tolak) pada 3 set pesan berlabel
               (evalsets/memory_router*.jsonl). Pembajakan = pesan 'cari' yang dialihkan ke alat lain.
  qa_beku    : jawaban dari catatan pada set uji beku (evalsets/memory_handwritten.jsonl, memory_complex*.jsonl) lewat agen penuh
               (pencarian, penjaga, koreksi) dengan model sebagai pembaca. Pembanding: gpt_sft_best.pt = 10/1/9 dan 28/9/32.
  kasus      : 26 kasus bertema (evalsets/memory_cases.jsonl): jam/ruang/tim, definisi, pembaruan fakta, pertanyaan lanjutan,
               tanya balik, dan pertanyaan yang HARUS ditolak. Tag 'nyata' = berasal dari kegagalan nyata yang sudah diperbaiki
               (uji regresi, bukan ujian bersih); tag 'segar' = ditulis untuk uji ini.
  ketahanan  : kasus yang lulus ditanyakan ulang dengan susunan berbeda (huruf kecil tanpa '?', spasi sebelum '?', kapital + '?').

Cara pakai:
  python tools/model_acceptance_eval.py --backend gpt100m                       # garis dasar: model sekarang
  python tools/model_acceptance_eval.py --backend hf --model <id atau folder>   # perlu: pip install transformers accelerate
  python tools/model_acceptance_eval.py --backend gpt100m --quick               # uji asap (kriteria tidak berlaku)
  --strict mengembalikan kode keluar 1 bila ada kriteria gagal. --criteria berkas.json menimpa kriteria bawaan.
Kriteria bawaan dibaca dari evalsets/acceptance_criteria.json (disetujui pengguna 2026-11-22); CRITERIA di kode hanya cadangan.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import memory_agent as M  # noqa: E402
import rag_lib as R  # noqa: E402

LABELS = ["urut", "hitung", "tanggal", "cari", "sapaan", "tolak"]
ROUTER_SETS = {"dev": "memory_router.jsonl", "heldout": "memory_router_heldout.jsonl", "test2": "memory_router_test2.jsonl"}
REFUSAL = "Maaf, jawabannya tidak terdapat dalam teks yang diberikan."

CRITERIA = {
    "router": {"akurasi_min_per_set": 0.90, "pembajakan_cari_maks": 0},
    "qa_beku": {"tulisan_tangan": {"benar_min": 14, "salah_maks": 1, "mengarang_maks": 0},
                "kalimat_rumit": {"benar_min": 42, "salah_maks": 9, "mengarang_maks": 1}},
    "kasus": {"lolos_min": 22, "abstain_wajib_lolos": True, "tim_wajib_lolos": True},
    "ketahanan": {"lolos_varian_min": 0.85},
}

TOOL_PROMPT = ("Anda memilih alat untuk pesan pengguna. Alat: urut (mengurutkan angka); hitung (operasi hitung); tanggal (hari dan selisih tanggal); "
               "cari (mencari jawaban di catatan pengguna); sapaan (menyapa, berterima kasih); tolak (menulis, menerjemahkan, merangkum). "
               "Jawab dengan SATU kata saja dari: urut, hitung, tanggal, cari, sapaan, tolak.")
READ_PROMPT = ("Anda membaca teks lalu menjawab pertanyaan. Jawab HANYA dengan potongan kata persis dari teks, sesingkat mungkin. "
               "Jika jawabannya tidak ada di teks, jawab persis: " + REFUSAL)


# ---------- antarmuka model ----------
class Backend:
    """Antarmuka minimal. read: satu catatan + pertanyaan -> kutipan atau penolakan. choose_tool: pesan -> satu dari LABELS."""
    name = "dasar"

    def read(self, ctx, question):
        raise NotImplementedError

    def choose_tool(self, message):
        raise NotImplementedError

    def info(self):
        return {}


def label_in(text):
    low = (text or "").lower()
    hits = [(low.find(l), l) for l in LABELS if l in low]
    return min(hits)[1] if hits else None


class Gpt100mBackend(Backend):
    """Model sekarang (gpt_sft_best.pt) lewat pembaca aplikasi; pemilih alat memakai peringkat peluang (cara terbaik yang diukur)."""

    def __init__(self, checkpoint):
        self.reader = M.Reader(checkpoint)
        self.tok = self.reader.tok
        self.name = f"gpt100m ({Path(checkpoint).parent.name})"
        import memory_router_eval as RE
        self._RE = RE

    def read(self, ctx, question):
        return self.reader.read(ctx, question)

    def choose_tool(self, message):
        return self._RE.rank_label(self.reader, self._RE.CONTEXT, message)

    def info(self):
        return {"parameter_juta": round(sum(p.numel() for p in self.reader.model.parameters()) / 1e6, 1)}


class HFBackend(Backend):
    """Model HuggingFace mana pun (perlu transformers). Pembaca dan pemilih alat memakai prompt chat; keluaran greedy."""

    def __init__(self, model_id, max_new=60):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise SystemExit("Backend 'hf' memerlukan transformers: pip install transformers accelerate")
        self.name = f"hf ({model_id})"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        self.max_new, self.torch = max_new, torch

    def _chat(self, system, user, max_new):
        tk = self.tokenizer
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) if getattr(tk, "chat_template", None) \
            else system + "\n\n" + user + "\nJawaban:"
        ids = tk(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**ids, max_new_tokens=max_new, do_sample=False)
        return tk.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def read(self, ctx, question):
        return self._chat(READ_PROMPT, f"Teks:\n{ctx}\n\nPertanyaan: {question}", self.max_new)

    def choose_tool(self, message):
        return label_in(self._chat(TOOL_PROMPT, message, 8))

    def info(self):
        return {"parameter_juta": round(sum(p.numel() for p in self.model.parameters()) / 1e6, 1)}


class RouterBackend(Backend):
    """Pemilih alat buatan sendiri (tools/memory_router.py) dengan kebijakan ambang; hanya untuk ujian 'router' (tidak membaca catatan)."""

    def __init__(self):
        import memory_router
        self.router = memory_router.Router.load()
        m = self.router.meta
        self.tau = (m["tau"], m["tau_sapaan"], m["tau_tolak"])
        self.name = "classifier alat sendiri (regresi logistik n-gram)"

    def choose_tool(self, message):
        return self.router.decide(message, *self.tau)[0]

    def read(self, ctx, question):
        raise NotImplementedError("backend 'router' hanya untuk ujian pemilih alat: gunakan --only router")

    def info(self):
        return {"parameter_juta": round(self.router.W.size / 1e6, 2)}


class _AgentReader:
    """Menyesuaikan Backend ke antarmuka pembaca agen; mencatat waktu tiap pembacaan."""

    def __init__(self, backend):
        self.backend, self.times = backend, []
        self.tok = getattr(backend, "tok", None)

    def read(self, ctx, question):
        t0 = time.time()
        try:
            return self.backend.read(ctx, question)
        finally:
            self.times.append(time.time() - t0)


# ---------- ujian ----------
def load(name):
    return [json.loads(l) for l in open(PROJECT_ROOT / "evalsets" / name, encoding="utf-8") if l.strip()]


def run_router(backend, limit=None):
    out = {}
    for name, fname in ROUTER_SETS.items():
        rows = load(fname)
        if limit:  # cuplikan merata di semua kelas (berkas diurutkan per kelas, jadi bukan N pertama)
            step = max(1, len(rows) // limit)
            rows = rows[::step][:limit * 2]
        preds = [backend.choose_tool(r["message"]) for r in rows]
        ok = sum(p == r["label"] for p, r in zip(preds, rows))
        hijack = sum(1 for p, r in zip(preds, rows) if r["label"] == "cari" and p != "cari")
        sebaran = {}
        for p in preds:
            sebaran[p] = sebaran.get(p, 0) + 1
        out[name] = {"benar": ok, "n": len(rows), "akurasi": ok / len(rows), "pembajakan_cari": hijack, "sebaran": sebaran}
    return out


def _agent(store, reader):
    return M.MemoryAgent(store, reader)


def run_qa_frozen(reader, limit=None):
    unans_hw = json.loads((PROJECT_ROOT / "evalsets" / "memory_handwritten_unans.json").read_text(encoding="utf-8"))
    suites = {"tulisan_tangan": (load("memory_handwritten.jsonl"), unans_hw),
              "kalimat_rumit": (load("memory_complex.jsonl"), [r["question"] for r in load("memory_complex_unans.jsonl")])}
    out = {}
    for name, (items, unans) in suites.items():
        store = M.MemoryStore()
        for n in dict.fromkeys(r["note"] for r in items):
            store.teach(n)
        agent = _agent(store, reader)
        items, unans = (items[:limit], unans[:max(1, limit // 3)]) if limit else (items, unans)
        t = {"benar": 0, "salah": 0, "belum_tahu": 0, "mengarang": 0, "n_terjawab": len(items), "n_tak_terjawab": len(unans)}
        for r in items:
            o = agent.ask(r["question"])
            a = o["answer"] if o["status"] in ("ok", "conflict") else None
            t["belum_tahu" if a is None else "benar" if R.norm(r["answer"]) in R.norm(a) else "salah"] += 1
        for q in unans:
            t["mengarang"] += agent.ask(q)["status"] in ("ok", "conflict")
        out[name] = t
    return out


def eval_case(agent, case, question=None):
    o = agent.ask(question or case["question"], case.get("history"))
    st, ans = o["status"], o.get("answer", "")
    if case.get("clarify"):
        ok = st == "clarify"
    elif case.get("abstain"):
        ok = st not in ("ok", "conflict")
    else:
        ok = st in ("ok", "conflict") and R.norm(case["answer"]) in R.norm(ans)
    return {"lolos": bool(ok), "status": st, "jawaban": ans if st in ("ok", "conflict") else None}


def _case_agent(case, reader):
    store = M.MemoryStore()
    for n in case["notes"]:
        store.teach(n)
    return _agent(store, reader)


def variants(q):
    base = q.rstrip("?").strip()
    return [base.lower(), base + " ?", base[:1].upper() + base[1:] + "?"]


def run_cases(reader, limit=None):
    cases = load("memory_cases.jsonl")[:limit] if limit else load("memory_cases.jsonl")
    rows, var_ok, var_n = [], 0, 0
    for c in cases:
        res = eval_case(_case_agent(c, reader), c)
        rows.append({"id": c["id"], "tag": c["tag"], "jenis": "tanya_balik" if c.get("clarify") else "tolak" if c.get("abstain") else "jawab", **res})
        if res["lolos"] and not c.get("clarify"):
            for v in variants(c["question"]):
                var_n += 1
                var_ok += eval_case(_case_agent(c, reader), c, v)["lolos"]
    return rows, {"lolos_varian": var_ok / var_n if var_n else 0.0, "varian_ok": var_ok, "varian_n": var_n}


# ---------- kriteria ----------
def evaluate_criteria(res, criteria):
    """Kembalikan daftar (nama, nilai, syarat, lolos) dari hasil ujian yang tersedia."""
    out = []
    if "router" in res:
        c = criteria["router"]
        for name, r in res["router"].items():
            out.append((f"router/{name}: akurasi", f"{r['benar']}/{r['n']} ({100 * r['akurasi']:.1f}%)", f">= {100 * c['akurasi_min_per_set']:.0f}%", r["akurasi"] >= c["akurasi_min_per_set"]))
        hij = sum(r["pembajakan_cari"] for r in res["router"].values())
        out.append(("router: pesan 'cari' dibajak alat", str(hij), f"<= {c['pembajakan_cari_maks']}", hij <= c["pembajakan_cari_maks"]))
    if "qa_beku" in res:
        for name, r in res["qa_beku"].items():
            c = criteria["qa_beku"][name]
            out.append((f"qa_beku/{name}: benar", f"{r['benar']}/{r['n_terjawab']}", f">= {c['benar_min']}", r["benar"] >= c["benar_min"]))
            out.append((f"qa_beku/{name}: salah", str(r["salah"]), f"<= {c['salah_maks']}", r["salah"] <= c["salah_maks"]))
            out.append((f"qa_beku/{name}: mengarang (tak terjawab)", f"{r['mengarang']}/{r['n_tak_terjawab']}", f"<= {c['mengarang_maks']}", r["mengarang"] <= c["mengarang_maks"]))
    if "kasus" in res:
        c, rows = criteria["kasus"], res["kasus"]
        n_ok = sum(r["lolos"] for r in rows)
        out.append(("kasus: lolos", f"{n_ok}/{len(rows)}", f">= {c['lolos_min']}", n_ok >= c["lolos_min"]))
        if c.get("abstain_wajib_lolos"):
            ab = [r for r in rows if r["jenis"] == "tolak"]
            out.append(("kasus: pertanyaan yang harus ditolak", f"{sum(r['lolos'] for r in ab)}/{len(ab)}", "semua", bool(ab) and all(r["lolos"] for r in ab)))
        if c.get("tim_wajib_lolos"):
            tim = [r for r in rows if r["id"].startswith("rapat-tim") or r["id"] == "rapat-tim-mawar-diadakan"]
            out.append(("kasus: nama tim (gagal di model sekarang)", f"{sum(r['lolos'] for r in tim)}/{len(tim)}", "semua", bool(tim) and all(r["lolos"] for r in tim)))
    if "ketahanan" in res:
        c, r = criteria["ketahanan"], res["ketahanan"]
        out.append(("ketahanan: varian pertanyaan lolos", f"{r['varian_ok']}/{r['varian_n']} ({100 * r['lolos_varian']:.1f}%)", f">= {100 * c['lolos_varian_min']:.0f}%", r["lolos_varian"] >= c["lolos_varian_min"]))
    return out


DEFAULT_CRITERIA_FILE = PROJECT_ROOT / "evalsets" / "acceptance_criteria.json"


def load_criteria(path=None):
    """Kriteria dari --criteria; bila tidak ada, dari evalsets/acceptance_criteria.json (disetujui pengguna); bila berkas hilang, CRITERIA di kode."""
    f = Path(path) if path else DEFAULT_CRITERIA_FILE
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    if path:
        raise SystemExit(f"Berkas kriteria tidak ditemukan: {path}")
    return CRITERIA


def run_all(backend, only=None, quick=False, criteria=None):
    criteria = criteria or CRITERIA
    only = set(only or ["router", "qa_beku", "kasus", "ketahanan"])
    limit = 6 if quick else None
    reader = backend.reader if isinstance(backend, Gpt100mBackend) else _AgentReader(backend)
    wrapped = _AgentReader(backend) if isinstance(backend, Gpt100mBackend) else reader
    res = {"model": backend.name, "info": backend.info(), "cepat": quick}
    if "router" in only:
        res["router"] = run_router(backend, limit)
    if "qa_beku" in only:
        res["qa_beku"] = run_qa_frozen(wrapped, limit)
    if only & {"kasus", "ketahanan"}:
        rows, rob = run_cases(wrapped, limit)
        if "kasus" in only:
            res["kasus"] = rows
        if "ketahanan" in only:
            res["ketahanan"] = rob
    if wrapped.times:
        res["detik_per_baca_median"] = round(statistics.median(wrapped.times), 3)
    res["kriteria"] = [{"nama": n, "nilai": v, "syarat": s, "lolos": bool(ok)} for n, v, s, ok in evaluate_criteria(res, criteria)]
    res["semua_lolos"] = all(k["lolos"] for k in res["kriteria"]) if res["kriteria"] else False
    return res


def print_report(res):
    print(f"\nModel: {res['model']} {res.get('info') or ''}" + ("   [UJI CEPAT: kriteria tidak berlaku]" if res.get("cepat") else ""))
    if "detik_per_baca_median" in res:
        print(f"Median waktu per pembacaan: {res['detik_per_baca_median']} detik (informasi, bukan kriteria)")
    if "router" in res:
        print("\nPemilih alat:")
        for n, r in res["router"].items():
            print(f"  {n:8s} {r['benar']:3d}/{r['n']:<3d} {100 * r['akurasi']:5.1f}%  pembajakan 'cari': {r['pembajakan_cari']}  sebaran: {r['sebaran']}")
    if "qa_beku" in res:
        print("\nJawaban dari catatan (set uji beku):")
        for n, r in res["qa_beku"].items():
            print(f"  {n:15s} benar {r['benar']:3d}  salah {r['salah']:3d}  belum tahu {r['belum_tahu']:3d} | tak terjawab: mengarang {r['mengarang']}/{r['n_tak_terjawab']}")
    if "kasus" in res:
        gagal = [r for r in res["kasus"] if not r["lolos"]]
        print(f"\nKasus: {sum(r['lolos'] for r in res['kasus'])}/{len(res['kasus'])} lolos" + (". Gagal: " + ", ".join(f"{r['id']}[{r['status']}]" for r in gagal) if gagal else ""))
    print("\nKriteria:")
    for k in res["kriteria"]:
        print(f"  {'LULUS' if k['lolos'] else 'GAGAL'}  {k['nama']:52s} {k['nilai']:>16s}   syarat {k['syarat']}")
    if res["kriteria"] and not res.get("cepat"):
        print("\nKEPUTUSAN: " + ("SEMUA KRITERIA LULUS" if res["semua_lolos"] else f"{sum(not k['lolos'] for k in res['kriteria'])} dari {len(res['kriteria'])} kriteria GAGAL"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Uji kelulusan model untuk aplikasi Memori")
    ap.add_argument("--backend", choices=["gpt100m", "hf", "router"], default="gpt100m")
    ap.add_argument("--model", default=None, help="gpt100m: jalur checkpoint (bawaan sft_mem2). hf: id atau folder model.")
    ap.add_argument("--only", default=None, help="koma: router,qa_beku,kasus,ketahanan")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--criteria", default=None, help="berkas JSON yang menimpa CRITERIA")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    criteria = load_criteria(args.criteria)
    if args.backend == "gpt100m":
        backend = Gpt100mBackend(args.model or str(PROJECT_ROOT / "reports/checkpoints_100m_sft_mem2/gpt_sft_best.pt"))
    elif args.backend == "router":
        backend = RouterBackend()
        if args.only is None:
            args.only = "router"
    else:
        if not args.model:
            raise SystemExit("--backend hf memerlukan --model <id atau folder>")
        backend = HFBackend(args.model)
    res = run_all(backend, args.only.split(",") if args.only else None, args.quick, criteria)
    print_report(res)
    out = Path(args.out) if args.out else PROJECT_ROOT / "reports" / f"acceptance_{'cepat_' if args.quick else ''}{args.backend}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nLaporan: {out}")
    return 1 if (args.strict and not res["semua_lolos"]) else 0


if __name__ == "__main__":
    sys.exit(main())
