"""
Ukur apakah model gpt_sft_best.pt bisa menjadi PEMILIH ALAT (router), seperti AI modern yang membiarkan model memutuskan
apakah pesan ditangani alat tertentu atau dicari di catatan. Dibandingkan dengan pemilih aturan (kode) yang dipakai aplikasi.

Set uji: evalsets/memory_router.jsonl (dibekukan; dibuat sekali oleh skrip ini bila belum ada).
Enam kelas: urut, hitung, tanggal, cari (tanya catatan), sapaan, tolak (permintaan tulis/terjemah yang tidak didukung).
Tiap kelas 12 pesan; 'para' = True untuk pesan yang sengaja memakai kata di luar daftar kata pemicu aturan (uji generalisasi).

Cara model dipakai (tiga cara, semuanya zero/one-shot karena konteks hanya 256 token):
  gen      : model menjawab bebas "Alat apa yang tepat ...?", jawabannya dipetakan ke kelas (kata kelas pertama yang muncul)
  rank     : kelas dengan peluang tertinggi (jumlah log-peluang token kelas setelah 'Jawaban:')
  rank+contoh : sama, dengan satu contoh per kelas di konteks
  python tools/memory_router_eval.py
"""
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import memory_agent as M
import memory_tools as T
import rag_lib as R

LABELS = ["urut", "hitung", "tanggal", "cari", "sapaan", "tolak"]
SET_PATH = PROJECT_ROOT / "evalsets" / "memory_router.jsonl"

# (pesan, para)
DATA = {
    "urut": [("urutkan 5, 3, 9, 1", False), ("sortir 8 2 6", False), ("mengurutkan deret 15, 4, 23", False), ("urutkan angka di catatan 2", False),
             ("Urutkan bilangan lima tiga sembilan satu", False), ("urutkan 7 7 1 4 dari yang terkecil", False),
             ("tolong susun angka 40 12 7 dari kecil ke besar", True), ("susun dari yang terbesar: 3 9 1 7", True),
             ("rapikan urutan 100 20 3 dari kecil", True), ("buat berurutan angka 33 11 22", True),
             ("angka 9 4 6 2 tolong dibuat menurun", True), ("mana yang paling kecil sampai paling besar: 12 5 8", True)],
    "hitung": [("berapa 12 ditambah 30", False), ("hitung 7 x 8", False), ("100 dibagi 4", False), ("2 pangkat 10", False),
               ("berapa dua juta dikali tiga", False), ("15 dikurangi 6 hasilnya?", False),
               ("tambahkan 25 dengan 17", True), ("kalau 9 dikali 9 jadi berapa", True), ("total dari 12 dan 18 berapa?", True),
               ("lima ratus ribu dikurangi dua puluh ribu sisanya berapa", True), ("kurangi 90 dengan 45", True), ("jumlah 4 5 dan 6 ada berapa", True)],
    "tanggal": [("hari apa 17 Agustus 1945", False), ("tanggal 25 Desember 2026 jatuh hari apa", False), ("1 Januari 2000 itu hari apa?", False),
                ("selisih 5 Mei 2025 dan 20 Mei 2025", False), ("berapa hari dari 1 Januari 2026 sampai 1 Maret 2026", False),
                ("jarak antara 10 Oktober 2024 dan 10 November 2024", False),
                ("tanggal 4 Juli 2026 itu Senin atau bukan, hari apa ya", True), ("sudah berapa lama sejak 3 Maret 2026 sampai 3 April 2026", True),
                ("17 Agustus 2030 jatuh pada hari apakah", True), ("rentang waktu 1 Mei 2026 hingga 31 Mei 2026 ada berapa hari", True),
                ("kapan hari Natal 2027 jatuhnya, 25 Desember 2027 hari apa", True), ("antara 2 Februari 2026 dan 2 Maret 2026 beda berapa hari", True)],
    "cari": [("Di mana Bu Rina tinggal?", False), ("Siapa ketua koperasi Maju?", False), ("Kapan rapat tim dimulai?", False), ("Apa pekerjaan Pak Joko?", False),
             ("Berapa jumlah anggota koperasi?", False), ("apa itu gravitasi", False),
             ("Bu Rina bekerja di mana", True), ("lambang negara indonesia apa?", True), ("siapa yang menyimpan kunci gudang", True),
             ("ruang rapat tim finance di mana", True), ("apa warna rumah Bu Rina", True), ("kota asal Pak Slamet apa", True)],
    "sapaan": [("halo", False), ("hai", False), ("selamat pagi", False), ("terima kasih banyak", False), ("makasih", False), ("apa kabar", False),
               ("assalamualaikum", False), ("hello", False), ("hei teman", False), ("selamat malam semua", False), ("thanks", False), ("selamat siang", False)],
    "tolak": [("tuliskan puisi tentang hujan", False), ("terjemahkan ke bahasa Inggris: selamat pagi", False), ("ringkaskan artikel ini", False),
              ("buatkan surat lamaran kerja", False), ("ceritakan dongeng kancil", False), ("rangkum rapat kemarin", False),
              ("tolong buat email permintaan maaf", True), ("buatkan slogan untuk toko roti", True), ("karang lagu tentang persahabatan", True),
              ("tulis cerita pendek tentang kucing", True), ("ubah kalimat ini ke bahasa Jawa: apa kabar", True), ("susun pidato singkat untuk wisuda", True)],
}

CONTEXT = ("Alat yang tersedia: urut (mengurutkan angka); hitung (operasi hitung); tanggal (hari dan selisih tanggal); "
           "cari (mencari jawaban di catatan); sapaan (menyapa); tolak (menulis, menerjemahkan, merangkum).")
EXAMPLES = ("Contoh: urutkan 3 1 2 = urut; berapa 4 ditambah 5 = hitung; hari apa 1 Mei 2026 = tanggal; Di mana Bu Rina tinggal = cari; "
            "selamat pagi = sapaan; tuliskan puisi = tolak.")


def build_set():
    if SET_PATH.exists():
        return [json.loads(l) for l in open(SET_PATH, encoding="utf-8") if l.strip()]
    rows = [{"message": m, "label": lab, "para": para} for lab, items in DATA.items() for m, para in items]
    SET_PATH.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return rows


def rule_router(msg, notes):
    """Pemilih aturan yang dipakai aplikasi, dipetakan ke enam kelas."""
    if T.GREETING.match(msg.strip()):
        return "sapaan"
    r = T.route(msg, None, notes)
    if r is None:
        return "cari"
    if r["kind"] == "task":
        return "tolak"
    return {"pengurut angka": "urut", "kalkulator": "hitung", "tanggal": "tanggal"}.get(r["tool"], "urut" if "angka" in (r["tool"] or "") else "hitung")


def question_for(msg):
    return f'Alat apa yang tepat untuk pesan: "{msg}"?'


def label_in(text):
    low = text.lower()
    hits = [(low.find(l), l) for l in LABELS if l in low]
    return min(hits)[1] if hits else None


def rank_label(reader, ctx, msg, mean=False):
    """Kelas dengan log-peluang tertinggi untuk teks yang ditambahkan setelah 'Jawaban:'. Batas token antara prompt dan kata kelas
    bisa bergeser (tokenizer menggabungkan ': ' berbeda), jadi dinilai mulai dari awalan token yang sama antara prompt dan prompt+kelas."""
    prompt = R.TEMPLATE_PROMPT.format(p=R.make_prompt(ctx, question_for(msg)))
    tok, model, dev = reader.tok, reader.model, reader.device
    base = tok.encode_ids(prompt, add_bos=True)
    best, best_score = None, -1e18
    for lab in LABELS:
        full = tok.encode_ids(prompt + " " + lab, add_bos=True)
        if len(full) > 256:
            continue
        k = 0
        while k < min(len(base), len(full)) and base[k] == full[k]:
            k += 1
        x = torch.tensor([full], dtype=torch.long, device=dev)
        with torch.inference_mode():
            logits = model(x, past_key_values=None, use_cache=False, position_offset=0)
        logp = torch.log_softmax(logits[0].float(), dim=-1)
        parts = [float(logp[i - 1, full[i]]) for i in range(max(k, 1), len(full))]
        score = (sum(parts) / len(parts)) if mean else sum(parts)
        if score > best_score:
            best, best_score = lab, score
    return best


def accuracy(preds, rows):
    n = len(rows)
    ok = sum(p == r["label"] for p, r in zip(preds, rows))
    para = [(p, r) for p, r in zip(preds, rows) if r["para"]]
    okp = sum(p == r["label"] for p, r in para)
    return ok, n, okp, len(para)


def main():
    rows = build_set()
    notes = [(2, "Skor ujian: 78, 95, 60, 88.")]
    reader = M.Reader(str(PROJECT_ROOT / "reports/checkpoints_100m_sft_mem2/gpt_sft_best.pt"))
    methods = {
        "aturan (kode, dipakai aplikasi)": [rule_router(r["message"], notes) for r in rows],
        "model: jawab bebas (gen)": [label_in(reader.read(CONTEXT, question_for(r["message"]), max_new=8) or "") for r in rows],
        "model: peringkat peluang (rank)": [rank_label(reader, CONTEXT, r["message"]) for r in rows],
        "model: rank, rata-rata per token": [rank_label(reader, CONTEXT, r["message"], mean=True) for r in rows],
        "model: rank + satu contoh/kelas": [rank_label(reader, CONTEXT + " " + EXAMPLES, r["message"]) for r in rows],
    }
    print(f"Set uji: {len(rows)} pesan, 6 kelas x 12; peluang acak = {100 / 6:.1f}%\n")
    print(f"{'metode':36s}{'semua':>14s}{'pesan parafrase':>20s}")
    report = {}
    for name, preds in methods.items():
        ok, n, okp, npara = accuracy(preds, rows)
        print(f"{name:36s}{ok:>6d}/{n} {100 * ok / n:5.1f}%{okp:>8d}/{npara} {100 * okp / npara:5.1f}%")
        per = {lab: sum(p == r["label"] for p, r in zip(preds, rows) if r["label"] == lab) for lab in LABELS}
        report[name] = {"benar": ok, "n": n, "benar_parafrase": okp, "n_parafrase": npara, "per_kelas_benar_dari_12": per,
                        "prediksi": preds}
    print("\nBenar per kelas (dari 12):")
    print(f"{'metode':36s}" + "".join(f"{l:>9s}" for l in LABELS))
    for name, rep in report.items():
        print(f"{name:36s}" + "".join(f"{rep['per_kelas_benar_dari_12'][l]:>9d}" for l in LABELS))
    # kesalahan paling berbahaya: pesan alat dikirim ke 'cari' atau sebaliknya, serta pesan yang diberi alat salah
    for name in list(methods)[1:]:
        preds = methods[name]
        dist = {}
        for p, r in zip(preds, rows):
            dist[p] = dist.get(p, 0) + 1
        print(f"\nSebaran prediksi {name}: {dict(sorted(dist.items(), key=lambda kv: -kv[1]))}")
    out = PROJECT_ROOT / "reports" / "eval_router.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nLaporan: {out}")


if __name__ == "__main__":
    main()
