"""
Uji "diajari lalu ditanya": memori catatan pendek + BM25 + pembaca (model SFT).

Fakta buatan dengan nama fiktif (jawaban tidak mungkin berasal dari pengetahuan hafalan model).
Untuk tiap ukuran memori: soal terjawab (jawaban ada di memori) dan soal tak terjawab (orang/proyek
yang tidak pernah diajarkan; nama depan sengaja sama dengan orang di memori agar sulit).
Strategi: oracle (hanya catatan yang benar), top1, top3 (digabung berurutan).
"""
import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "apps"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import rag_lib as R
from config import get_config
from generate_gpt import _load_checkpoint_payload, _load_model, _set_seed, generate_text

FIRST = "Sari Budi Dewi Agus Rina Joko Wati Hendra Lestari Bayu Putri Rahmat Maya Dedi Ani Eko Tuti Fajar Indah Ketut Nur Yusuf Citra Andi Mega Surya Lina Rudi Ayu Taufik".split()
SYL = ["su", "ka", "ra", "ma", "ti", "ba", "lu", "ge", "no", "pa", "di", "ja", "wi", "sa", "ke", "mu", "la", "ro"]
JOBS = ["dokter gigi", "tukang kayu", "guru olahraga", "pilot", "penjahit", "petani kopi", "juru masak", "arsitek", "nelayan", "montir"]
FOODS = ["gudeg", "rawon", "pempek", "rendang", "sate padang", "soto betawi", "bakso", "gado-gado", "ayam betutu", "tahu campur"]
COLORS = ["merah", "biru", "hijau", "kuning", "hitam", "putih", "oranye", "ungu"]
ANIMALS = ["kucing", "anjing", "burung", "kelinci", "kura-kura"]
MONTHS = ["Januari", "Maret", "Mei", "Juli", "September", "November"]


def word(rng, n=3):
    return "".join(rng.choice(SYL) for _ in range(n))


def person(rng, used):
    while True:
        n = f"{rng.choice(FIRST)} {word(rng, 2).capitalize()}{word(rng, 1)}"
        if n not in used:
            used.add(n)
            return n


def make_facts(rng, count):
    """Kembalikan daftar (note, question, answer, key) minimal 'count' catatan."""
    facts, used, persons = [], set(), []
    while len(facts) < count:
        p = person(rng, used)
        persons.append(p)
        attrs = rng.sample(range(10), 3)
        city, year = word(rng, 3).capitalize(), rng.randint(1950, 2005)
        pet, floor = word(rng, 2).capitalize(), rng.randint(2, 30)
        proj, team = f"Proyek {word(rng, 2).capitalize()}-{rng.randint(2, 99)}", f"{word(rng, 2).capitalize()}{word(rng, 1)}"
        item, price = f"{word(rng, 3).capitalize()}", rng.randint(5, 95)
        table = {
            0: (f"{p} lahir di kota {city}.", f"Di kota mana {p} dilahirkan?", city),
            1: (f"{p} lahir pada tahun {year}.", f"Pada tahun berapa {p} dilahirkan?", str(year)),
            2: (f"{p} bekerja sebagai {rng.choice(JOBS)}.", None, None),
            3: (f"Makanan favorit {p} adalah {rng.choice(FOODS)}.", None, None),
            4: (f"{p} memelihara seekor {rng.choice(ANIMALS)} bernama {pet}.", f"Siapa nama hewan peliharaan {p}?", pet),
            5: (f"Ruang kerja {p} berada di lantai {floor}.", f"Di lantai berapa ruang kerja {p}?", str(floor)),
            6: (f"{p} mengendarai mobil berwarna {rng.choice(COLORS)}.", None, None),
            7: (f"Tenggat {proj} jatuh pada tanggal {rng.randint(1, 28)} {rng.choice(MONTHS)}.", f"Kapan batas waktu {proj}?", None),
            8: (f"Harga satu kotak {item} adalah {price} ribu rupiah.", f"Berapa harga satu kotak {item}?", str(price)),
            9: (f"{p} adalah ketua tim {team}.", f"Siapa ketua tim {team}?", p),
        }
        for a in attrs:
            note, q, ans = table[a]
            if a == 2:
                job = note.split("sebagai ")[1].rstrip(".")
                q, ans = f"Apa pekerjaan {p}?", job
            elif a == 3:
                food = note.split("adalah ")[1].rstrip(".")
                q, ans = f"Apa makanan kesukaan {p}?", food
            elif a == 6:
                col = note.split("berwarna ")[1].rstrip(".")
                q, ans = f"Apa warna mobil {p}?", col
            elif a == 7:
                ans = " ".join(note.split("tanggal ")[1].rstrip(".").split())
            facts.append((note, q, ans, p))
    return facts[:count], persons


def build_memory(notes):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE VIRTUAL TABLE passages USING fts5(title, text, src UNINDEXED, tokenize='unicode61 remove_diacritics 2')")
    con.executemany("INSERT INTO passages(title,text,src) VALUES ('',?,'mem')", [(n,) for n in notes])
    return con


def ask(model, tok, ctx, q, device):
    prompt = R.TEMPLATE_PROMPT.format(p=R.make_prompt(ctx, q))
    _set_seed(1)
    text = generate_text(model, tok, prompt, 30, 0.0, 0, 1.0, 1.0, device)
    k = text.find("Jawaban:", max(0, len(prompt) - 20))
    return (text[k + len("Jawaban:"):] if k >= 0 else text[len(prompt):]).strip()


def refusal(a):
    return (not a.strip()) or bool(R.REFUSE.search(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="reports/checkpoints_100m_sft_rag/gpt_sft_best.pt")
    ap.add_argument("--sizes", default="20,200,2000")
    ap.add_argument("--n-answerable", type=int, default=60)
    ap.add_argument("--n-unanswerable", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="reports/eval_memory.json")
    args = ap.parse_args()

    device = torch.device("cpu")
    cfg = get_config("production_100m")
    ckpt = Path(args.checkpoint)
    payload = _load_checkpoint_payload(ckpt)
    model, tok = _load_model(ckpt, cfg, device, payload=payload)
    del payload

    report = {}
    for size in [int(s) for s in args.sizes.split(",")]:
        rng = random.Random(args.seed + size)
        facts, persons = make_facts(rng, size)
        notes = [f[0] for f in facts]
        con = build_memory(notes)
        ans_q = rng.sample(facts, min(args.n_answerable, len(facts)))
        un_q = []
        for _ in range(args.n_unanswerable):
            known = rng.choice(persons)
            fake = f"{known.split()[0]} {word(rng, 2).capitalize()}{word(rng, 1)}"
            un_q.append(rng.choice([f"Di kota mana {fake} dilahirkan?", f"Apa pekerjaan {fake}?",
                                    f"Apa makanan kesukaan {fake}?", f"Di lantai berapa ruang kerja {fake}?"]))
        stats = {"oracle": 0, "top1": 0, "top3": 0}
        recall = {1: 0, 3: 0}
        rows = []
        for note, q, gold, _ in ans_q:
            hits = [h["text"] for h in R.retrieve(con, q, 3)]
            g = R.norm(gold)
            for k in recall:
                recall[k] += int(any(g in R.norm(h) or note == h for h in hits[:k]))
            outs = {"oracle": ask(model, tok, note, q, device),
                    "top1": ask(model, tok, hits[0], q, device) if hits else "",
                    "top3": ask(model, tok, " ".join(hits), q, device) if hits else ""}
            for s, a in outs.items():
                stats[s] += int(g in R.norm(a) and not refusal(a))
            rows.append({"q": q, "gold": gold, "note": note, "hit1": hits[:1] == [note], **outs})
        un_ref = {"top1": 0, "top3": 0}
        for q in un_q:
            hits = [h["text"] for h in R.retrieve(con, q, 3)]
            if not hits:
                for s in un_ref: un_ref[s] += 1
                continue
            un_ref["top1"] += int(refusal(ask(model, tok, hits[0], q, device)))
            un_ref["top3"] += int(refusal(ask(model, tok, " ".join(hits), q, device)))
        n, nu = len(ans_q), len(un_q)
        report[size] = {
            "notes": size, "answerable": n, "unanswerable": nu,
            "retrieval_recall@1": recall[1] / n, "retrieval_recall@3": recall[3] / n,
            "acc_oracle": stats["oracle"] / n, "acc_top1": stats["top1"] / n, "acc_top3": stats["top3"] / n,
            "refusal_unknown_top1": un_ref["top1"] / nu, "refusal_unknown_top3": un_ref["top3"] / nu,
        }
        print(json.dumps(report[size], indent=1), flush=True)
        report[size]["rows"] = rows
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
