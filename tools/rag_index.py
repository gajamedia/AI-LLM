"""
Bangun indeks pencarian (SQLite FTS5, ranking BM25) dari korpus Wikipedia jsonl.

Kalimat berurutan dari artikel yang sama digabung menjadi potongan (passage) >= --passage-chars
karakter. Tabel FTS5 'passages' berkolom (title, text) + src (tidak diindeks: 'wiki' / 'extra').
Opsional --extra: JSONL berisi {"context": ...} (mis. data/sft/sft_qa_eval.jsonl) yang ditambahkan
sebagai potongan agar evaluasi punya jawaban yang pasti ada di indeks.
"""
import argparse
import json
import sqlite3
import time
from pathlib import Path


def passages_from_wiki(path, passage_chars, max_chars, max_lines):
    cur_id, cur_title, buf, size = None, None, [], 0
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if max_lines and n > max_lines:
                break
            try:
                d = json.loads(line)
            except ValueError:
                continue
            text = (d.get("text") or "").strip()
            if not text:
                continue
            sid = d.get("source_id")
            if cur_id is not None and sid != cur_id and buf:
                yield cur_title, " ".join(buf)[:max_chars]
                buf, size = [], 0
            cur_id, cur_title = sid, d.get("source_title") or ""
            buf.append(text)
            size += len(text) + 1
            if size >= passage_chars:
                yield cur_title, " ".join(buf)[:max_chars]
                buf, size = [], 0
            if n % 1_000_000 == 0:
                print(f"  baca {n:,} baris", flush=True)
    if buf:
        yield cur_title, " ".join(buf)[:max_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", default="data/corpus_dump_20260901/wiki.jsonl")
    ap.add_argument("--db", default="data/rag/wiki_fts.db")
    ap.add_argument("--extra", default=None)
    ap.add_argument("--passage-chars", type=int, default=450)
    ap.add_argument("--max-chars", type=int, default=800)
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=120)
    args = ap.parse_args()

    db = Path(args.db)
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    con = sqlite3.connect(str(db))
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("CREATE VIRTUAL TABLE passages USING fts5(title, text, src UNINDEXED, tokenize='unicode61 remove_diacritics 2')")

    t0, total, batch = time.time(), 0, []
    for title, text in passages_from_wiki(args.wiki, args.passage_chars, args.max_chars, args.max_lines):
        if len(text) < args.min_chars:
            continue
        batch.append((title, text, "wiki"))
        if len(batch) >= 20000:
            con.executemany("INSERT INTO passages(title,text,src) VALUES (?,?,?)", batch)
            con.commit(); total += len(batch); batch = []
            print(f"  indeks {total:,} potongan, {time.time()-t0:.0f}s", flush=True)
    if batch:
        con.executemany("INSERT INTO passages(title,text,src) VALUES (?,?,?)", batch)
        total += len(batch)
    n_extra = 0
    if args.extra:
        rows = [json.loads(l) for l in open(args.extra, encoding="utf-8")]
        con.executemany("INSERT INTO passages(title,text,src) VALUES (?,?,?)",
                        [("", r["context"], "extra") for r in rows])
        n_extra = len(rows)
    con.commit()
    print(f"[rag] mengoptimalkan indeks ...", flush=True)
    con.execute("INSERT INTO passages(passages) VALUES('optimize')")
    con.commit()
    con.close()
    print(f"[rag] selesai: {total:,} potongan wiki + {n_extra} potongan tambahan -> {db} "
          f"({db.stat().st_size/1e9:.2f} GB) dalam {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
