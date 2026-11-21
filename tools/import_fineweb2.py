#!/usr/bin/env python3
"""Konversi shard Parquet FineWeb-2 menjadi JSONL korpus (streaming, tanpa reservoir).

Sampling deterministik berbasis hash URL (bukan baris awal) supaya tidak bias
ke satu blok dump/domain. Skema keluaran sama dengan tools/import_hf_parquet.py.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-parquet", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-2")
    p.add_argument("--revision", required=True)
    p.add_argument("--license", default="odc-by-1.0")
    p.add_argument("--keep-fraction", type=float, default=1.0,
                   help="Fraksi dokumen yang dipertahankan (hash URL), 0<x<=1")
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=100_000)
    p.add_argument("--min-language-score", type=float, default=0.9)
    p.add_argument("--max-docs", type=int, default=0, help="Batas dokumen keluaran (0=tak terbatas)")
    a = p.parse_args()
    if not 0 < a.keep_fraction <= 1:
        p.error("--keep-fraction harus di (0,1]")
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(a.input_parquet)
    names = pf.schema_arrow.names
    cols = [c for c in ("text", "url", "id", "dump", "language_score") if c in names]
    if "text" not in cols:
        raise SystemExit(f"kolom text tidak ada: {names}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    threshold = int(a.keep_fraction * 2**32)
    scanned = kept = chars = low_score = short = 0
    with a.output.open("w", encoding="utf-8", newline="\n") as out:
        for batch in pf.iter_batches(batch_size=2048, columns=cols):
            d = {c: batch.column(i).to_pylist() for i, c in enumerate(cols)}
            for i in range(batch.num_rows):
                scanned += 1
                url = str(d["url"][i]) if "url" in d else f"row{scanned}"
                h = hashlib.sha256(url.encode("utf-8")).digest()
                if int.from_bytes(h[:4], "big") >= threshold:
                    continue
                score = d["language_score"][i] if "language_score" in d else None
                if score is not None and score < a.min_language_score:
                    low_score += 1
                    continue
                text = d["text"][i]
                if not isinstance(text, str):
                    continue
                text = re.sub(r"[ \t\r\f\v]+", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
                if len(text) < a.min_chars or len(text) > a.max_chars:
                    short += 1
                    continue
                row = {
                    "text": text,
                    "source_id": f"hf:{a.dataset}:ind_Latn:{h.hex()[:16]}",
                    "source_title": url, "source_url": url,
                    "license": a.license, "language": "id",
                    "metadata": {"dataset": a.dataset, "revision": a.revision,
                                 "file": a.input_parquet.name,
                                 "dump": d["dump"][i] if "dump" in d else None,
                                 "language_score": score},
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
                chars += len(text)
                if a.max_docs and kept >= a.max_docs:
                    break
            if a.max_docs and kept >= a.max_docs:
                break
            if scanned % 200_000 < 2048:
                print(f"scanned={scanned:,} kept={kept:,} chars={chars:,}", flush=True)
    est = chars * 0.288
    print(json.dumps({"scanned": scanned, "kept": kept, "chars": chars, "low_score": low_score,
                      "too_short_or_long": short, "est_tokens_0.288": int(est)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
