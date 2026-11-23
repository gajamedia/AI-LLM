#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build Indonesian router datasets from MASSIVE Parquet and INTISARI Chat V4.

MASSIVE is loaded from the Hugging Face converted-Parquet branch. This avoids
executing the legacy massive.py dataset loading script and works with modern
versions of `datasets`.

Outputs:
  evalsets_online.jsonl                 {"text": ..., "label": ...}
  router_tulisan_anda_online.jsonl      {"message": ..., "label": ..., "family": ..., "salah_ketik": false}
  ulasan_label_online.md                manual-review sheet (No, Pesan, Label, Benar?)
  dataset_sources_manifest.csv          complete provenance and mapping audit
  dataset_summary.json                  counts, source status, and policy notes
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

MASSIVE_REPO = "AmazonScience/massive"
INTISARI_REPO = "INTISARI/intisari-indonesian-chat-v4"
MASSIVE_PARQUET_BASE = (
    "https://huggingface.co/datasets/AmazonScience/massive/resolve/"
    "refs%2Fconvert%2Fparquet/id-ID"
)

LABEL_FAMILY = {
    "urut": "ordering",
    "hitung": "calculation",
    "tanggal": "datetime",
    "cari": "information_retrieval",
    "sapaan": "social",
    "tolak": "out_of_scope",
}

# Deliberately explicit mapping. Any MASSIVE intent not listed here is sent
# to manual review; it is NEVER automatically assigned to `tolak`.
MASSIVE_MAP = {
    "qa_maths": "hitung",
    "datetime_query": "tanggal",
    "datetime_convert": "tanggal",
    "calendar_query": "tanggal",
    "calendar_set": "tanggal",
    "calendar_remove": "tanggal",
    "alarm_query": "tanggal",
    "alarm_set": "tanggal",
    "alarm_remove": "tanggal",
    "general_greet": "sapaan",
    "qa_factoid": "cari",
    "qa_definition": "cari",
    "qa_stock": "cari",
    "news_query": "cari",
    "weather_query": "cari",
    "transport_query": "cari",
    "recommendation_locations": "cari",
    "recommendation_events": "cari",
    "recommendation_movies": "cari",
    "email_query": "cari",
    "email_querycontact": "cari",
    "social_query": "cari",
    "music_query": "cari",
    "lists_query": "cari",
    "cooking_query": "cari",
    "takeaway_query": "cari",
}


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower()
    return re.sub(r"\s+", " ", value).strip()


def compact_record(message: str, label: str) -> dict[str, Any]:
    return {
        "message": message.strip(),
        "label": label,
        "family": LABEL_FAMILY[label],
        "salah_ketik": False,
    }


def load_massive():
    """Load the exact three Parquet splits validated by the user's test script."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Library datasets belum terpasang. Jalankan: "
            "python -m pip install -r requirements.txt"
        ) from exc

    print("Memuat MASSIVE id-ID dari branch Parquet Hugging Face...")
    data_files = {
        "train": f"{MASSIVE_PARQUET_BASE}/train/0000.parquet",
        "validation": f"{MASSIVE_PARQUET_BASE}/validation/0000.parquet",
        "test": f"{MASSIVE_PARQUET_BASE}/test/0000.parquet",
    }
    dataset = load_dataset("parquet", data_files=data_files)

    expected = {"train": 11514, "validation": 2033, "test": 2974}
    for split, expected_count in expected.items():
        if split not in dataset:
            raise RuntimeError(f"Split MASSIVE '{split}' tidak ditemukan.")
        actual = len(dataset[split])
        print(f"  {split}: {actual:,} baris")
        if actual != expected_count:
            print(
                f"  PERINGATAN: jumlah {split} saat ini {actual:,}; "
                f"referensi tes sebelumnya {expected_count:,}."
            )
    return dataset


def massive_intent_name(split_dataset, raw_intent: Any) -> str:
    """Convert ClassLabel integer IDs into their canonical intent names."""
    feature = split_dataset.features.get("intent")
    if feature is not None and hasattr(feature, "int2str"):
        try:
            return str(feature.int2str(int(raw_intent)))
        except (TypeError, ValueError, IndexError):
            pass
    # Fallback for datasets where the intent column is already textual.
    return str(raw_intent) if raw_intent is not None else ""


def get_massive_text(row: dict[str, Any]) -> str:
    for key in ("utt", "text", "sentence", "query"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def load_intisari():
    """Load INTISARI's public JSON/JSONL/Parquet files; failure is non-fatal."""
    from huggingface_hub import HfApi
    from datasets import load_dataset

    print("Mencari file INTISARI Chat V4...")
    files = HfApi().list_repo_files(repo_id=INTISARI_REPO, repo_type="dataset")
    candidates = [
        name for name in files
        if Path(name).suffix.lower() in {".parquet", ".json", ".jsonl"}
        and not Path(name).name.lower().startswith(("readme", "dataset_card"))
    ]
    if not candidates:
        raise RuntimeError("Tidak menemukan file JSON/JSONL/Parquet INTISARI.")

    parquet = [name for name in candidates if name.lower().endswith(".parquet")]
    chosen = sorted(parquet if parquet else candidates)
    suffixes = {Path(name).suffix.lower() for name in chosen}
    if suffixes == {".parquet"}:
        loader = "parquet"
    elif suffixes <= {".json", ".jsonl"}:
        loader = "json"
    else:
        raise RuntimeError(f"Campuran format file INTISARI tidak didukung: {suffixes}")

    def resolve(name: str) -> str:
        safe = "/".join(quote(part, safe="") for part in name.split("/"))
        return f"https://huggingface.co/datasets/{INTISARI_REPO}/resolve/main/{safe}"

    urls = [resolve(name) for name in chosen]
    print(f"  File INTISARI: {len(urls)} ({loader})")
    return load_dataset(loader, data_files={"train": urls})


def extract_intisari_user_turns(row: dict[str, Any]):
    messages = row.get("messages") or []
    if isinstance(messages, list):
        for turn_index, message in enumerate(messages):
            if isinstance(message, dict) and str(message.get("role", "")).lower() == "user":
                content = str(message.get("content") or "").strip()
                if content:
                    yield turn_index, content


def label_intisari_conservative(message: str) -> str | None:
    """Only short, unambiguous social formulas are auto-labeled as sapaan."""
    text = norm(message)
    patterns = [
        r"^(halo|hai|hi|hello|pagi|selamat pagi|siang|selamat siang|sore|selamat sore|malam|selamat malam|assalamualaikum)[!.? ]*$",
        r"^(apa kabar|gimana kabar(ku)?|kabarku baik|baik, terima kasih)[!.? ]*$",
        r"^(makasih|terima kasih|thanks|thank you|tengkyu)[!.? ]*$",
        r"^(dadah|sampai jumpa|sampai ketemu|bye|goodbye)[!.? ]*$",
    ]
    return "sapaan" if any(re.fullmatch(pattern, text) for pattern in patterns) else None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_review(path: Path, rows: list[dict[str, Any]], max_rows: int) -> None:
    shown = rows[:max_rows]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Ulasan Label Dataset Online\n\n")
        handle.write(
            "Tentukan label router yang tepat untuk setiap pesan. "
            "Isi kolom **Label** dengan salah satu: `urut`, `hitung`, `tanggal`, "
            "`cari`, `sapaan`, atau `tolak`. Isi **Benar?** dengan `ya`, `tidak`, "
            "atau `ragu`. Baris yang berasal dari split `test` hanya untuk audit/evaluasi; "
            "jangan masukkan ke data training router.\n\n"
        )
        handle.write(f"Total kandidat: **{len(rows):,}**. Ditampilkan: **{len(shown):,}**.\n\n")
        handle.write("| No | Pesan | Label | Benar? |\n")
        handle.write("|---:|---|---|---|\n")
        for number, row in enumerate(shown, 1):
            message = str(row.get("message", "")).replace("|", "\\|")
            message = re.sub(r"\s+", " ", message).replace("<", "&lt;").replace(">", "&gt;")
            candidate = row.get("candidate_label") or ""
            # For unmapped rows label remains blank for the reviewer to decide.
            handle.write(f"| {number} | {message} | {candidate} | |\n")
        if len(rows) > len(shown):
            handle.write(
                "\nCatatan: daftar lengkap seluruh kandidat tersedia pada "
                "`dataset_sources_manifest.csv`; file ini dibatasi oleh opsi "
                "`--max-review-rows`.\n"
            )


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "message", "source", "split", "source_id", "source_label",
        "candidate_label", "mapping", "review_reason",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build online Indonesian router datasets.")
    parser.add_argument("--output", default="generated_online", help="Folder output")
    parser.add_argument(
        "--max-review-rows", type=int, default=10000,
        help="Maksimum baris pada Markdown review (default: 10000; manifest tetap lengkap)",
    )
    parser.add_argument(
        "--skip-intisari", action="store_true",
        help="Lewati INTISARI dan hanya gunakan MASSIVE",
    )
    args = parser.parse_args()
    if args.max_review_rows < 0:
        parser.error("--max-review-rows tidak boleh negatif")

    try:
        import datasets
        import huggingface_hub
    except ImportError:
        print("Dependensi belum lengkap. Jalankan: python -m pip install -r requirements.txt", file=sys.stderr)
        return 2

    print(f"huggingface_hub={huggingface_hub.__version__} | datasets={datasets.__version__}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    massive = load_massive()
    eval_rows: list[dict[str, str]] = []
    router_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    seen_eval: set[str] = set()
    seen_router: set[str] = set()
    intent_stats: dict[str, Counter] = defaultdict(Counter)

    for split_name in ("train", "validation", "test"):
        if split_name not in massive:
            continue
        split_data = massive[split_name]
        for row_index, row in enumerate(split_data):
            message = get_massive_text(row)
            if not message:
                continue
            source_label = massive_intent_name(split_data, row.get("intent"))
            candidate = MASSIVE_MAP.get(source_label)
            key = norm(message)
            entry = {
                "message": message,
                "source": MASSIVE_REPO,
                "split": split_name,
                "source_id": str(row.get("id", row_index)),
                "source_label": source_label,
                "candidate_label": candidate or "",
                "mapping": "explicit_intent_map" if candidate else "unmapped_manual_review",
                "review_reason": "intent belum dipetakan secara eksplisit" if not candidate else "",
            }
            manifest.append(entry)
            intent_stats[split_name][source_label] += 1

            if candidate and split_name == "test":
                if key not in seen_eval:
                    eval_rows.append({"text": message, "label": candidate})
                    seen_eval.add(key)
            elif candidate and split_name in ("train", "validation"):
                if key not in seen_router:
                    router_rows.append(compact_record(message, candidate))
                    seen_router.add(key)
            elif not candidate:
                # Unmapped rows from every split go to review only. Test rows
                # are explicitly excluded from router training by construction.
                review_rows.append(entry)

    intisari_status = {"loaded": False, "records": 0, "error": None}
    if not args.skip_intisari:
        try:
            intisari = load_intisari()
            intisari_split = "train" if "train" in intisari else next(iter(intisari.keys()))
            intisari_status["loaded"] = True
            intisari_status["records"] = len(intisari[intisari_split])
            intisari_seen: set[str] = set()
            for row_index, row in enumerate(intisari[intisari_split]):
                for turn_index, message in extract_intisari_user_turns(row):
                    key = norm(message)
                    if not key or key in intisari_seen:
                        continue
                    intisari_seen.add(key)
                    candidate = label_intisari_conservative(message)
                    entry = {
                        "message": message,
                        "source": INTISARI_REPO,
                        "split": intisari_split,
                        "source_id": f"{row_index}:turn{turn_index}",
                        "source_label": "conversation_user_turn",
                        "candidate_label": candidate or "",
                        "mapping": "conservative_social_pattern" if candidate else "unmapped_manual_review",
                        "review_reason": "bukan formula sosial yang jelas" if not candidate else "",
                    }
                    manifest.append(entry)
                    if candidate:
                        if key not in seen_router:
                            router_rows.append(compact_record(message, candidate))
                            seen_router.add(key)
                    else:
                        review_rows.append(entry)
        except Exception as exc:  # optional source: MASSIVE outputs remain usable
            intisari_status["error"] = f"{type(exc).__name__}: {exc}"
            print(f"PERINGATAN: INTISARI tidak berhasil dimuat: {intisari_status['error']}")
            print("Builder tetap melanjutkan dengan data MASSIVE.")
    else:
        intisari_status["error"] = "dilewati melalui --skip-intisari"

    write_jsonl(output / "evalsets_online.jsonl", eval_rows)
    write_jsonl(output / "router_tulisan_anda_online.jsonl", router_rows)
    write_review(output / "ulasan_label_online.md", review_rows, args.max_review_rows)
    write_manifest(output / "dataset_sources_manifest.csv", manifest)

    router_counts = Counter(row["label"] for row in router_rows)
    eval_counts = Counter(row["label"] for row in eval_rows)
    review_by_source = Counter(row["source"] for row in review_rows)
    review_by_split = Counter(f"{row['source']}:{row['split']}" for row in review_rows)
    summary = {
        "builder_version": "5.0.0",
        "libraries": {"datasets": datasets.__version__, "huggingface_hub": huggingface_hub.__version__},
        "sources": [
            {
                "name": MASSIVE_REPO,
                "format": "Parquet via refs/convert/parquet",
                "language_config": "id-ID",
                "license": "CC BY 4.0",
                "url": "https://huggingface.co/datasets/AmazonScience/massive",
                "split_counts": {name: len(massive[name]) for name in ("train", "validation", "test") if name in massive},
            },
            {
                "name": INTISARI_REPO,
                "format": "Parquet/JSON/JSONL discovered from Hub",
                "license": "Periksa dataset card dan ketentuan penggunaan sebelum redistribusi",
                "url": "https://huggingface.co/datasets/INTISARI/intisari-indonesian-chat-v4",
                **intisari_status,
            },
        ],
        "router_records": len(router_rows),
        "router_label_counts": dict(router_counts),
        "eval_records": len(eval_rows),
        "eval_label_counts": dict(eval_counts),
        "manual_review_records": len(review_rows),
        "manual_review_by_source": dict(review_by_source),
        "manual_review_by_source_split": dict(review_by_split),
        "massive_intent_counts_by_split": {split: dict(counts) for split, counts in intent_stats.items()},
        "policies": [
            "MASSIVE intent integer IDs are converted with the Parquet ClassLabel.int2str metadata.",
            "Only explicit MASSIVE_MAP entries are auto-labeled.",
            "All unmapped MASSIVE intents go to manual review; none are automatically labeled tolak.",
            "MASSIVE train+validation mapped rows may enter router_tulisan_anda_online.jsonl.",
            "MASSIVE test mapped rows enter evalsets_online.jsonl only.",
            "Unmapped MASSIVE test rows appear in review/manifest only and are never inserted into router training output.",
            "INTISARI only auto-labels short, unambiguous social formulas as sapaan; all other user turns go to manual review.",
            "No automatic mapping is made to urut or tolak.",
            "Review Markdown follows the four-column template: No, Pesan, Label, Benar?.",
            "Keep evalsets separate from router training data and verify source license/attribution before redistribution.",
        ],
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== BUILD SELESAI ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutput: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
