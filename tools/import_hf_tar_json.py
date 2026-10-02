#!/usr/bin/env python3
"""Sample JSON documents from a pinned Hugging Face .tar.gz corpus."""

import argparse
import hashlib
import json
import random
import re
import tarfile
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


TEXT_KEYS = ("text", "content", "article_text", "article", "body", "news")
URL_KEYS = ("url", "link", "article_url", "source_url")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--text-key", help="Optional explicit article-text field")
    parser.add_argument("--license", required=True)
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_rows < 1:
        parser.error("--max-rows must be positive")

    raw = args.input_archive or args.output.with_suffix(".source.tgz")
    if args.input_archive:
        if not raw.is_file():
            raise FileNotFoundError(raw)
    else:
        raw.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{args.dataset}/resolve/{args.revision}/{args.file}?download=true"
        request = urllib.request.Request(url, headers={"User-Agent": "hai07-corpus-import/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, raw.open("wb") as target:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)

    reservoir = []
    counts = {"members_seen": 0, "json_members": 0, "malformed_json": 0,
              "missing_text": 0, "short_text": 0, "eligible": 0}
    rng = random.Random(args.seed)
    chosen_text_key = args.text_key
    with tarfile.open(raw, mode="r|gz") as archive:
        for member in archive:
            counts["members_seen"] += 1
            if not member.isfile() or not member.name.lower().endswith(".json"):
                continue
            fileobj = archive.extractfile(member)
            if fileobj is None:
                continue
            try:
                record = json.load(fileobj)
            except (json.JSONDecodeError, UnicodeDecodeError):
                counts["malformed_json"] += 1
                continue
            counts["json_members"] += 1
            if not isinstance(record, dict):
                counts["missing_text"] += 1
                continue
            if chosen_text_key is None:
                chosen_text_key = next((key for key in TEXT_KEYS if isinstance(record.get(key), str)), None)
                if chosen_text_key is None:
                    string_fields = [(key, value) for key, value in record.items()
                                     if isinstance(value, str) and len(value) > 40]
                    if string_fields:
                        chosen_text_key = max(string_fields, key=lambda pair: len(pair[1]))[0]
            value = record.get(chosen_text_key) if chosen_text_key else None
            if not isinstance(value, str):
                counts["missing_text"] += 1
                continue
            text = re.sub(r"\s+", " ", value).strip()
            if len(text) < 40:
                counts["short_text"] += 1
                continue
            counts["eligible"] += 1
            source_url = next((str(record[key]).strip() for key in URL_KEYS
                               if isinstance(record.get(key), str) and record[key].strip()), "")
            publisher = str(record.get("publisher") or record.get("source") or "")
            if not publisher and source_url:
                hostname = (urlsplit(source_url).hostname or "").lower()
                publisher = next((domain for domain in (
                    "detik.com", "kompas.com", "tempo.co", "cnnindonesia.com",
                    "sindonews.com", "republika.co.id", "poskotanews.com"
                ) if hostname == domain or hostname.endswith("." + domain)), hostname)
            source_key = source_url or member.name
            row = {
                "text": text,
                "source_id": f"hf:{args.dataset}:{hashlib.sha256(source_key.encode('utf-8')).hexdigest()[:20]}",
                "source_title": str(record.get("title") or member.name),
                "source_url": source_url or f"https://huggingface.co/datasets/{args.dataset}/blob/{args.revision}/{args.file}",
                "license": args.license,
                "language": "id",
                "metadata": {"dataset": args.dataset, "revision": args.revision,
                             "archive_file": args.file, "archive_member": member.name,
                             "text_key": chosen_text_key, "publisher": publisher,
                             "source": publisher, "domain": "news",
                             "original_record": {k: v for k, v in record.items() if k != chosen_text_key}},
            }
            item = (counts["eligible"], row)
            if len(reservoir) < args.max_rows:
                reservoir.append(item)
            else:
                replacement = rng.randrange(counts["eligible"])
                if replacement < args.max_rows:
                    reservoir[replacement] = item

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reservoir.sort(key=lambda pair: pair[0])
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for _, row in reservoir:
            output.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    manifest = {
        "dataset": args.dataset,
        "revision": args.revision,
        "file": args.file,
        "license": args.license,
        "text_key": chosen_text_key,
        **counts,
        "rows_written": len(reservoir),
        "sampling": "reservoir sample across all JSON members, sorted by eligible-record order",
        "seed": args.seed,
        "archive_path": str(raw),
        "archive_bytes": raw.stat().st_size,
        "archive_sha256": sha256(raw),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
