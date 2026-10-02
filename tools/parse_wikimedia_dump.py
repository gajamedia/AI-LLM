#!/usr/bin/env python3
"""Stream a Wikimedia multistream XML dump into attributed sentence JSONL."""

import argparse
import bz2
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

try:
    from lxml import etree
except ImportError as exc:
    raise SystemExit("Install dump parser dependencies: python -m pip install -r tools/requirements-corpus.txt") from exc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "data" / "corpus_dump_20260901" / "vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))
try:
    import mwparserfromhell
except ImportError as exc:
    raise SystemExit("Install dump parser dependencies: python -m pip install -r tools/requirements-corpus.txt") from exc

WS_RE = re.compile(r"\s+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
RESIDUAL_MARKUP_RE = re.compile(
    r"(?:\{\{|\}\}|\[\[|\]\]|\{\||\|\}|"
    r"\b(?:jmpl|thumb|thumbnail|ka|kiri|kanan)\s*\||"
    r"\\(?:displaystyle|begin|frac|sum|int|vec|tfrac|mathrm|mathbf|cdots|left|right)\b|"
    r"\bGalat:\s*\{\{|^\s*\|[-+])",
    re.IGNORECASE,
)


def child_text(element, name: str) -> str:
    child = element.find(f"{{*}}{name}")
    return (child.text or "") if child is not None else ""


def clean_wikitext(wikitext: str) -> str:
    wikitext = COMMENT_RE.sub(" ", wikitext)
    code = mwparserfromhell.parse(wikitext)
    for tag in code.filter_tags(recursive=True):
        if str(tag.tag).strip().lower() in {"ref", "references", "gallery", "math", "syntaxhighlight", "source", "nowiki"}:
            code.remove(tag)
    file_namespaces = {"berkas", "file", "gambar", "image", "media"}
    for link in code.filter_wikilinks(recursive=True):
        namespace = str(link.title).strip().partition(":")[0].casefold()
        if namespace in file_namespaces:
            code.remove(link)
    plain = code.strip_code(normalize=True, collapse=True)
    return WS_RE.sub(" ", plain).strip()


def sentence_rows(text: str, min_chars: int):
    for paragraph in re.split(r"\n+", text):
        for sentence in SENTENCE_RE.split(paragraph):
            sentence = WS_RE.sub(" ", sentence).strip(" \t\r\n*#;:")
            if (len(sentence) >= min_chars
                    and sum(char.isalpha() for char in sentence) >= min_chars * 0.55
                    and not RESIDUAL_MARKUP_RE.search(sentence)):
                yield sentence


def progress_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".progress.json")


def write_progress(output_path: Path, pages_scanned: int, stats: Counter) -> None:
    """Persist parse progress after the output stream has been flushed."""
    payload = {"pages_scanned": int(pages_scanned), "stats": dict(stats)}
    target = progress_path(output_path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def load_resume_state(output_path: Path) -> tuple[int, Counter]:
    target = progress_path(output_path)
    if not target.exists() or not output_path.exists():
        return 0, Counter()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"[warn] Ignoring unreadable progress file: {target}", flush=True)
        return 0, Counter()
    pages = int(payload.get("pages_scanned", 0))
    if pages <= 0 or output_path.stat().st_size == 0:
        return 0, Counter()
    return pages, Counter(payload.get("stats", {}))


def clear_page(page) -> None:
    page.clear()
    while page.getprevious() is not None:
        del page.getparent()[0]


def parse_dump(dump_path: Path, output_path: Path, wiki: str, snapshot: str,
               min_page_chars: int = 300, min_sentence_chars: int = 40,
               max_page_chars: int = 2_000_000, max_pages: int | None = None,
               resume: bool = False) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_pages = 0
    stats = Counter()
    mode = "w"
    if resume:
        start_pages, stats = load_resume_state(output_path)
        if start_pages:
            mode = "a"
            print(
                f"Resuming: skipping {start_pages:,} already-scanned pages; "
                f"carried stats={dict(stats)}",
                flush=True,
            )
    stream_position = 0
    stopped_early = False
    with bz2.BZ2File(dump_path, "rb") as compressed, output_path.open(mode, encoding="utf-8", newline="\n") as out:
        context = etree.iterparse(compressed, events=("end",), tag="{*}page", huge_tree=True)
        for _, page in context:
            stream_position += 1
            if stream_position <= start_pages:
                clear_page(page)
                if max_pages and stream_position >= max_pages:
                    stopped_early = True
                    break
                continue

            stats["pages_scanned"] += 1
            namespace = child_text(page, "ns")
            if namespace == "0":
                title = child_text(page, "title").strip()
                page_id = child_text(page, "id").strip()
                revision = page.find("{*}revision")
                if not title or not page_id or revision is None:
                    stats["missing_page_metadata"] += 1
                else:
                    revision_id = child_text(revision, "id").strip()
                    timestamp = child_text(revision, "timestamp").strip()
                    wiki_text = revision.find("{*}text")
                    if wiki_text is None:
                        wiki_text = revision.find(".//{*}slot/{*}text")
                    raw = (wiki_text.text or "") if wiki_text is not None else ""
                    if raw.lstrip().upper().startswith("#REDIRECT"):
                        stats["redirects"] += 1
                    elif not raw or len(raw) > max_page_chars:
                        stats["empty_or_oversize"] += 1
                    else:
                        try:
                            plain = clean_wikitext(raw)
                        except Exception:
                            stats["parse_errors"] += 1
                            plain = ""
                        if len(plain) < min_page_chars:
                            stats["short_or_empty_after_cleaning"] += 1
                        else:
                            page_sentences = list(sentence_rows(plain, min_sentence_chars))
                            if len(page_sentences) < 2:
                                stats["too_few_sentences"] += 1
                            else:
                                url = f"https://{wiki.removesuffix('wiki')}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                                for sentence in page_sentences:
                                    record = {
                                        "source_id": f"{wiki}:{page_id}",
                                        "source_title": title,
                                        "source_url": url,
                                        "revision_id": revision_id,
                                        "revision_timestamp": timestamp,
                                        "dump_snapshot": snapshot,
                                        "text": sentence,
                                    }
                                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                                    stats["sentences_written"] += 1
                                stats["pages_written"] += 1
            clear_page(page)

            if max_pages and stream_position >= max_pages:
                stopped_early = True
                break
            if stats["pages_scanned"] % 50_000 == 0:
                out.flush()
                write_progress(output_path, stream_position, stats)
                print(f"Scanned {stats['pages_scanned']:,} pages; wrote {stats['sentences_written']:,} sentences", flush=True)
        out.flush()
        write_progress(output_path, stream_position, stats)

    result = {
        "wiki": wiki,
        "snapshot": snapshot,
        "dump_file": str(dump_path),
        "output_file": str(output_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stream_pages_scanned": stream_position,
        "stopped_early": stopped_early,
        "filters": {
            "namespace": 0,
            "minimum_page_characters": min_page_chars,
            "minimum_sentence_characters": min_sentence_chars,
            "maximum_wikitext_characters": max_page_chars,
            "require_at_least_sentences_per_page": 2,
            "redirects_excluded": True,
        },
        "stats": dict(stats),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not stopped_early:
        progress_path(output_path).unlink(missing_ok=True)
    else:
        print(f"[info] Stopped early at page {stream_position:,}; progress kept for --resume.", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/corpus_dump_20260901/corpus.jsonl"))
    parser.add_argument("--wiki", default="idwiki")
    parser.add_argument("--snapshot", default="20260901")
    parser.add_argument("--min-page-chars", type=int, default=300)
    parser.add_argument("--min-sentence-chars", type=int, default=40)
    parser.add_argument("--max-page-chars", type=int, default=2_000_000)
    parser.add_argument("--max-pages", type=int, default=None, help="Optional early stop after this many XML pages.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a previous run from its .progress.json sidecar instead of "
        "rewriting the output from scratch. --max-pages stays absolute.",
    )
    args = parser.parse_args()
    parse_dump(args.input, args.output, args.wiki, args.snapshot,
               args.min_page_chars, args.min_sentence_chars, args.max_page_chars, args.max_pages,
               resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
