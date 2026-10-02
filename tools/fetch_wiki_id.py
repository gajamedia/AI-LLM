"""
Fetch Wikipedia Indonesia articles about specified topics
and append to corpus.txt

Usage:
    python tools/fetch_wiki_id.py
    python tools/fetch_wiki_id.py --topics "Indonesia,Sejarah Indonesia,Budaya Indonesia"
    python tools/fetch_wiki_id.py --limit 50
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WIKI_API = "https://id.wikipedia.org/w/api.php"
USER_AGENT = "HAI-LLM/1.0 (Indonesian corpus builder; sequential requests)"


def _write_records(path: Path, records, output_format: str, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        with open(path, "r", encoding="utf-8") as existing:
            first_line = next((line for line in existing if line.strip()), "")
        existing_is_jsonl = first_line.lstrip().startswith("{")
        if existing_is_jsonl != (output_format == "jsonl"):
            raise ValueError(
                f"Existing corpus format does not match --output-format: {path}. "
                "Choose a new output path or use the matching format."
            )
    mode = "a" if append and path.exists() else "w"
    with open(path, mode, encoding="utf-8") as f:
        if output_format == "jsonl":
            for title, text in records:
                json.dump({
                    "source_id": title.casefold(),
                    "source_title": title,
                    "source_url": "https://id.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
                    "text": text,
                }, f, ensure_ascii=False)
                f.write("\n")
        else:
            for _, text in records:
                f.write(text + "\n")


def _request_json(params: dict, retries: int = 3) -> dict:
    """Make sequential API requests and honor Wikimedia's Retry-After header."""
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(retries + 1):
        response = requests.get(WIKI_API, params=params, headers=headers, timeout=30)
        if response.status_code == 429 and attempt < retries:
            retry_after = response.headers.get("Retry-After")
            try:
                delay = max(1.0, float(retry_after)) if retry_after else min(120.0, 15.0 * (2 ** attempt))
            except ValueError:
                delay = min(120.0, 15.0 * (2 ** attempt))
            print(f"Wikipedia API rate limit (429); waiting {delay:g}s as requested.", flush=True)
            time.sleep(delay)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("Wikipedia API retry loop ended unexpectedly.")


def search_articles(query: str, limit: int = 20) -> list:
    """Search Wikipedia articles by query."""
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": min(limit, 50),
        "srprop": "snippet",
    }
    data = _request_json(params)
    return [r["title"] for r in data.get("query", {}).get("search", [])]


def get_article_extract(title: str, sentences: int = 5) -> str | None:
    """
    Get plain text extract of a Wikipedia article.
    Returns first `sentences` sentences.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "exsentences": sentences,
        "titles": title,
    }
    data = _request_json(params)
    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1":
            continue
        extract = page.get("extract", "").strip()
        if extract:
            return extract
    return None


def get_article_extracts(titles: list[str], sentences: int = 10) -> list[tuple[str, str]]:
    """Fetch lead extracts for up to 20 titles in one API request."""
    if not titles:
        return []
    if len(titles) > 20:
        raise ValueError("The TextExtracts API supports at most 20 extracts per batch.")
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "exsentences": min(10, max(1, int(sentences))),
        "exlimit": len(titles),
        "titles": "|".join(titles),
    }
    pages = _request_json(params).get("query", {}).get("pages", {})
    results = []
    for page in pages.values():
        title = str(page.get("title", "")).strip()
        extract = str(page.get("extract", "")).strip()
        if title and extract:
            results.append((title, extract))
    return results


def get_random_article_extracts(batch_size: int = 10, chars: int = 6000) -> list[tuple[str, str]]:
    """Fetch lead extracts for a batch of random main-namespace Indonesian pages."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": min(max(1, batch_size), 20),
        "prop": "extracts",
        "explaintext": True,
        "exchars": max(500, chars),
    }
    pages = _request_json(params).get("query", {}).get("pages", {})
    results = []
    for page in pages.values():
        title = str(page.get("title", "")).strip()
        extract = str(page.get("extract", "")).strip()
        if title and extract:
            results.append((title, extract))
    return results


def clean_text(text: str) -> str:
    """Clean Wikipedia text into simple sentences."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip lines that are likely references, notes, etc.
        if line.startswith("^") or line.startswith("["):
            continue
        # Split paragraphs into sentences while preserving Indonesian punctuation.
        sentences = re.split(r"(?<=[.!?])\s+", line)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue
            # Remove citations like [1], [2], etc.
            sent = re.sub(r"\[\d+\]", "", sent)
            sent = re.sub(r"\([^)]*\)", "", sent).strip()
            sent = sent.strip()
            if len(sent) > 30:
                cleaned.append(sent)
    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Wikipedia Indonesia articles for corpus"
    )
    parser.add_argument(
        "--topics",
        default="Indonesia,Sejarah Indonesia,Budaya Indonesia,Geografi Indonesia,"
                "Ekonomi Indonesia,Pendidikan di Indonesia,Bahasa Indonesia,"
                "Pariwisata Indonesia,Makanan Indonesia,Teknologi Indonesia",
        help="Comma-separated topics to search",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max articles per topic (default: 10)",
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=5,
        help="Sentences per article (maximum: 10; default: 5)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: data/corpus.txt)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=True,
        help="Append to existing corpus (default: True)",
    )
    parser.add_argument(
        "--output-format",
        choices=("text", "jsonl"),
        default="text",
        help="Use jsonl to retain article source IDs for grouped train/validation splits.",
    )
    parser.add_argument(
        "--target-total-lines",
        type=int,
        default=None,
        help="Fetch random Indonesian Wikipedia pages until the output reaches this many unique non-empty lines.",
    )
    parser.add_argument(
        "--random-batch-size",
        type=int,
        default=10,
        help="Random Wikipedia pages per API request when --target-total-lines is used (1-20).",
    )
    parser.add_argument(
        "--max-request-errors",
        type=int,
        default=10,
        help="Stop after this many consecutive failed/empty API batches; saved JSONL rows are retained.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=2.0,
        help="Minimum delay in seconds between sequential API requests (default: 2).",
    )
    args = parser.parse_args()

    default_name = "corpus.jsonl" if args.output_format == "jsonl" else "corpus.txt"
    output_path = Path(args.output or PROJECT_ROOT / "data" / default_name)
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]

    print("=" * 60)
    print("Wikipedia Indonesia Corpus Fetcher")
    print("=" * 60)
    print(f"Topics: {topics}")
    print(f"Articles per topic: {args.limit}")
    print(f"Sentences per article: {args.sentences}")
    print(f"Output: {output_path}")
    print()

    if args.target_total_lines is not None:
        if args.target_total_lines < 1:
            raise SystemExit("--target-total-lines must be >= 1.")

        existing_sentences = []
        seen_titles = set()
        if output_path.exists():
            for line in output_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                if args.output_format == "jsonl":
                    try:
                        row = json.loads(line)
                        existing_sentences.append(str(row.get("text", "")).strip())
                        title = str(row.get("source_title", "")).strip()
                        if title:
                            seen_titles.add(title.casefold())
                    except json.JSONDecodeError:
                        raise SystemExit(f"Existing corpus is not valid JSONL: {output_path}")
                else:
                    existing_sentences.append(line.strip())
        seen_sentences = {line.casefold() for line in existing_sentences}
        added_total = 0
        consecutive_errors = 0
        print(f"Existing unique lines: {len(seen_sentences)}")
        print(f"Target unique lines: {args.target_total_lines}")

        while len(seen_sentences) < args.target_total_lines:
            remaining = args.target_total_lines - len(seen_sentences)
            print(f"Fetching random pages; remaining lines: {remaining}")
            try:
                articles = get_random_article_extracts(
                    batch_size=args.random_batch_size,
                )
            except requests.RequestException as exc:
                consecutive_errors += 1
                print(f"Random fetch error {consecutive_errors}/{args.max_request_errors}: {exc}; retrying after delay.", flush=True)
                if consecutive_errors >= args.max_request_errors:
                    raise SystemExit("Stopped after repeated API errors; any downloaded rows remain saved.")
                time.sleep(3)
                continue

            if not articles:
                consecutive_errors += 1
                print(f"No random extracts returned ({consecutive_errors}/{args.max_request_errors}); retrying after delay.", flush=True)
                if consecutive_errors >= args.max_request_errors:
                    raise SystemExit("Stopped after repeated empty API batches; any downloaded rows remain saved.")
                time.sleep(2)
                continue
            consecutive_errors = 0

            batch_records = []
            for title, extract in articles:
                title_key = title.casefold()
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)
                for sentence in clean_text(extract):
                    key = sentence.casefold()
                    if key in seen_sentences:
                        continue
                    seen_sentences.add(key)
                    batch_records.append((title, sentence))
                    added_total += 1
                    if len(seen_sentences) >= args.target_total_lines:
                        break
                if len(seen_sentences) >= args.target_total_lines:
                    break
            # Commit each response batch so interruption/network loss keeps progress.
            if batch_records:
                _write_records(output_path, batch_records, args.output_format, append=True)
            time.sleep(max(0.0, args.request_delay))

        print(f"Corpus saved to: {output_path}")
        print(f"Total unique non-empty lines: {len(seen_sentences)}")
        print(f"Added {added_total} Indonesian Wikipedia sentences")
        return

    all_articles = []
    seen_titles = set()
    seen_sentences = set()
    if output_path.exists() and args.output_format == "jsonl":
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Existing corpus is not valid JSONL: {output_path}") from exc
            title = str(row.get("source_title", "")).strip()
            text = str(row.get("text", "")).strip()
            if title:
                seen_titles.add(title.casefold())
            if text:
                seen_sentences.add(text.casefold())

    for topic in topics:
        print(f"[{topic}] Searching...")
        try:
            titles = search_articles(topic, limit=args.limit)
            new_titles = [t for t in titles if t.casefold() not in seen_titles]
            for t in new_titles:
                seen_titles.add(t.casefold())
            all_articles.extend(new_titles)
            print(f"  Found {len(new_titles)} new articles")
            time.sleep(max(0.0, args.request_delay))
        except Exception as e:
            print(f"  Error searching '{topic}': {e}")

    # Remove duplicates while preserving order
    seen = set()
    unique_articles = []
    for a in all_articles:
        if a not in seen:
            seen.add(a)
            unique_articles.append(a)

    print(f"\nTotal unique articles to fetch: {len(unique_articles)}")
    print()

    # Fetch article extracts
    added_count = 0
    total_batches = (len(unique_articles) + 19) // 20
    for batch_index in range(total_batches):
        titles = unique_articles[batch_index * 20:(batch_index + 1) * 20]
        print(f"[{batch_index + 1}/{total_batches}] Fetching extracts for {len(titles)} articles")
        try:
            article_extracts = get_article_extracts(titles, sentences=args.sentences)
            batch_records = []
            for title, extract in article_extracts:
                cleaned = clean_text(extract)
                new_records = []
                for sentence in cleaned:
                    normalized = sentence.casefold()
                    if normalized in seen_sentences:
                        continue
                    seen_sentences.add(normalized)
                    new_records.append((title, sentence))
                if new_records:
                    batch_records.extend(new_records)
                added_count += len(new_records)
            if batch_records:
                _write_records(output_path, batch_records, args.output_format, append=True)
            print(f"  -> {len(article_extracts)} extracts, {len(batch_records)} unique sentences")
            time.sleep(max(0.0, args.request_delay))
        except Exception as e:
            print(f"  Error in article batch: {e}; stopping. Saved batches are retained.")
            break

    print(f"\n{'='*60}")
    print(f"Total new sentences: {added_count}")

    if added_count:
        with open(output_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
        print(f"Corpus saved to: {output_path}")
        print(f"Total lines in corpus: {total_lines}")
        print(f"Added {added_count} Indonesian Wikipedia sentences")
    else:
        print("No new sentences fetched!")

    print("Done!")


if __name__ == "__main__":
    main()
