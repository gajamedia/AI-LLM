#!/usr/bin/env python3
"""Fast streaming corpus cleaner for large files."""

import re
import sys
from pathlib import Path


def clean_line(line: str) -> str | None:
    """Clean a single line, return None if line should be removed."""
    # Remove HTML tags
    line = re.sub(r'<[^>]+>', '', line)
    # Remove HTML entities
    line = re.sub(r'&\w+;', '', line)
    line = re.sub(r'&#\d+;', '', line)
    # Remove [[wiki links]] - keep inner text
    line = re.sub(r'\[\[([^\]|]+)(\|[^\]]+)?\]\]', r'\1', line)
    # Remove {{templates}}
    line = re.sub(r'\{\{[^}]*\}\}', '', line)
    # Remove <ref> tags
    line = re.sub(r'<ref[^>]*>.*?</ref>', '', line, flags=re.DOTALL)
    line = re.sub(r'<ref[^/]*/>', '', line)
    # Remove footnote markers [1], [2], etc.
    line = re.sub(r'\[\d+\]', '', line)
    # Remove URLs
    line = re.sub(r'https?://\S+', '', line)
    line = re.sub(r'www\.\S+', '', line)
    # Remove emails
    line = re.sub(r'[\w.-]+@[\w.-]+\.\w+', '', line)
    # Normalize dashes
    line = line.replace('–', ' - ').replace('—', ' - ')
    # Remove arrows
    line = line.replace('→', '').replace('←', '').replace('↑', '').replace('↓', '')
    # Normalize whitespace
    line = re.sub(r'\s+', ' ', line).strip()
    
    # Filter: too short
    if len(line) < 10:
        return None
    # Filter: only numbers/punctuation
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', line)
    if len(cleaned) < 5:
        return None
    
    return line


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/corpus.txt')
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('data/corpus_clean.txt')
    
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    
    total = 0
    kept = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            total += 1
            if total % 500000 == 0:
                print(f"  Processed {total:,} lines...", flush=True)
            
            result = clean_line(line.rstrip('\n'))
            if result is not None:
                fout.write(result + '\n')
                kept += 1
    
    removed = total - kept
    pct = (removed / max(total, 1)) * 100
    print(f"\nDone! {total:,} -> {kept:,} ({removed:,} removed, {pct:.1f}%)")


if __name__ == '__main__':
    main()
