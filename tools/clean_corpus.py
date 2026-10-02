#!/usr/bin/env python3
"""Clean and normalize corpus.txt for LLM training.

Usage:
    python tools/clean_corpus.py [--backup] [--dry-run]
"""

import argparse
import re
import shutil
from pathlib import Path


def clean_line(line: str) -> str:
    """Clean a single line of text."""
    # Remove Wikipedia-style references and markers
    line = re.sub(r'\[butuh rujukan\]', '', line)
    line = re.sub(r'\[diperlukan rujukan\]', '', line)
    line = re.sub(r'\[perlu diperjelas\]', '', line)
    line = re.sub(r'\[page needed\]', '', line)
    line = re.sub(r'\[citation needed\]', '', line)
    line = re.sub(r'\[sunting\]', '', line)
    line = re.sub(r'\[sumber?\]', '', line)
    
    # Remove ref tags and their content
    line = re.sub(r'<ref[^>]*>.*?</ref>', '', line)
    line = re.sub(r'<ref[^/]*/>', '', line)
    line = re.sub(r'<ref[^>]*>', '', line)
    
    # Remove HTML-like tags
    line = re.sub(r'<[^>]+>', '', line)
    
    # Remove footnote markers like [1], [2], etc.
    line = re.sub(r'\[\d+\]', '', line)
    
    # Remove Wikipedia arrows and special characters
    line = line.replace('→', '')
    line = line.replace('←', '')
    line = line.replace('–', ' - ')
    line = line.replace('—', ' - ')
    
    # Normalize whitespace
    line = re.sub(r'\s+', ' ', line)
    line = line.strip()
    
    return line


def has_relevant_content(line: str) -> bool:
    """Check if line has meaningful content (not just punctuation/numbers)."""
    # Remove all non-alphanumeric characters for check
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', line)
    # Must have at least some letters
    return len(cleaned) >= 5


def remove_duplicate_lines(lines: list[str]) -> list[str]:
    """Remove exact duplicate lines, keeping first occurrence."""
    seen = set()
    unique = []
    for line in lines:
        normalized = line.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(line)
    return unique


def clean_corpus(input_path: Path, output_path: Path, dry_run: bool = False) -> dict:
    """Main cleaning pipeline."""
    stats = {
        'original_lines': 0,
        'blank_removed': 0,
        'duplicate_removed': 0,
        'too_short_removed': 0,
        'irrelevant_removed': 0,
        'final_lines': 0,
    }
    
    # Read original
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    
    stats['original_lines'] = len(raw_lines)
    
    # Step 1: Remove blank lines
    lines = [l.rstrip('\n') for l in raw_lines if l.strip()]
    stats['blank_removed'] = stats['original_lines'] - len(lines)
    
    # Step 2: Clean each line
    lines = [clean_line(l) for l in lines]
    
    # Step 3: Remove too-short lines (< 10 chars after cleaning)
    before = len(lines)
    lines = [l for l in lines if len(l) >= 10]
    stats['too_short_removed'] = before - len(lines)
    
    # Step 4: Remove lines without relevant content
    before = len(lines)
    lines = [l for l in lines if has_relevant_content(l)]
    stats['irrelevant_removed'] = before - len(lines)
    
    # Step 5: Remove duplicates
    before = len(lines)
    lines = remove_duplicate_lines(lines)
    stats['duplicate_removed'] = before - len(lines)
    
    stats['final_lines'] = len(lines)
    
    # Write output
    if not dry_run:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Clean corpus.txt')
    parser.add_argument('--backup', action='store_true', help='Create backup before cleaning')
    parser.add_argument('--dry-run', action='store_true', help='Show stats without writing')
    parser.add_argument('--input', default='data/corpus.txt', help='Input file')
    parser.add_argument('--output', default='data/corpus_clean.txt', help='Output file')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return
    
    # Backup
    if args.backup and not args.dry_run:
        backup_path = input_path.with_suffix('.txt.bak')
        shutil.copy2(input_path, backup_path)
        print(f"Backup created: {backup_path}")
    
    # Clean
    stats = clean_corpus(input_path, output_path, args.dry_run)
    
    # Report
    print(f"\n=== Corpus Cleaning Report ===")
    print(f"Original lines:     {stats['original_lines']:,}")
    print(f"Blank removed:      {stats['blank_removed']:,}")
    print(f"Too short removed:  {stats['too_short_removed']:,}")
    print(f"Irrelevant removed: {stats['irrelevant_removed']:,}")
    print(f"Duplicates removed: {stats['duplicate_removed']:,}")
    print(f"Final lines:        {stats['final_lines']:,}")
    print(f"Reduction:          {stats['original_lines'] - stats['final_lines']:,} lines ({(1 - stats['final_lines']/max(stats['original_lines'],1))*100:.1f}%)")
    
    if not args.dry_run:
        print(f"\nOutput written to: {output_path}")
    
    return stats


if __name__ == '__main__':
    main()
