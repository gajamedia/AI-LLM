#!/usr/bin/env python3
"""Deep clean corpus for LLM training.

Removes:
- HTML tags and artifacts
- Wikipedia markup ([ref], [[wiki links]], footnotes, metadata)
- Broken Unicode and unusual characters
- PII (emails, phone numbers, URLs)
- Duplicate lines
- Lines with only numbers/punctuation
- Wiki template/boilerplate text
- Normalizes punctuation and whitespace
"""

import re
import unicodedata
import hashlib
from pathlib import Path


# === UNICODE NORMALIZATION ===

# Map special unicode chars to ASCII equivalents or remove
UNICODE_MAP = {
    '\u2018': "'",   # left single quote
    '\u2019': "'",   # right single quote
    '\u201C': '"',   # left double quote
    '\u201D': '"',   # right double quote
    '\u2013': '-',   # en dash
    '\u2014': ' - ', # em dash
    '\u2026': '...',  # ellipsis
    '\u2022': '',    # bullet
    '\u00AB': '"',   # left guillemet
    '\u00BB': '"',   # right guillemet
    '\u00B0': ' derajat',  # degree
    '\u00B1': '+/-',  # plus-minus
    '\u00D7': 'x',   # multiplication
    '\u00F7': '/',   # division
    '\u2190': '',    # left arrow
    '\u2191': '',    # up arrow
    '\u2192': '',    # right arrow
    '\u2193': '',    # down arrow
    '\u2020': '',    # dagger
    '\u2021': '',    # double dagger
    '\u00A7': '',    # section sign
    '\u00B6': '',    # pilcrow
    '\u00A9': '',    # copyright
    '\u00AE': '',    # registered
    '\u2122': 'TM',  # trademark
    '\u00B2': '2',   # superscript 2
    '\u00B3': '3',   # superscript 3
    '\u00BC': '1/4', # fraction 1/4
    '\u00BD': '1/2', # fraction 1/2
    '\u00BE': '3/4', # fraction 3/4
    '\u00E9': 'e',   # e-acute (common in names)
}


def normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC and replace special chars."""
    # Normalize to NFC form
    text = unicodedata.normalize('NFC', text)
    
    # Replace known special characters
    for old, new in UNICODE_MAP.items():
        text = text.replace(old, new)
    
    # Remove remaining unusual unicode (keep Latin, basic punctuation, digits)
    cleaned = []
    for ch in text:
        cp = ord(ch)
        # Keep: ASCII, Latin Extended, common punctuation, digits, space
        if (cp < 128 or                           # ASCII
            (0x00C0 <= cp <= 0x024F) or            # Latin Extended
            (0x0300 <= cp <= 0x036F) or            # Combining diacriticals
            (0x0400 <= cp <= 0x04FF) or            # Cyrillic
            ch in '.,;:!?()[]{}"\'-/\\@#$%&*+=<>~`^|'):
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append(ch)
        else:
            # Skip unknown unicode
            pass
    
    return ''.join(cleaned)


# === TEXT CLEANING ===

def remove_html(text: str) -> str:
    """Remove HTML tags and entities."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove HTML entities
    text = re.sub(r'&\w+;', '', text)
    text = re.sub(r'&#\d+;', '', text)
    return text


def remove_wiki_markup(text: str) -> str:
    """Remove Wikipedia-specific markup."""
    # Remove [[wiki links]] - keep inner text
    text = re.sub(r'\[\[([^\]|]+)(\|[^\]]+)?\]\]', r'\1', text)
    
    # Remove {{templates}} and {displaystyle}
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'\{displaystyle[^}]*\}', '', text)
    text = re.sub(r'\{[^}]{1,3}\}', '', text)
    
    # Remove <ref> tags and content
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*/>', '', text)
    text = re.sub(r'<ref[^>]*>', '', text)
    
    # Remove other XML-like tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove footnote markers [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove wiki arrows and symbols
    text = text.replace('→', '')
    text = text.replace('←', '')
    text = text.replace('↑', '')
    text = text.replace('↓', '')
    
    # Remove common wiki metadata text
    wiki_patterns = [
        r'(?i)\(diperlukan\s*rujukan\)',
        r'(?i)\(butuh\s*rujukan\)',
        r'(?i)\(perlu\s*diperjelas\)',
        r'(?i)\(sumber\s*perlu\s*ditambahkan\)',
        r'(?i)\(page\s*needed\)',
        r'(?i)\(citation\s*needed\)',
        r'(?i)referensi\s*(dan|atau)\s*pranala\s*luar',
        r'(?i)pranala\s*luar',
    ]
    for pat in wiki_patterns:
        text = re.sub(pat, '', text)
    
    return text


def remove_pii(text: str) -> str:
    """Remove/redact personally identifiable information."""
    # Remove email addresses
    text = re.sub(r'[\w.-]+@[\w.-]+\.\w+', '[EMAIL]', text)
    
    # Remove phone numbers (Indonesian patterns)
    text = re.sub(r'(?:\+62|62|0)\d{2,3}[-.\s]?\d{6,8}', '[TELEPON]', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove IP addresses
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '[IP]', text)
    
    return text


def clean_punctuation(text: str) -> str:
    """Normalize and clean punctuation."""
    # Fix double spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix double commas, periods, etc.
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\.\s*\.', '.', text)
    
    # Fix space before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # Fix space after opening bracket
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    
    # Remove lines that are just numbers or very short junk
    text = text.strip()
    
    return text


def remove_repeated_words(text: str) -> str:
    """Remove repeated words like 'adalah adalah', 'akar akar', etc."""
    # Match word repeated with space in between (case-insensitive, handle different cases)
    # e.g., "kota Kota" -> "kota", "di di" -> "di"
    text = re.sub(r'\b(\w+)(?:\s+\1)+\b', r'\1', text, flags=re.IGNORECASE)
    # Also handle "DI DI" -> "DI" (all caps)
    text = re.sub(r'\b([A-Z]{2,})(?:\s+\1)+\b', r'\1', text)
    return text


def remove_wiki_tableMarkup(text: str) -> str:
    """Remove Wikipedia table markup."""
    # Remove table syntax like |- style="..." |content]]
    text = re.sub(r'\|-[^|]*\|', '', text)
    # Remove remaining ]] artifacts
    text = text.replace(']]', '')
    # Remove remaining [[ artifacts  
    text = text.replace('[[', '')
    # Remove {| and |}
    text = re.sub(r'\{\|[^}]*\|\}', '', text)
    return text


def is_valid_line(line: str) -> bool:
    """Check if a line is valid (not junk)."""
    line = line.strip()
    
    # Empty line
    if not line:
        return False
    
    # Too short (less than 10 chars)
    if len(line) < 10:
        return False
    
    # Only numbers/punctuation
    if re.match(r'^[\d\s.,;:!?()\-/+=%*#@$&]+$', line):
        return False
    
    # Only special characters
    alpha_count = sum(1 for c in line if c.isalpha())
    if alpha_count < 5:
        return False
    
    # Wikipedia table markup
    if line.startswith('|-') or line.startswith('|}') or line.startswith('{|'):
        return False
    
    # Mathematical formulas (very specific pattern)
    if re.match(r'^[\d\s+\-*/^()=<>≤≥∞∑∫]+$', line):
        return False
    
    # Code fragments (incomplete or out of context)
    if re.match(r'^(?:SERVER_URL|API_KEY|PASSWORD|SECRET|TOKEN)\s*=', line):
        return False
    
    # Lines ending with incomplete syntax
    if re.search(r'[=\"]\s*$', line) and len(line) < 30:
        return False
    
    return True


def deduplicate_lines(lines: list[str]) -> list[str]:
    """Remove exact and near-duplicate lines."""
    seen_hashes = set()
    seen_normalized = set()
    unique = []
    
    for line in lines:
        # Exact duplicate check
        h = hashlib.md5(line.strip().encode()).hexdigest()
        if h in seen_hashes:
            continue
        
        # Normalized duplicate check (ignore case, spaces, punctuation)
        norm = re.sub(r'[^a-z0-9]', '', line.lower())
        if norm in seen_normalized:
            continue
        
        seen_hashes.add(h)
        seen_normalized.add(norm)
        unique.append(line)
    
    return unique


# === MAIN PIPELINE ===

def deep_clean(input_path: str, output_path: str, dry_run: bool = False) -> dict:
    """Main cleaning pipeline."""
    stats = {
        'original': 0,
        'empty_removed': 0,
        'invalid_removed': 0,
        'duplicate_removed': 0,
        'final': 0,
    }
    
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    
    stats['original'] = len(raw_lines)
    
    # Step 1: Basic cleanup on each line
    cleaned = []
    for line in raw_lines:
        line = line.rstrip('\n')
        
        # Apply all cleaning functions
        line = remove_html(line)
        line = remove_wiki_markup(line)
        line = remove_wiki_tableMarkup(line)
        line = remove_pii(line)
        line = normalize_unicode(line)
        line = clean_punctuation(line)
        line = remove_repeated_words(line)
        
        cleaned.append(line)
    
    # Step 2: Remove empty and invalid lines
    valid = []
    for line in cleaned:
        if not line.strip():
            stats['empty_removed'] += 1
            continue
        if not is_valid_line(line):
            stats['invalid_removed'] += 1
            continue
        valid.append(line.strip())
    
    # Step 3: Deduplicate
    unique = deduplicate_lines(valid)
    stats['duplicate_removed'] = len(valid) - len(unique)
    
    stats['final'] = len(unique)
    
    # Write output
    if not dry_run:
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in unique:
                f.write(line + '\n')
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Deep clean corpus')
    parser.add_argument('--input', default='data/corpus.txt')
    parser.add_argument('--output', default='data/corpus_final.txt')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--backup', action='store_true')
    args = parser.parse_args()
    
    import shutil
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return
    
    if args.backup:
        backup = input_path.with_suffix('.txt.bak2')
        shutil.copy2(input_path, backup)
        print(f"Backup: {backup}")
    
    stats = deep_clean(str(input_path), str(output_path), args.dry_run)
    
    print(f"\n{'='*50}")
    print(f"DEEP CLEAN REPORT")
    print(f"{'='*50}")
    print(f"Original lines:     {stats['original']:>8,}")
    print(f"Empty removed:      {stats['empty_removed']:>8,}")
    print(f"Invalid removed:    {stats['invalid_removed']:>8,}")
    print(f"Duplicates removed: {stats['duplicate_removed']:>8,}")
    print(f"{'='*50}")
    print(f"Final lines:        {stats['final']:>8,}")
    reduction = stats['original'] - stats['final']
    pct = (1 - stats['final']/max(stats['original'],1)) * 100
    print(f"Total reduction:    {reduction:>8,} ({pct:.1f}%)")
    print(f"{'='*50}")
    
    if not args.dry_run:
        print(f"\nOutput: {output_path}")


if __name__ == '__main__':
    main()
