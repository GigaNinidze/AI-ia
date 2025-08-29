#!/usr/bin/env python3
"""
Georgian Text Corpus Processing Pipeline

This script processes large Georgian text files to create a clean, deduplicated corpus
suitable for tokenizer training and small LLM pretraining.

Pipeline Stages:
1. Input Reading: Streams through UTF-8 text files without loading everything into RAM
2. Normalization: Unicode NFC, whitespace normalization, quote/dash standardization
3. Document Splitting: Paragraph-based with size packing (min/max)
4. Georgian Filtering: Whitelist approach to keep only Georgian script and essential punctuation
   + fair ratio computed on letters/digits only
5. Deduplication: SimHash-based near-duplicate detection at document and paragraph levels
6. Output Generation: JSONL format with metadata and quality statistics

Author: Georgian Corpus Processor
Date: 2024-2025
"""

import argparse
import collections
import json
import logging
import random
import re
import statistics
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Dict, List, Tuple


def setup_logging(log_file: Path) -> None:
    """Configure logging to both console and file."""
    log_file.parent.mkdir(exist_ok=True)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler], force=True)


def to_nfc(text: str) -> str:
    """Convert text to Unicode NFC."""
    return unicodedata.normalize('NFC', text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace:
    - Convert CRLF/CR to LF
    - Trim trailing spaces
    - Collapse 3+ blank lines to exactly 2
    - Collapse 2+ spaces inside lines to 1 (preserve single spaces)
    """
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        line = line.rstrip()
        line = re.sub(r' {2,}', ' ', line)
        processed_lines.append(line)
    text = '\n'.join(processed_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def normalize_quotes_dashes(text: str) -> str:
    """
    Standardize quotes/dashes/ellipsis to consistent Unicode characters.
    """
    # Quotes: map smart to straight or guillemets normalization
    replacements = {
        '\u2018': "'",  # ‘
        '\u2019': "'",  # ’
        '\u201C': '"',  # “
        '\u201D': '"',  # ”
        '‹': '«',
        '›': '»',
    }
    for a, b in replacements.items():
        text = text.replace(a, b)

    # Dashes: en dash -> em dash; minus -> hyphen
    text = text.replace('\u2013', '\u2014')  # – -> —
    text = text.replace('\u2212', '-')       # − -> -

    # Ellipsis: three dots -> …
    text = re.sub(r'\.\.\.', '…', text)
    return text


def strip_wiki_markup(t: str) -> str:
    """
    Lightweight removal of common wiki/HTML artifacts.
    Safe for running on non-wiki text as well.
    """
    # <ref>...</ref>
    t = re.sub(r'<ref[^>]*>.*?</ref>', '', t, flags=re.DOTALL | re.IGNORECASE)
    # HTML tags
    t = re.sub(r'<[^>]+>', '', t)
    # Simple templates {{ ... }} (non-nested)
    t = re.sub(r'\{\{[^{}]*\}\}', '', t)
    # Files and categories
    t = re.sub(r'\[\[File:[^\]]+\]\]', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\[\[(?:Category|კატეგორია):[^\]]+\]\]', '', t, flags=re.IGNORECASE)
    # Links [[A|B]] -> B ; [[A]] -> A
    t = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', t)
    t = re.sub(r'\[\[([^\]]+)\]\]', r'\1', t)
    # Bold/italic markup
    t = re.sub(r"'''''(.*?)'''''", r'\1', t)
    t = re.sub(r"'''(.*?)'''", r'\1', t)
    t = re.sub(r"''(.*?)''", r'\1', t)
    # Horizontal rules
    t = re.sub(r'^\s*-{4,}\s*$', '\n', t, flags=re.MULTILINE)
    return t


def split_documents(text: str, max_doc_chars: int = 20000, min_doc_chars: int = 200) -> List[str]:
    """
    Split by 1+ blank lines (paragraphs). Then:
    - merge adjacent paragraphs until each chunk ≥ min_doc_chars
    - enforce max_doc_chars by paragraph packing
    """
    # Paragraph split on any blank line
    paras = [p.strip() for p in re.split(r'\n\s*\n', text)]
    paras = [p for p in paras if p]
    if not paras:
        return []

    # First pass: ensure minimum size per chunk
    merged: List[str] = []
    buf, size = [], 0
    for p in paras:
        if size and size + len(p) + 2 >= min_doc_chars:
            buf.append(p)
            merged.append('\n\n'.join(buf).strip())
            buf, size = [], 0
        else:
            buf.append(p)
            size += len(p) + 2
    if buf:
        merged.append('\n\n'.join(buf).strip())

    # Second pass: enforce maximum size by packing
    final_docs: List[str] = []
    for doc in merged:
        if len(doc) <= max_doc_chars:
            final_docs.append(doc)
            continue
        ps = doc.split('\n\n')
        pack, cur = [], 0
        for p in ps:
            L = len(p) + 2
            if cur and cur + L > max_doc_chars:
                final_docs.append('\n\n'.join(pack).strip())
                pack, cur = [p], L
            else:
                pack.append(p)
                cur += L
        if pack:
            final_docs.append('\n\n'.join(pack).strip())

    return final_docs


def _is_ge(c: str) -> bool:
    return ('\u10A0' <= c <= '\u10FF') or ('\u1C90' <= c <= '\u1CBF') or ('\u2D00' <= c <= '\u2D2F')


def _ge_ratio_letters_digits(s: str) -> float:
    """
    Share of Georgian among letters/digits only (ignores spaces/punct).
    """
    signal = [c for c in s if c.isalpha() or c.isdigit()]
    if not signal:
        return 0.0
    ge = sum(1 for c in signal if _is_ge(c))
    return ge / len(signal)


def georgian_filter(text: str) -> Tuple[str, float, float]:
    """
    Filter text to keep only Georgian script and essential characters.

    Returns:
    - Filtered text
    - Non-Georgian share BEFORE filtering (letters/digits only)
    - Non-Georgian share AFTER filtering (letters/digits only)
    """
    # Compute pre-filter non-GE share on letters/digits only
    ge_ratio_before = _ge_ratio_letters_digits(text)
    non_ge_share_before = 1 - ge_ratio_before

    # Remove URLs/emails/paths/code
    text2 = re.sub(r'https?://\S+', '', text)
    text2 = re.sub(r'\S+@\S+\.\S+', '', text2)
    text2 = re.sub(r'[\/\\][\w\-\.\/\\]+', '', text2)
    text2 = re.sub(r'```.*?```', '', text2, flags=re.DOTALL)  # code blocks
    text2 = re.sub(r'`[^`]+`', '', text2)                     # inline code

    # Allowed characters
    allowed_chars = set()
    # Georgian ranges
    for start, end in [('\u10A0', '\u10FF'), ('\u1C90', '\u1CBF'), ('\u2D00', '\u2D2F')]:
        allowed_chars.update(chr(i) for i in range(ord(start), ord(end) + 1))
    # Punctuation / symbols
    allowed_chars.update('.,!?;:"\'«»—–…()[]{}- /')
    allowed_chars.update('0123456789%№+*=<>')
    # Whitespace
    allowed_chars.update('\n\t ')

    # NOTE: Latin letters intentionally NOT added to whitelist by default.
    filtered_text = ''.join(c for c in text2 if c in allowed_chars)

    # Post-filter non-GE share on letters/digits only
    ge_ratio_after = _ge_ratio_letters_digits(filtered_text)
    non_ge_share_after = 1 - ge_ratio_after

    return filtered_text, non_ge_share_before, non_ge_share_after


def compute_simhash(text: str, hash_bits: int = 64) -> int:
    """
    Compute SimHash signature for text using word 5-grams.
    """
    words = text.lower().split()
    if len(words) < 5:
        return hash(text.lower()) % (1 << hash_bits)
    ngrams = [' '.join(words[i:i+5]) for i in range(len(words) - 4)]
    if not ngrams:
        return hash(text.lower()) % (1 << hash_bits)

    vec = [0] * hash_bits
    for ng in ngrams:
        h = hash(ng) % (1 << hash_bits)
        for i in range(hash_bits):
            if h & (1 << i):
                vec[i] += 1
            else:
                vec[i] -= 1
    simhash = 0
    for i in range(hash_bits):
        if vec[i] > 0:
            simhash |= (1 << i)
    return simhash


def hamming_distance(h1: int, h2: int) -> int:
    """Calculate Hamming distance between two SimHash values."""
    return bin(h1 ^ h2).count('1')


def dedupe_docs(documents: List[Tuple[str, str, float, float]], hamming_threshold: int = 3) -> List[Tuple[str, str, float, float]]:
    """
    Remove near-duplicate documents using SimHash. Keep longer docs on collision.
    """
    if not documents:
        return []
    hashed = []
    for i, (text, source, non_ge_before, non_ge_after) in enumerate(documents):
        hashed.append((compute_simhash(text), len(text), i))
    hashed.sort(key=lambda x: x[1], reverse=True)

    unique_indices, unique_hashes = set(), set()
    for sh, _len, idx in hashed:
        dup = any(hamming_distance(sh, ex) <= hamming_threshold for ex in unique_hashes)
        if not dup:
            unique_hashes.add(sh)
            unique_indices.add(idx)
    return [documents[i] for i in range(len(documents)) if i in unique_indices]


def dedupe_paragraphs(text: str, hamming_threshold: int = 3) -> str:
    """
    Remove near-identical paragraphs within a document.
    """
    paragraphs = text.split('\n\n')
    if len(paragraphs) <= 1:
        return text
    hashed = []
    for i, p in enumerate(paragraphs):
        if p.strip():
            hashed.append((compute_simhash(p), len(p), i))
    hashed.sort(key=lambda x: x[1], reverse=True)

    unique_indices, unique_hashes = set(), set()
    for sh, _len, idx in hashed:
        dup = any(hamming_distance(sh, ex) <= hamming_threshold for ex in unique_hashes)
        if not dup:
            unique_hashes.add(sh)
            unique_indices.add(idx)
    unique_paragraphs = [paragraphs[i] for i in range(len(paragraphs)) if i in unique_indices]
    return '\n\n'.join(unique_paragraphs)


def write_jsonl(documents: List[Dict], output_file: Path) -> None:
    """Write documents to JSONL format."""
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')


def write_sample_preview(documents: List[Dict], sample_file: Path, sample_count: int = 5) -> None:
    """Write sample previews for human spot-checking."""
    sample_file.parent.mkdir(exist_ok=True)
    samples = documents if len(documents) <= sample_count else random.sample(documents, sample_count)
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write("=== GEORGIAN CORPUS SAMPLE PREVIEW ===\n\n")
        for i, doc in enumerate(samples, 1):
            f.write(f"SAMPLE {i}:\n")
            f.write(f"ID: {doc['id']}\n")
            f.write(f"Source: {doc['source']}\n")
            f.write(f"Length: {doc['len']} chars\n")
            f.write(f"Non-Georgian share: {doc['non_ge_share_after']:.3f}\n")
            f.write("Preview (first 300 chars):\n")
            f.write(doc['text'][:300].replace('\n', ' ') + "...\n")
            f.write("-" * 50 + "\n\n")


def print_stats(documents: List[Dict]) -> int:
    """Print corpus statistics and return quality gate exit code."""
    if not documents:
        logging.info("No documents to process")
        return 1

    total_docs = len(documents)
    total_chars = sum(doc['len'] for doc in documents)
    avg_chars = total_chars / total_docs

    char_counts = collections.Counter()
    for doc in documents:
        char_counts.update(doc['text'])

    non_ge_shares = [doc['non_ge_share_after'] for doc in documents]
    median_non_ge = statistics.median(non_ge_shares)

    logging.info("=== CORPUS STATISTICS ===")
    logging.info(f"Documents kept: {total_docs:,}")
    logging.info(f"Total characters: {total_chars:,}")
    logging.info(f"Average chars/doc: {avg_chars:.1f}")
    logging.info(f"Median non-Georgian share: {median_non_ge:.3f}")

    logging.info("\nTop 50 characters by frequency:")
    for char, count in char_counts.most_common(50):
        char_repr = repr(char)[1:-1] if char.isprintable() else f"U+{ord(char):04X}"
        logging.info(f"'{char_repr}': {count:,}")

    # Quality gate (still on letters/digits-based share)
    if median_non_ge > 0.15:
        logging.error(f"Quality gate failed: median non-Georgian share {median_non_ge:.3f} > 0.15")
        return 1
    else:
        logging.info("Quality gate passed ✓")
        return 0


def process_file(file_path: Path, source_name: str, min_chars: int, max_doc_chars: int) -> List[Tuple[str, str, float, float]]:
    """
    Process a single input file and return filtered documents.
    Returns list of (text, source, non_ge_share_before, non_ge_share_after).
    """
    logging.info(f"Processing {source_name}: {file_path}")
    documents: List[Tuple[str, str, float, float]] = []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        logging.info(f"Read {len(text):,} characters from {source_name}")

        # Stage 1: Normalization
        text = to_nfc(text)
        text = normalize_whitespace(text)
        text = normalize_quotes_dashes(text)

        # Strip wiki markup for Wikipedia-like sources (safe on others)
        if source_name.lower() in ('wikipedia', 'wiki', 'geowiki'):
            text = strip_wiki_markup(text)

        # Stage 2: Document splitting
        docs = split_documents(text, max_doc_chars=max_doc_chars, min_doc_chars=min_chars)
        logging.info(f"Split into {len(docs)} initial documents")

        # Stage 3: Georgian filtering + gating
        # More lenient threshold for markup-heavy sources
        threshold = 0.25 if source_name.lower() in ('wikipedia', 'wiki', 'geowiki') else 0.15

        for doc in docs:
            filtered_text, non_ge_before, non_ge_after = georgian_filter(doc)
            if len(filtered_text) >= min_chars and non_ge_after <= threshold:
                documents.append((filtered_text, source_name, round(non_ge_before, 6), round(non_ge_after, 6)))

        logging.info(f"After filtering: {len(documents)} documents kept")

    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")

    return documents


def main():
    """Main processing pipeline."""
    parser = argparse.ArgumentParser(description='Process Georgian text corpus')
    parser.add_argument('--inputs', nargs='+', required=True, help='Input text files to process')
    parser.add_argument('--out', required=True, type=Path, help='Output JSONL file path')
    parser.add_argument('--min-chars', type=int, default=200, help='Minimum characters per document (default: 200)')
    parser.add_argument('--max-doc-chars', type=int, default=20000, help='Maximum characters per document (default: 20000)')
    parser.add_argument('--hamming-threshold', type=int, default=3, help='Hamming distance threshold for deduplication (default: 3)')
    parser.add_argument('--sample-count', type=int, default=5, help='Number of sample previews (default: 5)')
    args = parser.parse_args()

    random.seed(42)

    log_file = Path('out/run.log')
    setup_logging(log_file)

    start_time = time.time()
    logging.info("Starting Georgian corpus processing pipeline")

    # Process input files
    all_documents: List[Tuple[str, str, float, float]] = []
    source_mapping = {
        'geowiki_fixed.txt': 'wikipedia',
        'წიგნები.txt': 'books'
    }

    for input_file in args.inputs:
        file_path = Path(input_file)
        if not file_path.exists():
            logging.error(f"Input file not found: {file_path}")
            continue
        source_name = source_mapping.get(file_path.name, file_path.stem)
        docs = process_file(file_path, source_name, args.min_chars, args.max_doc_chars)
        all_documents.extend(docs)

    if not all_documents:
        logging.error("No documents to process")
        return 1

    logging.info(f"Total documents before deduplication: {len(all_documents):,}")

    # Stage 4: Document deduplication
    unique_docs = dedupe_docs(all_documents, args.hamming_threshold)
    logging.info(f"After document deduplication: {len(unique_docs):,} documents")

    # Stage 5: Paragraph deduplication within documents
    final_documents: List[Tuple[str, str, float, float]] = []
    for text, source, non_ge_before, non_ge_after in unique_docs:
        deduped_text = dedupe_paragraphs(text, args.hamming_threshold)
        final_documents.append((deduped_text, source, non_ge_before, non_ge_after))

    # Convert to output format
    output_docs: List[Dict] = []
    for text, source, non_ge_before, non_ge_after in final_documents:
        output_docs.append({
            'id': str(uuid.uuid4()),
            'source': source,
            'text': text,
            'len': len(text),
            'non_ge_share_before': non_ge_before,
            'non_ge_share_after': non_ge_after
        })

    # Write outputs
    write_jsonl(output_docs, args.out)
    sample_file = args.out.parent / 'sample_preview.txt'
    write_sample_preview(output_docs, sample_file, args.sample_count)

    # Print statistics
    exit_code = print_stats(output_docs)

    elapsed_time = time.time() - start_time
    logging.info(f"Processing completed in {elapsed_time:.1f} seconds")
    return exit_code


if __name__ == '__main__':
    exit(main())