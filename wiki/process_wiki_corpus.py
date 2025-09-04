#!/usr/bin/env python3
"""
Advanced Georgian Wikipedia corpus processor for LLM training.
Processes 99,996 articles from wiki/articles/ folder.
"""

import os
import json
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Set, Optional
import random
from collections import Counter

# Configuration
ARTICLES_DIR = Path("articles")
OUTPUT_DIR = Path("../corpus/processed")
MIN_ARTICLE_LENGTH = 100  # Minimum characters
MAX_ARTICLE_LENGTH = 50000  # Maximum characters to avoid outliers
MIN_GEORGIAN_RATIO = 0.7  # At least 70% Georgian characters

# Georgian Unicode ranges
GEORGIAN_RANGES = [
    (0x10A0, 0x10FF),  # Georgian (Mkhedruli and Asomtavruli)
    (0x2D00, 0x2D2F),  # Georgian Supplement
    (0x1C90, 0x1CBF),  # Georgian Extended
]

# Allowed punctuation and symbols
ALLOWED_CHARS = set('.,!?;:"\'«»—–…()[]{}- \n\r\t0123456789')

def is_georgian_char(char: str) -> bool:
    """Check if character is Georgian."""
    if not char:
        return False
    code_point = ord(char)
    return any(start <= code_point <= end for start, end in GEORGIAN_RANGES)

def calculate_georgian_ratio(text: str) -> float:
    """Calculate ratio of Georgian characters in text."""
    if not text:
        return 0.0
    
    georgian_count = sum(1 for c in text if is_georgian_char(c))
    total_chars = len([c for c in text if not c.isspace()])
    
    return georgian_count / total_chars if total_chars > 0 else 0.0

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Unicode normalization
    text = unicodedata.normalize('NFC', text)
    
    # Remove wiki markup patterns
    patterns = [
        r'\{\{[^}]*\}\}',  # Templates
        r'\[\[[^\]]*\]\]',  # Wiki links
        r'<[^>]*>',  # HTML tags
        r'=+\s*[^=]*\s*=+',  # Section headers
        r'^\s*\*+\s*',  # List items
        r'^\s*#+\s*',  # Numbered lists
        r'https?://[^\s]+',  # URLs
        r'\{\|[^}]*\|\}',  # Tables
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE)
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Filter characters to keep only Georgian and allowed chars
    filtered_chars = []
    for char in text:
        if is_georgian_char(char) or char in ALLOWED_CHARS:
            filtered_chars.append(char)
    
    text = ''.join(filtered_chars).strip()
    return text

def process_article(file_path: Path) -> Optional[Dict]:
    """Process a single article file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean the content
        cleaned = clean_text(content)
        
        # Quality checks
        if len(cleaned) < MIN_ARTICLE_LENGTH:
            return None
            
        if len(cleaned) > MAX_ARTICLE_LENGTH:
            cleaned = cleaned[:MAX_ARTICLE_LENGTH]
        
        georgian_ratio = calculate_georgian_ratio(cleaned)
        if georgian_ratio < MIN_GEORGIAN_RATIO:
            return None
        
        # Extract title from filename
        title = file_path.stem.replace('_', ' ')
        
        return {
            'title': title,
            'text': cleaned,
            'length': len(cleaned),
            'georgian_ratio': georgian_ratio,
            'word_count': len(cleaned.split()),
            'source_file': str(file_path)
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def deduplicate_by_content(articles: List[Dict]) -> List[Dict]:
    """Simple deduplication by exact content match."""
    seen_content = set()
    unique_articles = []
    
    for article in articles:
        content_hash = hash(article['text'])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_articles.append(article)
    
    return unique_articles

def create_corpus_splits(articles: List[Dict], train_ratio=0.9, val_ratio=0.05):
    """Split corpus into train/validation/test sets."""
    random.shuffle(articles)
    
    total = len(articles)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_articles = articles[:train_end]
    val_articles = articles[train_end:val_end]
    test_articles = articles[val_end:]
    
    return train_articles, val_articles, test_articles

def save_jsonl(articles: List[Dict], output_path: Path):
    """Save articles in JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for article in articles:
            json.dump(article, f, ensure_ascii=False)
            f.write('\n')

def save_text(articles: List[Dict], output_path: Path):
    """Save articles as plain text for tokenizer training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(article['text'])
            f.write('\n\n')

def get_corpus_stats(articles: List[Dict]) -> Dict:
    """Calculate corpus statistics."""
    total_chars = sum(a['length'] for a in articles)
    total_words = sum(a['word_count'] for a in articles)
    georgian_ratios = [a['georgian_ratio'] for a in articles]
    
    # Character frequency
    char_freq = Counter()
    for article in articles:
        char_freq.update(article['text'])
    
    return {
        'total_articles': len(articles),
        'total_characters': total_chars,
        'total_words': total_words,
        'avg_chars_per_article': total_chars / len(articles) if articles else 0,
        'avg_words_per_article': total_words / len(articles) if articles else 0,
        'avg_georgian_ratio': sum(georgian_ratios) / len(georgian_ratios) if georgian_ratios else 0,
        'size_mb': total_chars / (1024 * 1024),
        'top_characters': dict(char_freq.most_common(50))
    }

def main():
    """Main processing function."""
    print("🚀 Processing Georgian Wikipedia Corpus...")
    print(f"📁 Source: {ARTICLES_DIR}")
    print(f"📁 Output: {OUTPUT_DIR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all article files
    article_files = list(ARTICLES_DIR.glob("*.txt"))
    print(f"📄 Found {len(article_files)} article files")
    
    # Process articles
    processed_articles = []
    skipped = 0
    
    print("🔄 Processing articles...")
    for i, file_path in enumerate(article_files):
        if i % 1000 == 0:
            print(f"   Progress: {i}/{len(article_files)} ({i/len(article_files)*100:.1f}%)")
        
        article = process_article(file_path)
        if article:
            processed_articles.append(article)
        else:
            skipped += 1
    
    print(f"✅ Processed {len(processed_articles)} articles")
    print(f"⏭️  Skipped {skipped} articles (too short, low Georgian ratio, etc.)")
    
    # Deduplicate
    print("🔄 Deduplicating...")
    unique_articles = deduplicate_by_content(processed_articles)
    duplicates_removed = len(processed_articles) - len(unique_articles)
    print(f"🗑️  Removed {duplicates_removed} duplicate articles")
    
    # Calculate statistics
    stats = get_corpus_stats(unique_articles)
    print(f"\n📊 Corpus Statistics:")
    print(f"   Total articles: {stats['total_articles']:,}")
    print(f"   Total characters: {stats['total_characters']:,}")
    print(f"   Total words: {stats['total_words']:,}")
    print(f"   Size: {stats['size_mb']:.1f} MB")
    print(f"   Average Georgian ratio: {stats['avg_georgian_ratio']:.2%}")
    print(f"   Average chars/article: {stats['avg_chars_per_article']:.0f}")
    
    # Create splits
    print("🔄 Creating train/val/test splits...")
    train_articles, val_articles, test_articles = create_corpus_splits(unique_articles)
    
    print(f"   Train: {len(train_articles)} articles")
    print(f"   Validation: {len(val_articles)} articles")
    print(f"   Test: {len(test_articles)} articles")
    
    # Save in different formats
    print("💾 Saving corpus files...")
    
    # JSONL format (for analysis)
    save_jsonl(train_articles, OUTPUT_DIR / "train.jsonl")
    save_jsonl(val_articles, OUTPUT_DIR / "val.jsonl")
    save_jsonl(test_articles, OUTPUT_DIR / "test.jsonl")
    save_jsonl(unique_articles, OUTPUT_DIR / "full_corpus.jsonl")
    
    # Plain text format (for tokenizer training)
    save_text(train_articles, OUTPUT_DIR / "train.txt")
    save_text(val_articles, OUTPUT_DIR / "val.txt")
    save_text(test_articles, OUTPUT_DIR / "test.txt")
    save_text(unique_articles, OUTPUT_DIR / "full_corpus.txt")
    
    # Save statistics
    with open(OUTPUT_DIR / "stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Corpus processing complete!")
    print(f"📁 Output files saved to: {OUTPUT_DIR}")
    print(f"🎯 Ready for LLM training with {stats['size_mb']:.1f}MB of high-quality Georgian text!")

if __name__ == "__main__":
    main()
