#!/usr/bin/env python3
"""
Advanced Georgian text cleaner to achieve 100% Georgian character ratio.
Removes all non-Georgian content while preserving numbers and essential punctuation.
"""

import os
import re
import unicodedata
from pathlib import Path
from collections import Counter
import json

# Configuration
ARTICLES_DIR = Path("articles")
OUTPUT_DIR = Path("../corpus/clean_georgian")
MIN_ARTICLE_LENGTH = 200  # Minimum characters after cleaning
MIN_PARAGRAPH_LENGTH = 50  # Minimum paragraph length

# Georgian Unicode ranges (comprehensive)
GEORGIAN_RANGES = [
    (0x10A0, 0x10FF),  # Georgian (Mkhedruli and Asomtavruli)
    (0x2D00, 0x2D2F),  # Georgian Supplement  
    (0x1C90, 0x1CBF),  # Georgian Extended (Mtavruli)
]

# Allowed non-Georgian characters
ALLOWED_PUNCTUATION = set('.,!?;:"\'«»—–…()[]{}-')
ALLOWED_NUMBERS = set('0123456789')
ALLOWED_WHITESPACE = set(' \n\r\t')
ALLOWED_SYMBOLS = set('№§©®™°±×÷=€$£¥₾%')

# All allowed characters
ALLOWED_CHARS = ALLOWED_PUNCTUATION | ALLOWED_NUMBERS | ALLOWED_WHITESPACE | ALLOWED_SYMBOLS

def is_georgian_char(char: str) -> bool:
    """Check if character is in Georgian Unicode ranges."""
    if not char:
        return False
    code_point = ord(char)
    return any(start <= code_point <= end for start, end in GEORGIAN_RANGES)

def is_allowed_char(char: str) -> bool:
    """Check if character is allowed (Georgian + numbers + punctuation)."""
    return is_georgian_char(char) or char in ALLOWED_CHARS

def remove_wiki_markup(text: str) -> str:
    """Remove all Wikipedia markup patterns."""
    if not text:
        return ""
    
    # Remove templates and infoboxes
    text = re.sub(r'\{\{[^}]*\}\}', '', text, flags=re.DOTALL)
    
    # Remove wiki links but keep the text
    text = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', text)  # [[link|text]] -> text
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # [[text]] -> text
    
    # Remove external links
    text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]*)\]', r'\1', text)
    text = re.sub(r'https?://[^\s]+', '', text)
    
    # Remove references
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/?>', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove section headers
    text = re.sub(r'^=+\s*[^=]*\s*=+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove list markers
    text = re.sub(r'^\s*[\*#:;]+\s*', '', text, flags=re.MULTILINE)
    
    # Remove table markup
    text = re.sub(r'\{\|[^}]*\|\}', '', text, flags=re.DOTALL)
    text = re.sub(r'^\s*\|.*$', '', text, flags=re.MULTILINE)
    
    # Remove categories
    text = re.sub(r'\[\[კატეგორია:[^\]]*\]\]', '', text)
    text = re.sub(r'\[\[Category:[^\]]*\]\]', '', text)
    
    # Remove redirects
    text = re.sub(r'#REDIRECT.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'#გადამისამართება.*$', '', text, flags=re.MULTILINE)
    
    # Remove file references
    text = re.sub(r'\[\[(ფაილი|სურათი|File|Image):[^\]]*\]\]', '', text)
    
    # Remove magic words
    text = re.sub(r'__[A-Z_]+__', '', text)
    
    return text

def filter_georgian_only(text: str) -> str:
    """Filter text to keep only Georgian characters and allowed symbols."""
    if not text:
        return ""
    
    # Filter character by character
    filtered_chars = []
    for char in text:
        if is_allowed_char(char):
            filtered_chars.append(char)
    
    return ''.join(filtered_chars)

def normalize_text(text: str) -> str:
    """Normalize Georgian text."""
    if not text:
        return ""
    
    # Unicode normalization (NFC)
    text = unicodedata.normalize('NFC', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')  # Smart quotes to straight
    text = text.replace(''', "'").replace(''', "'")  # Smart apostrophes
    
    # Normalize dashes
    text = text.replace('–', '—').replace('−', '-')
    
    return text.strip()

def clean_article(text: str) -> str:
    """Complete cleaning pipeline for a single article."""
    if not text:
        return ""
    
    # Step 1: Remove wiki markup
    text = remove_wiki_markup(text)
    
    # Step 2: Filter to Georgian + allowed chars only
    text = filter_georgian_only(text)
    
    # Step 3: Normalize
    text = normalize_text(text)
    
    # Step 4: Split into paragraphs and filter short ones
    paragraphs = []
    for para in text.split('\n\n'):
        para = para.strip()
        if len(para) >= MIN_PARAGRAPH_LENGTH:
            paragraphs.append(para)
    
    # Rejoin paragraphs
    cleaned_text = '\n\n'.join(paragraphs)
    
    return cleaned_text

def calculate_georgian_ratio(text: str) -> float:
    """Calculate the ratio of Georgian characters (excluding whitespace and punctuation)."""
    if not text:
        return 0.0
    
    # Count only letters (excluding punctuation, numbers, whitespace)
    letters_only = [c for c in text if c.isalpha()]
    if not letters_only:
        return 0.0
    
    georgian_letters = [c for c in letters_only if is_georgian_char(c)]
    return len(georgian_letters) / len(letters_only)

def process_all_articles():
    """Process all Wikipedia articles to create clean Georgian corpus."""
    print("🧹 Starting comprehensive Georgian text cleaning...")
    print(f"📁 Source: {ARTICLES_DIR}")
    print(f"📁 Output: {OUTPUT_DIR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all article files
    article_files = list(ARTICLES_DIR.glob("*.txt"))
    print(f"📄 Found {len(article_files):,} article files")
    
    # Processing statistics
    stats = {
        'total_files': len(article_files),
        'processed': 0,
        'kept': 0,
        'too_short': 0,
        'low_georgian_ratio': 0,
        'empty_after_cleaning': 0,
        'total_chars_before': 0,
        'total_chars_after': 0,
        'georgian_ratio_distribution': [],
    }
    
    cleaned_articles = []
    
    print("🔄 Processing articles...")
    for i, file_path in enumerate(article_files):
        if i % 5000 == 0:
            print(f"   Progress: {i:,}/{len(article_files):,} ({i/len(article_files)*100:.1f}%)")
        
        try:
            # Read original article
            with open(file_path, 'r', encoding='utf-8') as f:
                original_text = f.read()
            
            stats['total_chars_before'] += len(original_text)
            stats['processed'] += 1
            
            # Clean the article
            cleaned_text = clean_article(original_text)
            
            if not cleaned_text:
                stats['empty_after_cleaning'] += 1
                continue
            
            if len(cleaned_text) < MIN_ARTICLE_LENGTH:
                stats['too_short'] += 1
                continue
            
            # Check Georgian ratio
            georgian_ratio = calculate_georgian_ratio(cleaned_text)
            stats['georgian_ratio_distribution'].append(georgian_ratio)
            
            # We want very high Georgian ratio (>95%)
            if georgian_ratio < 0.95:
                stats['low_georgian_ratio'] += 1
                continue
            
            # Article passed all filters
            stats['kept'] += 1
            stats['total_chars_after'] += len(cleaned_text)
            
            # Extract title from filename
            title = file_path.stem.replace('_', ' ')
            
            article_data = {
                'title': title,
                'text': cleaned_text,
                'length': len(cleaned_text),
                'word_count': len(cleaned_text.split()),
                'georgian_ratio': georgian_ratio,
                'source_file': str(file_path)
            }
            
            cleaned_articles.append(article_data)
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue
    
    # Save cleaned articles
    print(f"\n💾 Saving {len(cleaned_articles):,} cleaned articles...")
    
    # Save as JSONL
    jsonl_path = OUTPUT_DIR / "clean_georgian_corpus.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for article in cleaned_articles:
            json.dump(article, f, ensure_ascii=False)
            f.write('\n')
    
    # Save as plain text for tokenizer training
    txt_path = OUTPUT_DIR / "clean_georgian_corpus.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        for article in cleaned_articles:
            f.write(article['text'])
            f.write('\n\n')
    
    # Calculate final statistics
    avg_georgian_ratio = sum(stats['georgian_ratio_distribution']) / len(stats['georgian_ratio_distribution']) if stats['georgian_ratio_distribution'] else 0
    size_reduction = (1 - stats['total_chars_after'] / stats['total_chars_before']) * 100 if stats['total_chars_before'] > 0 else 0
    final_size_mb = stats['total_chars_after'] / (1024 * 1024)
    
    # Save processing statistics
    processing_stats = {
        **stats,
        'avg_georgian_ratio': avg_georgian_ratio,
        'size_reduction_percent': size_reduction,
        'final_size_mb': final_size_mb,
        'final_size_gb': final_size_mb / 1024,
        'estimated_tokens': stats['total_chars_after'] / 4,  # Rough estimate
    }
    
    with open(OUTPUT_DIR / "processing_stats.json", 'w', encoding='utf-8') as f:
        json.dump(processing_stats, f, ensure_ascii=False, indent=2)
    
    # Print final report
    print(f"\n📊 CLEANING RESULTS:")
    print(f"=" * 50)
    print(f"📄 Original articles: {stats['total_files']:,}")
    print(f"✅ Articles kept: {stats['kept']:,} ({stats['kept']/stats['total_files']*100:.1f}%)")
    print(f"❌ Rejected articles:")
    print(f"   📏 Too short: {stats['too_short']:,}")
    print(f"   🔤 Low Georgian ratio: {stats['low_georgian_ratio']:,}")
    print(f"   📭 Empty after cleaning: {stats['empty_after_cleaning']:,}")
    
    print(f"\n📊 SIZE ANALYSIS:")
    print(f"📦 Original size: {stats['total_chars_before']/1024/1024:.1f} MB")
    print(f"✨ Cleaned size: {final_size_mb:.1f} MB")
    print(f"📉 Size reduction: {size_reduction:.1f}%")
    
    print(f"\n🎯 QUALITY METRICS:")
    print(f"🇬🇪 Average Georgian ratio: {avg_georgian_ratio:.1%}")
    print(f"📝 Estimated tokens: {stats['total_chars_after']/4:,.0f}")
    print(f"📚 Average article length: {stats['total_chars_after']/stats['kept']:.0f} chars")
    
    print(f"\n💾 OUTPUT FILES:")
    print(f"📄 JSONL: {jsonl_path}")
    print(f"📝 Text: {txt_path}")
    print(f"📊 Stats: {OUTPUT_DIR / 'processing_stats.json'}")
    
    print(f"\n🎉 GEORGIAN CORPUS READY FOR LLM TRAINING!")
    
    # Quality validation
    if avg_georgian_ratio >= 0.98:
        print(f"🟢 EXCELLENT: {avg_georgian_ratio:.1%} Georgian ratio achieved!")
    elif avg_georgian_ratio >= 0.95:
        print(f"🟡 GOOD: {avg_georgian_ratio:.1%} Georgian ratio (target: >95%)")
    else:
        print(f"🔴 NEEDS MORE FILTERING: {avg_georgian_ratio:.1%} Georgian ratio")
    
    return cleaned_articles, processing_stats

def sample_cleaned_articles(articles, n_samples=5):
    """Show samples of cleaned articles."""
    print(f"\n📖 CLEANED ARTICLE SAMPLES:")
    print("=" * 50)
    
    samples = articles[:n_samples] if len(articles) >= n_samples else articles
    
    for i, article in enumerate(samples, 1):
        print(f"\n📄 Sample {i}: {article['title']}")
        print(f"📏 Length: {article['length']:,} characters")
        print(f"🇬🇪 Georgian ratio: {article['georgian_ratio']:.1%}")
        print(f"📝 Text preview:")
        
        # Show first 300 characters
        preview = article['text'][:300]
        if len(article['text']) > 300:
            preview += "..."
        print(f"   {preview}")
        print("-" * 50)

def validate_corpus_quality(articles):
    """Validate the final corpus quality."""
    print(f"\n🔍 CORPUS QUALITY VALIDATION:")
    print("=" * 50)
    
    if not articles:
        print("❌ No articles in corpus!")
        return False
    
    # Calculate overall statistics
    total_chars = sum(a['length'] for a in articles)
    total_words = sum(a['word_count'] for a in articles)
    avg_georgian_ratio = sum(a['georgian_ratio'] for a in articles) / len(articles)
    
    # Character analysis on sample
    sample_text = ' '.join(a['text'] for a in articles[:100])  # Sample of 100 articles
    char_freq = Counter(sample_text)
    
    # Count character types
    georgian_chars = sum(count for char, count in char_freq.items() if is_georgian_char(char))
    total_letters = sum(count for char, count in char_freq.items() if char.isalpha())
    actual_ratio = georgian_chars / total_letters if total_letters > 0 else 0
    
    print(f"📊 Final Corpus Statistics:")
    print(f"   📄 Articles: {len(articles):,}")
    print(f"   📝 Total characters: {total_chars:,}")
    print(f"   🔤 Total words: {total_words:,}")
    print(f"   📦 Size: {total_chars/1024/1024:.1f} MB")
    print(f"   🇬🇪 Georgian ratio: {actual_ratio:.1%}")
    print(f"   📏 Avg article length: {total_chars/len(articles):.0f} chars")
    
    # Top characters (should be mostly Georgian)
    print(f"\n🔤 Top 20 characters:")
    for char, count in char_freq.most_common(20):
        if char.isprintable() and char != ' ':
            is_geo = "🇬🇪" if is_georgian_char(char) else "❓"
            print(f"   {is_geo} '{char}': {count:,}")
    
    # Quality check
    quality_passed = actual_ratio >= 0.98
    
    if quality_passed:
        print(f"\n🎉 QUALITY CHECK PASSED!")
        print(f"✅ Georgian ratio {actual_ratio:.1%} meets target (>98%)")
    else:
        print(f"\n⚠️  QUALITY CHECK FAILED!")
        print(f"❌ Georgian ratio {actual_ratio:.1%} below target (>98%)")
    
    return quality_passed

def main():
    """Main cleaning function."""
    print("🧹 Georgian Wikipedia Corpus Cleaner")
    print("🎯 Goal: 100% Georgian character ratio for LLM training")
    print("=" * 60)
    
    if not ARTICLES_DIR.exists():
        print(f"❌ Articles directory not found: {ARTICLES_DIR}")
        return
    
    # Process all articles
    cleaned_articles, stats = process_all_articles()
    
    # Show samples
    if cleaned_articles:
        sample_cleaned_articles(cleaned_articles, 3)
    
    # Validate quality
    quality_ok = validate_corpus_quality(cleaned_articles)
    
    if quality_ok:
        print(f"\n🚀 NEXT STEPS:")
        print(f"1. Train tokenizer: python train_tokenizer.py")
        print(f"2. Train model: python improved_training.py")
        print(f"3. Or run full pipeline: python run_full_pipeline.py")
    else:
        print(f"\n🔧 NEEDS MORE CLEANING:")
        print(f"Consider adjusting filtering parameters and running again.")

if __name__ == "__main__":
    main()
