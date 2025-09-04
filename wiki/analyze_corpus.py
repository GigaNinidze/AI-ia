#!/usr/bin/env python3
"""
Analyze the Georgian Wikipedia corpus to understand data quality and characteristics.
"""

import os
import json
import random
from pathlib import Path
from collections import Counter

# Configuration
ARTICLES_DIR = Path("articles")
SAMPLE_SIZE = 1000  # Sample articles for detailed analysis

def analyze_file_sizes():
    """Analyze article file sizes."""
    print("📊 Analyzing article file sizes...")
    
    file_sizes = []
    for file_path in ARTICLES_DIR.glob("*.txt"):
        try:
            size = file_path.stat().st_size
            file_sizes.append(size)
        except:
            continue
    
    file_sizes.sort()
    
    print(f"📄 Total articles: {len(file_sizes):,}")
    print(f"📏 Size statistics:")
    print(f"   Min: {min(file_sizes):,} bytes")
    print(f"   Max: {max(file_sizes):,} bytes") 
    print(f"   Median: {file_sizes[len(file_sizes)//2]:,} bytes")
    print(f"   Average: {sum(file_sizes)/len(file_sizes):.0f} bytes")
    
    # Size distribution
    small = sum(1 for s in file_sizes if s < 1000)  # < 1KB
    medium = sum(1 for s in file_sizes if 1000 <= s < 10000)  # 1-10KB
    large = sum(1 for s in file_sizes if s >= 10000)  # > 10KB
    
    print(f"📈 Size distribution:")
    print(f"   Small (< 1KB): {small:,} ({small/len(file_sizes)*100:.1f}%)")
    print(f"   Medium (1-10KB): {medium:,} ({medium/len(file_sizes)*100:.1f}%)")
    print(f"   Large (> 10KB): {large:,} ({large/len(file_sizes)*100:.1f}%)")
    
    return file_sizes

def sample_articles(n_samples=10):
    """Sample random articles for inspection."""
    print(f"\n📖 Sampling {n_samples} random articles...")
    
    article_files = list(ARTICLES_DIR.glob("*.txt"))
    samples = random.sample(article_files, min(n_samples, len(article_files)))
    
    for i, file_path in enumerate(samples, 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            title = file_path.stem.replace('_', ' ')
            print(f"\n📄 Sample {i}: {title}")
            print(f"   Length: {len(content):,} characters")
            print(f"   Preview: {content[:200]}...")
            
        except Exception as e:
            print(f"   ❌ Error reading {file_path}: {e}")

def analyze_content_quality():
    """Analyze content quality across random sample."""
    print(f"\n🔍 Analyzing content quality (sample of {SAMPLE_SIZE} articles)...")
    
    article_files = list(ARTICLES_DIR.glob("*.txt"))
    sample_files = random.sample(article_files, min(SAMPLE_SIZE, len(article_files)))
    
    stats = {
        'total_chars': 0,
        'total_words': 0,
        'georgian_chars': 0,
        'articles_processed': 0,
        'empty_articles': 0,
        'short_articles': 0,  # < 100 chars
        'medium_articles': 0,  # 100-1000 chars
        'long_articles': 0,   # > 1000 chars
        'char_frequency': Counter(),
        'word_frequency': Counter(),
    }
    
    for file_path in sample_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                stats['empty_articles'] += 1
                continue
            
            # Basic stats
            stats['total_chars'] += len(content)
            words = content.split()
            stats['total_words'] += len(words)
            stats['articles_processed'] += 1
            
            # Character analysis
            for char in content:
                stats['char_frequency'][char] += 1
                if is_georgian_char(char):
                    stats['georgian_chars'] += 1
            
            # Word frequency (top words)
            for word in words[:10]:  # Limit to avoid huge counters
                if len(word) > 2:  # Skip very short words
                    stats['word_frequency'][word] += 1
            
            # Length categorization
            if len(content) < 100:
                stats['short_articles'] += 1
            elif len(content) < 1000:
                stats['medium_articles'] += 1
            else:
                stats['long_articles'] += 1
                
        except Exception as e:
            print(f"   ❌ Error processing {file_path}: {e}")
    
    # Calculate ratios
    georgian_ratio = stats['georgian_chars'] / stats['total_chars'] if stats['total_chars'] > 0 else 0
    
    print(f"📊 Content Quality Analysis:")
    print(f"   Articles processed: {stats['articles_processed']:,}")
    print(f"   Empty articles: {stats['empty_articles']:,}")
    print(f"   Total characters: {stats['total_chars']:,}")
    print(f"   Total words: {stats['total_words']:,}")
    print(f"   Georgian character ratio: {georgian_ratio:.1%}")
    print(f"   Avg chars/article: {stats['total_chars']/stats['articles_processed']:.0f}")
    print(f"   Avg words/article: {stats['total_words']/stats['articles_processed']:.0f}")
    
    print(f"\n📏 Article length distribution:")
    print(f"   Short (< 100 chars): {stats['short_articles']:,}")
    print(f"   Medium (100-1000 chars): {stats['medium_articles']:,}")
    print(f"   Long (> 1000 chars): {stats['long_articles']:,}")
    
    print(f"\n🔤 Top 20 characters:")
    for char, count in stats['char_frequency'].most_common(20):
        if char.isprintable() and char != ' ':
            print(f"   '{char}': {count:,}")
    
    print(f"\n📝 Top 20 words:")
    for word, count in stats['word_frequency'].most_common(20):
        if len(word) > 2:
            print(f"   '{word}': {count:,}")
    
    return stats

def is_georgian_char(char: str) -> bool:
    """Check if character is Georgian."""
    if not char:
        return False
    code_point = ord(char)
    georgian_ranges = [
        (0x10A0, 0x10FF),  # Georgian
        (0x2D00, 0x2D2F),  # Georgian Supplement
        (0x1C90, 0x1CBF),  # Georgian Extended
    ]
    return any(start <= code_point <= end for start, end in georgian_ranges)

def main():
    """Main analysis function."""
    print("🔍 Georgian Wikipedia Corpus Analysis")
    print("=" * 50)
    
    if not ARTICLES_DIR.exists():
        print(f"❌ Articles directory not found: {ARTICLES_DIR}")
        sys.exit(1)
    
    # File size analysis
    file_sizes = analyze_file_sizes()
    
    # Sample articles
    sample_articles(5)
    
    # Content quality analysis
    content_stats = analyze_content_quality()
    
    # Estimate total corpus size
    avg_size = sum(file_sizes) / len(file_sizes)
    total_articles = len(list(ARTICLES_DIR.glob("*.txt")))
    estimated_total_chars = (content_stats['total_chars'] / content_stats['articles_processed']) * total_articles
    estimated_mb = estimated_total_chars / (1024 * 1024)
    
    print(f"\n🎯 FINAL ESTIMATES:")
    print(f"   Total articles: {total_articles:,}")
    print(f"   Estimated total size: {estimated_mb:.0f} MB")
    print(f"   Estimated tokens: {estimated_total_chars / 4:.0f} (assuming 4 chars/token)")
    print(f"   Quality: {'🟢 Excellent' if content_stats['georgian_chars']/content_stats['total_chars'] > 0.8 else '🟡 Good' if content_stats['georgian_chars']/content_stats['total_chars'] > 0.6 else '🔴 Needs filtering'}")
    
    print(f"\n🚀 RECOMMENDATIONS:")
    if estimated_mb > 500:
        print("   📈 You have MORE than enough data for excellent LLM training!")
        print("   🎯 Consider using subsets for faster iteration")
    elif estimated_mb > 100:
        print("   ✅ Perfect amount of data for small LLM training!")
    else:
        print("   ⚠️  Consider gathering more data for better results")
    
    print(f"   🔄 Next steps:")
    print(f"   1. Run: python process_wiki_corpus.py")
    print(f"   2. Run: python train_tokenizer.py")  
    print(f"   3. Run: python improved_training.py")

if __name__ == "__main__":
    main()
