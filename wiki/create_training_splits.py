#!/usr/bin/env python3
"""
Create train/validation/test splits from the clean Georgian corpus.
"""

import json
import random
from pathlib import Path

# Configuration
INPUT_JSONL = Path("../corpus/clean_georgian/clean_georgian_corpus.jsonl")
OUTPUT_DIR = Path("../corpus/clean_georgian")

# Split ratios
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

def create_splits():
    """Create train/val/test splits."""
    print("🔄 Creating train/validation/test splits...")
    
    # Load all articles
    articles = []
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                article = json.loads(line.strip())
                articles.append(article)
            except json.JSONDecodeError:
                continue
    
    print(f"📚 Loaded {len(articles):,} articles")
    
    # Shuffle for random splits
    random.seed(42)  # Reproducible splits
    random.shuffle(articles)
    
    # Calculate split sizes
    total = len(articles)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    train_articles = articles[:train_end]
    val_articles = articles[train_end:val_end]
    test_articles = articles[val_end:]
    
    print(f"📊 Split sizes:")
    print(f"   Train: {len(train_articles):,} articles ({len(train_articles)/total:.1%})")
    print(f"   Validation: {len(val_articles):,} articles ({len(val_articles)/total:.1%})")
    print(f"   Test: {len(test_articles):,} articles ({len(test_articles)/total:.1%})")
    
    # Save splits
    splits = {
        'train': train_articles,
        'val': val_articles,
        'test': test_articles
    }
    
    for split_name, split_articles in splits.items():
        # Save JSONL
        jsonl_path = OUTPUT_DIR / f"{split_name}.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for article in split_articles:
                json.dump(article, f, ensure_ascii=False)
                f.write('\n')
        
        # Save plain text
        txt_path = OUTPUT_DIR / f"{split_name}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for article in split_articles:
                f.write(article['text'])
                f.write('\n\n')
        
        # Calculate size
        total_chars = sum(len(a['text']) for a in split_articles)
        size_mb = total_chars / (1024 * 1024)
        
        print(f"💾 {split_name.upper()}: {len(split_articles):,} articles, {size_mb:.1f}MB")
        print(f"   📄 JSONL: {jsonl_path}")
        print(f"   📝 Text: {txt_path}")
    
    print("✅ Splits created successfully!")

if __name__ == "__main__":
    create_splits()
