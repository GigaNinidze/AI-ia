#!/usr/bin/env python3
"""
Train a SentencePiece tokenizer optimized for Georgian text.
"""

import sentencepiece as spm
from pathlib import Path
import json

# Configuration
INPUT_TEXT = Path("../corpus/clean_georgian/clean_georgian_corpus.txt")
OUTPUT_DIR = Path("../tokens")
MODEL_PREFIX = "ge_tokenizer_clean"

# Tokenizer parameters optimized for Georgian
VOCAB_SIZE = 32000
CHARACTER_COVERAGE = 0.9995  # High coverage for Georgian
MODEL_TYPE = "bpe"  # Byte-pair encoding

def train_tokenizer():
    """Train SentencePiece tokenizer on Georgian corpus."""
    print("🚀 Training Georgian SentencePiece tokenizer...")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Training parameters
    train_params = [
        f"--input={INPUT_TEXT}",
        f"--model_prefix={OUTPUT_DIR / MODEL_PREFIX}",
        f"--vocab_size={VOCAB_SIZE}",
        f"--character_coverage={CHARACTER_COVERAGE}",
        f"--model_type={MODEL_TYPE}",
        "--pad_id=0",
        "--unk_id=1", 
        "--bos_id=2",
        "--eos_id=3",
        "--train_extremely_large_corpus=true",
        "--shuffle_input_sentence=true",
        "--max_sentence_length=16384",
        "--num_threads=8",
        "--split_digits=true",
        "--allow_whitespace_only_pieces=true",
        "--byte_fallback=true",
    ]
    
    # Train the tokenizer
    print(f"📚 Training on: {INPUT_TEXT}")
    print(f"📁 Output: {OUTPUT_DIR / MODEL_PREFIX}")
    print(f"🔤 Vocabulary size: {VOCAB_SIZE:,}")
    
    spm.SentencePieceTrainer.train(" ".join(train_params))
    
    # Test the tokenizer
    print("\n🧪 Testing tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(str(OUTPUT_DIR / f"{MODEL_PREFIX}.model"))
    
    test_texts = [
        "გამარჯობა, როგორ ხარ?",
        "ხელოვნური ინტელექტი არის ტექნოლოგია რომელიც ეხმარება ადამიანებს.",
        "საქართველო არის ქვეყანა კავკასიაში.",
        "მათემატიკა და ფიზიკა არის მნიშვნელოვანი საგნები.",
    ]
    
    for text in test_texts:
        tokens = sp.encode(text, out_type=str)
        token_ids = sp.encode(text, out_type=int)
        decoded = sp.decode(token_ids)
        
        print(f"Original: {text}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Decoded: {decoded}")
        print(f"Token count: {len(tokens)}")
        print("-" * 50)
    
    # Save tokenizer config
    config = {
        "model_file": f"{MODEL_PREFIX}.model",
        "vocab_file": f"{MODEL_PREFIX}.vocab", 
        "vocab_size": VOCAB_SIZE,
        "character_coverage": CHARACTER_COVERAGE,
        "model_type": MODEL_TYPE,
        "special_tokens": {
            "pad_id": 0,
            "unk_id": 1,
            "bos_id": 2,
            "eos_id": 3
        },
        "trained_on": str(INPUT_TEXT),
        "training_corpus_size_mb": INPUT_TEXT.stat().st_size / (1024 * 1024) if INPUT_TEXT.exists() else 0
    }
    
    with open(OUTPUT_DIR / "tokenizer_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Tokenizer training complete!")
    print(f"📁 Model saved: {OUTPUT_DIR / MODEL_PREFIX}.model")
    print(f"📁 Config saved: {OUTPUT_DIR / 'tokenizer_config.json'}")

if __name__ == "__main__":
    train_tokenizer()
