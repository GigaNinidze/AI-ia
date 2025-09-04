#!/usr/bin/env python3
"""
Complete pipeline for processing Georgian Wikipedia and training LLM.
"""

import subprocess
import sys
from pathlib import Path
import time

def run_command(cmd: str, description: str):
    """Run a command with progress indication."""
    print(f"\n🔄 {description}")
    print(f"💻 Running: {cmd}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        print(f"✅ Completed in {elapsed:.1f}s")
        if result.stdout:
            print(f"📤 Output: {result.stdout[:500]}...")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ Failed after {elapsed:.1f}s")
        print(f"📤 Error: {e.stderr}")
        return False

def check_prerequisites():
    """Check if all required files exist."""
    required_files = [
        Path("articles"),
        Path("clean_georgian_corpus.py"),
        Path("train_tokenizer.py"),
        Path("improved_training.py"),
    ]
    
    missing = [f for f in required_files if not f.exists()]
    if missing:
        print(f"❌ Missing required files: {missing}")
        return False
    
    print("✅ All prerequisites found")
    return True

def main():
    """Run the complete pipeline."""
    print("🚀 Georgian Wikipedia LLM Training Pipeline")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Step 1: Clean Georgian corpus
    print("\n🧹 STEP 1: Cleaning Georgian Wikipedia Corpus")
    if not run_command("python clean_georgian_corpus.py", "Cleaning 233,926 Wikipedia articles to 100% Georgian"):
        print("❌ Corpus cleaning failed!")
        sys.exit(1)
    
    # Step 2: Train tokenizer
    print("\n🔤 STEP 2: Training SentencePiece Tokenizer")
    if not run_command("python train_tokenizer.py", "Training Georgian tokenizer"):
        print("❌ Tokenizer training failed!")
        sys.exit(1)
    
    # Step 3: Train improved model
    print("\n🧠 STEP 3: Training Improved Georgian LLM")
    print("⚠️  This will take several hours on M3 Mac...")
    
    if not run_command("python improved_training.py", "Training Georgian language model"):
        print("❌ Model training failed!")
        sys.exit(1)
    
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("📁 Check the following directories:")
    print("   📊 Processed corpus: ../corpus/processed/")
    print("   🔤 Tokenizer: ../tokens/")
    print("   🧠 Model checkpoints: ../checkpoints/improved-ge-llm/")
    print("\n🚀 Your Georgian LLM is ready!")

if __name__ == "__main__":
    main()
