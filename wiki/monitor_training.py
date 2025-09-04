#!/usr/bin/env python3
"""
Monitor the training progress of the Georgian LLM.
"""

import os
import time
from pathlib import Path
import json

CHECKPOINT_DIR = Path("../checkpoints/clean-georgian-llm")

def monitor_training():
    """Monitor training progress."""
    print("📊 Georgian LLM Training Monitor")
    print("=" * 40)
    
    if not CHECKPOINT_DIR.exists():
        print("❌ Checkpoint directory not found. Training not started yet.")
        return
    
    print(f"📁 Monitoring: {CHECKPOINT_DIR}")
    print("🔄 Checking for checkpoints...")
    
    while True:
        # List all checkpoint files
        checkpoints = list(CHECKPOINT_DIR.glob("*.pt"))
        
        if checkpoints:
            print(f"\n📊 Found {len(checkpoints)} checkpoints:")
            
            for ckpt in sorted(checkpoints):
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                mod_time = time.ctime(ckpt.stat().st_mtime)
                print(f"   📄 {ckpt.name} ({size_mb:.1f}MB) - {mod_time}")
            
            # Check if best model exists
            best_model = CHECKPOINT_DIR / "best_model.pt"
            if best_model.exists():
                print(f"\n🏆 Best model found: {best_model}")
                size_mb = best_model.stat().st_size / (1024 * 1024)
                print(f"   📦 Size: {size_mb:.1f}MB")
        else:
            print("⏳ No checkpoints found yet. Training in progress...")
        
        print(f"\n🕐 Last check: {time.strftime('%H:%M:%S')}")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped.")
            break

if __name__ == "__main__":
    monitor_training()
