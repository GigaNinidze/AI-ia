#!/usr/bin/env python3
"""
Improved training script for Georgian LLM using the processed Wikipedia corpus.
"""

import os
import math
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

# Paths
PROCESSED_DATA_DIR = Path("../corpus/clean_georgian")
TOKENIZER_MODEL = Path("../tokens/ge_tokenizer_clean.model")
CHECKPOINT_DIR = Path("../checkpoints/clean-georgian-llm")

# Training configuration for 1.8GB corpus
TOTAL_TOKENS_ESTIMATE = 300_000_000  # ~300M tokens (much larger!)
TARGET_PARAMS = int(TOTAL_TOKENS_ESTIMATE / 20)  # Chinchilla optimal: ~15M params

# Enhanced hyperparameters for larger corpus
SEQ_LEN = 1024  # Longer context
BATCH_SIZE = 4  # Larger batch
ACCUM_STEPS = 16  # More accumulation
LR = 1e-4  # Lower learning rate for stability
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000  # More warmup
MAX_STEPS = 50000  # Much longer training
EVAL_EVERY = 1000
SAVE_EVERY = 5000

# Model configuration
DROPOUT = 0.1
VOCAB_PAD = 0
VOCAB_UNK = 1
VOCAB_BOS = 2
VOCAB_EOS = 3

# Device setup
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class ImprovedTextDataset(Dataset):
    """
    Enhanced dataset for Georgian text with better tokenization and packing.
    """
    def __init__(self, jsonl_path: Path, tokenizer_path: Path, seq_len: int = 1024):
        self.seq_len = seq_len
        
        # Load tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(tokenizer_path))
        
        # Load and tokenize all text
        print(f"📚 Loading data from {jsonl_path}")
        all_tokens = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f):
                if line_no % 1000 == 0:
                    print(f"   Processing line {line_no}")
                
                try:
                    article = json.loads(line.strip())
                    text = article['text']
                    
                    # Tokenize
                    tokens = self.sp.encode(text, out_type=int)
                    
                    # Add special tokens
                    tokens = [VOCAB_BOS] + tokens + [VOCAB_EOS]
                    all_tokens.extend(tokens)
                    
                except json.JSONDecodeError:
                    continue
        
        print(f"📊 Total tokens: {len(all_tokens):,}")
        
        # Pack into sequences
        num_sequences = len(all_tokens) // seq_len
        all_tokens = all_tokens[:num_sequences * seq_len]
        
        self.data = torch.tensor(all_tokens, dtype=torch.long).view(num_sequences, seq_len)
        print(f"📦 Created {num_sequences:,} sequences of length {seq_len}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        return {
            "input_ids": sequence,
            "labels": sequence.clone()
        }

class ImprovedGPTConfig:
    """Enhanced configuration for larger model."""
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, dropout=0.1):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout

class EnhancedAttention(nn.Module):
    """Enhanced multi-head attention with better efficiency."""
    def __init__(self, cfg: ImprovedGPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        
        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        
        # Flash attention for efficiency (if available)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate QKV
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.transpose(2, 3).unbind(dim=2)  # (B, n_head, T, head_dim)
        
        if self.flash:
            # Use flash attention if available
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True
            )
        else:
            # Manual attention
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float('-inf'))
            att = torch.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class EnhancedBlock(nn.Module):
    """Enhanced transformer block with better normalization."""
    def __init__(self, cfg: ImprovedGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = EnhancedAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        
        # Larger MLP for better capacity
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False),
            nn.Dropout(cfg.dropout),
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class ImprovedGPT(nn.Module):
    """Enhanced GPT model for Georgian language."""
    def __init__(self, cfg: ImprovedGPTConfig):
        super().__init__()
        self.cfg = cfg
        
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        
        self.blocks = nn.ModuleList([EnhancedBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.tok_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None):
        B, T = input_ids.size()
        
        # Token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss

def estimate_model_size(vocab_size: int, n_layer: int, n_embd: int, block_size: int) -> int:
    """Estimate model parameters."""
    # Embeddings: vocab_size * n_embd + block_size * n_embd
    embeddings = (vocab_size + block_size) * n_embd
    
    # Each layer: attention (4 * n_embd^2) + MLP (8 * n_embd^2) + LayerNorm (2 * n_embd)
    per_layer = 12 * n_embd * n_embd + 2 * n_embd
    layers = n_layer * per_layer
    
    # Final LayerNorm
    final_ln = n_embd
    
    return embeddings + layers + final_ln

def find_optimal_architecture(vocab_size: int, target_params: int, block_size: int = 1024):
    """Find optimal architecture for target parameter count."""
    best_config = None
    best_diff = float('inf')
    
    # Search space
    layer_options = range(6, 25)  # 6-24 layers
    head_options = [4, 6, 8, 12, 16]  # Number of heads
    embd_options = range(256, 1025, 64)  # 256-1024, step 64
    
    for n_layer in layer_options:
        for n_head in head_options:
            for n_embd in embd_options:
                if n_embd % n_head != 0:
                    continue
                
                params = estimate_model_size(vocab_size, n_layer, n_embd, block_size)
                diff = abs(params - target_params)
                
                if diff < best_diff:
                    best_diff = diff
                    best_config = (n_layer, n_head, n_embd, params)
    
    return best_config

def cosine_lr_schedule(step, max_steps, base_lr, warmup_steps):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return base_lr * (step / max(warmup_steps, 1))
    
    progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

def evaluate_model(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            _, loss = model(input_ids, labels)
            total_loss += loss.item()
            total_batches += 1
    
    model.train()
    avg_loss = total_loss / max(total_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))  # Cap to prevent overflow
    
    return avg_loss, perplexity

def main():
    """Main training function."""
    print("🚀 Starting improved Georgian LLM training...")
    
    # Create checkpoint directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    if not TOKENIZER_MODEL.exists():
        print(f"❌ Tokenizer not found: {TOKENIZER_MODEL}")
        print("🔧 Please run train_tokenizer.py first!")
        return
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_MODEL))
    vocab_size = sp.get_piece_size()
    print(f"🔤 Vocabulary size: {vocab_size:,}")
    
    # Find optimal architecture
    print(f"🎯 Target parameters: {TARGET_PARAMS:,}")
    n_layer, n_head, n_embd, actual_params = find_optimal_architecture(
        vocab_size, TARGET_PARAMS, SEQ_LEN
    )
    
    print(f"🏗️  Optimal architecture:")
    print(f"   Layers: {n_layer}")
    print(f"   Heads: {n_head}")
    print(f"   Embedding dim: {n_embd}")
    print(f"   Actual parameters: {actual_params:,}")
    
    # Load datasets
    train_path = PROCESSED_DATA_DIR / "train.jsonl"
    val_path = PROCESSED_DATA_DIR / "val.jsonl"
    
    if not train_path.exists() or not val_path.exists():
        print(f"❌ Processed data not found!")
        print("🔧 Please run process_wiki_corpus.py first!")
        return
    
    print("📚 Loading datasets...")
    train_dataset = ImprovedTextDataset(train_path, TOKENIZER_MODEL, SEQ_LEN)
    val_dataset = ImprovedTextDataset(val_path, TOKENIZER_MODEL, SEQ_LEN)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    cfg = ImprovedGPTConfig(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=SEQ_LEN,
        dropout=DROPOUT
    )
    
    model = ImprovedGPT(cfg).to(DEVICE)
    
    # Count actual parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🎯 Model parameters: {total_params:,}")
    print(f"📱 Device: {DEVICE}")
    
    # Optimizer with better settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95)  # Better betas for large models
    )
    
    # Training loop
    print(f"🏋️  Starting training for {MAX_STEPS:,} steps...")
    
    model.train()
    best_val_loss = float('inf')
    global_step = 0
    
    for step in range(1, MAX_STEPS + 1):
        # Get batch
        try:
            batch = next(iter(train_loader))
        except StopIteration:
            continue
        
        input_ids = batch['input_ids'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Gradient accumulation
        total_loss = 0
        for micro_step in range(ACCUM_STEPS):
            # Get micro-batch (if we have enough data)
            start_idx = micro_step * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            
            if end_idx <= len(input_ids):
                micro_input = input_ids[start_idx:end_idx]
                micro_labels = labels[start_idx:end_idx]
            else:
                micro_input = input_ids
                micro_labels = labels
            
            _, loss = model(micro_input, micro_labels)
            loss = loss / ACCUM_STEPS
            loss.backward()
            total_loss += loss.item()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Learning rate schedule
        lr_now = cosine_lr_schedule(global_step, MAX_STEPS, LR, WARMUP_STEPS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now
        
        optimizer.step()
        global_step += 1
        
        # Logging
        if step % 100 == 0:
            print(f"[Step {step:,}] Loss: {total_loss:.4f}, LR: {lr_now:.6f}")
        
        # Evaluation
        if step % EVAL_EVERY == 0 or step == 1:
            print("🧪 Evaluating...")
            val_loss, val_ppl = evaluate_model(model, val_loader, DEVICE)
            print(f"[Eval {step:,}] Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best checkpoint
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': cfg.__dict__,
                    'step': step,
                    'val_loss': val_loss,
                    'val_ppl': val_ppl
                }, CHECKPOINT_DIR / "best_model.pt")
                print(f"💾 Saved best model (val_loss: {val_loss:.4f})")
        
        # Regular checkpoints
        if step % SAVE_EVERY == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg.__dict__,
                'step': step,
            }, CHECKPOINT_DIR / f"checkpoint_step_{step}.pt")
            print(f"💾 Saved checkpoint at step {step:,}")
    
    print("🎉 Training completed!")

if __name__ == "__main__":
    main()
