#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, sys, random
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

# ===== Paths (edit if needed) =====
CKPT_PATH = Path("/Users/us1ndiso/Desktop/ქართული-ენა/checkpoints/small-ge-llm/step_7000.pt")
SP_MODEL_PATH = Path("/Users/us1ndiso/Desktop/ქართული-ენა/tokens/ge_tokenizer.model")

# ===== Generation defaults =====
PROMPT = "საქართველო არის  "
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.9
REP_PENALTY = 1.05
EOS_ID = 3
SEED = 42

# ----------------------------
# Model definitions (same as train.py)
# ----------------------------
class GPTConfig:
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, dropout, pad_id, tie_weights=True):
        self.vocab_size=vocab_size
        self.n_layer=n_layer
        self.n_head=n_head
        self.n_embd=n_embd
        self.block_size=block_size
        self.dropout=dropout
        self.pad_id=pad_id
        self.tie_weights=tie_weights

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.key   = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.query = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.value = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj  = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.n_head= cfg.n_head
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B,T,C = x.size()
        H = self.n_head
        head_dim = C // H
        k = self.key(x).view(B,T,H,head_dim).transpose(1,2)
        q = self.query(x).view(B,T,H,head_dim).transpose(1,2)
        v = self.value(x).view(B,T,H,head_dim).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(head_dim)
        mask = torch.tril(torch.ones(T,T, device=x.device)).view(1,1,T,T)
        att = att.masked_fill(mask==0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.proj(y)
        return y

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4*cfg.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4*cfg.n_embd, cfg.n_embd, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, elementwise_affine=False)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

    def forward(self, input_ids, labels=None):
        B,T = input_ids.size()
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, None

# ----------------------------
# Helpers
# ----------------------------
def top_k_top_p_filter(logits, top_k=50, top_p=0.9):
    if top_k > 0:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = -float("inf")
    if 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        cutoff = cum > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits[cutoff] = -float("inf")
        logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits

@torch.no_grad()
def generate(model, sp, prompt):
    ids = sp.encode(prompt, out_type=int)
    generated = ids.copy()
    block_size = model.cfg.block_size

    for _ in range(MAX_NEW_TOKENS):
        ctx = generated[-block_size:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = model(x)
        next_logits = logits[0, -1, :] / TEMPERATURE

        # repetition penalty
        if REP_PENALTY > 1.0:
            for t in set(generated):
                next_logits[t] /= REP_PENALTY

        next_logits = top_k_top_p_filter(next_logits, TOP_K, TOP_P)
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()

        generated.append(next_id)
        sys.stdout.write(sp.decode([next_id]))
        sys.stdout.flush()

        if EOS_ID is not None and next_id == EOS_ID:
            break
    print()
    return sp.decode(generated)

# ----------------------------
# Main
# ----------------------------
def pick_device():
    if torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = pick_device()

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(str(SP_MODEL_PATH))

    # Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=device)
    cfg = GPTConfig(**ckpt["cfg"])
    model = TinyGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Run
    print(f"[Prompt] {PROMPT}\n[Generation]:")
    out = generate(model, sp, PROMPT)
    print("\n---\nFull output:\n", out)