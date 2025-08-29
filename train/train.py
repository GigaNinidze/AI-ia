#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, json, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sentencepiece as spm

# ============ Paths ============
DATA_JSONL = Path("/Users/us1ndiso/Desktop/ქართული-ენა/corpus/out/clean_ge.jsonl")
SP_MODEL   = Path("/Users/us1ndiso/Desktop/ქართული-ენა/tokens/ge_tokenizer.model")
OUTDIR     = Path("/Users/us1ndiso/Desktop/ქართული-ენა/checkpoints/small-ge-llm")

# ============ Chinchilla target ============
TOTAL_TOKENS = 14_069_351
TARGET_PARAMS = int(TOTAL_TOKENS / 20)  # ≈ 703,467

# ============ Base training hyperparams (M3-friendly) ============
SEQ_LEN   = 512
VOCAB_PAD = 0    # must match tokenizer_config (PAD=0, UNK=1, BOS=2, EOS=3)
VOCAB_UNK = 1
VOCAB_BOS = 2
VOCAB_EOS = 3

DROPOUT = 0.1

BATCH_SIZE = 2
ACCUM_STEPS = 8
LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
MAX_STEPS = 15000
EVAL_EVERY = 200
SAVE_EVERY = 1000
RNG_SEED = 42

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

# ============ Data ============
def stream_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                obj=json.loads(line)
            except json.JSONDecodeError:
                continue
            txt=obj.get("text")
            if isinstance(txt,str) and txt.strip():
                yield txt.replace("\r\n","\n").replace("\r","\n")

def split_train_val(path: Path, val_ratio=0.05):
    lines=list(stream_jsonl(path))
    random.shuffle(lines)
    n=len(lines)
    n_val=max(1000, int(n*val_ratio))
    return lines[n_val:], lines[:n_val]

class CausalTextDataset(Dataset):
    """
    Packs consecutive texts into fixed-length sequences for causal LM.
    Buffer approach: concatenate tokenized lines with EOS and chunk into SEQ_LEN.
    """
    def __init__(self, texts: List[str], spm_model: Path, seq_len: int=512, bos_id=2, eos_id=3):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(spm_model))
        self.seq_len = seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id

        token_stream = []
        for t in texts:
            ids = self.sp.encode(t, out_type=int)
            token_stream.extend([bos_id] + ids + [eos_id])

        n_full = (len(token_stream) // seq_len) * seq_len
        token_stream = token_stream[:n_full]
        self.data = torch.tensor(token_stream, dtype=torch.long)

    def __len__(self):
        return self.data.numel() // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start:start+self.seq_len]
        y = x.clone()
        return {"input_ids": x, "labels": y}

def collate_batch(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([s["input_ids"] for s in samples], dim=0)
    labels    = torch.stack([s["labels"] for s in samples], dim=0)
    return {"input_ids": input_ids, "labels": labels}

# ============ Tiny GPT (decoder-only Transformer) ============
class GPTConfig:
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, dropout, pad_id, tie_weights: bool = True):
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
        k = self.key(x).view(B,T,H,head_dim).transpose(1,2)   # (B,H,T,hd)
        q = self.query(x).view(B,T,H,head_dim).transpose(1,2) # (B,H,T,hd)
        v = self.value(x).view(B,T,H,head_dim).transpose(1,2) # (B,H,T,hd)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(head_dim)  # (B,H,T,T)
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
        # elementwise_affine=False keeps parameter count tight and stable
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
        # final LN without affine to avoid extra params
        self.ln_f = nn.LayerNorm(cfg.n_embd, elementwise_affine=False)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying (decoder-only standard trick)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        B,T = input_ids.size()
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
        return logits, loss

# ============ Param estimation & autosizing ============
def estimate_params_tied(vocab_size:int, n_layer:int, n_embd:int, block_size:int) -> int:
    """
    Approx params with tied embeddings and no LN/MLP biases:
    per-layer ~ 12 * n_embd^2 (Q,K,V,Proj + MLP), embeddings ~ (vocab+block_size)*n_embd
    """
    per_layer = 12 * (n_embd ** 2)
    layers = n_layer * per_layer
    embeddings = (vocab_size + block_size) * n_embd
    return layers + embeddings

def choose_architecture(vocab_size:int, block_size:int, target_params:int) -> Tuple[int,int,int,int]:
    """
    Search tiny configs appropriate for large vocab (e.g., 32k) to hit Chinchilla target.
    Allows very small embeddings and even single-head attention.
    """
    best = None
    # Include 1 head for very small embeddings; keep options modest for M3
    head_candidates = [1, 2, 4, 8]
    layer_candidates = list(range(1, 33))  # 1..32
    grid=[]
    for e in range(8, 257, 4):   # 8..256, step 4 to include 20
        for h in head_candidates:
            if e % h == 0:
                grid.append((h, e))

    for n_layer in layer_candidates:
        for n_head, n_embd in grid:
            params = estimate_params_tied(vocab_size, n_layer, n_embd, block_size)
            diff = params - target_params
            score = (abs(diff), diff > 0)  # prefer under (False < True)
            if best is None or score < best[0]:
                best = (score, (n_layer, n_head, n_embd, params))

    (_, (L, H, E, P)) = best
    return L, H, E, P

# ============ Training utils ============
def cosine_lr(step, max_steps, base_lr, warmup):
    if step < warmup:
        return base_lr * (step / max(warmup, 1))
    progress = (step - warmup) / max(1, (max_steps - warmup))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

def evaluate(model, loader):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(DEVICE)
            _, loss = model(**batch)
            total += loss.item()
            count += 1
    model.train()
    avg = total / max(count,1)
    ppl = math.exp(min(20, avg))  # guard overflow
    return avg, ppl

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# ============ Main ============
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # SentencePiece vocab size (you said it's 32,000; we still read it to be safe)
    sp = spm.SentencePieceProcessor()
    sp.load(str(SP_MODEL))
    vocab_size = sp.get_piece_size()
    print(f"[vocab] size reported by SP: {vocab_size}")
    assert VOCAB_EOS < vocab_size, "Special token ids must be within vocab size."

    # Autosize by Chinchilla
    n_layer, n_head, n_embd, est_params = choose_architecture(vocab_size, SEQ_LEN, TARGET_PARAMS)

    print(f"[chinchilla] tokens={TOTAL_TOKENS:,}  target_params≈{TARGET_PARAMS:,}")
    print(f"[autosize] n_layer={n_layer} n_head={n_head} n_embd={n_embd}  est_params≈{est_params:,}")

    # Split data
    train_texts, val_texts = split_train_val(DATA_JSONL, val_ratio=0.05)

    # Datasets & loaders
    train_ds = CausalTextDataset(train_texts, SP_MODEL, SEQ_LEN, VOCAB_BOS, VOCAB_EOS)
    val_ds   = CausalTextDataset(val_texts,   SP_MODEL, SEQ_LEN, VOCAB_BOS, VOCAB_EOS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collate_batch)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=collate_batch)

    # Model
    cfg = GPTConfig(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
                    n_embd=n_embd, block_size=SEQ_LEN, dropout=DROPOUT, pad_id=VOCAB_PAD, tie_weights=True)
    model = TinyGPT(cfg).to(DEVICE)
    actual_params = count_params(model)
    print(f"[model] actual_params={actual_params:,}  (tied embeddings, no LN/MLP biases)  device={DEVICE}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    global_step = 0
    train_iter = iter(train_loader)

    for step in range(1, MAX_STEPS+1):
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUM_STEPS):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            for k in batch:
                batch[k] = batch[k].to(DEVICE)
            _, loss = model(**batch)
            (loss / ACCUM_STEPS).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr_now = cosine_lr(global_step, MAX_STEPS, LR, WARMUP_STEPS)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        opt.step()
        global_step += 1

        if step % 50 == 0:
            print(f"[step {step}] train_loss={loss.item():.4f} lr={lr_now:.6f}")

        if step % EVAL_EVERY == 0 or step == 1:
            val_loss, val_ppl = evaluate(model, val_loader)
            print(f"[eval {step}] val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "step": step
                }, OUTDIR / "best.pt")
                print(f"[save] best checkpoint @ step {step}")

        if step % SAVE_EVERY == 0:
            torch.save({
                "model_state": model.state_dict(),
                "cfg": cfg.__dict__,
                "step": step
            }, OUTDIR / f"step_{step}.pt")

    print("Done.")

if __name__ == "__main__":
    main()