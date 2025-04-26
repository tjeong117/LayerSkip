import os, math, random, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt, numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import (GPT2TokenizerFast,
                          TopKLogitsWarper,
                          TopPLogitsWarper)

SEQ_LEN        = 128
BATCH_SIZE     = 8
EPOCHS         = 3
LR             = 5e-5
WEIGHT_DECAY   = 0.01
GRAD_CLIP      = 1.0
MAX_DROP       = 0.5
TAU_VAL        = 0.45
LAMBDA_CONF    = 0.5
TRAIN_LIM      = 20_000
VAL_LIM        = 2_000
GEN_TOP_K      = 50
GEN_TOP_P      = 0.95
GEN_TEMP       = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = GPT2TokenizerFast.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
VOCAB = tok.vocab_size

def keep_layer(p: float, training: bool, device):
    return True if (not training or p == 0.0) else (torch.rand(1, device=device) > p)

class TransformerBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
        self.mlp  = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.ln1(x + h)
        return self.ln2(x + self.mlp(x))

class MoE(nn.Module):
    def __init__(self, d, hidden, n_exp=4, k=2):
        super().__init__()
        self.k = k
        self.router = nn.Linear(d, n_exp)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d)) for _ in range(n_exp)])

    def forward(self, x):
        B, T, D = x.shape
        scores = self.router(x)
        top_s, top_i = scores.topk(self.k, dim=-1)
        probs = F.softmax(top_s, dim=-1)
        all_out = torch.stack([e(x) for e in self.experts], dim=2)
        gathered = torch.gather(all_out, 2, top_i.unsqueeze(-1).expand(-1, -1, -1, D))
        return (probs.unsqueeze(-1) * gathered).sum(dim=2)

class MoEBlock(nn.Module):
    def __init__(self, d, hidden, n_exp=4, k=2):
        super().__init__()
        self.tr = TransformerBlock(d)
        self.moe = MoE(d, hidden, n_exp, k)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        return self.ln(self.moe(self.tr(x)))

@dataclass
class LayerSpec:
    module: nn.Module
    drop: float

class MoETextGen(nn.Module):
    def __init__(self, vocab, d=512, L=12, n_exp=4, k=2, max_drop=MAX_DROP):
        super().__init__()
        self.token = nn.Embedding(vocab, d)
        self.pos   = nn.Embedding(1024, d)

        # gradual dropout probabilities
        specs: List[LayerSpec] = []
        for i in range(L):
            p = max(0, (i - 3) / (L - 4)) * max_drop
            specs.append(LayerSpec(MoEBlock(d, 4*d, n_exp, k), p))

        self.blocks = nn.ModuleList([s.module for s in specs])
        self.p_drop = [s.drop for s in specs]

        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)    # shared head

    def forward(self, idx, training=False, collect_exit_layers=False, tau=TAU_VAL):
        B, T = idx.shape
        h = self.token(idx) + self.pos(torch.arange(T, device=idx.device).unsqueeze(0))

        logits_layers = []
        exit_layer_tensor = None

        for i, (blk, p) in enumerate(zip(self.blocks, self.p_drop)):
            if keep_layer(p, self.training, idx.device):          # LayerSkip drop
                h = blk(h)

            # shared exit head
            logits = self.head(self.ln_f(h))
            logits_layers.append(logits)

            # early exit
            if not self.training and collect_exit_layers:
                conf = logits[:, -1, :].softmax(-1).max(-1).values  # max-prob of next token
                if exit_layer_tensor is None:
                    exit_layer_tensor = torch.full((B,), len(self.blocks) - 1, device=idx.device)
                mask = (conf > tau) & (exit_layer_tensor == len(self.blocks) - 1)
                exit_layer_tensor = torch.where(mask,torch.full_like(exit_layer_tensor, i),exit_layer_tensor)
        return logits_layers, exit_layer_tensor

# data
def get_loaders():
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def enc(b):
        ids = tok(b["text"], truncation=True, padding="max_length",
                  max_length=SEQ_LEN + 1, return_tensors="pt").input_ids
        return {"input_ids": ids[:, :-1], "labels": ids[:, 1:]}

    ds = ds.map(enc, batched=True, remove_columns=["text"])
    ds.set_format(type="torch")
    tr = ds.shuffle(seed=42).select(range(TRAIN_LIM))
    va = ds.select(range(VAL_LIM))
    return (DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(va, batch_size=BATCH_SIZE))

train_loader, val_loader = get_loaders()

# defining models
model = MoETextGen(VOCAB).to(device)
opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, len(train_loader) * EPOCHS)
CE, BCE = nn.CrossEntropyLoss(), nn.BCELoss()

# samples text
warpers = [TopKLogitsWarper(GEN_TOP_K), TopPLogitsWarper(GEN_TOP_P)]
def sample_text(prompt="In a distant future, humanity has learned to", max_new=60):
    model.eval()
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        for _ in range(max_new):
            logits, _ = model(ids, collect_exit_layers=False, tau=TAU_VAL)
            logits = logits[-1][:, -1, :] / GEN_TEMP
            for w in warpers:
                logits = w(None, logits)
            probs = logits.softmax(-1)
            nxt = torch.multinomial(probs, 1).squeeze(-1)
            if nxt.item() == tok.eos_token_id:
                break
            ids = torch.cat([ids, nxt.unsqueeze(0)], dim=1)
    return tok.decode(ids[0], skip_special_tokens=True)

# train loss, validation loss, perplexity, accuracy, average exit layer
stats = {k: [] for k in ["tr", "vl", "ppl", "acc", "exit"]}

# training
for ep in range(1, EPOCHS + 1):
    model.train()
    tot_loss = 0
    n_tok = 0
    for b in tqdm(train_loader, desc=f"Train {ep}/{EPOCHS}"):
        opt.zero_grad()
        inp, lab = b["input_ids"].to(device), b["labels"].to(device)
        outs, _ = model(inp, training=True)

        # layerskip loss
        loss_main = loss_conf = 0.0
        for log in outs:
            sl, sb = log[:, :-1, :], lab[:, :-1]
            loss_main += CE(sl.reshape(-1, VOCAB), sb.reshape(-1))

            probs = sl.softmax(-1).max(-1).values
            correct = (sl.argmax(-1) == sb).float()
            loss_conf += BCE(probs.view(-1), correct.view(-1))

        loss = (loss_main + LAMBDA_CONF * loss_conf) / len(outs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()

        tot_loss += loss.item() * inp.numel()
        n_tok    += inp.numel()
    stats["tr"].append(tot_loss / n_tok)

    # validate model
    model.eval()
    vl = vt = cor = ex_sum = ex_tok = 0
    with torch.no_grad():
        for b in tqdm(val_loader, desc=f"Val   {ep}/{EPOCHS}"):
            inp, lab = b["input_ids"].to(device), b["labels"].to(device)
            outs, ex = model(inp, collect_exit_layers=True, tau=TAU_VAL)

            log = outs[-1]
            sl, sb = log[:, :-1, :], lab[:, :-1]
            vl += CE(sl.reshape(-1, VOCAB), sb.reshape(-1)).item() * sb.numel()
            vt += sb.numel()
            cor += (sl.argmax(-1) == sb).float().sum().item()

            # get stats for layerskip
            if ex is not None:
                seq_len = (inp != tok.pad_token_id).sum(-1)
                ex_sum += (ex * seq_len).sum().item()   # sum of exit_layer * token_count
                ex_tok += seq_len.sum().item()

    stats["vl"].append(vl / vt)
    stats["ppl"].append(math.exp(vl / vt))
    stats["acc"].append(cor / vt)
    stats["exit"].append(ex_sum / ex_tok if ex_tok else len(model.blocks))

    print(f"\nEpoch {ep}: "
          f"Train={stats['tr'][-1]:.4f}  Val={stats['vl'][-1]:.4f}  "
          f"PPL={stats['ppl'][-1]:.2f}  Acc={stats['acc'][-1]*100:.1f}%  "
          f"Exit/tok={stats['exit'][-1]:.2f}\n")

    print("Sample:\n", sample_text(), "\n" + "-" * 80)

# results
plt.figure(figsize=(14, 4))
plt.subplot(131)
plt.plot(stats['tr'], label='train')
plt.plot(stats['vl'], label='val')
plt.title("loss"); plt.legend()

plt.subplot(132)
plt.plot(stats['ppl'])
plt.title("val PPL")

plt.subplot(133)
plt.plot(stats['exit'])
plt.title("avg exit / token")

plt.tight_layout()
plt.show()


# def trace_generation(prompt, max_new=7, tau=TAU_VAL):
#     model.eval()
#     ids = tok(prompt, return_tensors="pt").input_ids.to(device)
#     for step in range(max_new):
#         logits_layers, ex = model(ids, collect_exit_layers=True, tau=tau)
#         deepest = int(ex.item())
#         confs = [l[:, -1, :].softmax(-1).max().item() for l in logits_layers]
#         print(f"\nSTEP {step}  (exit @ layer {deepest})")
#         print(" ".join(f"{c:.2f}" for c in confs))
#         next_token = torch.multinomial(
#             logits_layers[deepest][:, -1, :].softmax(-1), 1
#         )
#         if next_token.item() == tok.eos_token_id:
#             break
#         ids = torch.cat([ids, next_token], dim=1)
#     print("\nFINAL TEXT:\n", tok.decode(ids[0], skip_special_tokens=True))

# trace_generation("My favorite type of car is a")
