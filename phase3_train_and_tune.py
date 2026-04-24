"""
CSCI 182.06 — Natural Language Processing Final Project
Phase 3: Hyperparameter Tuning
Author Dataset: Sabrina Carpenter (Top 50 Songs)

Runs a grid search over key hyperparameters using LyricsAttentionModel
(the full attention architecture from phase3_model.py).
Results are saved to dataset/tuning_results.csv.
"""

import csv
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

DATASET_DIR = "dataset"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameter search space
# NOTE: num_heads must divide embed_dim evenly — all combos below satisfy this
HYPERPARAM_GRID = {
    "embed_dim"    : [64, 128],
    "num_heads"    : [2, 4],      # 2 and 4 both divide 64 and 128
    "ff_dim"       : [128, 256],
    "learning_rate": [1e-3, 5e-4],
    "batch_size"   : [64],
    "epochs"       : [15],
}

# ──────────────────────────────────────────────────────────────────────
# 1. LOAD PHASE 2 OUTPUTS
# ──────────────────────────────────────────────────────────────────────

with open(f"{DATASET_DIR}/vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)

VOCAB_SIZE = len(vocab["word2idx"])

data = torch.load(f"{DATASET_DIR}/dataset.pt", weights_only=True)
X, Y       = data["X"], data["Y"]
SEQ_LENGTH = data["seq_len"]

print(f"Vocab size: {VOCAB_SIZE:,}  |  Dataset: {len(Y):,} pairs  |  Seq len: {SEQ_LENGTH}")

# ──────────────────────────────────────────────────────────────────────
# 2. DATASET
# ──────────────────────────────────────────────────────────────────────

class LyricsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ──────────────────────────────────────────────────────────────────────
# 3. MODEL (same architecture as phase3_model.py)
# ──────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.q_proj    = nn.Linear(embed_dim, embed_dim)
        self.k_proj    = nn.Linear(embed_dim, embed_dim)
        self.v_proj    = nn.Linear(embed_dim, embed_dim)
        self.out_proj  = nn.Linear(embed_dim, embed_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, E = x.size()
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out  = torch.matmul(attn, V)
        out  = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention    = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1        = nn.LayerNorm(embed_dim)
        self.norm2        = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.attention(x, mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class LyricsAttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        self.transformer_block  = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        self.dropout            = nn.Dropout(dropout)
        self.output_layer       = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.size()
        pos  = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        out  = self.dropout(self.token_embedding(x) + self.position_embedding(pos))
        out  = self.transformer_block(out)
        out  = out.mean(dim=1)
        return self.output_layer(out)

# ──────────────────────────────────────────────────────────────────────
# 4. TRAIN FUNCTION
# ──────────────────────────────────────────────────────────────────────

def train_model(config):
    dataloader = DataLoader(
        LyricsDataset(X, Y),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    model = LyricsAttentionModel(
        vocab_size = VOCAB_SIZE,
        embed_dim  = config["embed_dim"],
        seq_len    = SEQ_LENGTH,
        num_heads  = config["num_heads"],
        ff_dim     = config["ff_dim"],
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch:>2}/{config['epochs']}  |  Loss: {avg_loss:.4f}")

    return avg_loss

# ──────────────────────────────────────────────────────────────────────
# 5. GRID SEARCH
# ──────────────────────────────────────────────────────────────────────

results = []
keys    = list(HYPERPARAM_GRID.keys())

for combo in product(*HYPERPARAM_GRID.values()):
    config = dict(zip(keys, combo))
    print(f"\n{'='*50}")
    print(f"Experiment: {config}")
    final_loss = train_model(config)
    results.append({**config, "final_loss": round(final_loss, 4)})

# ──────────────────────────────────────────────────────────────────────
# 6. SAVE RESULTS & REPORT BEST
# ──────────────────────────────────────────────────────────────────────

results_path = os.path.join(DATASET_DIR, "tuning_results.csv")
with open(results_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved tuning results → {results_path}")

best = min(results, key=lambda r: r["final_loss"])
print(f"\nBest config  : {best}")
print(f"Best loss    : {best['final_loss']}")
