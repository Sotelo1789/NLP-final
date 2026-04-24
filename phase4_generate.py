"""
CSCI 182.06 — Natural Language Processing Final Project
Phase 4: Text Generation & Evaluation
Author Dataset: Sabrina Carpenter (Top 50 Songs)

Loads the trained attention model and generates lyrics using:
  - Greedy decoding   (always picks the highest-probability word)
  - Temperature       (controls randomness — low=safe, high=creative)
  - Top-k sampling    (only sample from the k most likely next words)

All generated samples are saved to dataset/generated_samples.txt
"""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

DATASET_DIR  = "dataset"
MODEL_PATH   = f"{DATASET_DIR}/model_attention.pt"
VOCAB_PATH   = f"{DATASET_DIR}/vocab.json"
OUTPUT_PATH  = f"{DATASET_DIR}/generated_samples.txt"

SEQ_LENGTH   = 10    # must match Phase 2/3
EMBED_DIM    = 64
NUM_HEADS    = 4
FF_DIM       = 128
DROPOUT      = 0.1
GENERATE_LEN = 40   # words to generate per sample

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────
# 1. LOAD VOCAB
# ──────────────────────────────────────────────────────────────────────

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

word2idx  = vocab["word2idx"]
idx2word  = {int(k): v for k, v in vocab["idx2word"].items()}
VOCAB_SIZE = len(word2idx)
print(f"Vocab size: {VOCAB_SIZE:,}")

# ──────────────────────────────────────────────────────────────────────
# 2. MODEL DEFINITION (mirrors phase3_model.py)
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
        out  = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, -1)
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
# 3. LOAD TRAINED WEIGHTS
# ──────────────────────────────────────────────────────────────────────

model = LyricsAttentionModel(
    vocab_size = VOCAB_SIZE,
    embed_dim  = EMBED_DIM,
    seq_len    = SEQ_LENGTH,
    num_heads  = NUM_HEADS,
    ff_dim     = FF_DIM,
    dropout    = DROPOUT,
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print(f"Loaded model from {MODEL_PATH}\n")

# ──────────────────────────────────────────────────────────────────────
# 4. GENERATION HELPERS
# ──────────────────────────────────────────────────────────────────────

def encode_seed(seed_phrase: str, seq_len: int) -> list[int]:
    """
    Convert a seed string to a list of SEQ_LENGTH token IDs.
    Pads with <UNK> (0) on the left if the phrase is shorter than seq_len.
    Trims to the last seq_len tokens if longer.
    """
    tokens = seed_phrase.lower().split()[-seq_len:]
    ids    = [word2idx.get(t, 0) for t in tokens]
    ids    = [0] * (seq_len - len(ids)) + ids   # left-pad
    return ids


def generate(
    seed_phrase : str,
    n_words     : int   = GENERATE_LEN,
    temperature : float = 1.0,
    top_k       : int   = 0,
) -> str:
    """
    Generate n_words of text from a seed phrase.

    temperature:
        < 1.0  →  more focused / repetitive (model sticks to high-prob words)
        = 1.0  →  sample directly from the model's distribution
        > 1.0  →  more creative / random (model spreads probability more evenly)

    top_k:
        0      →  sample from the full vocabulary distribution
        k > 0  →  keep only the top-k most likely words before sampling
                  (prevents very unlikely words from ever being chosen)
    """
    ids = encode_seed(seed_phrase, SEQ_LENGTH)

    generated = []
    with torch.no_grad():
        for _ in range(n_words):
            x_in = torch.tensor([ids], dtype=torch.long).to(DEVICE)  # (1, seq_len)
            logits = model(x_in)                                       # (1, vocab_size)

            # ── Apply temperature ────────────────────────────────────
            # Dividing logits by temperature before softmax:
            #   low T  → logits become more extreme → winner-takes-all
            #   high T → logits flatten out → more uniform distribution
            logits = logits / temperature

            # ── Apply top-k filtering ────────────────────────────────
            # Zero out all logits except the top-k, forcing the model
            # to only ever sample from the k most probable next words
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold      = top_k_vals[:, -1].unsqueeze(-1)  # kth largest value
                logits         = logits.masked_fill(logits < threshold, float("-inf"))

            # ── Sample from distribution ─────────────────────────────
            probs   = F.softmax(logits, dim=-1)           # convert to probabilities
            next_id = torch.multinomial(probs, 1).item()  # sample one token

            generated.append(idx2word[next_id])
            ids = ids[1:] + [next_id]   # slide context window forward by 1

    return " ".join(generated)

# ──────────────────────────────────────────────────────────────────────
# 5. RUN EXPERIMENTS WITH DIFFERENT SETTINGS
# ──────────────────────────────────────────────────────────────────────

seeds = [
    "i know that you",
    "baby please don't",
    "late at night i'm thinking",
]

experiments = [
    {"label": "Greedy (temp=0.5, top_k=1)",    "temperature": 0.5, "top_k": 1},
    {"label": "Focused (temp=0.7, top_k=10)",   "temperature": 0.7, "top_k": 10},
    {"label": "Balanced (temp=1.0, top_k=20)",  "temperature": 1.0, "top_k": 20},
    {"label": "Creative (temp=1.3, top_k=40)",  "temperature": 1.3, "top_k": 40},
    {"label": "Wild (temp=1.6, top_k=0)",       "temperature": 1.6, "top_k": 0},
]

output_lines = []
output_lines.append("=" * 65)
output_lines.append("PHASE 4 — GENERATED LYRICS SAMPLES")
output_lines.append("Sabrina Carpenter Lyrics Language Model")
output_lines.append("=" * 65)

for seed in seeds:
    output_lines.append(f"\nSEED: \"{seed}\"")
    output_lines.append("-" * 65)
    for exp in experiments:
        result = generate(
            seed_phrase = seed,
            n_words     = GENERATE_LEN,
            temperature = exp["temperature"],
            top_k       = exp["top_k"],
        )
        output_lines.append(f"\n[{exp['label']}]")
        output_lines.append(f"{seed} {result}")

output_lines.append("\n" + "=" * 65)

# ──────────────────────────────────────────────────────────────────────
# 6. PRINT & SAVE
# ──────────────────────────────────────────────────────────────────────

full_output = "\n".join(output_lines)
print(full_output)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(full_output)

print(f"\nSaved generated samples → {OUTPUT_PATH}")
