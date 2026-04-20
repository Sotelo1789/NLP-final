"""
CSCI 182.06 — Natural Language Processing Final Project
Phase 2: Vocabulary & Dataset Construction
Author Dataset: Sabrina Carpenter (Top 50 Songs)

This script:
1. Builds word2idx / idx2word from cleaned_lyrics.txt
2. Creates (X, Y) sliding-window pairs  →  X = seq of token IDs, Y = next token ID
3. Wraps everything in a PyTorch Dataset + DataLoader
4. Saves tokenized outputs to dataset/ for use in Phase 3+
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

CLEANED_TEXT_PATH = "cleaned_lyrics.txt"
OUTPUT_DIR        = "dataset"
SEQ_LENGTH        = 10    # tokens fed as context (X)
BATCH_SIZE        = 64
MIN_FREQ          = 1     # minimum occurrences to keep a word

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 1. LOAD CLEANED TEXT
# ──────────────────────────────────────────────────────────────────────

with open(CLEANED_TEXT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

tokens = text.split()
print(f"Total tokens in corpus: {len(tokens):,}")

# ──────────────────────────────────────────────────────────────────────
# 2. BUILD VOCABULARY
# ──────────────────────────────────────────────────────────────────────

freq = Counter(tokens)

# Reserve index 0 for <UNK> so rare/unseen words have a valid ID
word2idx = {"<UNK>": 0}
for word, count in sorted(freq.items()):          # sorted → deterministic order
    if count >= MIN_FREQ and word not in word2idx:
        word2idx[word] = len(word2idx)

idx2word = {idx: word for word, idx in word2idx.items()}

VOCAB_SIZE = len(word2idx)
print(f"Vocabulary size (incl. <UNK>): {VOCAB_SIZE:,}")

# ──────────────────────────────────────────────────────────────────────
# 3. ENCODE FULL CORPUS
# ──────────────────────────────────────────────────────────────────────

encoded = [word2idx.get(t, 0) for t in tokens]   # unknown words → 0

# ──────────────────────────────────────────────────────────────────────
# 4. BUILD SLIDING-WINDOW (X, Y) PAIRS
# ──────────────────────────────────────────────────────────────────────

class LyricsDataset(Dataset):
    """
    Sliding window over an encoded token sequence.
    X: SEQ_LENGTH consecutive token IDs
    Y: the single token ID that follows
    """

    def __init__(self, encoded_tokens: list[int], seq_len: int):
        self.seq_len = seq_len
        xs, ys = [], []
        for i in range(len(encoded_tokens) - seq_len):
            xs.append(encoded_tokens[i : i + seq_len])
            ys.append(encoded_tokens[i + seq_len])
        self.X = torch.tensor(xs, dtype=torch.long)
        self.Y = torch.tensor(ys, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


dataset = LyricsDataset(encoded, SEQ_LENGTH)
print(f"Total (X, Y) pairs: {len(dataset):,}")
print(f"  X shape: {dataset.X.shape}  (num_pairs × seq_len)")
print(f"  Y shape: {dataset.Y.shape}  (num_pairs,)")

# ──────────────────────────────────────────────────────────────────────
# 5. DATALOADER
# ──────────────────────────────────────────────────────────────────────

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"\nDataLoader ready")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  Batches/epoch: {len(dataloader):,}")

# ── quick sanity check ────────────────────────────────────────────────
sample_x, sample_y = next(iter(dataloader))
print(f"\nSample batch shapes → X: {sample_x.shape}, Y: {sample_y.shape}")
print("Sample X tokens :", sample_x[0].tolist())
print("Sample X words  :", [idx2word[i.item()] for i in sample_x[0]])
print("Sample Y token  :", sample_y[0].item(), "→", idx2word[sample_y[0].item()])

# ──────────────────────────────────────────────────────────────────────
# 6. SAVE TOKENIZED OUTPUTS
# ──────────────────────────────────────────────────────────────────────

# vocab dicts  →  dataset/vocab.json
vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump({"word2idx": word2idx, "idx2word": idx2word}, f, ensure_ascii=False, indent=2)
print(f"\nSaved vocab to {vocab_path}")

# encoded token IDs  →  dataset/encoded_tokens.json
encoded_path = os.path.join(OUTPUT_DIR, "encoded_tokens.json")
with open(encoded_path, "w") as f:
    json.dump(encoded, f)
print(f"Saved encoded tokens to {encoded_path}")

# X / Y tensors  →  dataset/dataset.pt
tensor_path = os.path.join(OUTPUT_DIR, "dataset.pt")
torch.save({"X": dataset.X, "Y": dataset.Y, "seq_len": SEQ_LENGTH}, tensor_path)
print(f"Saved X/Y tensors to {tensor_path}")

print(f"\nAll Phase 2 outputs saved to '{OUTPUT_DIR}/':")
print(f"  vocab.json          — word2idx & idx2word ({VOCAB_SIZE:,} tokens)")
print(f"  encoded_tokens.json — full corpus as token IDs ({len(encoded):,} tokens)")
print(f"  dataset.pt          — X/Y tensors ready for model training")
