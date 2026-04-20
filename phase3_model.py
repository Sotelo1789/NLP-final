"""
CSCI 182.06 — Natural Language Processing Final Project
Phase 3: Model Architecture & Training
Author Dataset: Sabrina Carpenter (Top 50 Songs)

Architecture:
  Embedding  →  Flatten  →  MLP (hidden layers + ReLU)  →  Linear  →  vocab_size logits

This script loads the Phase 2 outputs from dataset/ and trains the model.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

DATASET_DIR  = "dataset"
EMBED_DIM    = 64      # size of each word embedding vector
HIDDEN_DIM   = 128     # neurons in each MLP hidden layer
SEQ_LENGTH   = 10      # must match Phase 2
BATCH_SIZE   = 64
EPOCHS       = 20
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ──────────────────────────────────────────────────────────────────────
# 1. LOAD PHASE 2 OUTPUTS
# ──────────────────────────────────────────────────────────────────────

with open(f"{DATASET_DIR}/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

word2idx = vocab["word2idx"]
idx2word = {int(k): v for k, v in vocab["idx2word"].items()}
VOCAB_SIZE = len(word2idx)
print(f"Vocab size: {VOCAB_SIZE:,}")

data = torch.load(f"{DATASET_DIR}/dataset.pt", weights_only=True)
X, Y = data["X"], data["Y"]
print(f"Dataset: {X.shape[0]:,} pairs  |  X: {X.shape}  Y: {Y.shape}")

# ──────────────────────────────────────────────────────────────────────
# 2. DATASET & DATALOADER
# ──────────────────────────────────────────────────────────────────────

class LyricsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


dataset    = LyricsDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ──────────────────────────────────────────────────────────────────────
# 3. MODEL — EMBEDDING + MLP
# ──────────────────────────────────────────────────────────────────────

class LyricsModel(nn.Module):
    """
    Embedding layer turns token IDs into dense vectors.
    All SEQ_LENGTH vectors are flattened and passed through two MLP hidden
    layers before a final projection to vocab_size logits.
    """

    def __init__(self, vocab_size: int, embed_dim: int, seq_len: int, hidden_dim: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # input to MLP = flattened embeddings: seq_len * embed_dim
        input_dim = seq_len * embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)              # (batch, seq_len, embed_dim)
        flat     = embedded.view(x.size(0), -1)   # (batch, seq_len * embed_dim)
        hidden   = self.mlp(flat)                 # (batch, hidden_dim)
        logits   = self.output_layer(hidden)      # (batch, vocab_size)
        return logits


model = LyricsModel(VOCAB_SIZE, EMBED_DIM, SEQ_LENGTH, HIDDEN_DIM).to(DEVICE)
print(f"\nModel architecture:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ──────────────────────────────────────────────────────────────────────
# 4. TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nTraining for {EPOCHS} epochs...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch:>3}/{EPOCHS}  |  Loss: {avg_loss:.4f}")

# ──────────────────────────────────────────────────────────────────────
# 5. SAVE MODEL
# ──────────────────────────────────────────────────────────────────────

torch.save(model.state_dict(), f"{DATASET_DIR}/model.pt")
print(f"\nModel saved to {DATASET_DIR}/model.pt")
