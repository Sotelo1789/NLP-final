"""
CSCI 182.06 — Natural Language Processing Final Project
Phase 3 (Revised): Model Architecture & Training — with Multi-Head Attention
Author Dataset: Sabrina Carpenter (Top 50 Songs)

Architecture:
  Embedding  →  Multi-Head Self-Attention  →  LayerNorm  →  Feed-Forward  →  LayerNorm
             →  Mean Pooling over sequence  →  Linear (vocab_size logits)

Setting NUM_HEADS = 1 gives plain self-attention.
Setting NUM_HEADS > 1 (e.g. 4 or 8) gives multi-head attention.
embed_dim must be divisible by NUM_HEADS.
"""

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

DATASET_DIR   = "dataset"
EMBED_DIM     = 64       # must be divisible by NUM_HEADS
NUM_HEADS     = 4        # set to 1 for plain self-attention, >1 for multi-head
FF_DIM        = 128      # feed-forward hidden size inside the transformer block
DROPOUT       = 0.1      # dropout rate for regularization
SEQ_LENGTH    = 10       # must match Phase 2
BATCH_SIZE    = 64
EPOCHS        = 20
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"Attention mode: {'Multi-Head' if NUM_HEADS > 1 else 'Self'}-Attention  "
      f"({NUM_HEADS} head{'s' if NUM_HEADS > 1 else ''})")

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
# 3. MULTI-HEAD SELF-ATTENTION MODULE
# ──────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product attention with multiple heads.

    When num_heads=1 this is equivalent to plain self-attention.
    When num_heads>1 each head learns to attend to different parts of
    the context, then all heads are concatenated and projected back.

    Args:
        embed_dim : total embedding / model dimension
        num_heads : number of parallel attention heads
                    (embed_dim must be divisible by num_heads)

    Input  shape: (batch, seq_len, embed_dim)
    Output shape: (batch, seq_len, embed_dim)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads   # dimension per individual head

        # Linear projections for Q, K, V
        self.q_proj  = nn.Linear(embed_dim, embed_dim)
        self.k_proj  = nn.Linear(embed_dim, embed_dim)
        self.v_proj  = nn.Linear(embed_dim, embed_dim)

        # Final output projection that merges all heads back to embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x    : (batch, seq_len, embed_dim)
        mask : (batch, 1, 1, seq_len) or None
        """
        batch_size, seq_len, _ = x.size()

        # ── Project inputs to Q, K, V ────────────────────────────────
        Q = self.q_proj(x)   # (batch, seq_len, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # ── Reshape for multi-head: split embed_dim across heads ─────
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose → (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # ── Scaled dot-product attention ─────────────────────────────
        # scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)   # attention distribution over positions
        attn_weights = self.dropout(attn_weights)

        # weighted sum of values: (batch, num_heads, seq_len, head_dim)
        out = torch.matmul(attn_weights, V)

        # ── Merge heads back ─────────────────────────────────────────
        out = out.transpose(1, 2).contiguous()           # (batch, seq_len, num_heads, head_dim)
        out = out.view(batch_size, seq_len, self.embed_dim)  # (batch, seq_len, embed_dim)

        # Final projection
        out = self.out_proj(out)
        return out


# ──────────────────────────────────────────────────────────────────────
# 4. TRANSFORMER BLOCK
#    Attention → Add & Norm → Feed-Forward → Add & Norm
# ──────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One encoder-style transformer block:
        1. Multi-Head Self-Attention
        2. Residual connection + LayerNorm
        3. Position-wise Feed-Forward (Linear → ReLU → Linear)
        4. Residual connection + LayerNorm
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1      = nn.LayerNorm(embed_dim)
        self.norm2      = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Sub-layer 1: self-attention with residual
        attended = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attended))

        # Sub-layer 2: feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


# ──────────────────────────────────────────────────────────────────────
# 5. FULL LYRICS LANGUAGE MODEL
# ──────────────────────────────────────────────────────────────────────

class LyricsAttentionModel(nn.Module):
    """
    Token Embedding  +  Positional Embedding
        ↓
    TransformerBlock  (Multi-Head Self-Attention + FFN + residuals)
        ↓
    Mean pooling over the sequence dimension
        ↓
    Linear → vocab_size logits
    (probability distribution over the next token after softmax)
    """

    def __init__(
        self,
        vocab_size : int,
        embed_dim  : int,
        seq_len    : int,
        num_heads  : int,
        ff_dim     : int,
        dropout    : float = 0.1,
    ):
        super().__init__()

        # Maps each token ID → a dense vector of size embed_dim
        self.token_embedding    = nn.Embedding(vocab_size, embed_dim)

        # Learned positional embeddings: one vector per sequence position
        self.position_embedding = nn.Embedding(seq_len, embed_dim)

        # Core transformer block (attention + feed-forward + norms)
        self.transformer_block  = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)

        self.dropout = nn.Dropout(dropout)

        # ── Final linear layer (required by spec) ────────────────────
        # Projects the pooled context vector to vocab_size logits.
        # After applying softmax these logits become a probability
        # distribution representing the likelihood of each word being
        # the next token — matching the y ground-truth one-hot format.
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        """
        x      : (batch, seq_len)  — token IDs
        returns: logits (batch, vocab_size)
        """
        batch_size, seq_len = x.size()

        # Position indices [0, 1, ..., seq_len-1] broadcasted across batch
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # Token + positional embeddings combined
        out = self.dropout(
            self.token_embedding(x) + self.position_embedding(positions)
        )                                          # (batch, seq_len, embed_dim)

        # Transformer block: attention over all positions in the context
        out = self.transformer_block(out)          # (batch, seq_len, embed_dim)

        # Mean pooling: average across the sequence to get one context vector
        out = out.mean(dim=1)                      # (batch, embed_dim)

        # Project to vocabulary size → raw logits (not yet softmaxed)
        logits = self.output_layer(out)            # (batch, vocab_size)
        return logits


# ──────────────────────────────────────────────────────────────────────
# 6. INSTANTIATE MODEL
# ──────────────────────────────────────────────────────────────────────

model = LyricsAttentionModel(
    vocab_size = VOCAB_SIZE,
    embed_dim  = EMBED_DIM,
    seq_len    = SEQ_LENGTH,
    num_heads  = NUM_HEADS,
    ff_dim     = FF_DIM,
    dropout    = DROPOUT,
).to(DEVICE)

print(f"\nModel architecture:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ──────────────────────────────────────────────────────────────────────
# 7. TRAINING LOOP
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
        logits = model(x_batch)            # (batch, vocab_size)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch:>3}/{EPOCHS}  |  Loss: {avg_loss:.4f}")

# ──────────────────────────────────────────────────────────────────────
# 8. SAVE MODEL
# ──────────────────────────────────────────────────────────────────────

os.makedirs(DATASET_DIR, exist_ok=True)
torch.save(model.state_dict(), f"{DATASET_DIR}/model_attention.pt")
print(f"\nModel saved to {DATASET_DIR}/model_attention.pt")

# ──────────────────────────────────────────────────────────────────────
# 9. SPEC COMPLIANCE CHECK — output shape
# ──────────────────────────────────────────────────────────────────────

model.eval()
sample_x = X[:4].to(DEVICE)
logits = model(sample_x)
print(f"\nOutput shape check: {logits.shape}")
print(f"Expected:           torch.Size([4, {VOCAB_SIZE}])")
assert logits.shape == (4, VOCAB_SIZE), "Shape mismatch — check output_layer!"
print("Shape check PASSED")

# ──────────────────────────────────────────────────────────────────────
# 10. GENERATION PREVIEW
# ──────────────────────────────────────────────────────────────────────

print("\n--- Generation Preview ---")

model.eval()
seed_phrase = "i don't wanna be"          # change this to any phrase you like
seed_tokens = seed_phrase.lower().split()

# take only the last SEQ_LENGTH words as context
context = seed_tokens[-SEQ_LENGTH:]

# convert to IDs, padding with 0 (<UNK>) if shorter than SEQ_LENGTH
ids = [word2idx.get(w, 0) for w in context]
ids = [0] * (SEQ_LENGTH - len(ids)) + ids   # left-pad if needed

print(f"Seed: '{seed_phrase}'")
print("Generated: ", end="")

generated_words = []
for _ in range(20):                        # generate 20 words
    x_in = torch.tensor([ids]).to(DEVICE)
    with torch.no_grad():
        logits = model(x_in)
    next_id = torch.argmax(logits, dim=-1).item()
    word = idx2word[next_id]
    print(word, end=" ")
    generated_words.append(word)
    ids = ids[1:] + [next_id]             # slide the window forward

print("\n--------------------------")