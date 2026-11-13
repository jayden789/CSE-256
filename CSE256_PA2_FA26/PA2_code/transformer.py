# add all  your Encoder and Decoder code here

import math
import torch
from torch import nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, n_embd, head_size, causal):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.causal = causal

    def forward(self, x, key_mask=None):
        T = x.shape[1]
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
        if self.causal:
            future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(future, float('-inf'))
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, :], float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return attn @ v, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, causal):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, n_embd // n_head, causal) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, key_mask=None):
        outs, maps = zip(*[h(x, key_mask) for h in self.heads])
        return self.proj(torch.cat(outs, dim=-1)), list(maps)


class FeedForward(nn.Module):
    def __init__(self, n_embd, n_hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_embd))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_ff_hidden, causal):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, causal)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, n_ff_hidden)

    def forward(self, x, key_mask=None):
        a, maps = self.attn(self.ln1(x), key_mask)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, maps


class Embeddings(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
        positions = torch.arange(idx.shape[1], device=idx.device)
        return self.tok(idx) + self.pos(positions)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, n_ff_hidden=100, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = Embeddings(vocab_size, n_embd, block_size)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, n_ff_hidden, causal=False) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, idx):
        key_mask = idx != self.pad_idx
        x = self.embed(idx)
        attn_maps = []
        for block in self.blocks:
            x, maps = block(x, key_mask)
            attn_maps.extend(maps)
        return self.ln_f(x), attn_maps


class EncoderClassifier(nn.Module):
    def __init__(self, encoder, n_input, n_hidden, n_output):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(nn.Linear(n_input, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output))

    def forward(self, idx):
        h, _ = self.encoder(idx)
        mask = (idx != self.encoder.pad_idx).unsqueeze(-1).float()
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.classifier(mean)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, n_ff_hidden=100):
        super().__init__()
        self.embed = Embeddings(vocab_size, n_embd, block_size)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, n_ff_hidden, causal=True) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """Returns the mean cross-entropy loss if targets are given, otherwise (logits, attention maps)."""
        x = self.embed(idx)
        attn_maps = []
        for block in self.blocks:
            x, maps = block(x)
            attn_maps.extend(maps)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, attn_maps
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
