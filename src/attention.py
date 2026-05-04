"""
attention.py — Self-attention and cross-attention in one module.

Reading this file in order gives the complete picture:
  1. scaled_dot_product_attention() — the shared math kernel
  2. SelfAttention                  — Q, K, V all from the same sequence
  3. CrossAttention                 — Q from target, K/V from source
  4. MultiHeadSelfAttention         — parallel heads, self
  5. MultiHeadCrossAttention        — parallel heads, cross

The ONLY difference between self and cross attention:
  Self:   Q = W_Q(x),  K = W_K(x),  V = W_V(x)      one input
  Cross:  Q = W_Q(x),  K = W_K(c),  V = W_V(c)      two inputs (x=target, c=context/source)

Everything else — the scaled dot-product, softmax, weighted sum,
multi-head splitting, output projection — is identical.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Shared kernel
# ─────────────────────────────────────────────────────────────────────────────

def scaled_dot_product_attention(
    Q    : torch.Tensor,           # (batch, heads, q_len, d_k)
    K    : torch.Tensor,           # (batch, heads, k_len, d_k)
    V    : torch.Tensor,           # (batch, heads, k_len, d_v)
    mask : Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention.

    This is the exact same function for both self and cross attention.
    The difference is only in WHERE Q, K, V come from before this call.

    For self-attention:    Q, K, V all derived from the same sequence
    For cross-attention:   Q derived from target; K, V from source

    Returns:
        output:  (batch, heads, q_len, d_v)
        weights: (batch, heads, q_len, k_len)  ← this is the alignment matrix
    """
    d_k    = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)

    return torch.matmul(weights, V), weights


# ─────────────────────────────────────────────────────────────────────────────
# Single-head versions (explicit, for educational clarity)
# ─────────────────────────────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    """
    Single-head self-attention.

    One sequence in → context-aware representation out.
    Every position can attend to every other position in the SAME sequence.

    Used in: encoder layers, decoder self-attention layers.

    Args:
        emb_dim (int): input/output dimension
        dropout (float): attention dropout
    """

    def __init__(self, emb_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.W_Q  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_K  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_V  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_O  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x    : torch.Tensor,                  # (batch, seq_len, emb_dim)
        mask : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        Q = self.W_Q(x).unsqueeze(1)  # (B, 1, T, D)
        K = self.W_K(x).unsqueeze(1)
        V = self.W_V(x).unsqueeze(1)

        out, w = scaled_dot_product_attention(Q, K, V, mask)
        out    = self.drop(out.squeeze(1))    # (B, T, D)
        return self.W_O(out), w.squeeze(1)    # (B, T, D), (B, T, T)


class CrossAttention(nn.Module):
    """
    Single-head cross-attention.

    TWO sequences in:
        x       = target sequence  (provides Queries)
        context = source sequence  (provides Keys and Values)

    Each target position asks: "which source positions are most useful to me?"
    The answer is the attention weight vector for that position.

    Used in: encoder-decoder attention in translation, summarisation, etc.

    Key insight:
        The target sequence drives WHAT IS BEING ASKED (Q).
        The source sequence drives WHAT IS AVAILABLE TO ANSWER (K, V).

    Args:
        emb_dim (int): dimension for both sequences (must match)
        dropout (float): attention dropout
    """

    def __init__(self, emb_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.W_Q  = nn.Linear(emb_dim, emb_dim, bias=False)  # projects target
        self.W_K  = nn.Linear(emb_dim, emb_dim, bias=False)  # projects source
        self.W_V  = nn.Linear(emb_dim, emb_dim, bias=False)  # projects source
        self.W_O  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x       : torch.Tensor,                  # (batch, tgt_len, emb_dim) — target
        context : torch.Tensor,                  # (batch, src_len, emb_dim) — source
        mask    : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:       target sequence  (batch, tgt_len, emb_dim)
            context: source sequence  (batch, src_len, emb_dim)
            mask:    optional padding mask on the source  (batch, 1, 1, src_len)

        Returns:
            output:  (batch, tgt_len, emb_dim)
                     each target position enriched with relevant source info
            weights: (batch, tgt_len, src_len)
                     ← THIS IS THE ALIGNMENT MATRIX
                     weights[b, i, j] = how much target pos i attended to source pos j
        """
        # Q comes from TARGET  — what each output position is looking for
        Q = self.W_Q(x).unsqueeze(1)         # (B, 1, tgt_len, D)

        # K and V come from SOURCE  — what the encoder has available
        K = self.W_K(context).unsqueeze(1)   # (B, 1, src_len, D)
        V = self.W_V(context).unsqueeze(1)   # (B, 1, src_len, D)

        out, w = scaled_dot_product_attention(Q, K, V, mask)
        out    = self.drop(out.squeeze(1))    # (B, tgt_len, D)
        return self.W_O(out), w.squeeze(1)    # weights: (B, tgt_len, src_len)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-head versions
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention.

    Runs h self-attention heads in parallel. Each head can specialise
    in a different type of relationship within the sequence.

    Args:
        emb_dim (int): model dimension (must be divisible by n_heads)
        n_heads (int): number of parallel heads
        dropout (float): attention dropout
    """

    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert emb_dim % n_heads == 0
        self.emb_dim  = emb_dim
        self.n_heads  = n_heads
        self.head_dim = emb_dim // n_heads

        self.W_Q  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_K  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_V  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_O  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x    : torch.Tensor,
        mask : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        Q = self._split(self.W_Q(x), B, T)
        K = self._split(self.W_K(x), B, T)
        V = self._split(self.W_V(x), B, T)
        out, w = scaled_dot_product_attention(Q, K, V, mask)
        out    = self.drop(out)
        merged = out.transpose(1, 2).contiguous().view(B, T, self.emb_dim)
        return self.W_O(merged), w

    def _split(self, x, B, T):
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def __repr__(self):
        return (f"MultiHeadSelfAttention(emb_dim={self.emb_dim}, "
                f"n_heads={self.n_heads}, head_dim={self.head_dim})")


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention.

    Same as MultiHeadSelfAttention but takes TWO sequences.
    Q from target, K/V from source — split across h heads.

    The weight matrix has shape (batch, n_heads, tgt_len, src_len).
    Reading head h, row i: which source positions did target position i
    attend to, according to head h?

    Args:
        emb_dim (int): model dimension (must be divisible by n_heads)
        n_heads (int): number of parallel heads
        dropout (float): attention dropout
    """

    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert emb_dim % n_heads == 0
        self.emb_dim  = emb_dim
        self.n_heads  = n_heads
        self.head_dim = emb_dim // n_heads

        self.W_Q  = nn.Linear(emb_dim, emb_dim, bias=False)  # target → Q
        self.W_K  = nn.Linear(emb_dim, emb_dim, bias=False)  # source → K
        self.W_V  = nn.Linear(emb_dim, emb_dim, bias=False)  # source → V
        self.W_O  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x       : torch.Tensor,               # (batch, tgt_len, emb_dim)
        context : torch.Tensor,               # (batch, src_len, emb_dim)
        mask    : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output:  (batch, tgt_len, emb_dim)
            weights: (batch, n_heads, tgt_len, src_len)  ← alignment matrix per head
        """
        B, T_tgt, _ = x.shape
        B, T_src, _ = context.shape

        Q = self._split_tgt(self.W_Q(x),       B, T_tgt)
        K = self._split_src(self.W_K(context), B, T_src)
        V = self._split_src(self.W_V(context), B, T_src)

        out, w = scaled_dot_product_attention(Q, K, V, mask)
        out    = self.drop(out)
        merged = out.transpose(1, 2).contiguous().view(B, T_tgt, self.emb_dim)
        return self.W_O(merged), w

    def _split_tgt(self, x, B, T):
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _split_src(self, x, B, T):
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def __repr__(self):
        return (f"MultiHeadCrossAttention(emb_dim={self.emb_dim}, "
                f"n_heads={self.n_heads}, head_dim={self.head_dim})")
