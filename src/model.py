"""
model.py — Minimal encoder-decoder with cross-attention.

Architecture:

    SOURCE sequence
        │
        ▼
    ┌─────────────────────────────────┐
    │  Encoder                        │
    │  Embedding + PE                 │
    │  MultiHeadSelfAttention         │  source attends to itself
    │  FeedForward                    │
    └─────────────────────────────────┘
        │  encoder_output  (batch, src_len, emb_dim)
        │
        │  ┌─────────────────────────────────┐
        │  │  Decoder                        │
        │  │  Embedding + PE                 │
        │  │  MultiHeadSelfAttention (causal)│  target attends to past target tokens
        │  │                                 │
        └──┤  MultiHeadCrossAttention        │  ← THE NEW PIECE
           │    Q = decoder state            │    target queries the encoder output
           │    K = encoder output           │
           │    V = encoder output           │
           │  FeedForward                    │
           └─────────────────────────────────┘
                │
                ▼
            Linear → vocab logits

The cross-attention layer is where the decoder "reads" the encoder.
Its weight matrix (tgt_len × src_len) is the alignment:
entry [i, j] tells you how much target position i attended to source position j.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List

from src.attention import MultiHeadSelfAttention, MultiHeadCrossAttention
from src.dataset   import VOCAB_SIZE, PAD_IDX


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""
    def __init__(self, emb_dim: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, emb_dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, emb_dim, 2).float()
                        * -(math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or 4 * emb_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, ff_dim), nn.GELU(),
            nn.Linear(ff_dim, emb_dim), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


# ── Encoder block ─────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """
    One encoder block: self-attention + feedforward.
    No masking — the encoder sees the full source sequence bidirectionally.
    """
    def __init__(self, emb_dim, n_heads, ff_dim=None, dropout=0.1):
        super().__init__()
        self.attn  = MultiHeadSelfAttention(emb_dim, n_heads, dropout)
        self.ff    = FeedForward(emb_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out, _ = self.attn(self.norm1(x), src_mask)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class Encoder(nn.Module):
    """
    Encodes a source sequence into a rich contextual representation.
    The decoder will read this representation via cross-attention.
    """
    def __init__(self, vocab_size, emb_dim, n_heads, n_layers,
                 ff_dim=None, max_len=64, dropout=0.1, pad_idx=PAD_IDX):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pe        = PositionalEncoding(emb_dim, max_len, dropout)
        self.blocks    = nn.ModuleList([
            EncoderBlock(emb_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, src, src_mask=None):
        x = self.pe(self.embedding(src))
        for block in self.blocks:
            x = block(x, src_mask)
        return self.norm(x)   # (batch, src_len, emb_dim)


# ── Decoder block ─────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    One decoder block:
        1. Causal self-attention  — target attends to past target positions
        2. Cross-attention        — target attends to encoder output  ← NEW
        3. FeedForward

    The cross-attention weights are returned for visualisation.
    They form the alignment matrix: which source tokens did each
    output token rely on?
    """
    def __init__(self, emb_dim, n_heads, ff_dim=None, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadSelfAttention(emb_dim, n_heads, dropout)
        self.cross_attn = MultiHeadCrossAttention(emb_dim, n_heads, dropout)
        self.ff         = FeedForward(emb_dim, ff_dim, dropout)
        self.norm1      = nn.LayerNorm(emb_dim)
        self.norm2      = nn.LayerNorm(emb_dim)
        self.norm3      = nn.LayerNorm(emb_dim)
        self.drop       = nn.Dropout(dropout)

    def forward(
        self,
        x            : torch.Tensor,               # (batch, tgt_len, emb_dim)
        encoder_out  : torch.Tensor,               # (batch, src_len, emb_dim)
        causal_mask  : Optional[torch.Tensor] = None,
        src_pad_mask : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Causal self-attention on the target sequence
        sa_out, _     = self.self_attn(self.norm1(x), causal_mask)
        x             = x + self.drop(sa_out)

        # 2. Cross-attention: Q from decoder, K/V from encoder
        ca_out, cross_w = self.cross_attn(self.norm2(x), encoder_out, src_pad_mask)
        x               = x + self.drop(ca_out)

        # 3. Feedforward
        x = x + self.drop(self.ff(self.norm3(x)))

        return x, cross_w   # cross_w: (batch, n_heads, tgt_len, src_len)


class Decoder(nn.Module):
    """
    Decodes a target sequence conditioned on the encoder output.
    Returns cross-attention weights from every layer for visualisation.
    """
    def __init__(self, vocab_size, emb_dim, n_heads, n_layers,
                 ff_dim=None, max_len=64, dropout=0.1, pad_idx=PAD_IDX):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pe        = PositionalEncoding(emb_dim, max_len, dropout)
        self.blocks    = nn.ModuleList([
            DecoderBlock(emb_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def _causal_mask(self, T, device):
        return torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

    def forward(self, tgt, encoder_out, src_pad_mask=None):
        T           = tgt.size(1)
        causal_mask = self._causal_mask(T, tgt.device)
        x           = self.pe(self.embedding(tgt))
        all_cross_w = []

        for block in self.blocks:
            x, cross_w = block(x, encoder_out, causal_mask, src_pad_mask)
            all_cross_w.append(cross_w)

        return self.norm(x), all_cross_w


# ── Full encoder-decoder model ────────────────────────────────────────────────

class EncoderDecoder(nn.Module):
    """
    Complete encoder-decoder model for the sequence reversal task.

    Args:
        vocab_size (int): shared vocabulary size
        emb_dim    (int): embedding dimension
        n_heads    (int): attention heads
        n_layers   (int): encoder and decoder layers each
        ff_dim     (int): feedforward inner dimension
        max_len    (int): max sequence length
        dropout    (float): dropout
        pad_idx    (int): padding token index
    """

    def __init__(
        self,
        vocab_size : int   = VOCAB_SIZE,
        emb_dim    : int   = 32,
        n_heads    : int   = 2,
        n_layers   : int   = 2,
        ff_dim     : int   = None,
        max_len    : int   = 64,
        dropout    : float = 0.1,
        pad_idx    : int   = PAD_IDX,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.encoder = Encoder(vocab_size, emb_dim, n_heads, n_layers,
                               ff_dim, max_len, dropout, pad_idx)
        self.decoder = Decoder(vocab_size, emb_dim, n_heads, n_layers,
                               ff_dim, max_len, dropout, pad_idx)
        self.head    = nn.Linear(emb_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def _src_pad_mask(self, src):
        """Masks padding positions in the source. Shape: (B, 1, 1, src_len)"""
        return (src == self.pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        src : torch.Tensor,   # (batch, src_len)
        tgt : torch.Tensor,   # (batch, tgt_len)
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            logits:       (batch, tgt_len, vocab_size)
            all_cross_w:  list[n_layers] of (batch, n_heads, tgt_len, src_len)
        """
        src_pad_mask          = self._src_pad_mask(src)
        encoder_out           = self.encoder(src, src_pad_mask)
        decoder_out, cross_ws = self.decoder(tgt, encoder_out, src_pad_mask)
        return self.head(decoder_out), cross_ws

    @torch.no_grad()
    def translate(
        self,
        src        : torch.Tensor,   # (1, src_len)
        bos_idx    : int,
        eos_idx    : int,
        max_steps  : int = 20,
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Greedy autoregressive decoding.

        At each step:
            1. Run the full encoder on src (once, result is reused)
            2. Run the decoder on all generated tokens so far
            3. Take argmax of the last position's logits
            4. Append to the sequence

        Returns:
            generated_ids: list of token indices (without BOS)
            cross_weights: list[n_layers] of (1, n_heads, tgt_len, src_len)
                           at the FINAL decoding step — used for alignment plot
        """
        self.eval()
        src_pad_mask = self._src_pad_mask(src)
        encoder_out  = self.encoder(src, src_pad_mask)

        dec_ids = [bos_idx]
        final_cross_w = None

        for _ in range(max_steps):
            tgt = torch.tensor([dec_ids], dtype=torch.long, device=src.device)
            dec_out, cross_ws = self.decoder(tgt, encoder_out, src_pad_mask)
            logits            = self.head(dec_out)
            next_id           = logits[0, -1, :].argmax().item()
            dec_ids.append(next_id)
            final_cross_w = cross_ws
            if next_id == eos_idx:
                break

        return dec_ids[1:], final_cross_w   # strip BOS from output

    def __repr__(self):
        params = sum(p.numel() for p in self.parameters())
        d      = self.encoder.embedding.embedding_dim
        return (
            f"EncoderDecoder(\n"
            f"  emb_dim={d}, n_layers={len(self.encoder.blocks)},\n"
            f"  {self.encoder.blocks[0].attn},\n"
            f"  {self.decoder.blocks[0].cross_attn},\n"
            f"  parameters={params:,}\n"
            f")"
        )
