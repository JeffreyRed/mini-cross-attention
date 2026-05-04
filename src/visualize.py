"""
visualize.py — Cross-attention alignment matrix visualisation.

The central plot is plot_alignment():
  A heatmap of shape (tgt_len × src_len) where entry [i, j] shows
  how much target position i attended to source position j.

  For the reversal task, a well-trained model should produce a clear
  anti-diagonal pattern — position 0 in the output attended to the
  last position in the input, and so on.

  This is the same alignment matrix that made neural machine translation
  interpretable in Bahdanau et al. (2015) — the paper that introduced
  attention to seq2seq models before the transformer existed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List

PALETTE = {
    "bg":        "#0d1117",
    "grid":      "#21262d",
    "text":      "#e6edf3",
    "accent":    "#58a6ff",
    "highlight": "#f78166",
    "muted":     "#8b949e",
    "green":     "#3fb950",
    "yellow":    "#e3b341",
}


def plot_alignment(
    cross_weights  : "torch.Tensor",   # (n_heads, tgt_len, src_len)
    src_tokens     : List[str],
    tgt_tokens     : List[str],
    title          : str = "",
    save_path      : str = None,
) -> None:
    """
    Plots the cross-attention alignment matrix for all heads.

    For the reversal task you should see a near-perfect anti-diagonal.
    For a translation task you would see fuzzy diagonal-ish patterns
    reflecting word order correspondence between languages.

    Args:
        cross_weights: (n_heads, tgt_len, src_len) — from model.translate()
        src_tokens:    source token strings  (x-axis = keys)
        tgt_tokens:    target token strings  (y-axis = queries)
        title:         shown above the figure
        save_path:     optional save path
    """
    n_heads = cross_weights.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(5 * n_heads, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    if n_heads == 1:
        axes = [axes]

    for h, ax in enumerate(axes):
        w = cross_weights[h].numpy()   # (tgt_len, src_len)

        ax.set_facecolor(PALETTE["bg"])
        im = ax.imshow(w, cmap="Blues", vmin=0, vmax=1, aspect="auto")

        # Source tokens on x-axis (what the decoder attended TO)
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=0, fontsize=11,
                           color=PALETTE["accent"], fontfamily="monospace",
                           fontweight="bold")
        ax.set_xlabel("Source  (keys / values)", color=PALETTE["muted"], fontsize=9)

        # Target tokens on y-axis (what generated EACH output token)
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_yticklabels(tgt_tokens, fontsize=11,
                           color=PALETTE["highlight"], fontfamily="monospace",
                           fontweight="bold")
        ax.set_ylabel("Target  (queries)", color=PALETTE["muted"], fontsize=9)

        # Annotate each cell with its weight value
        for i in range(len(tgt_tokens)):
            for j in range(len(src_tokens)):
                val   = w[i, j]
                color = "white" if val > 0.5 else PALETTE["muted"]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

        for sp in ax.spines.values():
            sp.set_edgecolor(PALETTE["grid"])

        ax.set_title(
            f"Head {h}  —  Cross-Attention\n"
            f"Row i = 'to generate output[i], I looked at...'\n"
            f"Col j = 'source token j'",
            color=PALETTE["text"], fontsize=8, pad=8,
        )

    full_title = f"Alignment Matrix  ·  {title}" if title else "Alignment Matrix"
    fig.suptitle(full_title, color=PALETTE["text"], fontsize=12, y=1.03)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Alignment plot saved → {save_path}")
    plt.show()


def plot_self_vs_cross(
    self_weights  : "torch.Tensor",   # (n_heads, T, T) from encoder
    cross_weights : "torch.Tensor",   # (n_heads, tgt_len, src_len)
    src_tokens    : List[str],
    tgt_tokens    : List[str],
    save_path     : str = None,
) -> None:
    """
    Side-by-side comparison of self-attention and cross-attention.

    Self-attention  (left):  square matrix, same sequence attends to itself
    Cross-attention (right): rectangular matrix, target attends to source

    This plot makes the structural difference between the two immediately visible.
    """
    n_heads = self_weights.shape[0]
    fig, axes = plt.subplots(2, n_heads, figsize=(5 * n_heads, 10))
    fig.patch.set_facecolor(PALETTE["bg"])

    if n_heads == 1:
        axes = [[axes[0]], [axes[1]]]

    # Top row: self-attention (encoder)
    for h in range(n_heads):
        ax = axes[0][h]
        w  = self_weights[h].numpy()
        ax.set_facecolor(PALETTE["bg"])
        ax.imshow(w, cmap="Greens", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=0, fontsize=10,
                           color=PALETTE["accent"], fontfamily="monospace")
        ax.set_yticks(range(len(src_tokens)))
        ax.set_yticklabels(src_tokens, fontsize=10,
                           color=PALETTE["accent"], fontfamily="monospace")
        for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["grid"])
        ax.set_title(f"Self-Attention  Head {h}\n(source attends to itself)\n"
                     f"shape: {w.shape[0]}×{w.shape[1]}",
                     color=PALETTE["text"], fontsize=8, pad=6)

    # Bottom row: cross-attention (decoder)
    for h in range(n_heads):
        ax = axes[1][h]
        w  = cross_weights[h].numpy()
        ax.set_facecolor(PALETTE["bg"])
        ax.imshow(w, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=0, fontsize=10,
                           color=PALETTE["accent"], fontfamily="monospace")
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_yticklabels(tgt_tokens, fontsize=10,
                           color=PALETTE["highlight"], fontfamily="monospace")
        for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["grid"])
        ax.set_title(f"Cross-Attention  Head {h}\n(target queries source)\n"
                     f"shape: {w.shape[0]}×{w.shape[1]}",
                     color=PALETTE["text"], fontsize=8, pad=6)

    fig.suptitle(
        "Self-Attention vs Cross-Attention\n"
        "Self: square (T×T)   Cross: rectangular (tgt_len×src_len)",
        color=PALETTE["text"], fontsize=11, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Comparison plot saved → {save_path}")
    plt.show()


def plot_training(train_hist, val_hist, save_path=None):
    """Plots loss and token accuracy curves."""
    epochs     = list(range(1, len(train_hist) + 1))
    t_loss     = [h[0] for h in train_hist]
    v_loss     = [h[0] for h in val_hist]
    t_acc      = [h[2] for h in train_hist]
    v_acc      = [h[2] for h in val_hist]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(PALETTE["bg"])

    for ax, (tr, vl, ylabel, title) in zip(axes, [
        (t_loss, v_loss, "Cross-Entropy Loss", "Loss"),
        (t_acc,  v_acc,  "Token Accuracy",     "Accuracy  (higher = better)"),
    ]):
        ax.set_facecolor(PALETTE["bg"])
        ax.grid(color=PALETTE["grid"], linewidth=0.5)
        ax.plot(epochs, tr, color=PALETTE["accent"],    linewidth=2, label="Train")
        ax.plot(epochs, vl, color=PALETTE["highlight"], linewidth=2, label="Val")
        ax.set_xlabel("Epoch", color=PALETTE["muted"])
        ax.set_ylabel(ylabel,  color=PALETTE["muted"])
        ax.tick_params(colors=PALETTE["muted"])
        for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["grid"])
        ax.legend(facecolor=PALETTE["grid"], labelcolor=PALETTE["text"], fontsize=9)
        ax.set_title(title, color=PALETTE["text"], fontsize=11, pad=8, loc="left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Training curves saved → {save_path}")
    plt.show()
