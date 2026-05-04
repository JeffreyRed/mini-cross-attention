"""
main.py — End-to-end pipeline for mini-cross-attention.

Usage:
    python main.py

Pipeline:
    1.  Explain self vs cross attention side by side
    2.  Generate reversal dataset
    3.  Build encoder-decoder model
    4.  Train
    5.  Show alignment matrices for multiple examples
    6.  Side-by-side self vs cross attention comparison
    7.  Interactive demo: type a sequence, see the reversal + alignment
    8.  Save model
"""

import torch
from pathlib import Path

from src.dataset    import (
    generate_pairs, ReversalDataset,
    VOCAB, IDX2TOK, BOS_IDX, EOS_IDX, PAD_IDX, VOCAB_SIZE,
)
from src.model      import EncoderDecoder
from src.train      import train, evaluate
from src.visualize  import plot_alignment, plot_self_vs_cross, plot_training

# ── Config ────────────────────────────────────────────────────────────────────
EMB_DIM     = 32
N_HEADS     = 2
N_LAYERS    = 2
FF_DIM      = 64
MAX_LEN     = 16
EPOCHS      = 30
LR          = 3e-3
BATCH_SIZE  = 64
N_SAMPLES   = 2000
OUTPUTS_DIR = Path("outputs")
# ──────────────────────────────────────────────────────────────────────────────

LOWER_TO_VOCAB = {k.lower(): k for k in VOCAB}


def explain_difference() -> None:
    """Prints a clear side-by-side explanation of self vs cross attention."""
    print("\n── Self-Attention vs Cross-Attention ───────────────")
    print("""
  SELF-ATTENTION (used in encoder and decoder's first sub-layer)
  ─────────────────────────────────────────────────────────────
  One sequence in → same sequence attends to itself.

    Input: [A, B, C, D]
    Q = W_Q(input)   ← from input
    K = W_K(input)   ← from input
    V = W_V(input)   ← from input

    Weight matrix shape: (4 × 4)  — square
    Entry [i, j]: how much position i attended to position j
                  within the SAME sequence.

  CROSS-ATTENTION (used in decoder's second sub-layer)
  ─────────────────────────────────────────────────────────────
  TWO sequences: target and source.

    Target (decoder): [X, Y, Z]      ← provides Q
    Source (encoder): [A, B, C, D]   ← provides K, V

    Q = W_Q(target)   ← from TARGET
    K = W_K(source)   ← from SOURCE
    V = W_V(source)   ← from SOURCE

    Weight matrix shape: (3 × 4)  — rectangular
    Entry [i, j]: how much TARGET position i attended to SOURCE position j.

    This is the ALIGNMENT MATRIX — it shows which source tokens
    each output token relied on. After training it should show:
      output[0] (reversed last) → attended to source[-1]
      output[1]                 → attended to source[-2]
      ... anti-diagonal pattern
""")
    print("────────────────────────────────────────────────────\n")


def show_examples(model: EncoderDecoder, n: int = 6) -> None:
    """Shows n random reversal examples with accuracy."""
    from src.dataset import generate_pairs
    model.eval()
    pairs   = generate_pairs(n_samples=20)[:n]
    correct = 0

    print("── Reversal examples ───────────────────────────────")
    for src_ids, tgt_ids in pairs:
        src   = torch.tensor([src_ids])
        pred, _ = model.translate(src, BOS_IDX, EOS_IDX)
        expected = list(reversed(src_ids))

        src_str  = " ".join(IDX2TOK[i] for i in src_ids)
        pred_str = " ".join(IDX2TOK[i] for i in pred
                            if i not in (BOS_IDX, EOS_IDX, PAD_IDX))
        exp_str  = " ".join(IDX2TOK[i] for i in expected)
        ok       = pred[:len(expected)] == expected
        correct += int(ok)
        tick     = "✓" if ok else "✗"
        print(f"  {tick}  src: [{src_str}]  →  pred: [{pred_str}]  (expected: [{exp_str}])")

    print(f"\n  Accuracy: {correct}/{n}\n")
    print("────────────────────────────────────────────────────\n")


def show_alignment(model: EncoderDecoder, n_examples: int = 3) -> None:
    """Shows the cross-attention alignment matrix for n random examples."""
    from src.dataset import generate_pairs

    pairs = generate_pairs(n_samples=20, seed=99)[:n_examples]
    model.eval()

    for i, (src_ids, _) in enumerate(pairs):
        src      = torch.tensor([src_ids])
        pred, cross_ws = model.translate(src, BOS_IDX, EOS_IDX)

        src_toks  = [IDX2TOK[t] for t in src_ids]
        pred_toks = [IDX2TOK[t] for t in pred
                     if t not in (BOS_IDX, EOS_IDX, PAD_IDX)]

        if not pred_toks or cross_ws is None:
            continue

        # Use last decoder layer's cross-attention
        w = cross_ws[-1].squeeze(0)   # (n_heads, tgt_len, src_len)
        # Trim to actual lengths
        w = w[:, :len(pred_toks), :len(src_toks)]

        plot_alignment(
            w, src_toks, pred_toks,
            title     = f"[{' '.join(src_toks)}]  →  [{' '.join(pred_toks)}]",
            save_path = str(OUTPUTS_DIR / f"alignment_{i}.png"),
        )


def show_self_vs_cross(model: EncoderDecoder) -> None:
    """Shows self-attention and cross-attention side by side for one example."""
    from src.dataset import generate_pairs
    model.eval()

    src_ids  = [VOCAB["3"], VOCAB["7"], VOCAB["1"], VOCAB["5"]]
    src      = torch.tensor([src_ids])

    # Get encoder self-attention
    with torch.no_grad():
        enc_out = model.encoder(src)

    # Get encoder's self-attention weights by passing through first block manually
    with torch.no_grad():
        x = model.encoder.pe(model.encoder.embedding(src))
        _, self_w = model.encoder.blocks[0].attn(x)   # (1, n_heads, T, T)

    # Get cross-attention from translate
    pred, cross_ws = model.translate(src, BOS_IDX, EOS_IDX)

    src_toks  = [IDX2TOK[t] for t in src_ids]
    pred_toks = [IDX2TOK[t] for t in pred
                 if t not in (BOS_IDX, EOS_IDX, PAD_IDX)]

    if not pred_toks or cross_ws is None:
        print("  (skipping self vs cross plot — empty prediction)")
        return

    self_w_np  = self_w.squeeze(0)              # (n_heads, src, src)
    cross_w_np = cross_ws[-1].squeeze(0)        # (n_heads, tgt, src)
    cross_w_np = cross_w_np[:, :len(pred_toks), :len(src_toks)]

    plot_self_vs_cross(
        self_w_np, cross_w_np,
        src_toks, pred_toks,
        save_path = str(OUTPUTS_DIR / "self_vs_cross.png"),
    )


def interactive_demo(model: EncoderDecoder) -> None:
    """Interactive: type a sequence of digits, see reversal + alignment."""
    print("── Interactive demo ─────────────────────────────────")
    print("  Type a sequence of digits (e.g.  3 7 1 5)")
    print("  The model will reverse it and show the alignment matrix.")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            raw = input("  Sequence: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting.")
            break

        if raw.lower() in ("quit", "exit", "q"):
            break
        if not raw:
            continue

        tokens   = raw.split()
        unknown  = [t for t in tokens if t not in VOCAB]
        if unknown:
            print(f"  ✗ Unknown tokens: {unknown}. Use digits 0-9.\n")
            continue

        src_ids  = [VOCAB[t] for t in tokens]
        src      = torch.tensor([src_ids])
        pred, cross_ws = model.translate(src, BOS_IDX, EOS_IDX, max_steps=len(src_ids)+2)

        pred_toks = [IDX2TOK[t] for t in pred
                     if t not in (BOS_IDX, EOS_IDX, PAD_IDX)]
        expected  = list(reversed(tokens))

        print(f"\n  Input   : [{' '.join(tokens)}]")
        print(f"  Output  : [{' '.join(pred_toks)}]")
        print(f"  Expected: [{' '.join(expected)}]")
        ok = pred_toks == expected
        print(f"  {'✓ Correct' if ok else '✗ Wrong'}\n")

        if cross_ws is not None and pred_toks:
            w = cross_ws[-1].squeeze(0)[:, :len(pred_toks), :len(tokens)]
            plot_alignment(
                w, tokens, pred_toks,
                title = f"[{' '.join(tokens)}] → [{' '.join(pred_toks)}]",
            )

    print("────────────────────────────────────────────────────\n")


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # ── 1. Explain ────────────────────────────────────────────────────────────
    explain_difference()

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    print("── Dataset ─────────────────────────────────────────")
    pairs    = generate_pairs(n_samples=N_SAMPLES)
    train_ds = ReversalDataset(pairs, val_split=0.1, train=True)
    val_ds   = ReversalDataset(pairs, val_split=0.1, train=False)
    print(f"  Task: reverse a sequence of digits")
    print(f"  {train_ds}  (train)")
    print(f"  {val_ds}    (val)")
    print(f"  Vocab: {list(VOCAB.keys())}\n")
    print("  Example pairs:")
    for src_ids, tgt_ids in pairs[:3]:
        src_s = " ".join(IDX2TOK[i] for i in src_ids)
        tgt_s = " ".join(IDX2TOK[i] for i in tgt_ids
                         if i not in (BOS_IDX, EOS_IDX))
        print(f"    src: [{src_s}]  →  tgt: [{tgt_s}]")
    print()

    # ── 3. Model ──────────────────────────────────────────────────────────────
    print("── Model ───────────────────────────────────────────")
    model = EncoderDecoder(
        vocab_size = VOCAB_SIZE,
        emb_dim    = EMB_DIM,
        n_heads    = N_HEADS,
        n_layers   = N_LAYERS,
        ff_dim     = FF_DIM,
        max_len    = MAX_LEN,
    )
    print(model, "\n")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("── Training ────────────────────────────────────────")
    print("  After training, the cross-attention alignment should show")
    print("  a clear anti-diagonal: output[i] attends to source[n-1-i]\n")

    train_hist, val_hist = train(
        model, train_ds, val_ds,
        epochs      = EPOCHS,
        lr          = LR,
        batch_size  = BATCH_SIZE,
    )

    _, _, final_acc = evaluate(model, val_ds)
    print(f"\n  Final val accuracy: {final_acc:.3f}")

    plot_training(
        train_hist, val_hist,
        save_path = str(OUTPUTS_DIR / "training.png"),
    )

    # ── 5. Examples ───────────────────────────────────────────────────────────
    show_examples(model, n=8)

    # ── 6. Alignment matrices ─────────────────────────────────────────────────
    print("── Alignment matrices ──────────────────────────────")
    print("  Plotting cross-attention for 3 examples.")
    print("  Anti-diagonal = model correctly learned to reverse.\n")
    show_alignment(model, n_examples=3)

    # ── 7. Self vs Cross comparison ───────────────────────────────────────────
    print("── Self-Attention vs Cross-Attention (side by side) ─")
    show_self_vs_cross(model)

    # ── 8. Interactive ────────────────────────────────────────────────────────
    interactive_demo(model)

    # ── 9. Save ───────────────────────────────────────────────────────────────
    ckpt = OUTPUTS_DIR / "encoder_decoder.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Model saved → {ckpt}")


if __name__ == "__main__":
    main()
