# mini-cross-attention

> Cross-attention from scratch — the bridge between encoder and decoder.
> Step 6 of the mini-LLM series.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

---

## Series

| Step | Repository | What it builds |
|------|-----------|----------------|
| 1 | [mini-embedding](../mini-embedding) | Word vectors — Skip-gram Word2Vec |
| 2 | [mini-self-attention](../mini-self-attention) | Multi-head self-attention encoder block |
| 3 | [mini-transformer](../mini-transformer) | Positional encoding + stacked causal decoder |
| 4 | [mini-gpt](../mini-gpt) | Real corpus, overfitting, beam search, evaluation |
| 5 | [mini-chat](../mini-chat) | Instruction format, loss masking, chat interface |
| **6** | **mini-cross-attention** ← you are here | Cross-attention module + alignment visualisation |
| 7 | mini-translator _(coming)_ | English→Spanish encoder-decoder |

---

## The one new idea

Every previous project used **self-attention**: one sequence attends to itself.

```
Self-attention:
    Input:  [A, B, C, D]
    Q, K, V all derived from the same sequence
    Weight matrix: (4 × 4)  — square
    Answer: "how do positions within this sequence relate to each other?"
```

**Cross-attention** takes two sequences:

```
Cross-attention:
    Target: [X, Y, Z]       → provides Q  ("what am I looking for?")
    Source: [A, B, C, D]    → provides K, V  ("what do I have available?")
    Weight matrix: (3 × 4)  — rectangular
    Answer: "which source positions does each target position rely on?"
```

The weight matrix is called the **alignment matrix**. It is the most
interpretable output of any encoder-decoder model — it shows you, for each
generated token, which input tokens the model was looking at.

---

## Task: sequence reversal

To make alignment completely visible, this project trains an encoder-decoder
to reverse a sequence of digits:

```
Input:    [3, 7, 1, 5]
Output:   [5, 1, 7, 3]
```

A well-trained model produces a near-perfect anti-diagonal alignment:

```
Cross-attention weights:

       3     7     1     5       ← source tokens
  5  [0.00, 0.00, 0.00, 1.00]   ← output[0] = "5", attended to source[3]
  1  [0.00, 0.00, 1.00, 0.00]   ← output[1] = "1", attended to source[2]
  7  [0.00, 1.00, 0.00, 0.00]   ← output[2] = "7", attended to source[1]
  3  [1.00, 0.00, 0.00, 0.00]   ← output[3] = "3", attended to source[0]
```

This is the same alignment that appeared in neural machine translation —
when the model translated *"the cat"* to *"le chat"*, the alignment showed
*"cat"* attending strongly to *"chat"*.

---

## Architecture

```
Source sequence  [3, 7, 1, 5]
      │
      ▼
┌─────────────────────────────────────────┐
│  Encoder  (N blocks)                    │
│  Embedding + PositionalEncoding         │
│  MultiHeadSelfAttention  (no mask)      │  source attends to itself
│  FeedForward                            │
│  LayerNorm + residual                   │
└─────────────────────────────────────────┘
      │  encoder_output  (batch, src_len, emb_dim)
      │
      │    Target sequence  [BOS, 5, 1, 7]
      │          │
      │          ▼
      │    ┌─────────────────────────────────────────┐
      │    │  Decoder  (N blocks)                    │
      │    │  Embedding + PositionalEncoding         │
      │    │  MultiHeadSelfAttention  (causal mask)  │  target attends to past target
      │    │                                         │
      └────┤  MultiHeadCrossAttention               │  ← THE NEW PIECE
           │    Q = decoder hidden state             │
           │    K = encoder_output                   │
           │    V = encoder_output                   │
           │  FeedForward                            │
           │  LayerNorm + residual (×2)              │
           └─────────────────────────────────────────┘
                  │
                  ▼
             Linear → vocab logits → [5, 1, 7, 3, EOS]
```

---

## Project structure

```
mini-cross-attention/
│
├── src/
│   ├── attention.py    # SelfAttention, CrossAttention,
│   │                   # MultiHeadSelfAttention, MultiHeadCrossAttention
│   ├── dataset.py      # digit reversal dataset  (no external data needed)
│   ├── model.py        # Encoder, Decoder, EncoderDecoder
│   ├── train.py        # training loop + accuracy metric
│   └── visualize.py    # alignment heatmap, self vs cross comparison
│
├── outputs/
├── main.py
├── environment.yml
├── requirements.txt
├── THEORY.md
└── README.md
```

No `data/` folder — the dataset is generated programmatically.

---

## Quickstart

```bash
git clone https://github.com/your-username/mini-cross-attention.git
cd mini-cross-attention
conda env create -f environment.yml
conda activate mini-cross-attention
python main.py
```

Training takes **under 1 minute on CPU**.

---

## Configuration

| Parameter | Default | Notes |
|---|---|---|
| `EMB_DIM` | `32` | Small — task is simple |
| `N_HEADS` | `2` | |
| `N_LAYERS` | `2` | Encoder and decoder each |
| `EPOCHS` | `30` | Usually converges in 20 |
| `N_SAMPLES` | `2000` | Generated sequence pairs |

---

## Outputs

| File | Description |
|---|---|
| `alignment_0.png` | **Cross-attention alignment matrix — the key output** |
| `alignment_1.png` | Second example |
| `alignment_2.png` | Third example |
| `self_vs_cross.png` | Self-attention (square) vs cross-attention (rectangular) side by side |
| `training.png` | Loss and token accuracy curves |
| `encoder_decoder.pt` | Saved model weights |

---

## What to look for

**In `alignment_*.png`:**
Each plot is a grid of shape `(tgt_len × src_len)`.
- Blue = attended to that source position
- Row i = "to generate output token i, I looked at..."
- After full training: near-perfect anti-diagonal

**In `self_vs_cross.png`:**
Top row = encoder self-attention (green, square).
Bottom row = decoder cross-attention (blue, rectangular).
The shapes make the structural difference immediately obvious.

---

## Deep dive

See [`THEORY.md`](./THEORY.md) for:
- The full math of cross-attention (Q from target, K/V from source)
- Why cross-attention is necessary for encoder-decoder tasks
- How the alignment matrix emerges from training
- Why sequence reversal is the ideal demo task
- Line-by-line code walkthrough
- How this maps directly onto the mini-translator architecture

---

## References

- Bahdanau et al. (2015) — [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — introduced attention alignment
- Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — cross-attention in transformers
- Luong et al. (2015) — [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)

---

## License

MIT
# mini-cross-attention
