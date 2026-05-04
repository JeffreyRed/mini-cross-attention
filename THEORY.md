# Theory & Code Walkthrough — mini-cross-attention

> Step 6 of the mini-LLM series. Prerequisite: [mini-transformer](github.com/JeffreyRed/mini-transformer).

-----

## Table of Contents

1. [What this step adds](#1-what-this-step-adds)
1. [The problem self-attention cannot solve](#2-the-problem-self-attention-cannot-solve)
1. [Cross-attention — the math](#3-cross-attention--the-math)
1. [The alignment matrix](#4-the-alignment-matrix)
1. [Why the anti-diagonal appears — step by step](#5-why-the-anti-diagonal-appears--step-by-step)
1. [How encoder and decoder are trained — simultaneous, not sequential](#6-how-encoder-and-decoder-are-trained--simultaneous-not-sequential)
1. [The encoder-decoder architecture](#7-the-encoder-decoder-architecture)
1. [Why sequence reversal is the ideal demo task](#8-why-sequence-reversal-is-the-ideal-demo-task)
1. [Three types of attention in one model](#9-three-types-of-attention-in-one-model)
1. [Code walkthrough](#10-code-walkthrough)
1. [Full data flow](#11-full-data-flow)
1. [How this maps onto mini-translator](#12-how-this-maps-onto-mini-translator)
1. [Curiosity: combining two pretrained models](#13-curiosity-combining-two-pretrained-models)

-----

## 1. What this step adds

Every previous project used only self-attention: one sequence attends to itself.
This is sufficient for language modelling — the model reads its own context and
predicts the next token.

But some tasks require reading from one sequence while generating another:
translation, summarisation, question answering. For these, self-attention alone
is not enough. You need a mechanism that lets the decoder query the encoder.

That mechanism is **cross-attention**.

|                   |Self-attention                    |Cross-attention                              |
|-------------------|----------------------------------|---------------------------------------------|
|Inputs             |one sequence                      |two sequences                                |
|Q from             |same sequence                     |target sequence                              |
|K, V from          |same sequence                     |source sequence                              |
|Weight matrix shape|(T × T) square                    |(tgt_len × src_len) rectangular              |
|Used for           |encoding context within a sequence|reading one sequence while generating another|

-----

## 2. The problem self-attention cannot solve

Consider translation:

```
Source (English): "The cat sat on the mat"
Target (French):  "Le chat s'est assis sur le tapis"
```

When generating *“chat”* (French for cat), the decoder needs to look back at
the encoder’s representation of *“cat”* in the source. Self-attention alone
cannot do this — it can only attend within the sequence being generated.

The decoder needs a direct line to the encoder’s output. That line is
cross-attention.

More generally: any task where the input and output are **different sequences**
requires cross-attention. The decoder must be conditioned on the encoder,
not just on its own previous outputs.

-----

## 3. Cross-attention — the math

Given:

- Target sequence `X` of shape `(batch, tgt_len, emb_dim)` — the decoder’s current state
- Source sequence `C` of shape `(batch, src_len, emb_dim)` — the encoder’s output

Three projections:

```
Q = X @ W_Q      (batch, tgt_len, d_k)   ← from TARGET
K = C @ W_K      (batch, src_len, d_k)   ← from SOURCE
V = C @ W_V      (batch, src_len, d_v)   ← from SOURCE
```

Attention scores:

```
scores = Q @ K^T / sqrt(d_k)    (batch, tgt_len, src_len)
```

Note the shape: `tgt_len` rows × `src_len` columns.
Each row `i` asks: “how relevant is each source position to target position `i`?”

Attention weights:

```
weights = softmax(scores, dim=-1)    (batch, tgt_len, src_len)
```

Each row sums to 1. Entry `[i, j]` is the probability that target position
`i` attends to source position `j`.

Output:

```
out = weights @ V    (batch, tgt_len, d_v)
```

Each output position `i` gets a weighted sum of source value vectors,
weighted by how relevant each source position was.

**The only difference from self-attention:**
In self-attention, Q, K, V all come from `X`.
In cross-attention, Q comes from `X` (target), K and V come from `C` (source).
The computation is identical after that.

-----

## 4. The alignment matrix

The weight matrix `weights` of shape `(tgt_len × src_len)` is called the
**alignment matrix**. It is one of the most interpretable objects in deep learning.

For the reversal task `[3, 7, 1, 5] → [5, 1, 7, 3]`:

```
Ideal alignment after training:

         3     7     1     5       ← source positions
  5    [0.00, 0.00, 0.00, 1.00]   row 0: to generate "5", look at source[3]
  1    [0.00, 0.00, 1.00, 0.00]   row 1: to generate "1", look at source[2]
  7    [0.00, 1.00, 0.00, 0.00]   row 2: to generate "7", look at source[1]
  3    [1.00, 0.00, 0.00, 0.00]   row 3: to generate "3", look at source[0]
```

This anti-diagonal pattern is the model having learned to reverse sequences.
It is using cross-attention as a learned pointer into the source.

For machine translation the alignment is softer and more diagonal:

```
Translating "the cat" → "le chat":

         the    cat
  le   [0.80,  0.20]   "le" attended mostly to "the"
  chat [0.15,  0.85]   "chat" attended mostly to "cat"
```

The alignment is not perfect because languages differ in word order,
but it captures the correspondence between words across languages.

This alignment matrix was the key visualisation in Bahdanau et al. (2015),
the paper that introduced attention to sequence-to-sequence models — two years
before the transformer existed.

-----

## 5. Why the anti-diagonal appears — step by step

### The pattern is NOT programmed — the model discovered it

Nobody told the model to “look at position n-1-i”. The code contains no
rule about reversal. The only training signal is:

> “given these inputs, your output was wrong — here is the gradient to fix it”

The model was shown thousands of examples of `[A, B, C, D] → [D, C, B, A]`
and backpropagation adjusted W_Q, W_K, W_V until the cross-attention weights
naturally pointed at the right source position for each output step.
The anti-diagonal is the geometry that minimises the loss. The model discovered it.

-----

### Why the anti-diagonal specifically

Two things happen simultaneously:

**1. Output is generated left to right** — output token 0 first, then 1, then 2, then 3.

**2. The task requires reading the source right to left** — to produce the reversed
sequence, output token 0 needs source position n-1, output token 1 needs
source position n-2, and so on.

When you stack “which source position did each output step need?” as rows in a
grid, left-to-right output combined with right-to-left source reading naturally
draws a line from the top-right corner to the bottom-left — the anti-diagonal.

-----

### Concrete step-by-step trace: `[3, 7, 1, 5] → [5, 1, 7, 3]`

```
SOURCE:  [3,   7,   1,   5]
          pos0 pos1 pos2 pos3

DECODING (one token at a time, left to right):

Step 1 — generate output[0]
  decoder sees: [BOS]
  task: "what is the first token of the reversed sequence?"
  answer: the LAST source token = "5" at pos 3
  cross-attention weight row 0: [0.00, 0.00, 0.00, 0.97]  ← attends to pos 3

Step 2 — generate output[1]
  decoder sees: [BOS, 5]
  task: "what comes after 5 in the reversal?"
  answer: second-to-last source token = "1" at pos 2
  cross-attention weight row 1: [0.00, 0.00, 0.95, 0.05]  ← attends to pos 2

Step 3 — generate output[2]
  decoder sees: [BOS, 5, 1]
  task: "what comes next?"
  answer: "7" at pos 1
  cross-attention weight row 2: [0.00, 0.91, 0.07, 0.00]  ← attends to pos 1

Step 4 — generate output[3]
  decoder sees: [BOS, 5, 1, 7]
  task: "last token?"
  answer: first source token = "3" at pos 0
  cross-attention weight row 3: [0.98, 0.01, 0.00, 0.00]  ← attends to pos 0
```

Stack those four weight rows into a matrix:

```
Alignment matrix after training:

          3      7      1      5      ← source tokens (x-axis = keys)
  5   [ 0.00,  0.00,  0.00,  0.97 ]  ← output[0] looked at source[3]
  1   [ 0.00,  0.00,  0.95,  0.05 ]  ← output[1] looked at source[2]
  7   [ 0.00,  0.91,  0.07,  0.00 ]  ← output[2] looked at source[1]
  3   [ 0.98,  0.01,  0.00,  0.00 ]  ← output[3] looked at source[0]
       ↑                        ↑
    bottom-left               top-right
    bright cell               bright cell

Anti-diagonal: top-right → bottom-left
```

The bright cells run from top-right to bottom-left — the anti-diagonal.

-----

### Why “5” has a small value (0.05) at source position 3 in row 1

Real trained models are never perfectly 1.00. The softmax distributes a
tiny amount of probability everywhere. The 0.05 leak in row 1 (output “1”
attending slightly to source “5”) is noise — the model is 95% confident
about source position 2 but never completely certain. This is normal and
expected. The argmax (the brightest cell) is always correct.

-----

### The same pattern in translation

In `mini-translator` the source is English and the target is Spanish.
The alignment will not be anti-diagonal — Spanish and English have similar
word order (both roughly subject-verb-object), so the bright cells will form
a soft diagonal from top-left to bottom-right instead:

```
Translation: "the cat sat" → "el gato se sentó"

         the    cat    sat
  el   [ 0.85,  0.10,  0.05 ]   "el" looked mostly at "the"
  gato [ 0.10,  0.82,  0.08 ]   "gato" looked mostly at "cat"
  se   [ 0.05,  0.08,  0.87 ]   "se" looked mostly at "sat"
  sentó[ 0.02,  0.05,  0.93 ]   "sentó" looked mostly at "sat"
```

The mechanism is identical. Only the direction of the diagonal changes
because the word order relationship between the two languages differs.

-----

## 6. How encoder and decoder are trained — simultaneous, not sequential

### The question

“Do we train the encoder first, then the decoder, then combine them?
Or do we first train a Spanish model and an English model separately?”

### The answer

**Everything is trained simultaneously in a single pass.**
There is no pre-training phase, no sequential stages, no separate English or
Spanish model. One model, one loss function, one backward pass.

Here is what one training step looks like:

```
FORWARD PASS
────────────────────────────────────────────────────────────

Input pair:  source = [3, 7, 1, 5]   target = [BOS, 5, 1, 7]

Step A — Encoder reads the source
  embedding([3, 7, 1, 5])
  → 2 × EncoderBlock (self-attention, no mask)
  → encoder_output  shape: (1, 4, emb_dim)
  This is now fixed for all decoder steps.

Step B — Decoder reads the target (simultaneously)
  embedding([BOS, 5, 1, 7])
  → causal self-attention  (target attends to past target tokens)
  → cross-attention        Q = decoder state, K = V = encoder_output
  → feedforward
  → logits  shape: (1, 4, vocab_size)

Step C — Loss
  CrossEntropyLoss(logits, [5, 1, 7, 3, EOS])
  = one scalar number, e.g. 1.34

BACKWARD PASS
────────────────────────────────────────────────────────────

loss.backward() computes gradients for EVERY weight simultaneously:
  ∂loss/∂W_cross_Q    ← cross-attention query projection
  ∂loss/∂W_cross_K    ← cross-attention key projection
  ∂loss/∂W_cross_V    ← cross-attention value projection
  ∂loss/∂W_encoder    ← all encoder self-attention weights
  ∂loss/∂W_decoder    ← all decoder self-attention weights
  ∂loss/∂embedding    ← token embedding matrix

optimizer.step() applies all gradients in one update.
All weights move together toward lower loss.
```

The encoder learns to represent the source well **because that makes the
cross-attention more useful**. The decoder learns the target language
**because that minimises the loss**. The cross-attention learns to connect
them **because that is the only path for information to flow from source
to target**. All three needs are served by the same gradient signal.

-----

### What about two pretrained models?

A reasonable question: “I have a pretrained English model and a pretrained
Spanish model — can I combine them into a translator without training from scratch?”

This is called **model merging** or **cross-lingual transfer** and it is an
active research area. The short answer:

**You cannot simply concatenate them.** The two models learned independent
vector spaces. The English model maps “cat” to some vector `[0.3, -0.7, ...]`.
The Spanish model maps “gato” to some different vector in a completely
different coordinate system. There is no guarantee that semantically similar
words occupy nearby regions — the two spaces were never aligned.

**What you can do:**

1. **Fine-tune jointly** — use both models’ weights as initialisation, add
   a cross-attention bridge between them, then fine-tune on bilingual pairs.
   The weights are already good at their respective languages; training only
   needs to teach the bridge. Much faster than training from scratch.
1. **Vocabulary alignment** — find anchor pairs (words that appear in both
   vocabularies, like proper nouns or numbers) and use them to learn a linear
   mapping that rotates one embedding space to align with the other. This is
   how Facebook’s MUSE system worked.
1. **Shared embedding space** — train both languages simultaneously from the
   start with a shared vocabulary (multilingual BERT, mBERT) so the same
   embedding matrix handles both languages. Similar concepts end up nearby
   because they appear in similar contexts across both languages.

The cleanest approach for learning purposes is option 1: this is exactly
what `mini-translator` will do if you have pretrained weights to warm-start from.

-----

## 7. The encoder-decoder architecture

```
SOURCE  →  Encoder  →  encoder_output
                              │
TARGET  →  Decoder ←──────────┘
        →  logits
```

**Encoder:**

- Reads the full source sequence
- No masking — it sees the complete input bidirectionally
- Produces one contextual vector per source position
- These vectors are the K and V for every decoder cross-attention layer

**Decoder:**

- Generates the target sequence autoregressively
- Has three sub-layers per block:
1. **Causal self-attention** — attends to past target tokens (causal mask)
1. **Cross-attention** — Q from decoder, K/V from encoder output
1. **FeedForward**

The encoder runs **once** per input. Its output is fixed and reused at
every decoding step. The decoder runs **once per generated token**, each
time attending to the full encoder output via cross-attention.

-----

## 8. Why sequence reversal is the ideal demo task

Reversal was chosen deliberately:

1. **No external data needed** — sequences are generated programmatically
1. **Training is fast** — converges in under 30 epochs on CPU
1. **The alignment is perfectly interpretable** — anti-diagonal, no ambiguity
1. **Cross-attention is strictly necessary** — there is no way to reverse a
   sequence without looking back at the source; self-attention alone cannot do it
1. **Accuracy is measurable** — either the output is the exact reverse or it is not

Translation could also demonstrate alignment, but it requires a bilingual
corpus, takes longer to train, and the alignment is softer and harder to verify.
Reversal gives a clean, provable demonstration of the mechanism.

-----

## 9. Three types of attention in one model

The encoder-decoder model contains all three attention patterns:

```
ENCODER BLOCK:
┌──────────────────────────────────────────────┐
│  MultiHeadSelfAttention  (no mask)           │
│  Each source token attends to all others.    │
│  Weight matrix: (src_len × src_len) square   │
└──────────────────────────────────────────────┘

DECODER BLOCK:
┌──────────────────────────────────────────────┐
│  1. MultiHeadSelfAttention  (causal mask)    │
│     Each target token attends to past tokens  │
│     Weight matrix: (tgt_len × tgt_len) square │
│     Upper triangle masked out (no future)     │
│                                              │
│  2. MultiHeadCrossAttention                  │ ← NEW
│     Q from decoder, K/V from encoder output  │
│     Weight matrix: (tgt_len × src_len) rect. │
│     No masking — decoder can see all source  │
│                                              │
│  3. FeedForward                              │
└──────────────────────────────────────────────┘
```

The causal mask in decoder self-attention ensures the model cannot peek
at future target tokens during training. The cross-attention has no mask —
the decoder is allowed to look at any source position.

-----

## 10. Code walkthrough

### `attention.py`

The file defines four classes in order of increasing complexity:
`SelfAttention` → `CrossAttention` → `MultiHeadSelfAttention` → `MultiHeadCrossAttention`.

**`CrossAttention.forward()`** takes two arguments:

```python
def forward(self, x, context, mask=None):
    Q = self.W_Q(x).unsqueeze(1)         # Q from TARGET (x)
    K = self.W_K(context).unsqueeze(1)   # K from SOURCE (context)
    V = self.W_V(context).unsqueeze(1)   # V from SOURCE (context)
    out, w = scaled_dot_product_attention(Q, K, V, mask)
    return self.W_O(out.squeeze(1)), w.squeeze(1)
```

Compare with `SelfAttention.forward()`:

```python
def forward(self, x, mask=None):
    Q = self.W_Q(x).unsqueeze(1)   # Q from SELF
    K = self.W_K(x).unsqueeze(1)   # K from SELF
    V = self.W_V(x).unsqueeze(1)   # V from SELF
    ...
```

The only difference is that `CrossAttention` uses `context` for K and V.
`scaled_dot_product_attention()` is called identically in both cases.

**`MultiHeadCrossAttention`** splits Q differently from K and V because
they may have different sequence lengths:

```python
Q = self._split_tgt(self.W_Q(x),       B, T_tgt)   # (B, heads, tgt_len, head_dim)
K = self._split_src(self.W_K(context), B, T_src)   # (B, heads, src_len, head_dim)
V = self._split_src(self.W_V(context), B, T_src)   # (B, heads, src_len, head_dim)
```

The resulting attention weight matrix has shape `(B, heads, tgt_len, src_len)` —
one alignment matrix per head per example.

-----

### `dataset.py`

**`generate_pairs()`** creates source and target sequences:

```python
seq = [random.choice(symbols) for _ in range(length)]
src = seq
tgt = [BOS_IDX] + list(reversed(seq)) + [EOS_IDX]
```

The target is wrapped with BOS/EOS so the decoder learns:

- BOS → first reversed token (= last source token)
- last reversed token → EOS

**`ReversalDataset.__getitem__()`** applies the standard teacher-forcing shift:

```python
return (
    torch.tensor(src),        # source sequence
    torch.tensor(tgt[:-1]),   # decoder input:  [BOS, sn, ..., s1]
    torch.tensor(tgt[1:]),    # decoder target: [sn, ..., s1, s0, EOS]
)
```

At training time the decoder sees `[BOS, sn, ..., s1]` and must predict
`[sn, ..., s1, s0, EOS]`. At inference time it generates autoregressively.

-----

### `model.py`

**`DecoderBlock.forward()`** shows the three sub-layers in sequence:

```python
# 1. Causal self-attention
sa_out, _       = self.self_attn(self.norm1(x), causal_mask)
x               = x + self.drop(sa_out)

# 2. Cross-attention  ← the new piece
ca_out, cross_w = self.cross_attn(self.norm2(x), encoder_out, src_pad_mask)
x               = x + self.drop(ca_out)

# 3. FeedForward
x = x + self.drop(self.ff(self.norm3(x)))

return x, cross_w   # cross_w is returned for visualisation
```

**`EncoderDecoder.translate()`** implements greedy decoding:

```python
encoder_out = self.encoder(src, src_pad_mask)   # run ONCE

dec_ids = [bos_idx]
for _ in range(max_steps):
    tgt     = torch.tensor([dec_ids])
    dec_out, cross_ws = self.decoder(tgt, encoder_out, src_pad_mask)
    next_id  = self.head(dec_out)[0, -1, :].argmax().item()
    dec_ids.append(next_id)
    if next_id == eos_idx:
        break
```

The encoder output is computed once and reused at every decoding step —
this is why encoder-decoder models are efficient at inference: the source
is encoded once regardless of how many tokens are generated.

The final `cross_ws` contains the alignment matrices for every decoder layer
at the last decoding step. These are what `plot_alignment()` visualises.

-----

### `visualize.py`

**`plot_alignment()`** plots each head as a separate heatmap:

- X-axis = source tokens (keys — what the model attended TO)
- Y-axis = target tokens (queries — what generated EACH output token)
- Blue intensity = attention weight
- Numeric values annotated in each cell

For the reversal task, a well-trained model shows near-perfect
anti-diagonal weights. Each row has a single bright cell pointing
to the corresponding source position.

**`plot_self_vs_cross()`** puts both matrices side by side:

- Top row (green): encoder self-attention — `(src_len × src_len)` square
- Bottom row (blue): decoder cross-attention — `(tgt_len × src_len)` rectangular

The different shapes make the structural difference immediately obvious.

-----

## 11. Full data flow

Tracing one training step:

```
Source: [3, 7, 1, 5]   (token indices: [6, 10, 4, 8])
Target input:  [BOS, 5, 1, 7]   (teacher forcing)
Target target: [5, 1, 7, 3, EOS]

        ▼  Encoder
embedding([6, 10, 4, 8])  →  (1, 4, 32)
+ positional encoding
→ 2 × EncoderBlock (self-attention, no mask)
→ encoder_output  (1, 4, 32)   ← fixed, used by all decoder layers

        ▼  Decoder
embedding([BOS, 5, 1, 7])  →  (1, 4, 32)
+ positional encoding
→ DecoderBlock 1:
    causal self-attention  (sees past target tokens)
    cross-attention  Q=(1,4,32) K=V=encoder_output(1,4,32)
    → cross_w shape: (1, n_heads, 4, 4)  ← alignment matrix
    feedforward
→ DecoderBlock 2: same

→ head  →  logits  (1, 4, vocab_size)

        ▼  CrossEntropyLoss(logits, [5, 1, 7, 3, EOS])
scalar loss

        ▼  backward + Adam step
all weights updated
cross-attention W_Q, W_K, W_V learn to produce the anti-diagonal alignment
```

-----

## 12. How this maps onto mini-translator

`mini-translator` adds two things on top of this:

1. **A real bilingual corpus** — English→Spanish sentence pairs instead of
   synthetic digit sequences. The alignment will be diagonal-ish (similar
   word order) rather than anti-diagonal (reversed order).
1. **A shared or separate vocabulary** — source and target can share a
   vocabulary (simpler, used here) or have separate vocabularies
   (more realistic for different languages).

The architecture is identical:

- Encoder reads the English sentence
- Decoder generates the Spanish sentence token by token
- Cross-attention connects them
- The alignment matrix shows which English words each Spanish word attended to

When it works you will see alignments like:

```
         the   cat   sat   on   the   mat
  el    [0.9, 0.0,  0.0,  0.0, 0.0,  0.0]
  gato  [0.1, 0.85, 0.0,  0.0, 0.0,  0.0]
  se    [0.0, 0.0,  0.7,  0.0, 0.0,  0.0]
  sentó [0.0, 0.0,  0.8,  0.0, 0.0,  0.0]
  ...
```

This is the first step toward understanding how neural machine translation
actually works — not rule-based or statistical, but learned alignment from
paired examples.

-----

*Next: `mini-translator` — English→Spanish translation with full alignment visualisation.*

-----

## 13. Curiosity: combining two pretrained models

This is an advanced topic that goes beyond this series, but it is worth
understanding at a conceptual level because it connects directly to how
modern multilingual models are built.

### Why you cannot just glue two models together

Suppose you have:

- Model A: trained on English text, vocab size 5,000, embedding dim 64
- Model B: trained on Spanish text, vocab size 5,000, embedding dim 64

Each model learned its own embedding space independently. The English word
“dog” maps to some vector in Model A’s space. The Spanish word “perro” maps
to a vector in Model B’s space. But these two spaces are completely
unrelated — different random initialisations, different training data,
different gradient paths. There is no reason for “dog” and “perro” to be
anywhere near each other.

```
Model A embedding space:      Model B embedding space:
  "dog"   →  [0.3, -0.7]       "perro" → [-0.4,  0.2]
  "cat"   →  [0.4, -0.6]       "gato"  → [-0.3,  0.1]
  "house" →  [-0.2, 0.8]       "casa"  → [ 0.6, -0.5]
```

The relative geometry within each space is meaningful (dog and cat are
close in A, perro and gato are close in B), but there is no alignment
between the two spaces.

### Strategy 1: Fine-tuning with a cross-attention bridge

Add a cross-attention layer between Model A (encoder) and Model B (decoder).
Freeze both pretrained models initially. Train ONLY the cross-attention
weights on bilingual pairs. The cross-attention learns to project between
the two spaces.

```
English input → Model A (frozen) → encoder_output
                                         │
                                  cross-attention  ← only this is trained
                                         │
Spanish output ← Model B (frozen) ← decoder
```

Once the bridge is learned, unfreeze everything and fine-tune jointly
at a very low learning rate to let the spaces align gradually.

This is fast — the models already know their languages. You are only
teaching the connection.

### Strategy 2: Vocabulary alignment (MUSE approach)

Find anchor words — words that appear in both vocabularies and have stable
meanings across languages (numbers, proper nouns, cognates like “hotel”,
“internet”, “taxi”). Use these anchors to learn a rotation matrix R that
maps Model B’s space onto Model A’s space:

```
R × embed_B("perro") ≈ embed_A("dog")
R × embed_B("gato")  ≈ embed_A("cat")
```

Once R is learned, every Spanish word can be projected into the English
embedding space. The two models become compatible without any retraining.

### Strategy 3: Shared vocabulary from the start (mBERT, mT5, NLLB)

The cleanest approach for production systems: train one model on text
from many languages simultaneously, using a shared vocabulary that covers
all languages (typically byte-pair encoding, which handles any language).

Because the same embedding matrix handles “dog” and “perro”, and because
similar words appear in similar contexts across languages (if the training
data has aligned content), the model naturally learns to map semantically
similar concepts to nearby vectors regardless of language.

This is how Meta’s NLLB (No Language Left Behind) model translates between
200 language pairs — one model, one shared space, trained on everything
simultaneously.

### What this series will demonstrate

`mini-translator` will use Strategy 1 in spirit — a fresh encoder-decoder
trained on bilingual pairs from scratch, which is the simplest way to see
all the pieces working. If you want to experiment after completing the
series, replacing the encoder with a pretrained model and only training
the cross-attention bridge is a natural extension.

-----
