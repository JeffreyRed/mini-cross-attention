"""
dataset.py — Sequence reversal dataset for cross-attention demonstration.

Why sequence reversal?
  It is the simplest task that REQUIRES cross-attention to work.
  To generate output token i (which is source token [n-1-i]),
  the decoder must attend to a specific source position.
  After training, the cross-attention weight matrix will show a clear
  anti-diagonal pattern: output position 0 attends to source position n-1,
  output position 1 attends to source position n-2, etc.

  This makes the alignment completely interpretable and verifiable.

  Example:
    source:  [A, B, C, D, E]
    target:  [E, D, C, B, A]

    Expected cross-attention (anti-diagonal):
         A    B    C    D    E       ← source tokens
    E  [0.0, 0.0, 0.0, 0.0, 1.0]
    D  [0.0, 0.0, 0.0, 1.0, 0.0]
    C  [0.0, 0.0, 1.0, 0.0, 0.0]
    B  [0.0, 1.0, 0.0, 0.0, 0.0]
    A  [1.0, 0.0, 0.0, 0.0, 0.0]

Vocabulary:
  10 symbols (0-9) + PAD + BOS + EOS.
  Sequences of length 3-6 are generated randomly.
  Each (source, target) pair is a distinct training example.
"""

import torch
import random
from torch.utils.data import Dataset
from typing import List, Tuple, Dict


# Special token indices
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
VOCAB   = {
    "<PAD>": 0, "<BOS>": 1, "<EOS>": 2,
    "0": 3, "1": 4, "2": 5, "3": 6, "4": 7,
    "5": 8, "6": 9, "7": 10, "8": 11, "9": 12,
}
IDX2TOK = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)


def generate_pairs(
    n_samples  : int = 2000,
    min_len    : int = 3,
    max_len    : int = 6,
    seed       : int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """
    Generates (source, target) pairs where target = reverse(source).

    Returns:
        List of (source_ids, target_ids) tuples.
        source_ids: plain symbol sequence  [s0, s1, ..., sn]
        target_ids: BOS + reversed + EOS  [BOS, sn, ..., s0, EOS]
    """
    random.seed(seed)
    symbols = list(range(3, 3 + 10))   # token indices for "0"-"9"
    pairs   = []

    for _ in range(n_samples):
        length = random.randint(min_len, max_len)
        seq    = [random.choice(symbols) for _ in range(length)]
        src    = seq
        tgt    = [BOS_IDX] + list(reversed(seq)) + [EOS_IDX]
        pairs.append((src, tgt))

    return pairs


class ReversalDataset(Dataset):
    """
    Dataset of (source, decoder_input, decoder_target) triples for
    teacher-forced sequence reversal.

    decoder_input  = tgt[:-1]  = [BOS, sn, ..., s1]
    decoder_target = tgt[1:]   = [sn, ..., s1, s0, EOS]

    Args:
        pairs:      list of (source_ids, target_ids) from generate_pairs()
        val_split:  fraction held out for validation
        train:      if True, return train split; if False, return val split
        seed:       random seed for the split
    """

    def __init__(
        self,
        pairs     : List[Tuple],
        val_split : float = 0.1,
        train     : bool  = True,
        seed      : int   = 42,
    ) -> None:
        random.seed(seed)
        shuffled = pairs.copy()
        random.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_split))

        if train:
            self.pairs = shuffled[n_val:]
        else:
            self.pairs = shuffled[:n_val]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        src, tgt = self.pairs[idx]
        return (
            torch.tensor(src,      dtype=torch.long),   # source
            torch.tensor(tgt[:-1], dtype=torch.long),   # decoder input  (BOS ... last-1)
            torch.tensor(tgt[1:],  dtype=torch.long),   # decoder target (first+1 ... EOS)
        )

    def __repr__(self) -> str:
        return f"ReversalDataset(examples={len(self.pairs)})"


def collate_fn(batch, pad_idx=PAD_IDX):
    """Pads all three tensors to the longest in the batch."""
    srcs, dec_ins, dec_tgts = zip(*batch)

    def pad(seqs):
        max_len = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :s.size(0)] = s
        return out

    return pad(srcs), pad(dec_ins), pad(dec_tgts)
