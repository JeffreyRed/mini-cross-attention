"""
train.py — Training loop for the encoder-decoder reversal model.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Tuple

from src.model   import EncoderDecoder
from src.dataset import ReversalDataset, collate_fn, PAD_IDX


def cosine_lr(step, warmup, total, peak, min_lr=1e-5):
    if step < warmup:
        return peak * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, dataset, batch_size=64):
    model.eval()
    loader  = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    total, n = 0.0, 0
    correct, total_toks = 0, 0

    for src, dec_in, dec_tgt in loader:
        logits, _ = model(src, dec_in)
        loss = loss_fn(logits.transpose(1, 2), dec_tgt)
        total += loss.item(); n += 1

        # Token accuracy
        preds       = logits.argmax(-1)
        active      = dec_tgt != PAD_IDX
        correct    += (preds == dec_tgt)[active].sum().item()
        total_toks += active.sum().item()

    mean_loss = total / max(n, 1)
    accuracy  = correct / max(total_toks, 1)
    return mean_loss, math.exp(min(mean_loss, 30)), accuracy


def train(
    model          : EncoderDecoder,
    train_dataset  : ReversalDataset,
    val_dataset    : ReversalDataset,
    epochs         : int   = 30,
    lr             : float = 3e-3,
    batch_size     : int   = 64,
    warmup_steps   : int   = 200,
    verbose        : bool  = True,
) -> Tuple[list, list]:
    """
    Trains the encoder-decoder model.

    Returns:
        train_history : list of (loss, perplexity, accuracy) per epoch
        val_history   : same for validation
    """
    loader    = DataLoader(train_dataset, batch_size=batch_size,
                           shuffle=True, collate_fn=collate_fn)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))

    total_steps = epochs * len(loader)
    step        = 0
    train_hist  = []
    val_hist    = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for src, dec_in, dec_tgt in loader:
            step += 1
            for pg in optimizer.param_groups:
                pg["lr"] = cosine_lr(step, warmup_steps, total_steps, lr)

            optimizer.zero_grad()
            logits, _ = model(src, dec_in)
            loss = loss_fn(logits.transpose(1, 2), dec_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        t_loss = total_loss / len(loader)
        t_ppl  = math.exp(min(t_loss, 30))
        _, _, t_acc = evaluate(model, train_dataset, batch_size)
        train_hist.append((t_loss, t_ppl, t_acc))

        v_loss, v_ppl, v_acc = evaluate(model, val_dataset, batch_size)
        val_hist.append((v_loss, v_ppl, v_acc))

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"Epoch [{epoch:>3}/{epochs}]  "
                f"loss={t_loss:.4f}  ppl={t_ppl:.2f}  "
                f"train_acc={t_acc:.3f}  val_acc={v_acc:.3f}"
            )

    return train_hist, val_hist
