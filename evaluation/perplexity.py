"""Lesson 11 - is our model actually better than doing nothing?

Three numbers this file computes, cheapest-to-most-honest:

1. random baseline   - what an UNTRAINED model's loss looks like: ln(vocab_size).
2. bigram baseline    - a five-line frequency model. Beating this is the real
                        minimum bar; a lot of undertrained toy models don't.
3. our model's loss   - measured over EVERY window in val.bin exactly once
                        (not a random sample), for an honest final number.

Perplexity is just exp(loss) - the "effective number of choices" the model
faces at each step. Bits-per-byte is the fix for the fact that perplexity
is NOT comparable across different tokenizers/vocab sizes.
"""

import math

import numpy as np
import torch

DTYPE = np.uint16


def full_split_loss(model, bin_path, block_size, device, max_blocks=None, seed=1337):
    """Average loss over non-overlapping windows in a split.

    Trainer.estimate_loss() (Lesson 10) samples random windows - fine for a
    quick progress check mid-training. A final, reportable number deserves
    every token scored exactly once, with no gaps and no double-counting -
    that's what max_blocks=None (the default, used for the small val split)
    gives you. For the much larger train split, max_blocks caps the count
    (a fixed-seed random subset of the non-overlapping blocks) so this stays
    fast on CPU while still being a fair, reproducible estimate.
    """
    data = np.memmap(bin_path, dtype=DTYPE, mode="r")
    n_blocks = (len(data) - 1) // block_size
    block_ids = np.arange(n_blocks)
    if max_blocks is not None and max_blocks < n_blocks:
        rng = np.random.RandomState(seed)
        block_ids = rng.choice(n_blocks, size=max_blocks, replace=False)

    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for i in block_ids:
            start = int(i) * block_size
            x = torch.from_numpy(
                data[start:start + block_size].astype(np.int64)
            ).unsqueeze(0).to(device)
            y = torch.from_numpy(
                data[start + 1:start + block_size + 1].astype(np.int64)
            ).unsqueeze(0).to(device)
            _, loss = model(x, y)
            total_loss += loss.item() * block_size
            total_tokens += block_size

    return total_loss / total_tokens


def random_baseline_loss(vocab_size):
    """An untrained model assigns 1/vocab_size to every token - by
    definition its loss is exactly ln(vocab_size)."""
    return math.log(vocab_size)


def bigram_baseline_loss(train_bin_path, val_bin_path, vocab_size):
    """A tiny frequency model: P(next token | current token), counted
    directly from the training corpus, +1 (Laplace) smoothed.

    This is deliberately dumb - no neural network, just counting. If our
    ~10M-parameter model can't beat it, the network isn't earning its
    complexity.

    Memory note: builds a dense vocab_size x vocab_size table. At vocab
    8000 that's ~256MB (float32) - fine here, but would need a sparse
    representation for a much larger vocabulary.
    """
    train_data = np.memmap(train_bin_path, dtype=DTYPE, mode="r")
    counts = np.ones((vocab_size, vocab_size), dtype=np.float32)

    prev = train_data[:-1].astype(np.int64)
    nxt = train_data[1:].astype(np.int64)
    np.add.at(counts, (prev, nxt), 1)
    probs = counts / counts.sum(axis=1, keepdims=True)

    val_data = np.memmap(val_bin_path, dtype=DTYPE, mode="r")
    prev_v = val_data[:-1].astype(np.int64)
    nxt_v = val_data[1:].astype(np.int64)
    token_probs = probs[prev_v, nxt_v]

    return float(-np.log(token_probs).mean())


def perplexity(loss_nats):
    """exp(loss) - "how many roughly-equally-likely options is the model
    choosing between at each step." Lower is better; 1.0 would be perfect
    (impossible) prediction."""
    return math.exp(loss_nats)


def bits_per_byte(loss_nats, n_tokens, n_bytes):
    """The fix for "perplexity isn't comparable across tokenizers."

    Two models with different vocab sizes tokenize the same text into a
    different NUMBER of tokens, so their per-token loss isn't a fair
    comparison. Converting to bits-per-BYTE of the original text removes
    the tokenizer from the equation entirely.
    """
    total_bits = (loss_nats * n_tokens) / math.log(2)
    return total_bits / n_bytes
