"""Lesson 0's promise, finally paid off: one set_seed() used by every script.

Fixing the seed does not make training deterministic in a strict bit-for-bit
sense (cuDNN kernels still vary), but it removes the single biggest source of
"did that change actually help, or did I just get a luckier shuffle?" - which
is exactly the question the two-run comparison in the training window needs
to answer honestly.
"""

import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
