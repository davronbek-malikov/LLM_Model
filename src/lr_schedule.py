"""Lesson 10 - the learning-rate schedule: linear warmup, then cosine decay.

    lr
    |        ____
    |      /      \\___
    |    /             \\___
    |  /                    \\________
    | /
    |/________________________________ step
     0   warmup_steps            max_steps

Two failure modes this prevents:

1. No warmup: the first few steps take large, confident weight updates from
   a randomly-initialised model that does not deserve that confidence yet.
   This is exactly what causes the loss-spike-then-diverge pattern your
   roadmap calls "a very persuasive demonstration" - persuasive because it
   is easy to trigger by simply deleting the warmup.

2. No decay: a learning rate held at its peak for the whole run keeps the
   weights bouncing around near a minimum instead of settling into it, which
   shows up as a validation loss that stops improving well before it should.
"""

import math


def get_lr(step, config):
    """The learning rate for one optimizer step, as a pure function of step.

    Pure and stateless on purpose: resuming from a checkpoint just means
    calling this with a larger `step`, no separate state to save or restore.
    """
    if step < config.warmup_steps:
        # Linear ramp from 0 up to lr_max, reached exactly at warmup_steps.
        return config.lr_max * (step + 1) / config.warmup_steps

    if step >= config.max_steps:
        return config.lr_max * config.min_lr_ratio

    # Cosine decay from lr_max down to lr_max * min_lr_ratio over the
    # remaining steps after warmup.
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    lr_min = config.lr_max * config.min_lr_ratio
    return lr_min + coeff * (config.lr_max - lr_min)
