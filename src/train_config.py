"""Lesson 10 - the decisions that govern HOW we train, as opposed to WHAT we
train (that part is model_config.py).

Same philosophy as Lesson 7: write the numbers down before spending GPU
hours, so a run can be reproduced - or explained when it goes wrong - from
its config filename alone.
"""

from dataclasses import dataclass, fields

import yaml


@dataclass
class TrainConfig:
    # ---- batching -----------------------------------------------------
    # Micro-batch: how many sequences go through the model in one forward
    # pass. Limited by GPU memory, not by anything about the data.
    batch_size: int = 32

    # Effective batch size = batch_size * grad_accum_steps. Gradient
    # accumulation lets a small GPU reach a large effective batch by summing
    # gradients over several micro-batches before one optimizer step.
    grad_accum_steps: int = 4

    # ---- schedule -------------------------------------------------------
    # Training is measured in optimizer steps, one step = one effective
    # batch. max_steps * batch_size * grad_accum_steps * block_size is the
    # real unit that matters: tokens seen.
    max_steps: int = 3000
    warmup_steps: int = 100

    # Peak learning rate, reached at the end of warmup, then cosine-decayed
    # down to min_lr_ratio * lr_max for the rest of the run.
    lr_max: float = 3e-4
    min_lr_ratio: float = 0.1

    # ---- optimizer --------------------------------------------------------
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # ---- logging / checkpointing ------------------------------------------
    log_interval: int = 20          # print every N optimizer steps
    eval_interval: int = 250        # measure val loss every N steps
    eval_iters: int = 50            # batches averaged per val-loss estimate
    checkpoint_interval: int = 250  # save a resumable checkpoint every N steps

    # ---- misc ---------------------------------------------------------
    seed: int = 1337
    # "auto" picks bf16 on GPUs that support it, else fp16 on CUDA, else
    # fp32. A T4 (Colab's free GPU) has no real bf16 tensor cores, so "auto"
    # lands on fp16 there - explicit if you ever need to force one.
    precision: str = "auto"
    out_dir: str = "checkpoints"

    def tokens_per_step(self, block_size):
        return self.batch_size * self.grad_accum_steps * block_size


def load_train_config(path):
    """Read a TrainConfig from the same YAML file model_config.py reads.

    Unknown keys (the architecture ones) are ignored, mirroring
    model_config.load_config's handling of training keys - each loader
    reads only the slice of the file it understands.
    """
    with open(path, "r", encoding="utf-8") as f:
        values = yaml.safe_load(f)

    known = {f.name for f in fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in values.items() if k in known})
