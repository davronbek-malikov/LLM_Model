"""Lesson 10 - saving and resuming a run.

On Colab, the session can vanish with no warning at all - a disconnect, a
runtime recycle, a closed laptop lid. A checkpoint that only has the model
weights loses the run's PLACE: the optimizer's momentum and second-moment
estimates, which optimizer step we were on (and therefore where we are on
the learning-rate curve), and which seed produced this exact trajectory.
Restoring only the weights after a crash silently restarts optimisation from
a cold state while pretending training continued - worse than a clean
restart, because it looks fine on a loss curve until you look closely.

So every checkpoint here is the full picture: model, optimizer, step, the
configs that produced it, and the seed.
"""

import os
import tempfile
from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, step, best_val_loss,
                     model_config, train_config):
    """Write a checkpoint atomically.

    Writing straight to `path` and then losing power/connection mid-write
    leaves a corrupt file with the right name - the worst possible outcome,
    because --resume will pick it up and fail confusingly. Writing to a
    temp file first and renaming only once the write is complete means the
    checkpoint at `path` is always either the previous good one or the new
    good one, never a half-written third thing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "best_val_loss": best_val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model_config.to_dict(),
        "train_config": vars(train_config),
    }

    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)  # atomic on both POSIX and Windows
    except Exception:
        os.unlink(tmp_path)
        raise


def load_checkpoint(path, model, optimizer=None, map_location=None):
    """Restore a checkpoint, returning (step, best_val_loss) to resume from.

    `optimizer` is optional so the same function can load just the weights
    for a Lesson 9-style serving dry run, which has no optimizer at all.
    """
    # weights_only=True: this checkpoint holds only tensors and plain Python
    # dicts/ints/floats/strings, never custom classes, so the restricted
    # (safe) unpickler is sufficient and avoids arbitrary code execution on
    # load - worth keeping even for our own checkpoints, since one might get
    # copied around (Drive, a shared Kaggle dataset) before it is loaded.
    payload = torch.load(path, map_location=map_location, weights_only=True)

    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    return payload["step"], payload["best_val_loss"]
