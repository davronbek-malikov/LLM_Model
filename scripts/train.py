"""Lesson 10: teach our LLM.

    python scripts/train.py --config configs/run_01.yaml
    python scripts/train.py --config configs/run_01.yaml --resume checkpoints/last.pt

Do not run this until every check in tests/test_model.py passes. A pipeline
bug during training is far more expensive to find than one caught by those
four tests in a few seconds.

On Colab: the session can disconnect with no warning. Point --resume at
checkpoints/last.pt (ideally saved under a Drive-mounted path - pass a
different `out_dir` in the config if so) and re-run this exact command to
pick training back up from the last saved step, optimizer state included.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import torch

from src.checkpoint import load_checkpoint
from src.dataset_builder import DatasetBuilder
from src.model import OurLLM
from src.model_config import load_config
from src.seed import set_seed
from src.tokenizer import Tokenizer
from src.trainer import Trainer
from src.train_config import load_train_config

TRAIN_BIN = Path("dataset/train.bin")
VAL_BIN = Path("dataset/val.bin")
META_FILE = Path("dataset/meta.json")
TOKENIZER_FILE = Path("tokenizer/tokenizer.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/run_01.yaml"))
    parser.add_argument("--resume", type=Path, default=None,
                         help="path to a checkpoint (e.g. checkpoints/last.pt) to resume from")
    args = parser.parse_args()

    model_config = load_config(args.config)
    train_config = load_train_config(args.config)
    set_seed(train_config.seed)

    # Catch a stale config before it wastes GPU hours: the vocabulary the
    # model expects must be the vocabulary the dataset was actually built
    # with, or every embedding lookup past that boundary is garbage.
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    if meta["vocab_size"] != model_config.vocab_size:
        raise ValueError(
            "configs vocab_size ({}) does not match dataset/meta.json vocab_size "
            "({}) - re-run scripts/build_dataset.py or fix the config".format(
                model_config.vocab_size, meta["vocab_size"]
            )
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OurLLM(model_config)
    tokenizer = Tokenizer().load(TOKENIZER_FILE)
    dataset_builder = DatasetBuilder(tokenizer, seed=train_config.seed)

    trainer = Trainer(model, dataset_builder, TRAIN_BIN, VAL_BIN,
                       model_config, train_config, device)

    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        start_step, best_val_loss = load_checkpoint(
            args.resume, model, optimizer=trainer.optimizer, map_location=device
        )
        start_step += 1  # the saved step already completed
        print("Resumed from {} at step {}, best val loss so far {:.4f}".format(
            args.resume, start_step, best_val_loss
        ))

    print("=== Lesson 10: training {:,} parameters ===".format(model.num_parameters()))
    print()

    log_path = Path(train_config.out_dir) / "train_log.jsonl"
    best_val_loss = trainer.train(
        start_step=start_step, best_val_loss=best_val_loss, log_path=log_path
    )

    print()
    print("Training finished. Best val loss: {:.4f}".format(best_val_loss))
    print("Checkpoints saved under:", train_config.out_dir)
    print("Log saved to:", log_path)
    print()
    print("Next: Lesson 11 - evaluation, baselines and generation.")


if __name__ == "__main__":
    main()
