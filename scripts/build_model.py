"""Step 8 (Lesson 8): build our model from the Lesson 7 design.

    python scripts/build_model.py

This script does not train anything. It reads configs/run_01.yaml, builds
OurLLM from it, and prints the model plus its real parameter count - so we
can compare that real number against the estimate scripts/design_model.py
made in Lesson 7.

Before trusting this model with a single GPU-hour, run:

    python tests/test_model.py

and make sure all four checks pass.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.model_config import load_config
from src.model import OurLLM

CONFIG_FILE = Path("configs/run_01.yaml")

config = load_config(CONFIG_FILE)
model = OurLLM(config)

print("=== Our LLM ===")
print(model)
print()
print("Parameters (real count)        : {:,}".format(model.num_parameters()))
print("Parameters (Lesson 7 estimate) : {:,}".format(
    config.estimate_parameters()["total"]))
print()

# A tiny end-to-end smoke test: does a forward pass even run, at the right shape?
batch = torch.randint(0, config.vocab_size, (2, config.block_size))
logits, _ = model(batch)
print("Smoke test forward pass -> logits shape:", tuple(logits.shape))
print()
print("Next: run 'python tests/test_model.py' before Lesson 10.")
