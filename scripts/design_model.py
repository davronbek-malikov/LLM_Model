"""Step 7 (Lesson 7): design our model and check that we can afford it.

    python scripts/design_model.py

This script builds nothing. It reads configs/run_01.yaml, multiplies a few
numbers, and tells us whether the design is sensible BEFORE Lesson 8 turns it
into code and Lesson 10 spends ten GPU hours on it.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_config import load_config

CONFIG_FILE = Path("configs/run_01.yaml")
META_FILE = Path("dataset/meta.json")

config = load_config(CONFIG_FILE)
params = config.estimate_parameters()

print("=== Our LLM Model Design ===")
print()
print("Vocabulary size    :", config.vocab_size)
print("Context length     :", config.block_size)
print("Embedding dimension:", config.n_embd)
print("Layers             :", config.n_layer)
print("Attention heads    :", config.n_head, "(head dim", config.head_dim(), ")")
print("Dropout            :", config.dropout)
print("FFN multiplier     :", config.ffn_mult)
print("Tied weights       :", config.tie_weights)
print()

print("Estimated parameters")
print("  non-embedding :  {:>12,}".format(params["non_embedding"]))
print("  embedding     :  {:>12,}".format(params["embedding"]))
print("  position      :  {:>12,}".format(params["position"]))
print("  TOTAL         :  {:>12,}   (~{:.1f}M)".format(
    params["total"], params["total"] / 1e6))
print()

print("Rough training memory for weights + gradients + AdamW: {:.0f} MB".format(
    config.estimate_training_memory_mb()))
print("(activations add more on top, and they grow with batch size)")
print()

# The compute-optimal check. About 20 training tokens per parameter is the
# standard rule of thumb. Fewer means the model is bigger than our data can
# teach; many more means we could have afforded a bigger model.
if META_FILE.exists():
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    n_tokens = meta["n_tokens_train"]
    ratio = n_tokens / params["total"]

    print("Training tokens available: {:,}".format(n_tokens))
    print("Tokens per parameter     : {:.1f}   (about 20 is the target)".format(ratio))

    if ratio < 10:
        print("-> The model is large for this corpus. Shrink it, or get more data.")
    elif ratio > 40:
        print("-> Plenty of data for this model. We could afford to go bigger.")
    else:
        print("-> Reasonably balanced.")
else:
    print("No dataset/meta.json yet - run scripts/build_dataset.py to check")
    print("the tokens-per-parameter ratio against a real corpus.")

print()
print("Why this model?")
print()
print("We selected this configuration based on:")
print("- our dataset size (how many tokens we actually have)")
print("- available compute (which GPU, for how many hours)")
print("- memory limitations (it has to fit while training, not just exist)")
print("- inference cost (a smaller model is cheaper to serve, forever)")
print()
print("A bigger model is not automatically a better model.")
print("Next lesson: we build this design in code.")
