"""Step 5: train our own tokenizer on the final corpus.

    python scripts/train_tokenizer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.jsonl_io import read_jsonl
from src.tokenizer import Tokenizer

INPUT = Path("data/final/tinystories.jsonl")
OUTPUT = Path("tokenizer/tokenizer.json")

VOCAB_SIZE = 8000

# BPE statistics settle quickly: 200k stories produce almost the same merges
# as 2 million, in a fraction of the time.
MAX_DOCUMENTS = 200000

documents = read_jsonl(INPUT, limit=MAX_DOCUMENTS)
texts = [doc["text"] for doc in documents]
print("Training on", len(texts), "documents")

tokenizer = Tokenizer()
tokenizer.train(texts, vocab_size=VOCAB_SIZE)
tokenizer.save(OUTPUT)

# Round-trip test: encode then decode must give back exactly the original.
# If this fails, every token ID in the dataset is suspect.
sample = "Once upon a time there was a little girl."
ids = tokenizer.encode(sample)
back = tokenizer.decode(ids)

print()
print("Text   :", sample)
print("IDs    :", ids)
print("Decoded:", back)
print("Round-trip OK:", back == sample)
