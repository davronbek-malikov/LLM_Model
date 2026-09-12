"""Stage 6 - the dataset.

This turns our final corpus into what the training loop actually reads:

    final.jsonl  ->  split  ->  encode  ->  train.bin / val.bin / meta.json

Three ideas live in this file, and all three are exam material.

1. SPLIT BEFORE ENCODING, at document level.
   If we split after packing, a single story can end up half in train and half
   in validation. Validation loss then measures memory, not learning.

2. PACKING.
   We glue all documents together into one long stream of token IDs, with
   <eos> between them, and store it as a flat uint16 array. Training then cuts
   random windows out of that stream. This is far faster than tokenizing text
   on the fly, and uint16 halves the disk size (it holds any ID below 65,536).

3. THE INPUT/TARGET SHIFT.
       x = tokens[i     : i + T]
       y = tokens[i + 1 : i + T + 1]
   y is x moved one step to the left. At every position the model sees x and
   must predict y - so one window of length T gives T training signals, not
   one. Draw this on paper. Getting it off by one is the most common silent
   bug in the whole project, and it does not crash; it just trains a worse
   model.
"""

import json
import random
import time
from pathlib import Path

import numpy as np

from src.jsonl_io import read_jsonl

# uint16 = 2 bytes per token. Valid as long as vocab_size <= 65,536.
DTYPE = np.uint16


class DatasetBuilder:
    def __init__(self, tokenizer, val_ratio=0.01, seed=1337):
        self.tokenizer = tokenizer
        self.val_ratio = val_ratio
        self.seed = seed

    def split_documents(self, documents):
        """Shuffle, then cut off a validation slice - at DOCUMENT level."""
        indices = list(range(len(documents)))

        # A fixed seed means the same split every time the script runs.
        # Reproducibility is not a nice extra; without it you cannot tell a
        # real improvement from a lucky shuffle.
        random.Random(self.seed).shuffle(indices)

        n_val = max(1, int(len(documents) * self.val_ratio))
        val_docs = [documents[i] for i in indices[:n_val]]
        train_docs = [documents[i] for i in indices[n_val:]]
        return train_docs, val_docs

    def encode_to_bin(self, documents, output_path):
        """Encode documents into one flat binary file of token IDs."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        eos_id = self.tokenizer.eos_id()
        n_tokens = 0

        with open(output_path, "wb") as f:
            for doc in documents:
                ids = self.tokenizer.encode(doc["text"])
                ids.append(eos_id)          # mark where this document ends

                array = np.array(ids, dtype=DTYPE)
                array.tofile(f)             # write straight to disk
                n_tokens += array.size

        print(output_path.name, "->", n_tokens, "tokens")
        return n_tokens

    def get_batch(self, bin_path, batch_size, block_size):
        """Cut random windows out of the token stream.

        This is the function the training loop will call in Lesson 10. We write
        it now because it is the clearest possible demonstration of the shift.
        """
        # memmap: the file stays on disk, only the slices we ask for are read.
        # A 4 GB token file does not need 4 GB of RAM.
        data = np.memmap(bin_path, dtype=DTYPE, mode="r")

        # -1 so that y still has one token left to look at past the end of x.
        starts = np.random.randint(0, len(data) - block_size - 1, size=batch_size)

        x = np.stack([data[i:i + block_size].astype(np.int64) for i in starts])
        y = np.stack([data[i + 1:i + block_size + 1].astype(np.int64) for i in starts])
        return x, y

    def run(self, input_path, train_bin, val_bin, meta_path, block_size=256,
            limit=None):
        documents = read_jsonl(Path(input_path), limit=limit)
        print("Documents:", len(documents))

        train_docs, val_docs = self.split_documents(documents)
        print("Split ->", len(train_docs), "train /", len(val_docs), "val")

        n_train = self.encode_to_bin(train_docs, train_bin)
        n_val = self.encode_to_bin(val_docs, val_bin)

        # meta.json is how Lesson 7 and Lesson 8 find out what this dataset is
        # without re-reading gigabytes of tokens.
        meta = {
            "vocab_size": self.tokenizer.vocab_size(),
            "eos_id": self.tokenizer.eos_id(),
            "dtype": np.dtype(DTYPE).name,
            "block_size": block_size,
            "seed": self.seed,
            "val_ratio": self.val_ratio,
            "n_docs_train": len(train_docs),
            "n_docs_val": len(val_docs),
            "n_tokens_train": int(n_train),
            "n_tokens_val": int(n_val),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        meta_path = Path(meta_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print("Saved", meta_path)
        return meta
