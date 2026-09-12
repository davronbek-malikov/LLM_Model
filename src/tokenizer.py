"""Stage 5 - the tokenizer.

The model never sees text. It sees integers. The tokenizer is the only thing
that knows how to go between the two:

    "Once upon a time"  --encode-->  [3891, 552, 261, 1204]
    [3891, 552, 261, 1204]  --decode-->  "Once upon a time"

We use byte-level BPE from the Hugging Face `tokenizers` library. We do not
write the BPE algorithm ourselves - that is a separate lesson, and a library
that a thousand projects depend on has fewer bugs than our afternoon.

We train on our FINAL corpus only, never on raw text. A tokenizer is a mirror
of the text it was trained on: train it on cookie banners and it will happily
learn a single token for "Accept all cookies".
"""

from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import models, pre_tokenizers, decoders, trainers

# <eos> marks the end of a document. It is the only special token our training
# loop actually uses, so it is the only one we add. Unused special tokens are
# vocabulary slots paid for and never spent.
SPECIAL_TOKENS = ["<eos>"]


class Tokenizer:
    def __init__(self):
        self.tokenizer = None

    def train(self, texts, vocab_size=8000, min_frequency=2):
        """Learn a vocabulary from our corpus."""
        tokenizer = HFTokenizer(models.BPE())

        # Byte-level: the alphabet is the 256 possible bytes, so ANY text can
        # be encoded. There is no <unk> token because there cannot be an
        # unknown character.
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            # A pair seen only once is noise, not a pattern worth a token.
            min_frequency=min_frequency,
            show_progress=True,
        )

        tokenizer.train_from_iterator(texts, trainer=trainer)
        self.tokenizer = tokenizer

        print("Trained tokenizer. Vocabulary size:", self.vocab_size())
        return self

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))
        print("Saved tokenizer to", path)

    def load(self, path):
        self.tokenizer = HFTokenizer.from_file(str(path))
        return self

    def encode(self, text):
        """text -> list of token IDs"""
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        """list of token IDs -> text"""
        return self.tokenizer.decode(ids)

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def eos_id(self):
        """The ID of <eos>. The dataset builder needs it to mark document
        boundaries."""
        return self.tokenizer.token_to_id("<eos>")
