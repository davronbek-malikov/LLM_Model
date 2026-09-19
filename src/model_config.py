"""Lesson 7 - the design of our model.

There is no model in this file. There is no attention, no transformer block,
no training loop. There are only the DECISIONS, written down, plus enough
arithmetic to find out whether those decisions are affordable.

    Architecture
          |
    Parameter count
          |
    Memory requirement
          |
    Training feasibility

That chain is the whole lesson. A model you cannot fit in your GPU is not a
design, it is a wish - and the only way to find out before burning ten hours
of compute is to multiply the numbers first.

Lesson 8 will read this config and build the real model from it.
"""

from dataclasses import dataclass, asdict, fields

import yaml


@dataclass
class ModelConfig:
    """The eight numbers that define our model.

    The values below are an EXAMPLE, sized for a free Colab T4. Yours should
    come from your own dataset, your own GPU and your own patience.
    """

    # How many different tokens exist. Comes from the tokenizer, not from
    # taste - read it out of dataset/meta.json.
    vocab_size: int = 8000

    # Context length: how many tokens the model can look back at.
    # Attention cost grows with the SQUARE of this number, so doubling it
    # roughly quadruples the cost of the attention step.
    block_size: int = 256

    # Embedding dimension - the "width" of the model. Every parameter count
    # below depends on its square, so this is the most expensive dial here.
    n_embd: int = 384

    # How many transformer blocks are stacked - the "depth" of the model.
    n_layer: int = 6

    # Attention heads per block. n_embd must divide evenly by n_head, because
    # the heads split the embedding between them.
    n_head: int = 6

    # Randomly zeroes activations during training to fight overfitting.
    # A small model on a small corpus WILL overfit, so we keep some.
    dropout: float = 0.1

    # The feed-forward layer is this many times wider than n_embd.
    # 4 is the classic GPT-2 choice.
    ffn_mult: int = 4

    # Reuse the embedding table as the output layer. Free quality, and it
    # saves vocab_size * n_embd parameters.
    tie_weights: bool = True

    def head_dim(self):
        """Size of one attention head."""
        return self.n_embd // self.n_head

    def check(self):
        """Catch the two mistakes that would break Lesson 8 on line one."""
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                "n_embd ({}) must divide evenly by n_head ({})".format(
                    self.n_embd, self.n_head)
            )
        if self.vocab_size > 65536:
            raise ValueError(
                "vocab_size above 65536 does not fit in the uint16 token files"
            )

    def estimate_parameters(self):
        """A rough parameter count. Deliberately rough.

        Non-embedding parameters, per transformer block, come out at about
        12 * n_embd^2:

            attention  Q, K, V and the output projection  ->  4 * n_embd^2
            feed-forward  n_embd -> 4*n_embd -> n_embd    ->  8 * n_embd^2

        so the stack of blocks is roughly 12 * n_layer * n_embd^2. Biases and
        normalisation layers add a few thousand more; at this scale they do
        not change any decision, so we ignore them.

        The embedding table is a separate cost: vocab_size * n_embd. With
        tie_weights=True the output layer reuses it, so we count it once.
        """
        non_embedding = 12 * self.n_layer * self.n_embd ** 2
        embedding = self.vocab_size * self.n_embd
        position = self.block_size * self.n_embd   # learned position embeddings

        total = non_embedding + embedding + position
        if not self.tie_weights:
            total += self.vocab_size * self.n_embd   # a separate output layer

        return {
            "non_embedding": non_embedding,
            "embedding": embedding,
            "position": position,
            "total": total,
        }

    def estimate_training_memory_mb(self):
        """Very rough GPU memory for the weights during training.

        Each parameter is stored four times over: the weight itself, its
        gradient, and the two running averages AdamW keeps. At 4 bytes each
        that is 16 bytes per parameter.

        Activations cost more on top of this and depend on batch size, so treat
        this as a floor, not a budget.
        """
        total = self.estimate_parameters()["total"]
        return total * 16 / (1024 ** 2)

    def to_dict(self):
        return asdict(self)


def load_config(path):
    """Read a ModelConfig from a YAML file.

    The config lives in a file and not inside a .py file for one reason: six
    weeks from now, the only question that matters is "which settings produced
    this checkpoint?", and a filename can answer it.

    From Lesson 10 onward, one run's YAML holds BOTH architecture fields (read
    here) and training fields (read by src/train_config.py) side by side, so
    a single filename still names the whole experiment. Unknown keys - the
    training ones - are ignored here rather than raising, so this loader does
    not need to change every time a training hyperparameter is added.
    """
    with open(path, "r", encoding="utf-8") as f:
        values = yaml.safe_load(f)

    known = {f.name for f in fields(ModelConfig)}
    config = ModelConfig(**{k: v for k, v in values.items() if k in known})
    config.check()
    return config
