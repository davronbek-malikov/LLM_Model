"""Lesson 9 - the inference engine: load once, generate many times.

    prompt -> tokenizer.encode -> token IDs
          -> model forward (looped)  -> logits for the last position
          -> sampling               -> next token ID
          -> append, repeat until max_tokens or <eos>
          -> tokenizer.decode -> text

The one rule that matters most for latency: the model and tokenizer are
loaded ONCE, in __init__, not per request. Loading a 10M-parameter model from
disk takes real time (hundreds of ms); doing that on every API call would
dominate the actual generation cost.

No KV cache here, on purpose - every new token re-runs the FULL sequence
through the model instead of reusing previous keys/values, which is
quadratic-ish waste for anything beyond a short prompt. For a model this
small, on a CPU-only free host, the difference isn't the bottleneck yet, but
it's worth being able to say precisely why a bigger model would need one.
"""

import time
from pathlib import Path

import torch

from src.checkpoint import load_checkpoint
from src.model import OurLLM
from src.model_config import load_config
from src.tokenizer import Tokenizer
from inference.sampling import sample_next_token


class InferenceEngine:
    def __init__(self, config_path, checkpoint_path, tokenizer_path, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.config = load_config(config_path)
        self.model = OurLLM(self.config).to(self.device)
        load_checkpoint(checkpoint_path, self.model, map_location=self.device)
        self.model.eval()

        self.tokenizer = Tokenizer().load(tokenizer_path)
        self.eos_id = self.tokenizer.eos_id()

        print("Inference engine ready: {:,} parameters on {}".format(
            self.model.num_parameters(), self.device
        ))

    @torch.no_grad()
    def generate(self, prompt, max_tokens=50, temperature=0.8, top_k=50,
                 top_p=0.95, repetition_penalty=1.1):
        """Returns (text, tokens_generated, latency_ms)."""
        start = time.perf_counter()

        ids = self.tokenizer.encode(prompt)
        generated = list(ids)
        idx = torch.tensor([ids], device=self.device)

        for _ in range(max_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self.model(idx_cond)
            next_logits = logits[0, -1, :]

            next_id = sample_next_token(
                next_logits, generated,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            if next_id == self.eos_id:
                break

            generated.append(next_id)
            idx = torch.cat(
                [idx, torch.tensor([[next_id]], device=self.device)], dim=1
            )

        text = self.tokenizer.decode(generated)
        latency_ms = (time.perf_counter() - start) * 1000
        tokens_generated = len(generated) - len(ids)
        return text, tokens_generated, latency_ms


def default_engine():
    """Convenience constructor using the repo's standard file layout."""
    return InferenceEngine(
        config_path=Path("configs/run_01.yaml"),
        checkpoint_path=Path("checkpoints/best.pt"),
        tokenizer_path=Path("tokenizer/tokenizer.json"),
    )
