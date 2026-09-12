"""Lesson 8, Part 4 - OurLLM: the full model.

    tokens    -> token embedding    -+
                                      +-> add -> N x TransformerBlock -> LayerNorm -> LM head -> logits
    positions -> position embedding -+

Everything from Month 1 becomes code here: embeddings, attention, the
feed-forward network, and the head that turns the final hidden state back
into a probability over the vocabulary.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.transformer_block import TransformerBlock


class OurLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            # Reuse the embedding table as the output layer - input tokens
            # and output predictions live in the same "meaning space", so
            # sharing the weights is free quality and fewer parameters to
            # train (see ModelConfig.estimate_parameters).
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Small random weights at the start, per the GPT-2 recipe."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_parameters(self):
        """The real parameter count, to compare against Lesson 7's estimate."""
        n = sum(p.numel() for p in self.parameters())
        if self.config.tie_weights:
            # lm_head.weight IS token_embedding.weight (same tensor) when
            # tied, so counting both would double-count it.
            n -= self.lm_head.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, "sequence longer than block_size"

        positions = torch.arange(T, device=idx.device)

        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Score every position at once: flatten (B, T, vocab) -> (B*T,
            # vocab) and (B, T) -> (B*T,), then one cross-entropy call scores
            # all B*T predictions together.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Greedy generation: always pick the single most likely next token.

        Sampling (temperature, top-k, top-p) arrives in Lesson 9 - this is
        deliberately the simplest possible version, just to prove the loop
        (encode -> forward -> pick token -> append -> repeat) works end to
        end.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :]  # only the last position matters
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
