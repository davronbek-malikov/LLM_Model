"""Lesson 9 - turning logits into a next token.

model.generate() (Lesson 8) always argmax's - deterministic, but boring and
prone to repeating itself. Real serving needs a choice of HOW to pick the
next token from the model's predicted distribution. These functions all take
the same input (one row of logits, shape (vocab_size,)) and return one token
id, so the engine can swap between them freely.
"""

import torch
import torch.nn.functional as F


def apply_temperature(logits, temperature):
    """Scale logits before softmax - the single knob with the biggest effect.

    temperature -> 0   : the distribution collapses toward argmax (greedy).
    temperature == 1   : the model's own probabilities, unchanged.
    temperature -> inf : the distribution flattens toward uniform (random).

    Dividing logits (not probabilities) by temperature is what makes this
    work: softmax(logits / T) sharpens the distribution for T < 1 and
    flattens it for T > 1, without ever needing to touch probabilities
    directly.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 - use top_k=1 for greedy")
    return logits / temperature


def top_k_filter(logits, k):
    """Keep only the k highest-probability tokens; -inf out everything else.

    Prevents the model from ever picking one of its many low-probability
    "junk" tokens, which is what causes a lot of incoherent generations at
    high temperature.
    """
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k)
    threshold = values[..., -1, None]
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def top_p_filter(logits, p):
    """Nucleus sampling: keep the smallest set of tokens whose probabilities
    sum to at least p, drop the rest.

    Where top-k always keeps exactly k tokens, top-p adapts: a peaked
    distribution keeps very few tokens, a flat one keeps many. This is the
    "distribution is very flat at one step and very peaked at the next"
    case your roadmap's viva question asks about.
    """
    if p is None or p >= 1.0:
        return logits

    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Shift right by one so the token that PUSHES the cumulative sum past p
    # is still included - otherwise we could end up with zero tokens kept.
    sorted_remove = cumulative > p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False

    remove_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        -1, sorted_idx, sorted_remove
    )
    return logits.masked_fill(remove_mask, float("-inf"))


def apply_repetition_penalty(logits, generated_ids, penalty):
    """Discourage tokens that already appeared in this generation.

    Divides the logit of any already-seen token by `penalty` (if positive)
    or multiplies (if negative), which pushes previously-used tokens toward
    a lower probability without banning them outright. penalty=1.0 is a
    no-op. This is what stops the classic "the the the the..." failure mode
    that pure greedy decoding is prone to.
    """
    if penalty == 1.0 or not generated_ids:
        return logits
    logits = logits.clone()
    seen = torch.tensor(sorted(set(generated_ids)), device=logits.device)
    scores = logits[..., seen]
    logits[..., seen] = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits


def sample_next_token(logits, generated_ids, temperature=0.8, top_k=50,
                       top_p=0.95, repetition_penalty=1.1):
    """The full pipeline: penalty -> temperature -> top-k -> top-p -> sample.

    `logits` is one row, shape (vocab_size,) - the model's raw output for the
    next position. Order matters: repetition penalty and temperature reshape
    the whole distribution first, then top-k/top-p trim the tail, and only
    then do we actually sample - trimming before reshaping would filter on
    the wrong (unadjusted) probabilities.
    """
    logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)
    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
