# Lesson 10 Teaching Guide — Pretraining: Teach Our LLM

Opening line for class: *the model and data have been ready since Lesson 8 -
today we finally spend GPU hours making the weights actually learn something.*

Six new files this lesson adds, plus one change to a Lesson 8 file. Suggested
teaching order below goes roughly bottom-up, the same way Lesson 8 built the
model piece by piece — each file is small enough to read in full on screen.

## 1. Reproducibility, first, before anything else

- [src/seed.py](src/seed.py) — the `set_seed()` helper Lesson 0 promised.
  One line each for `random`, `numpy`, and `torch`. Point out *why* it matters
  here specifically: the training window later asks for two runs that differ
  in exactly one variable — without a fixed seed, a student can't tell a real
  improvement from a lucky shuffle.

## 2. The learning-rate schedule (pure function, easy to reason about)

- [src/lr_schedule.py](src/lr_schedule.py) — `get_lr(step, config)`: linear
  warmup, then cosine decay. Worth actually plotting this on the whiteboard
  (x = step, y = lr) before showing the code — students should recognize the
  shape before they read the math. Ask: *what happens to the first few steps
  if we delete the warmup branch?* (Answer: the model takes large, confident
  updates before it's earned any confidence — the loss-spike-then-diverge
  failure mode.)

## 3. Parameter groups and the optimizer

- [src/model.py:39-59](src/model.py#L39-L59) — `configure_optimizers()`.
  This is the one Lesson-8 file that changed. Ask students to find the line
  that decides which parameters get weight decay (`param.dim() >= 2`) and
  explain *why* a bias or a norm gain is excluded — this is a genuine "aha"
  moment about the difference between a weight matrix and a per-channel
  scalar.

## 4. Checkpointing — save AND resume, not just save

- [src/checkpoint.py](src/checkpoint.py) — `save_checkpoint()` /
  `load_checkpoint()`. Two teaching points:
  - The atomic write (temp file + `os.replace`) — ask *what could go wrong
    if we wrote straight to the real path and Colab disconnected mid-write?*
  - `weights_only=True` on load — a real security habit, not just Lesson
    theatre: worth a 2-minute detour into why unpickling arbitrary Python
    objects is a code-execution risk, and why a checkpoint (tensors + plain
    dicts) doesn't need that risk at all.

## 5. The loop itself

- [src/trainer.py](src/trainer.py) is the one file worth walking top to
  bottom live. Key stops:
  - [Lines 26-46](src/trainer.py#L26-L46) `pick_precision()` — the T4-has-no-
    real-bf16 fact lands well as a concrete "know your hardware" moment.
  - [Lines 135-146](src/trainer.py#L135-L146) the gradient-accumulation loop
    — trace through with `grad_accum_steps=4` on the whiteboard: 4 forward/
    backward passes, one optimizer step, loss divided by 4 *before* backward.
    Ask what breaks if that division is forgotten (answer: it trains as if
    the batch were 4x larger than it is, effectively multiplying the
    learning rate).
  - [Lines 148-153](src/trainer.py#L148-L153) gradient clipping — this is
    where the roadmap's "why does clipping prevent loss spikes" question
    becomes concrete: show `grad_norm` in the printed log and point out it's
    exactly what's being clipped.
  - [Lines 158-177](src/trainer.py#L158-L177) the logging block — tokens/sec
    and ETA, computed from a real measured interval, not a static estimate.
    This directly answers the roadmap's "predict, then measure" requirement.

## 6. Tying it together

- [scripts/train.py](scripts/train.py) — the thin driver. Point out the
  vocab_size cross-check near the top (lines ~52-58) against
  `dataset/meta.json` — a two-line guard that catches a stale config *before*
  it wastes GPU hours, which is the whole spirit of the readiness checklist.
  Then show `--resume` and explain that resuming restores the optimizer
  state and step count, not just the weights — tie this directly to the
  viva question bank: *"does resuming restore your optimizer state and your
  learning-rate position, or only the weights?"*

## 7. Where the training hyperparameters live

- [configs/run_01.yaml](configs/run_01.yaml) (bottom half, under the
  `# ---- Lesson 10: training` comment) — same file as the architecture from
  Lesson 7, on purpose: one filename still names the whole experiment.
  `src/train_config.py` and `src/model_config.py` each read only the keys
  they understand from this one file.

## Live-run talking points (once training is actually running)

Currently running: 3000 steps, batch 32 × grad_accum 4 × block 256 =
32,768 tokens/step → ~98M tokens total against a 10.8M-token corpus, so
**this run sees the data about 9 times**, not once. Good moment to ask the
class: *is repeating a small corpus 9 times pretraining, or is it starting
to look like something else?* (There's no clean answer — flag it as exactly
the kind of overfitting-risk judgment call Lesson 11's evaluation exists to
catch, not something to resolve today.)
