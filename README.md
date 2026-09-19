# Our Own LLM

This project builds a small language model step by step.

Current stage:

```
Data
→ Cleaning
→ Filtering
→ Deduplication
→ Tokenizer
→ Dataset
→ Model Design
→ Model Code (built, all 4 sanity checks pass)
→ Pretraining       <- we are here
```

**Lesson 7:** designed the architecture of our own LLM.
**Lesson 8:** implemented the Transformer (`src/attention.py`, `mlp.py`,
`transformer_block.py`, `model.py`) - `tests/test_model.py` passes all four
checks (shape, random-init loss, overfit-one-batch, causality).
**Lesson 10:** training loop (`scripts/train.py`) - run it once the checks
above pass. Lesson 9 (serving dry run) and Lesson 11 (evaluation) are still
open.

---

## How the code is organised

```
src/       reusable tools   - one class per stage
scripts/   programs         - each one uses a class from src/
configs/   the settings for one run
```

That split is the point: a class is a tool, a script is a program that uses it.

---

## Setup

```bash
pip install -r requirements.txt
```

## Running the pipeline

Run these from the repository root, in order:

```bash
python scripts/download_data.py        # Hugging Face -> data/raw/
python scripts/clean_data.py           # -> data/cleaned/
python scripts/filter_data.py          # -> data/filtered/
python scripts/deduplicate_data.py     # -> data/final/
python scripts/train_tokenizer.py      # -> tokenizer/tokenizer.json
python scripts/build_dataset.py        # -> dataset/train.bin, val.bin, meta.json
```

## Model design and build (Lessons 7-8)

```bash
python scripts/design_model.py     # parameter/memory estimate vs. token budget
python scripts/build_model.py      # builds OurLLM, prints it + real param count
python tests/test_model.py         # 4 sanity checks - must all PASS before training
```

## Training (Lesson 10)

```bash
python scripts/train.py --config configs/run_01.yaml
```

`configs/run_01.yaml` holds both the architecture (read by
`src/model_config.py`) and the training hyperparameters (read by
`src/train_config.py`) for this run - one file names the whole experiment.

If the session disconnects (Colab, Kaggle), resume with:

```bash
python scripts/train.py --config configs/run_01.yaml --resume checkpoints/last.pt
```

This restores the model weights, optimiser state and step count, so training
picks up exactly where it left off rather than restarting cold.

---

## Two rules

1. **Raw/intermediate data and checkpoints never go into Git** - only the code
   that regenerates them. See [`.gitignore`](.gitignore). The one deliberate
   exception: the final `dataset/*.bin` + `meta.json` and `tokenizer/tokenizer.json`
   ARE committed, small (~22MB total) and frozen since Lesson 3, specifically
   so a fresh `git clone` on Colab/Kaggle can train immediately with no
   separate upload step.
2. **Every run gets its own config file.** Copy `run_01.yaml` to `run_02.yaml`;
   never edit a config in place to start a new experiment.

---

## Not built yet

Lesson 9 (serving a checkpoint behind a FastAPI + Docker dry run) and
Lesson 11 (evaluation against baselines, perplexity, generation samples,
`MODEL_CARD.md`) still belong to later lessons.
