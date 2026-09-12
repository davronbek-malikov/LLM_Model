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
→ Model Design      <- we are here
```

**Lesson 7:** we design the architecture of our own LLM.
**Lesson 8:** we will implement the Transformer.

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

## Today's lesson

```bash
python scripts/design_model.py
```

It reads [`configs/run_01.yaml`](configs/run_01.yaml), estimates the parameter
count and the memory it would need, and compares that against how many tokens
we actually have.

---

## Two rules

1. **Data and checkpoints never go into Git** - only the code that regenerates
   them. See [`.gitignore`](.gitignore).
2. **Every run gets its own config file.** Copy `run_01.yaml` to `run_02.yaml`;
   never edit a config in place to start a new experiment.

---

## Not built yet

Attention, the transformer block, the model, training, evaluation, inference
and deployment all belong to later lessons. Lesson 7 produces a design, not a
model.
