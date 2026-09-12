# Project Charter (Lesson 1)

**What are we building?**
A small (~10-15M parameter) decoder-only transformer trained on children's stories.

**Who is it for?**
Nobody in production — a teaching artefact, built to be explained end to end.

**What does "it works" mean?**
Validation loss clearly below the random baseline `ln(vocab_size)` and the
bigram baseline, plus generated text that stays grammatical for 2-3 sentences.

**What hardware do we have?**
Free-tier Colab/Kaggle T4 GPU, a few hours per day.

**What is our token budget?**
~10-15M training tokens (see [DATASET_CARD.md](DATASET_CARD.md)), which caps
the model at roughly 10-15M parameters per the 20-tokens/parameter rule.

**What are we explicitly NOT building?**
Instruction-following, a chat interface, multilingual support, anything
competing with GPT/Gemini.
