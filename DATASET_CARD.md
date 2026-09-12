# Dataset Card (Lesson 2 & 3) — FROZEN

| Field | Value |
|---|---|
| Source | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) |
| Licence | CDLA-Sharing-1.0 (permissive, attribution required) |
| Language | English |
| Raw documents | 50,000 |
| Raw characters | 44,557,053 |
| Estimated raw tokens | ~11.1M (chars/4) |
| After clean → filter → dedup | 49,888 docs |
| Final train tokens (real tokenizer) | 10,795,556 |
| Final val tokens | 108,754 |
| Mixture | 100% TinyStories (single source, no blend) |

**Why this dataset:** short, simple, synthetic stories — small enough for a
free GPU, clean enough that a tiny model can produce coherent output. Matches
the charter's "children's stories" scope directly.

**Frozen at:** Lesson 3 review. Changing the source after this point requires
instructor approval.
