"""Step 3: filter out low-quality documents.

    python scripts/filter_data.py

TinyStories are short, so min_chars is low here. On web text you would use a
much higher threshold. Change the numbers, re-run, and look at how many
documents each rule removed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_filter import DataFilter

INPUT = Path("data/cleaned/tinystories.jsonl")
OUTPUT = Path("data/filtered/tinystories.jsonl")

data_filter = DataFilter(
    min_chars=200,
    max_chars=100000,
    min_alpha_ratio=0.30,
    max_repeated_line_ratio=0.30,
)
data_filter.run(INPUT, OUTPUT)
