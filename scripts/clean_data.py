"""Step 2: clean the raw corpus.

    python scripts/clean_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_cleaner import DataCleaner

INPUT = Path("data/raw/tinystories.jsonl")
OUTPUT = Path("data/cleaned/tinystories.jsonl")

cleaner = DataCleaner()
cleaner.run(INPUT, OUTPUT)
