"""Step 4: remove duplicate documents. Produces our FINAL corpus.

    python scripts/deduplicate_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.deduplicator import Deduplicator

INPUT = Path("data/filtered/tinystories.jsonl")
OUTPUT = Path("data/final/tinystories.jsonl")

deduplicator = Deduplicator()
deduplicator.run(INPUT, OUTPUT)
