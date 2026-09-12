"""Step 1: download the raw corpus.

    python scripts/download_data.py
"""

import sys
from pathlib import Path

# Let Python find the src/ folder when this script is run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_downloader import DataDownloader

# Change this to download more or fewer stories. None = the whole dataset
# (over 2 million documents - slow, only do it when you mean it).
MAX_DOCUMENTS = 50000

OUTPUT = Path("data/raw/tinystories.jsonl")

downloader = DataDownloader(dataset_name="roneneldan/TinyStories", split="train")
downloader.run(OUTPUT, max_documents=MAX_DOCUMENTS)
