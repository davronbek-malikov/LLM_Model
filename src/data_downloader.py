"""Stage 1 - get the raw text.

We download TinyStories from the Hugging Face Hub. Short, simple children's
stories: small enough for a free GPU, clean enough that a tiny model can
actually produce something coherent.

This is deliberately not a web crawler. A real project would collect from many
sources; the lesson here is only the shape of the stage - a source goes in,
data/raw/<name>.jsonl comes out, and nothing else in the pipeline needs to know
where the text came from.
"""

from pathlib import Path

from datasets import load_dataset

from src.jsonl_io import write_jsonl


class DataDownloader:
    def __init__(self, dataset_name="roneneldan/TinyStories", split="train"):
        self.dataset_name = dataset_name
        self.split = split

    def run(self, output_path, max_documents=None):
        """Download the dataset and save it as JSONL.

        max_documents=50000 takes only the first 50k stories. The full set is
        over 2 million - fine for a real run, far too slow for a lesson where
        we want to see the whole pipeline work end to end in a few minutes.
        """
        output_path = Path(output_path)

        print("Downloading:", self.dataset_name, "| split:", self.split)

        # load_dataset caches on disk, so running this a second time
        # downloads nothing.
        dataset = load_dataset(self.dataset_name, split=self.split)

        documents = []
        for i, item in enumerate(dataset):
            if max_documents is not None and i >= max_documents:
                break
            # We keep only the text. Anything else the dataset carries is not
            # used by a pretraining pipeline.
            documents.append({"text": item["text"]})

        write_jsonl(output_path, documents)

        print("Saved", len(documents), "documents to", output_path)
        return len(documents)
