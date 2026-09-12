"""Stage 2 - cleaning.

Cleaning fixes the FORM of a document: broken characters, stray HTML, messy
whitespace. It does not judge whether the document is good - that is filtering,
the next stage, and keeping the two apart keeps both easy to explain.

Rule of thumb: cleaning changes text, filtering deletes documents.
"""

import re
import unicodedata
from pathlib import Path

from src.jsonl_io import read_jsonl, write_jsonl


class DataCleaner:
    def clean_text(self, text):
        """Run one document through every cleaning step, in order."""
        text = self.remove_html(text)
        text = self.normalize_unicode(text)
        text = self.remove_control_characters(text)
        text = self.normalize_whitespace(text)
        return text

    def remove_html(self, text):
        """Drop <tags>. Web text is full of them and they teach the model
        nothing about language."""
        return re.sub(r"<[^>]+>", " ", text)

    def normalize_unicode(self, text):
        """NFKC turns look-alike characters into one standard form.

        Without it the curly quote and the straight quote are two different
        characters, so the tokenizer wastes vocabulary slots learning both.
        """
        return unicodedata.normalize("NFKC", text)

    def remove_control_characters(self, text):
        """Remove invisible control characters, but keep newline and tab -
        those carry real structure."""
        return "".join(
            char for char in text
            if char in "\n\t" or not unicodedata.category(char).startswith("C")
        )

    def normalize_whitespace(self, text):
        """Collapse runs of spaces and blank lines.

        "hello      world" and "hello world" are the same sentence, but to a
        tokenizer they are different token sequences.
        """
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    def run(self, input_path, output_path, limit=None):
        """Clean every document and report what happened."""
        documents = read_jsonl(Path(input_path), limit=limit)

        cleaned = []
        empty_removed = 0

        for doc in documents:
            text = self.clean_text(doc["text"])

            # A document that is empty after cleaning was never text -
            # it was markup, or whitespace, or a broken line.
            if not text:
                empty_removed += 1
                continue

            cleaned.append({"text": text})

        write_jsonl(Path(output_path), cleaned)

        stats = {
            "input": len(documents),
            "empty_removed": empty_removed,
            "output": len(cleaned),
        }
        print("Cleaning:", stats)
        return stats
