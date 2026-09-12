"""Stage 3 - filtering.

Filtering decides which documents we keep. Every rule below is a HEURISTIC,
not a truth: each one will sometimes throw away a document we wanted. That is
the trade, and it is why the run() method counts each reason separately - when
a threshold is wrong, the counts are how you find out.

Always read a few of the documents you rejected. It is the fastest way to
discover that min_chars is too high.
"""

import re
from collections import Counter
from pathlib import Path

from src.jsonl_io import read_jsonl, write_jsonl

# Simple PII patterns. Not perfect - just enough to demonstrate the idea:
# mask what looks sensitive, count how often it happened.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")


class DataFilter:
    def __init__(self, min_chars=200, max_chars=100000, min_alpha_ratio=0.30,
                 max_repeated_line_ratio=0.30):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_alpha_ratio = min_alpha_ratio
        self.max_repeated_line_ratio = max_repeated_line_ratio

    def mask_pii(self, text):
        """Replace emails and phone numbers with a placeholder.

        We mask instead of dropping the document - one phone number should not
        cost us an otherwise good story.
        """
        text, n_email = EMAIL_RE.subn("<EMAIL>", text)
        text, n_phone = PHONE_RE.subn("<PHONE>", text)
        return text, n_email + n_phone

    def alphabetic_ratio(self, text):
        """Share of characters that are letters.

        A low value means the "document" is mostly numbers, symbols or a
        navigation menu - a price table, not prose.
        """
        if not text:
            return 0.0
        return sum(char.isalpha() for char in text) / len(text)

    def repeated_line_ratio(self, text):
        """Share of lines that appear more than once.

        High values mean boilerplate: a cookie banner, a menu, a template.
        The model would memorise the repetition instead of learning language.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) <= 1:
            return 0.0

        counts = Counter(lines)
        repeated = sum(count for count in counts.values() if count > 1)
        return repeated / len(lines)

    def reject_reason(self, text):
        """Return why this document should be dropped, or None to keep it.

        Returning the REASON instead of True/False is what lets us print a
        breakdown at the end instead of a single unhelpful number.
        """
        if len(text) < self.min_chars:
            return "too_short"
        if len(text) > self.max_chars:
            return "too_long"
        if self.alphabetic_ratio(text) < self.min_alpha_ratio:
            return "low_alpha_ratio"
        if self.repeated_line_ratio(text) > self.max_repeated_line_ratio:
            return "repeated_lines"
        return None

    def run(self, input_path, output_path, limit=None):
        documents = read_jsonl(Path(input_path), limit=limit)

        kept = []
        dropped = Counter()
        pii_masked = 0

        for doc in documents:
            text, n_pii = self.mask_pii(doc["text"])
            pii_masked += n_pii

            reason = self.reject_reason(text)
            if reason:
                dropped[reason] += 1
                continue
            kept.append({"text": text})

        write_jsonl(Path(output_path), kept)

        stats = {
            "input": len(documents),
            "dropped": dict(dropped),
            "pii_masked": pii_masked,
            "output": len(kept),
        }
        print("Filtering:", stats)
        return stats
