"""Stage 4 - deduplication.

    document A
    document B
    document A     <- we keep only the first copy

Why this matters:

1. A document seen 50 times is memorised, not learned.
2. Duplicates inflate the token count, so you think you have more data
   than you do.
3. If the same document lands in both train and validation, validation
   loss becomes a lie - the model has already seen the answer.

We do EXACT deduplication only: two documents are duplicates when their text
is identical. Near-duplicate detection (MinHash, LSH) finds documents that
differ by only a date or a name - a real and important problem, but you have
to understand the exact case first.
"""

import hashlib
from pathlib import Path

from src.jsonl_io import read_jsonl, write_jsonl


class Deduplicator:
    def shingles(self, text, k=5):
        """Break text into overlapping k-word chunks ("shingles").

        Two documents that share most of their shingles are near-duplicates -
        e.g. the same story with one name changed - even though their full
        text differs.
        """
        words = text.split()
        return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}

    def minhash_signature(self, shingle_set, num_hashes=8):
        """A short fingerprint that approximates shingle overlap.

        Real MinHash uses num_hashes independent hash functions; we simulate
        that cheaply by salting one hash function num_hashes times. Good
        enough to teach the idea, not tuned for a production LSH index.
        """
        if not shingle_set:
            return tuple([0] * num_hashes)
        return tuple(
            min(hash((salt, s)) for s in shingle_set)
            for salt in range(num_hashes)
        )

    def count_near_duplicates(self, documents):
        """Report (do not remove) documents whose MinHash signature exactly
        matches another document's - a stand-in for "very high similarity".
        Informational only: shows near-dup detection without an exact-copy
        requirement. A real pipeline would use many more hashes plus LSH
        buckets so it doesn't compare every pair directly."""
        signatures = [
            self.minhash_signature(self.shingles(doc["text"]))
            for doc in documents
        ]
        seen = set()
        near_dupes = 0
        for sig in signatures:
            if sig in seen:
                near_dupes += 1
            seen.add(sig)
        return near_dupes

    def document_hash(self, text):
        """A short fingerprint of the document.

        Why hash instead of storing the text itself? Because a set of two
        million full documents would not fit in memory, while a set of two
        million 32-character fingerprints comfortably does.
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def run(self, input_path, output_path, limit=None):
        documents = read_jsonl(Path(input_path), limit=limit)

        seen = set()
        unique = []
        duplicates = 0

        for doc in documents:
            fingerprint = self.document_hash(doc["text"])

            if fingerprint in seen:
                duplicates += 1
                continue

            seen.add(fingerprint)
            unique.append(doc)

        write_jsonl(Path(output_path), unique)

        # Reported, not removed - see count_near_duplicates() docstring.
        near_dupes = self.count_near_duplicates(unique)

        stats = {
            "input": len(documents),
            "duplicates_removed": duplicates,
            "near_duplicates_found": near_dupes,
            "output": len(unique),
        }
        print("Deduplication:", stats)
        return stats
