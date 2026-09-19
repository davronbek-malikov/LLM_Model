"""Lesson 11 - see the actual effect of temperature, with real weights.

    python evaluation/generate.py --prompt "Once upon a time"
    python evaluation/generate.py --prompt "Once upon a time" --temperatures 0.2 0.8 1.5

Lesson 8's model.generate() only does greedy decoding - useful for proving
the loop works, but every run gives the identical output. This script uses
the real sampling pipeline (inference/engine.py) so the difference between a
cautious, repetitive temperature-0.2 sample and a chaotic temperature-1.5
one is actually visible, as your roadmap's Lesson 9/11 material asks for.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows consoles default to cp1252, which cannot print every character a
# byte-level tokenizer can decode (an undertrained model especially can
# produce raw replacement characters). Forcing UTF-8 here means this script
# never crashes on output alone, on any platform.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from inference.engine import InferenceEngine

DEFAULT_TEMPERATURES = [0.2, 0.8, 1.5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperatures", type=float, nargs="+",
                         default=DEFAULT_TEMPERATURES)
    parser.add_argument("--checkpoint", type=Path,
                         default=Path("checkpoints/best.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/run_01.yaml"))
    parser.add_argument("--tokenizer", type=Path,
                         default=Path("tokenizer/tokenizer.json"))
    args = parser.parse_args()

    engine = InferenceEngine(args.config, args.checkpoint, args.tokenizer)

    print("Prompt:", repr(args.prompt))
    print()
    for temp in args.temperatures:
        text, n_tokens, latency_ms = engine.generate(
            args.prompt, max_tokens=args.max_tokens, temperature=temp
        )
        print("--- temperature = {} ---".format(temp))
        print(text)
        print("({} tokens, {:.0f} ms)".format(n_tokens, latency_ms))
        print()


if __name__ == "__main__":
    main()
