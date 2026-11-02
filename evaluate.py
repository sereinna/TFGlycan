#!/usr/bin/env python
"""Evaluate the TFGlycan model branches."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("MPLBACKEND", "Agg")
    from src.evaluate import main as evaluate
    evaluate()


if __name__ == "__main__":
    main()
