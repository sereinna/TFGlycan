#!/usr/bin/env python
"""Train a TFGlycan model branch."""
import argparse
import importlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BRANCHES = {"light": "light", "taxonomy": "taxonomy", "functional": "atom"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("branch", choices=BRANCHES)
    parser.add_argument("arguments", nargs=argparse.REMAINDER,
                        help="Branch arguments; use BRANCH --help to list them.")
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / "runtime_cache/torch_extensions"))
    sys.argv = [sys.argv[0] + " " + args.branch, *args.arguments]
    module = importlib.import_module("tfglycan." + BRANCHES[args.branch])
    module.main()


if __name__ == "__main__":
    main()
