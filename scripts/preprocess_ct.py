#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from lungnav.preprocess import preprocess_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(preprocess_directory(args.input_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
