#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from lungnav.synthetic import generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-cases", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(generate_synthetic_dataset(args.output_dir, args.num_cases), indent=2))


if __name__ == "__main__":
    main()
