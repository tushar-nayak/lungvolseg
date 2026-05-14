#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from lungvolseg.zenodo_covid_lung import run_zenodo_lung_pipeline


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="outputs/zenodo_lung")
    parser.add_argument("--epochs", type=_positive_int, default=8)
    parser.add_argument("--target-depth", type=_positive_int, default=96)
    parser.add_argument("--target-height", type=_positive_int, default=128)
    parser.add_argument("--target-width", type=_positive_int, default=128)
    parser.add_argument("--max-cases", type=_positive_int, default=None)
    args = parser.parse_args()

    result = run_zenodo_lung_pipeline(
        args.workspace,
        epochs=args.epochs,
        target_shape=(args.target_depth, args.target_height, args.target_width),
        max_cases=args.max_cases,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
