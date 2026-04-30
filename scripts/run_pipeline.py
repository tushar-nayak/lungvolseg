#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from lungnav.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--num-cases", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    result = run_pipeline(args.workspace, num_cases=args.num_cases, epochs=args.epochs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
