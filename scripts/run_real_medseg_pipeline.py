#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from lungnav.real_medseg import run_real_medseg_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="outputs/real_medseg")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--block-depth", type=int, default=16)
    parser.add_argument("--xy-size", type=int, default=128)
    args = parser.parse_args()

    result = run_real_medseg_pipeline(
        args.workspace,
        epochs=args.epochs,
        block_depth=args.block_depth,
        xy_size=args.xy_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
