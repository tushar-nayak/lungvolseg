#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from lungnav.real_medseg import download_medseg_dataset, prepare_medseg_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/real/medseg/raw")
    parser.add_argument("--cases-dir", default="data/real/medseg/cases")
    parser.add_argument("--block-depth", type=int, default=16)
    parser.add_argument("--xy-size", type=int, default=128)
    args = parser.parse_args()

    download_medseg_dataset(args.raw_dir)
    cases = prepare_medseg_cases(
        args.raw_dir,
        args.cases_dir,
        block_depth=args.block_depth,
        xy_size=args.xy_size,
    )
    print(json.dumps({"raw_dir": args.raw_dir, "cases_dir": args.cases_dir, "num_cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
