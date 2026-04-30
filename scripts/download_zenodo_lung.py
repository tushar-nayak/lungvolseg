#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from lungvolseg.zenodo_covid_lung import download_zenodo_lung_dataset, prepare_zenodo_lung_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/real/zenodo_lung/raw")
    parser.add_argument("--cases-dir", default="data/real/zenodo_lung/cases")
    parser.add_argument("--target-depth", type=int, default=96)
    parser.add_argument("--target-height", type=int, default=128)
    parser.add_argument("--target-width", type=int, default=128)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    download_zenodo_lung_dataset(args.raw_dir)
    cases = prepare_zenodo_lung_cases(
        args.raw_dir,
        args.cases_dir,
        target_shape=(args.target_depth, args.target_height, args.target_width),
        max_cases=args.max_cases,
    )
    print(json.dumps({"raw_dir": args.raw_dir, "cases_dir": args.cases_dir, "num_cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
