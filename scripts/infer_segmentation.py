#!/usr/bin/env python3
from __future__ import annotations

import argparse

from lungnav.inference import infer_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--output-label", required=True)
    args = parser.parse_args()
    print(infer_case(args.checkpoint, args.input_image, args.output_label))


if __name__ == "__main__":
    main()
