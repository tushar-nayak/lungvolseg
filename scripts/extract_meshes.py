#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from lungnav.mesh import extract_meshes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-image", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(extract_meshes(args.label_image, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
