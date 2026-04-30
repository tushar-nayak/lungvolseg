#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lungnav.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cases = [
        {
            "case_id": image_path.name.replace("_ct.nii.gz", ""),
            "image": str(image_path),
            "label": str(data_dir / image_path.name.replace("_ct.nii.gz", "_label.nii.gz")),
        }
        for image_path in sorted(data_dir.glob("*_ct.nii.gz"))
        if (data_dir / image_path.name.replace("_ct.nii.gz", "_label.nii.gz")).exists()
    ]
    print(json.dumps(train_model(cases, args.output_dir, epochs=args.epochs), indent=2))


if __name__ == "__main__":
    main()
