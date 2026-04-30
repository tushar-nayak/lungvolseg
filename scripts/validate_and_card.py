#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lungnav.metrics import compute_case_metrics, summarize_metrics
from lungnav.model_card import write_model_card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    prediction_dir = Path(args.prediction_dir)
    reference_dir = Path(args.reference_dir)
    output_dir = Path(args.output_dir)
    case_metrics = {}
    for prediction_path in sorted(prediction_dir.glob("*_pred.nii.gz")):
        case_id = prediction_path.name.replace("_pred.nii.gz", "")
        reference_path = reference_dir / f"{case_id}_label.nii.gz"
        case_metrics[case_id] = compute_case_metrics(prediction_path, reference_path)

    metrics_summary = summarize_metrics(case_metrics, output_dir / "metrics.json")
    training_summary = json.loads((Path(args.model_dir) / "training_summary.json").read_text(encoding="utf-8"))
    training_summary["spacing"] = "1.5 mm isotropic"
    card_path = write_model_card(output_dir / "model_card.md", training_summary, metrics_summary)
    print(json.dumps({"metrics": str(output_dir / "metrics.json"), "model_card": card_path}, indent=2))


if __name__ == "__main__":
    main()
