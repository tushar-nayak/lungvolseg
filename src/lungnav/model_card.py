from __future__ import annotations

from pathlib import Path

from .config import ensure_dir


def write_model_card(
    output_path: str | Path,
    training_summary: dict[str, object],
    metrics_summary: dict[str, object],
) -> str:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    mean_metrics = metrics_summary["mean"]
    content = f"""# Lung CT Navigation-Prep Model Card

## Overview

This model segments lungs and the central airway from chest CT volumes to support navigation-planning style preprocessing tasks such as anatomical review, surface generation, and downstream registration experiments.

## Intended use

- CT-based bronchoscopy planning research
- Synthetic workflow demonstrations for segmentation-to-mesh pipelines
- Surface export for visualization or geometry QA

## Model

- Architecture: MONAI 3D UNet
- Input: preprocessed chest CT volume resampled to {training_summary.get("spacing", "1.5 mm isotropic")}
- Output classes: background, lungs, airway

## Training data

- Dataset type: synthetic chest CT volumes included in this repo's demo path unless replaced by external data
- Train cases: {training_summary["num_train_cases"]}
- Validation cases: {training_summary["num_val_cases"]}
- Epochs: {training_summary["epochs"]}

## Validation

- Mean Dice, lungs: {mean_metrics["dice_lungs"]:.4f}
- Mean Dice, airway: {mean_metrics["dice_airway"]:.4f}
- 95th percentile Hausdorff, lungs: {mean_metrics["hd95_lungs"]:.4f}
- 95th percentile Hausdorff, airway: {mean_metrics["hd95_airway"]:.4f}

## Limitations

- The included demo data is synthetic and not clinically representative.
- Performance on real scanner data, pathology, noise, and acquisition variability is unknown without external validation.
- Mesh quality depends on label fidelity and may require post-processing for procedural planning use.

## Safety

This project is for research and engineering demonstration only. It is not validated for clinical decision-making or robotic bronchoscopy guidance.
"""

    output_path.write_text(content, encoding="utf-8")
    return str(output_path)
