from __future__ import annotations

from pathlib import Path

from .config import ensure_dir


def write_model_card(
    output_path: str | Path,
    training_summary: dict[str, object],
    metrics_summary: dict[str, object],
    dataset_name: str = "COVID-19 CT Lung and Infection Segmentation Dataset",
    class_names: dict[int, str] | None = None,
) -> str:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    mean_metrics = metrics_summary["mean"]
    class_names = class_names or training_summary.get("class_names") or {0: "background", 1: "lung"}
    output_classes = ", ".join(str(name) for _, name in sorted(class_names.items()))
    metric_lines = "\n".join(f"- Mean {name}: {value:.4f}" for name, value in sorted(mean_metrics.items()))
    content = f"""# Lung CT Navigation-Prep Model Card

## Overview

This model segments lung regions from full-volume chest CT scans to support navigation-prep engineering experiments such as anatomical review, surface generation, and downstream registration prototypes.

## Intended use

- CT-based bronchoscopy planning research
- Synthetic workflow demonstrations for segmentation-to-mesh pipelines
- Surface export for visualization or geometry QA

## Model

- Architecture: MONAI 3D UNet
- Input: preprocessed chest CT volume resampled to {training_summary.get("spacing", "1.5 mm isotropic")}
- Output classes: {output_classes}

## Training data

- Dataset type: {dataset_name}
- Train cases: {training_summary["num_train_cases"]}
- Validation cases: {training_summary["num_val_cases"]}
- Epochs: {training_summary["epochs"]}

## Validation

{metric_lines}

## Limitations

- The default training command is a smoke test and is not clinically representative.
- Performance across scanners, pathology, noise, and acquisition variability is unknown without broader external validation.
- Mesh quality depends on label fidelity and may require post-processing for procedural planning use.

## Safety

This project is for research and engineering demonstration only. It is not validated for clinical decision-making or robotic bronchoscopy guidance.
"""

    output_path.write_text(content, encoding="utf-8")
    return str(output_path)
