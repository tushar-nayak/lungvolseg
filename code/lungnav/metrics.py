from __future__ import annotations

import json
from pathlib import Path

import torch
from monai.metrics import compute_dice, compute_hausdorff_distance
from monai.transforms import AsDiscrete
import SimpleITK as sitk

from .config import CLASS_NAMES, ensure_dir


def _load_label_tensor(path: str | Path) -> torch.Tensor:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def compute_case_metrics(
    prediction_path: str | Path,
    reference_path: str | Path,
    class_names: dict[int, str] | None = None,
) -> dict[str, float]:
    class_names = class_names or CLASS_NAMES
    pred = _load_label_tensor(prediction_path)
    ref = _load_label_tensor(reference_path)
    pred_oh = AsDiscrete(to_onehot=len(class_names))(pred)
    ref_oh = AsDiscrete(to_onehot=len(class_names))(ref)

    dice = compute_dice(pred_oh, ref_oh, include_background=False).squeeze(0)
    hd95 = compute_hausdorff_distance(
        pred_oh,
        ref_oh,
        include_background=False,
        percentile=95.0,
    ).squeeze(0)

    results: dict[str, float] = {}
    for metric_index, label_value in enumerate(sorted(label for label in class_names if label != 0)):
        label_name = class_names[label_value]
        results[f"dice_{label_name}"] = float(dice[metric_index].item())
        results[f"hd95_{label_name}"] = float(hd95[metric_index].item())
    return results


def summarize_metrics(case_metrics: dict[str, dict[str, float]], output_path: str | Path) -> dict[str, object]:
    ensure_dir(Path(output_path).parent)
    metric_names = list(next(iter(case_metrics.values())).keys())
    summary = {
        "per_case": case_metrics,
        "mean": {
            key: sum(values[key] for values in case_metrics.values()) / len(case_metrics)
            for key in metric_names
        },
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
