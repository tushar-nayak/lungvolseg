from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .config import ensure_dir


def _make_case(shape: tuple[int, int, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.indices(shape).astype(np.float32)
    center = np.array(shape, dtype=np.float32) / 2.0

    image = np.full(shape, 60.0, dtype=np.float32)
    label = np.zeros(shape, dtype=np.uint8)

    left_center = center + np.array([0.0, -18.0 + rng.uniform(-3, 3), -16.0 + rng.uniform(-3, 3)])
    right_center = center + np.array([0.0, 18.0 + rng.uniform(-3, 3), -16.0 + rng.uniform(-3, 3)])
    axes = np.array([30.0, 20.0, 15.0]) + rng.uniform(-2.0, 2.0, size=3)

    for lung_center in (left_center, right_center):
        ellipsoid = (
            ((zz - lung_center[0]) / axes[0]) ** 2
            + ((yy - lung_center[1]) / axes[1]) ** 2
            + ((xx - lung_center[2]) / axes[2]) ** 2
        ) <= 1.0
        label[ellipsoid] = 1
        image[ellipsoid] = -820.0 + rng.normal(0.0, 35.0, size=int(np.sum(ellipsoid)))

    airway_radius = 4.0 + rng.uniform(-0.5, 0.5)
    airway = ((yy - center[1]) ** 2 + (xx - (center[2] + 8.0)) ** 2) <= airway_radius**2
    airway &= (zz > center[0] - 28.0) & (zz < center[0] + 22.0)
    label[airway] = 2
    image[airway] = -980.0

    image += rng.normal(0.0, 20.0, size=shape).astype(np.float32)
    return image, label


def _write_image(array: np.ndarray, path: Path, spacing: tuple[float, float, float]) -> None:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path))


def generate_synthetic_dataset(output_dir: str | Path, num_cases: int = 6) -> list[dict[str, str]]:
    output_dir = ensure_dir(output_dir)
    cases: list[dict[str, str]] = []

    for index in range(num_cases):
        case_id = f"case{index + 1:03d}"
        image, label = _make_case((96, 96, 96), seed=100 + index)
        image_path = output_dir / f"{case_id}_ct.nii.gz"
        label_path = output_dir / f"{case_id}_label.nii.gz"
        _write_image(image, image_path, spacing=(1.3, 1.3, 1.8))
        _write_image(label, label_path, spacing=(1.3, 1.3, 1.8))
        cases.append({"case_id": case_id, "image": str(image_path), "label": str(label_path)})

    return cases
