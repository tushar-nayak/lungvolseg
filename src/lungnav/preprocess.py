from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .config import DEFAULT_PATCH_SIZE, DEFAULT_SPACING, ensure_dir


def _resample(image: sitk.Image, spacing: tuple[float, float, float], is_label: bool) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(size * old_sp / new_sp))
        for size, old_sp, new_sp in zip(original_size, original_spacing, spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def preprocess_case(
    image_path: str | Path,
    label_path: str | Path,
    output_dir: str | Path,
    spacing: tuple[float, float, float] = DEFAULT_SPACING,
) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path))

    image = _resample(image, spacing=spacing, is_label=False)
    label = _resample(label, spacing=spacing, is_label=True)

    image_array = sitk.GetArrayFromImage(image).astype(np.float32)
    label_array = sitk.GetArrayFromImage(label).astype(np.uint8)
    image_array, label_array = _center_pad_or_crop(image_array, label_array, DEFAULT_PATCH_SIZE)
    image_array = np.clip(image_array, -1000.0, 400.0)
    image_array = (image_array + 1000.0) / 1400.0

    output_image = sitk.GetImageFromArray(image_array)
    output_image.SetSpacing(spacing)
    output_label = sitk.GetImageFromArray(label_array)
    output_label.SetSpacing(spacing)

    case_id = Path(str(image_path)).name.replace("_ct.nii.gz", "")
    image_out = output_dir / f"{case_id}_ct.nii.gz"
    label_out = output_dir / f"{case_id}_label.nii.gz"
    sitk.WriteImage(output_image, str(image_out))
    sitk.WriteImage(output_label, str(label_out))

    return {"case_id": case_id, "image": str(image_out), "label": str(label_out)}


def _center_pad_or_crop(
    image_array: np.ndarray,
    label_array: np.ndarray,
    target_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    pad_width = []
    crop_slices = []
    for current, target in zip(image_array.shape, target_shape):
        if current >= target:
            start = (current - target) // 2
            crop_slices.append(slice(start, start + target))
            pad_width.append((0, 0))
        else:
            crop_slices.append(slice(0, current))
            total_pad = target - current
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_width.append((pad_before, pad_after))

    image_array = image_array[tuple(crop_slices)]
    label_array = label_array[tuple(crop_slices)]
    image_array = np.pad(image_array, pad_width, mode="constant", constant_values=0.0)
    label_array = np.pad(label_array, pad_width, mode="constant", constant_values=0)
    return image_array, label_array


def preprocess_directory(input_dir: str | Path, output_dir: str | Path) -> list[dict[str, str]]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    cases = []
    for image_path in sorted(input_dir.glob("*_ct.nii.gz")):
        label_path = input_dir / image_path.name.replace("_ct.nii.gz", "_label.nii.gz")
        if not label_path.exists():
            continue
        cases.append(preprocess_case(image_path, label_path, output_dir))
    return cases
