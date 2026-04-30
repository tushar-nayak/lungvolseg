from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .config import ensure_dir
from .inference import infer_case
from .mesh import extract_meshes
from .metrics import compute_case_metrics, summarize_metrics
from .model_card import write_model_card
from .training import train_model

ZENODO_DATASET_NAME = "COVID-19 CT Lung and Infection Segmentation Dataset"
ZENODO_DOI = "10.5281/zenodo.3757476"
ZENODO_API = "https://zenodo.org/api/records/3757476"
ZENODO_CLASS_NAMES = {
    0: "background",
    1: "lung",
}
ZENODO_FILES = {
    "COVID-19-CT-Seg_20cases.zip": {
        "url": "https://zenodo.org/api/records/3757476/files/COVID-19-CT-Seg_20cases.zip/content",
        "md5": "873617e1fdcbe92f8aa0ce83a4798a1f",
    },
    "Lung_Mask.zip": {
        "url": "https://zenodo.org/api/records/3757476/files/Lung_Mask.zip/content",
        "md5": "972b3d9b4c7b64d2c518b936bd98cb47",
    },
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path, expected_md5: str) -> None:
    if destination.exists() and _md5(destination) == expected_md5:
        return
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    observed_md5 = _md5(tmp_path)
    if observed_md5 != expected_md5:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"MD5 mismatch for {destination.name}: expected {expected_md5}, got {observed_md5}")
    tmp_path.replace(destination)


def download_zenodo_lung_dataset(output_dir: str | Path) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    paths = {}
    for file_name, metadata in ZENODO_FILES.items():
        destination = output_dir / file_name
        _download_file(metadata["url"], destination, metadata["md5"])
        paths[file_name] = str(destination)

    metadata_path = output_dir / "dataset_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "name": ZENODO_DATASET_NAME,
                "doi": ZENODO_DOI,
                "api": ZENODO_API,
                "license": "CC-BY-4.0",
                "classes": ZENODO_CLASS_NAMES,
                "files": ZENODO_FILES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["metadata"] = str(metadata_path)
    return paths


def _extract_zip(zip_path: Path, output_dir: Path) -> Path:
    destination = ensure_dir(output_dir / zip_path.stem)
    marker = destination / ".extracted"
    if marker.exists():
        return destination
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
    marker.write_text("ok\n", encoding="utf-8")
    return destination


def _nifti_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.name.endswith(".nii") or path.name.endswith(".nii.gz")
    )


def _case_key(path: Path) -> str:
    name = path.name.lower()
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    for token in ("lungmask", "lung_mask", "mask", "segmentation", "ct", "image"):
        name = name.replace(token, "")
    return "".join(character for character in name if character.isalnum())


def _normalize_ct(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    array = np.clip(array, -1000.0, 400.0)
    return ((array + 1000.0) / 1400.0).astype(np.float32)


def _bbox_from_label(label: np.ndarray, margin: int = 12) -> tuple[slice, slice, slice]:
    coords = np.argwhere(label > 0)
    if coords.size == 0:
        return tuple(slice(0, size) for size in label.shape)  # type: ignore[return-value]
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + margin + 1, label.shape)
    return tuple(slice(int(start), int(stop)) for start, stop in zip(mins, maxs))  # type: ignore[return-value]


def _resize(array: np.ndarray, target_shape: tuple[int, int, int], is_label: bool) -> np.ndarray:
    image = sitk.GetImageFromArray(array)
    target_size = (target_shape[2], target_shape[1], target_shape[0])
    original_size = image.GetSize()
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(
        tuple(old_size / new_size for old_size, new_size in zip(original_size, target_size))
    )
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    resized = sitk.GetArrayFromImage(resampler.Execute(image))
    return resized.astype(np.uint8 if is_label else np.float32)


def prepare_zenodo_lung_cases(
    raw_dir: str | Path,
    output_dir: str | Path,
    target_shape: tuple[int, int, int] = (96, 128, 128),
    max_cases: int | None = None,
) -> list[dict[str, str]]:
    raw_dir = Path(raw_dir)
    output_dir = ensure_dir(output_dir)
    image_root = _extract_zip(raw_dir / "COVID-19-CT-Seg_20cases.zip", raw_dir / "extracted")
    mask_root = _extract_zip(raw_dir / "Lung_Mask.zip", raw_dir / "extracted")

    images = {_case_key(path): path for path in _nifti_files(image_root)}
    masks = {_case_key(path): path for path in _nifti_files(mask_root)}
    pairs = [(key, images[key], masks[key]) for key in sorted(images.keys() & masks.keys())]
    if max_cases is not None:
        pairs = pairs[:max_cases]
    if not pairs:
        raise ValueError("No matching CT/mask NIfTI pairs found after extracting Zenodo archives.")

    cases: list[dict[str, str]] = []
    for index, (_, image_path, label_path) in enumerate(pairs, start=1):
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
        label_array = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path))).astype(np.uint8)
        label_array = (label_array > 0).astype(np.uint8)
        crop = _bbox_from_label(label_array)
        image_array = _normalize_ct(image_array[crop])
        label_array = label_array[crop]
        image_array = _resize(image_array, target_shape=target_shape, is_label=False)
        label_array = _resize(label_array, target_shape=target_shape, is_label=True)

        case_id = f"zenodo_lung_{index:03d}"
        output_image_path = output_dir / f"{case_id}_ct.nii.gz"
        output_label_path = output_dir / f"{case_id}_label.nii.gz"
        output_image = sitk.GetImageFromArray(image_array)
        output_label = sitk.GetImageFromArray(label_array)
        output_image.SetSpacing((1.5, 1.5, 1.5))
        output_label.SetSpacing((1.5, 1.5, 1.5))
        sitk.WriteImage(output_image, str(output_image_path))
        sitk.WriteImage(output_label, str(output_label_path))
        cases.append({"case_id": case_id, "image": str(output_image_path), "label": str(output_label_path)})

    (output_dir / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    return cases


def run_zenodo_lung_pipeline(
    workspace: str | Path,
    epochs: int = 8,
    target_shape: tuple[int, int, int] = (96, 128, 128),
    max_cases: int | None = None,
) -> dict[str, object]:
    workspace = ensure_dir(workspace)
    raw_dir = ensure_dir(Path(workspace) / "raw")
    cases_dir = ensure_dir(Path(workspace) / "cases")
    checkpoint_dir = ensure_dir(Path(workspace) / "checkpoints")
    prediction_dir = ensure_dir(Path(workspace) / "predictions")
    mesh_dir = ensure_dir(Path(workspace) / "meshes")

    download_zenodo_lung_dataset(raw_dir)
    cases = prepare_zenodo_lung_cases(raw_dir, cases_dir, target_shape=target_shape, max_cases=max_cases)
    training_summary = train_model(
        cases,
        checkpoint_dir,
        epochs=epochs,
        num_classes=len(ZENODO_CLASS_NAMES),
        class_names=ZENODO_CLASS_NAMES,
    )
    training_summary["spacing"] = "1.5 mm isotropic after lung-region crop and resize"

    prediction_paths = {}
    for case in cases:
        prediction_path = Path(prediction_dir) / f'{case["case_id"]}_pred.nii.gz'
        infer_case(
            training_summary["checkpoint"],
            case["image"],
            prediction_path,
            num_classes=len(ZENODO_CLASS_NAMES),
            patch_size=target_shape,
        )
        prediction_paths[case["case_id"]] = str(prediction_path)
        extract_meshes(prediction_path, Path(mesh_dir) / case["case_id"], class_names=ZENODO_CLASS_NAMES)

    case_metrics = {
        case["case_id"]: compute_case_metrics(
            prediction_paths[case["case_id"]],
            case["label"],
            class_names=ZENODO_CLASS_NAMES,
        )
        for case in cases
    }
    metrics_summary = summarize_metrics(case_metrics, Path(workspace) / "metrics.json")
    model_card_path = write_model_card(
        Path(workspace) / "model_card.md",
        training_summary,
        metrics_summary,
        dataset_name=f"{ZENODO_DATASET_NAME}, DOI {ZENODO_DOI}",
        class_names=ZENODO_CLASS_NAMES,
    )

    return {
        "workspace": str(workspace),
        "dataset": f"{ZENODO_DATASET_NAME}, DOI {ZENODO_DOI}",
        "checkpoint": training_summary["checkpoint"],
        "metrics": str(Path(workspace) / "metrics.json"),
        "model_card": model_card_path,
        "meshes": str(mesh_dir),
        "num_cases": len(cases),
    }
