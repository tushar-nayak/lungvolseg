from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .config import ensure_dir
from .inference import infer_case
from .mesh import extract_meshes
from .metrics import compute_case_metrics, summarize_metrics
from .model_card import write_model_card
from .training import train_model

MEDSEG_ARTICLE_API = "https://api.figshare.com/v2/articles/13521488"
MEDSEG_DATASET_NAME = "MedSeg Covid Dataset 1, Figshare DOI 10.6084/m9.figshare.13521488.v2"
MEDSEG_CLASS_NAMES = {
    0: "background",
    1: "ground_glass",
    2: "consolidation",
    3: "pleural_effusion",
}
MEDSEG_FILES = {
    "tr_im.nii.gz": {
        "url": "https://ndownloader.figshare.com/files/25953977",
        "md5": "54f82e3a9ac01bf6cc2300ca53d820ae",
    },
    "tr_mask.nii.gz": {
        "url": "https://ndownloader.figshare.com/files/25953980",
        "md5": "d11996de45d06d332658115d8b3cf6fc",
    },
    "val_im.nii.gz": {
        "url": "https://ndownloader.figshare.com/files/25953983",
        "md5": "098e8ab8b2e85f7c66abf8a92721ee95",
    },
    "Test-Images-Clinical-Details.csv": {
        "url": "https://ndownloader.figshare.com/files/25953974",
        "md5": "35aefddbccfbb8fa3dee5be69258cb29",
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


def download_medseg_dataset(output_dir: str | Path) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    paths = {}
    for file_name, metadata in MEDSEG_FILES.items():
        destination = output_dir / file_name
        _download_file(metadata["url"], destination, metadata["md5"])
        paths[file_name] = str(destination)

    metadata_path = output_dir / "dataset_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "name": MEDSEG_DATASET_NAME,
                "article_api": MEDSEG_ARTICLE_API,
                "license": "CC0",
                "classes": MEDSEG_CLASS_NAMES,
                "files": MEDSEG_FILES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["metadata"] = str(metadata_path)
    return paths


def _normalize_image(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    low, high = np.percentile(array, [1.0, 99.0])
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    array = np.clip(array, low, high)
    return ((array - low) / (high - low)).astype(np.float32)


def _pad_depth(array: np.ndarray, target_depth: int) -> np.ndarray:
    if array.shape[0] >= target_depth:
        return array[:target_depth]
    pad_after = target_depth - array.shape[0]
    return np.pad(array, ((0, pad_after), (0, 0), (0, 0)), mode="constant")


def _resize_block(array: np.ndarray, xy_size: int, is_label: bool) -> np.ndarray:
    image = sitk.GetImageFromArray(array)
    original_size = image.GetSize()
    target_size = (xy_size, xy_size, array.shape[0])
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


def prepare_medseg_cases(
    raw_dir: str | Path,
    output_dir: str | Path,
    block_depth: int = 16,
    xy_size: int = 128,
) -> list[dict[str, str]]:
    raw_dir = Path(raw_dir)
    output_dir = ensure_dir(output_dir)
    image_array = sitk.GetArrayFromImage(sitk.ReadImage(str(raw_dir / "tr_im.nii.gz")))
    label_array = sitk.GetArrayFromImage(sitk.ReadImage(str(raw_dir / "tr_mask.nii.gz"))).astype(np.uint8)

    cases: list[dict[str, str]] = []
    for start in range(0, image_array.shape[0], block_depth):
        image_block = _pad_depth(image_array[start : start + block_depth], block_depth)
        label_block = _pad_depth(label_array[start : start + block_depth], block_depth)
        image_block = _resize_block(_normalize_image(image_block), xy_size=xy_size, is_label=False)
        label_block = _resize_block(label_block, xy_size=xy_size, is_label=True)

        case_id = f"medseg_{len(cases) + 1:03d}"
        image_path = output_dir / f"{case_id}_ct.nii.gz"
        label_path = output_dir / f"{case_id}_label.nii.gz"
        image = sitk.GetImageFromArray(image_block)
        label = sitk.GetImageFromArray(label_block)
        image.SetSpacing((1.0, 1.0, 5.0))
        label.SetSpacing((1.0, 1.0, 5.0))
        sitk.WriteImage(image, str(image_path))
        sitk.WriteImage(label, str(label_path))
        cases.append({"case_id": case_id, "image": str(image_path), "label": str(label_path)})

    (output_dir / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    return cases


def run_real_medseg_pipeline(
    workspace: str | Path,
    epochs: int = 1,
    block_depth: int = 16,
    xy_size: int = 128,
) -> dict[str, object]:
    workspace = ensure_dir(workspace)
    raw_dir = ensure_dir(Path(workspace) / "raw")
    cases_dir = ensure_dir(Path(workspace) / "cases")
    checkpoint_dir = ensure_dir(Path(workspace) / "checkpoints")
    prediction_dir = ensure_dir(Path(workspace) / "predictions")
    mesh_dir = ensure_dir(Path(workspace) / "meshes")
    patch_size = (block_depth, xy_size, xy_size)

    download_medseg_dataset(raw_dir)
    cases = prepare_medseg_cases(raw_dir, cases_dir, block_depth=block_depth, xy_size=xy_size)
    training_summary = train_model(
        cases,
        checkpoint_dir,
        epochs=epochs,
        num_classes=len(MEDSEG_CLASS_NAMES),
        class_names=MEDSEG_CLASS_NAMES,
    )
    training_summary["spacing"] = "1.0 x 1.0 x 5.0 pseudo-volume spacing"

    prediction_paths = {}
    for case in cases:
        prediction_path = Path(prediction_dir) / f'{case["case_id"]}_pred.nii.gz'
        infer_case(
            training_summary["checkpoint"],
            case["image"],
            prediction_path,
            num_classes=len(MEDSEG_CLASS_NAMES),
            patch_size=patch_size,
        )
        prediction_paths[case["case_id"]] = str(prediction_path)
        extract_meshes(prediction_path, Path(mesh_dir) / case["case_id"], class_names=MEDSEG_CLASS_NAMES)

    case_metrics = {
        case["case_id"]: compute_case_metrics(
            prediction_paths[case["case_id"]],
            case["label"],
            class_names=MEDSEG_CLASS_NAMES,
        )
        for case in cases
    }
    metrics_summary = summarize_metrics(case_metrics, Path(workspace) / "metrics.json")
    model_card_path = write_model_card(
        Path(workspace) / "model_card.md",
        training_summary,
        metrics_summary,
        dataset_name=MEDSEG_DATASET_NAME,
        class_names=MEDSEG_CLASS_NAMES,
    )

    return {
        "workspace": str(workspace),
        "dataset": MEDSEG_DATASET_NAME,
        "checkpoint": training_summary["checkpoint"],
        "metrics": str(Path(workspace) / "metrics.json"),
        "model_card": model_card_path,
        "meshes": str(mesh_dir),
        "num_cases": len(cases),
    }
