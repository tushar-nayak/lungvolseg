from __future__ import annotations

from pathlib import Path

from .config import ensure_dir
from .inference import infer_case
from .mesh import extract_meshes
from .metrics import compute_case_metrics, summarize_metrics
from .model_card import write_model_card
from .preprocess import preprocess_case
from .synthetic import generate_synthetic_dataset
from .training import train_model


def run_pipeline(workspace: str | Path, num_cases: int = 6, epochs: int = 2) -> dict[str, object]:
    workspace = ensure_dir(workspace)
    raw_dir = ensure_dir(Path(workspace) / "raw")
    preprocessed_dir = ensure_dir(Path(workspace) / "preprocessed")
    checkpoint_dir = ensure_dir(Path(workspace) / "checkpoints")
    prediction_dir = ensure_dir(Path(workspace) / "predictions")
    mesh_dir = ensure_dir(Path(workspace) / "meshes")

    raw_cases = generate_synthetic_dataset(raw_dir, num_cases=num_cases)
    processed_cases = [preprocess_case(case["image"], case["label"], preprocessed_dir) for case in raw_cases]

    training_summary = train_model(processed_cases, checkpoint_dir, epochs=epochs)
    prediction_paths = {}
    for case in processed_cases:
        prediction_path = Path(prediction_dir) / f'{case["case_id"]}_pred.nii.gz'
        infer_case(training_summary["checkpoint"], case["image"], prediction_path)
        prediction_paths[case["case_id"]] = str(prediction_path)
        extract_meshes(prediction_path, Path(mesh_dir) / case["case_id"])

    case_metrics = {
        case["case_id"]: compute_case_metrics(
            prediction_paths[case["case_id"]],
            case["label"],
        )
        for case in processed_cases
    }
    metrics_summary = summarize_metrics(case_metrics, Path(workspace) / "metrics.json")
    training_summary["spacing"] = "1.5 mm isotropic"
    model_card_path = write_model_card(Path(workspace) / "model_card.md", training_summary, metrics_summary)

    return {
        "workspace": str(workspace),
        "checkpoint": training_summary["checkpoint"],
        "metrics": str(Path(workspace) / "metrics.json"),
        "model_card": model_card_path,
        "meshes": str(mesh_dir),
    }
