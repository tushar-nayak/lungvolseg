#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_training_curve(history: list[dict], output_path: Path) -> None:
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_dice = [item["val_dice"] for item in history]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, train_loss, color="#b45309", linewidth=2.0, label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="#b45309")
    ax1.tick_params(axis="y", labelcolor="#b45309")
    ax1.grid(True, color="#d6d3d1", linewidth=0.8, alpha=0.8)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_dice, color="#0f766e", linewidth=2.0, label="Validation Dice")
    ax2.set_ylabel("Validation Dice", color="#0f766e")
    ax2.tick_params(axis="y", labelcolor="#0f766e")
    ax2.set_ylim(0.75, 1.0)

    fig.suptitle("Training history: full 25-epoch Zenodo lung run", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_case_metrics(per_case: dict[str, dict[str, float]], output_path: Path) -> None:
    case_ids = list(per_case.keys())
    dice = [per_case[case]["dice_lung"] for case in case_ids]
    hd95 = [per_case[case]["hd95_lung"] for case in case_ids]
    x = np.arange(len(case_ids))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.bar(x, dice, color="#0f766e")
    ax1.set_ylabel("Dice")
    ax1.set_ylim(0.7, 1.0)
    ax1.grid(True, axis="y", color="#d6d3d1", linewidth=0.8, alpha=0.8)
    ax1.set_title("Per-case performance on 20 prepared CT volumes")

    ax2.bar(x, hd95, color="#b45309")
    ax2.set_ylabel("HD95")
    ax2.set_xlabel("Case")
    ax2.grid(True, axis="y", color="#d6d3d1", linewidth=0.8, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([case.replace("zenodo_lung_", "") for case in case_ids], rotation=90)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_array(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def plot_overlay(case_id: str, base_dir: Path, output_path: Path, title: str) -> None:
    image = _load_array(base_dir / "cases" / f"{case_id}_ct.nii.gz")
    label = _load_array(base_dir / "cases" / f"{case_id}_label.nii.gz")
    pred = _load_array(base_dir / "predictions" / f"{case_id}_pred.nii.gz")
    slice_index = int(np.argmax(label.sum(axis=(1, 2))))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image[slice_index], cmap="gray", vmin=0.0, vmax=1.0)
    ax.contour(label[slice_index], levels=[0.5], colors=["#22c55e"], linewidths=1.8)
    ax.contour(pred[slice_index], levels=[0.5], colors=["#ef4444"], linewidths=1.3)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    run_dir = Path("outputs/zenodo_lung_full_e25")
    docs_dir = ensure_dir(Path("docs/assets/results"))

    metrics = load_json(run_dir / "metrics.json")
    training = load_json(run_dir / "checkpoints/training_summary.json")
    model_card = (run_dir / "model_card.md").read_text(encoding="utf-8")

    per_case = metrics["per_case"]
    ranked = sorted(
        (
            {
                "case_id": case_id,
                "dice_lung": values["dice_lung"],
                "hd95_lung": values["hd95_lung"],
            }
            for case_id, values in per_case.items()
        ),
        key=lambda item: item["dice_lung"],
        reverse=True,
    )

    summary = {
        "dataset": "COVID-19 CT Lung and Infection Segmentation Dataset",
        "doi": "10.5281/zenodo.3757476",
        "workspace": str(run_dir),
        "mean_metrics": metrics["mean"],
        "best_validation_dice": training["best_val_dice"],
        "epochs": training["epochs"],
        "device": training["device"],
        "num_train_cases": training["num_train_cases"],
        "num_val_cases": training["num_val_cases"],
        "best_cases": ranked[:3],
        "weakest_cases": ranked[-3:],
        "history": training["history"],
        "model_card_excerpt": model_card,
    }

    (docs_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (docs_dir / "metrics_full_e25.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (docs_dir / "training_summary_full_e25.json").write_text(json.dumps(training, indent=2), encoding="utf-8")

    plot_training_curve(training["history"], docs_dir / "training_curve.png")
    plot_case_metrics(per_case, docs_dir / "case_metrics.png")
    plot_overlay("zenodo_lung_006", run_dir, docs_dir / "best_case_overlay.png", "Best case: zenodo_lung_006")
    plot_overlay("zenodo_lung_017", run_dir, docs_dir / "weak_case_overlay.png", "Weakest case: zenodo_lung_017")


if __name__ == "__main__":
    main()
