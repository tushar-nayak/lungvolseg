from __future__ import annotations

import json
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
)

from .config import ensure_dir
from .model import build_model
from .runtime import resolve_device


def _make_transforms(train: bool) -> Compose:
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ]
    if train:
        transforms.append(RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2))
    return Compose(transforms)


def _split_cases(cases: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    pivot = max(1, int(round(len(cases) * 0.67)))
    pivot = min(pivot, len(cases) - 1) if len(cases) > 1 else 1
    return cases[:pivot], cases[pivot:]


def train_model(
    cases: list[dict[str, str]],
    output_dir: str | Path,
    epochs: int = 2,
    learning_rate: float = 1e-3,
) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)
    train_cases, val_cases = _split_cases(cases)
    if not val_cases:
        val_cases = train_cases[-1:]
        train_cases = train_cases[:-1] or train_cases

    device = resolve_device()
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    metric = DiceMetric(include_background=False, reduction="mean")

    train_loader = DataLoader(Dataset(train_cases, _make_transforms(train=True)), batch_size=1, shuffle=True)
    val_loader = DataLoader(Dataset(val_cases, _make_transforms(train=False)), batch_size=1, shuffle=False)

    best_score = -1.0
    history: list[dict[str, float]] = []
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].long().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        model.eval()
        metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                predictions = torch.argmax(logits, dim=1, keepdim=True)
                metric(y_pred=predictions, y=labels)

        val_dice = float(metric.aggregate().item())
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(len(train_loader), 1),
            "val_dice": val_dice,
        }
        history.append(epoch_record)

        if val_dice > best_score:
            best_score = val_dice
            torch.save(model.state_dict(), best_model_path)

    metadata = {
        "device": str(device),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "num_train_cases": len(train_cases),
        "num_val_cases": len(val_cases),
        "best_val_dice": best_score,
        "history": history,
        "checkpoint": str(best_model_path),
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata
