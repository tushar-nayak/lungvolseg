from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged

from .config import DEFAULT_PATCH_SIZE, ensure_dir
from .model import build_model
from .runtime import resolve_device


def infer_case(
    checkpoint_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    num_classes: int = 3,
    patch_size: tuple[int, int, int] = DEFAULT_PATCH_SIZE,
) -> str:
    device = resolve_device()
    model = build_model(out_channels=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            EnsureTyped(keys=["image"]),
        ]
    )
    batch = transforms({"image": str(image_path)})
    image = batch["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = sliding_window_inference(image, patch_size, 1, model)
        prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype("uint8")

    reference = sitk.ReadImage(str(image_path))
    if prediction.shape == tuple(reference.GetSize()):
        prediction = np.transpose(prediction, (2, 1, 0))
    output = sitk.GetImageFromArray(prediction)
    output.CopyInformation(reference)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    sitk.WriteImage(output, str(output_path))
    return str(output_path)
