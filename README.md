# Lung CT Navigation Prep Pipeline

Small end-to-end pipeline for lung CT navigation-prep tasks:

1. ingest a chest CT volume
2. preprocess it with SimpleITK
3. train or run a 3D MONAI segmentation model
4. extract anatomical surfaces with VTK
5. export meshes and preview renders
6. compute validation metrics
7. generate a short model card

The repo includes a synthetic chest CT generator so the workflow can run in an empty repository.

## Install

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Quick start

Run the full demo on synthetic data:

```bash
python3 scripts/run_pipeline.py \
  --workspace outputs/demo \
  --num-cases 6 \
  --epochs 2
```

Main outputs:

- `outputs/demo/raw/`: synthetic CT volumes and labels
- `outputs/demo/preprocessed/`: resampled and normalized volumes
- `outputs/demo/checkpoints/`: trained MONAI weights
- `outputs/demo/predictions/`: predicted segmentations
- `outputs/demo/meshes/`: `.stl` and `.vtp` anatomical meshes plus preview PNGs
- `outputs/demo/metrics.json`: validation metrics
- `outputs/demo/model_card.md`: short model card

Preview PNG rendering is disabled by default for headless environments. To opt in on a workstation with a working X/GL setup, run with `LUNGNAV_RENDER_PREVIEW=1`.

## Real MedSeg Dataset

The `real` branch adds a reproducible path for MedSeg Covid Dataset 1:

- Source: Figshare, `MedSeg Covid Dataset 1`
- DOI: `10.6084/m9.figshare.13521488.v2`
- License: `CC0`
- Size: about `167 MB`
- Labels: background, ground-glass opacity, consolidation, pleural effusion

Download and prepare the real data into small 3D pseudo-volumes:

```bash
python3 scripts/download_real_medseg.py \
  --raw-dir data/real/medseg/raw \
  --cases-dir data/real/medseg/cases
```

Run the complete real-data workflow:

```bash
python3 scripts/run_real_medseg_pipeline.py \
  --workspace outputs/real_medseg \
  --epochs 1 \
  --block-depth 16 \
  --xy-size 128
```

Main outputs:

- `outputs/real_medseg/raw/`: downloaded Figshare NIfTI files and metadata
- `outputs/real_medseg/cases/`: converted 3D training cases
- `outputs/real_medseg/checkpoints/`: trained MONAI checkpoint
- `outputs/real_medseg/predictions/`: predicted segmentations
- `outputs/real_medseg/meshes/`: VTK-exported `.stl` and `.vtp` surfaces
- `outputs/real_medseg/metrics.json`: validation metrics
- `outputs/real_medseg/model_card.md`: generated model card

The source dataset is 2D axial CT slices packed into NIfTI files. The real pipeline converts contiguous slices into compact 3D pseudo-volumes so the same 3D MONAI/VTK stack can run without downloading a large full-volume CT dataset.

## Real data layout

For external data, place cases as:

```text
data/
  raw/
    case001_ct.nii.gz
    case001_label.nii.gz
    case002_ct.nii.gz
    case002_label.nii.gz
```

Then preprocess:

```bash
python3 scripts/preprocess_ct.py --input-dir data/raw --output-dir data/preprocessed
```

Train:

```bash
python3 scripts/train_segmentation.py --data-dir data/preprocessed --output-dir outputs/train --epochs 50
```

Infer:

```bash
python3 scripts/infer_segmentation.py \
  --checkpoint outputs/train/best_model.pt \
  --input-image data/preprocessed/case001_ct.nii.gz \
  --output-label outputs/case001_pred.nii.gz
```

Extract meshes:

```bash
python3 scripts/extract_meshes.py \
  --label-image outputs/case001_pred.nii.gz \
  --output-dir outputs/case001_meshes
```

Validate and generate a model card:

```bash
python3 scripts/validate_and_card.py \
  --prediction-dir outputs/demo/predictions \
  --reference-dir outputs/demo/preprocessed \
  --model-dir outputs/demo/checkpoints \
  --output-dir outputs/demo
```

## Labels

- `0`: background
- `1`: lungs
- `2`: central airway

## Notes

- The synthetic data path is for demonstration only and is not clinically meaningful.
- The pipeline uses MONAI for training/inference, SimpleITK for spatial preprocessing and IO, and VTK for surface extraction/export.
