# Lung CT Navigation Prep Pipeline

Real-data lung CT navigation-prep pipeline for CT-based planning experiments:

1. download a public chest CT segmentation dataset
2. prepare compact 3D medical volumes with SimpleITK
3. train or run a 3D MONAI segmentation model
4. extract 3D surfaces with VTK
5. export meshes as `.stl` and `.vtp`
6. compute validation metrics
7. generate a short model card

This branch is intentionally real-data only.

## Install

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Quick Start

Run the complete MedSeg real-data workflow:

```bash
python3 scripts/run_real_medseg_pipeline.py \
  --workspace outputs/real_medseg \
  --epochs 1 \
  --block-depth 16 \
  --xy-size 128
```

The command downloads the dataset, verifies MD5 hashes, converts the CT slice stack into compact 3D cases, trains a MONAI 3D UNet, runs inference, exports VTK meshes, computes metrics, and writes a model card.

Main outputs:

- `outputs/real_medseg/raw/`: downloaded Figshare NIfTI files and metadata
- `outputs/real_medseg/cases/`: converted 3D training cases
- `outputs/real_medseg/checkpoints/best_model.pt`: trained checkpoint
- `outputs/real_medseg/predictions/`: predicted segmentation volumes
- `outputs/real_medseg/meshes/`: VTK-exported `.stl` and `.vtp` surfaces
- `outputs/real_medseg/metrics.json`: Dice and HD95 validation metrics
- `outputs/real_medseg/model_card.md`: generated model card

## Dataset

The real workflow uses MedSeg Covid Dataset 1:

- Source page: `https://medicalsegmentation.com/covid19/`
- Figshare record: `https://figshare.com/articles/dataset/MedSeg_Covid_Dataset_1/13521488`
- DOI: `10.6084/m9.figshare.13521488.v2`
- License: `CC0`
- Download size: about `167 MB`
- Files: `tr_im.nii.gz`, `tr_mask.nii.gz`, `val_im.nii.gz`, clinical details CSV

Labels:

- `0`: background
- `1`: ground-glass opacity
- `2`: consolidation
- `3`: pleural effusion

The source dataset is 2D axial CT slices packed into NIfTI files. This branch groups contiguous slices into compact 3D pseudo-volumes so the same 3D MONAI and VTK stack can run without requiring a large full-volume CT dataset.

To download and prepare only the dataset:

```bash
python3 scripts/download_real_medseg.py \
  --raw-dir data/real/medseg/raw \
  --cases-dir data/real/medseg/cases \
  --block-depth 16 \
  --xy-size 128
```

Downloaded data and generated outputs are ignored by Git.

## External NIfTI Data

For another real dataset, place paired NIfTI files like this:

```text
data/
  raw/
    case001_ct.nii.gz
    case001_label.nii.gz
    case002_ct.nii.gz
    case002_label.nii.gz
```

Preprocess:

```bash
python3 scripts/preprocess_ct.py --input-dir data/raw --output-dir data/preprocessed
```

Train:

```bash
python3 scripts/train_segmentation.py \
  --data-dir data/preprocessed \
  --output-dir outputs/train \
  --epochs 50
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

## Mesh Preview

Mesh files are always exported as `.stl` and `.vtp`. Preview PNG rendering is disabled by default for headless environments. On a workstation with a working X/GL setup, opt in with:

```bash
LUNGNAV_RENDER_PREVIEW=1 python3 scripts/run_real_medseg_pipeline.py --workspace outputs/real_medseg
```

## Notes

- The default `--epochs 1` command is a smoke test, not a clinically meaningful model.
- This project is for research and engineering demonstration only. It is not validated for clinical decision-making, diagnosis, or robotic bronchoscopy guidance.
- The pipeline uses MONAI for training and inference, SimpleITK for spatial image IO and preprocessing, and VTK for surface extraction and mesh export.
