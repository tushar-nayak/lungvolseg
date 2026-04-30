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

Run the complete Zenodo full-volume lung segmentation workflow:

```bash
python3 scripts/run_zenodo_lung_pipeline.py \
  --workspace outputs/zenodo_lung \
  --epochs 8 \
  --target-depth 96 \
  --target-height 128 \
  --target-width 128
```

The command downloads the dataset, verifies MD5 hashes, extracts 20 full CT volumes and lung masks, prepares lung-region 3D cases, trains a MONAI 3D UNet, runs inference, exports VTK meshes, computes metrics, and writes a model card.

Main outputs:

- `outputs/zenodo_lung/raw/`: downloaded Zenodo archives and metadata
- `outputs/zenodo_lung/cases/`: prepared 3D lung-region training cases
- `outputs/zenodo_lung/checkpoints/best_model.pt`: trained checkpoint
- `outputs/zenodo_lung/predictions/`: predicted segmentation volumes
- `outputs/zenodo_lung/meshes/`: VTK-exported `.stl` and `.vtp` lung surfaces
- `outputs/zenodo_lung/metrics.json`: Dice and HD95 validation metrics
- `outputs/zenodo_lung/model_card.md`: generated model card

## Dataset

The real workflow uses the COVID-19 CT Lung and Infection Segmentation Dataset:

- Zenodo record: `https://zenodo.org/records/3757476`
- DOI: `10.5281/zenodo.3757476`
- License: `CC-BY-4.0`
- Download size: about `1.1 GB` for CT volumes plus `10 MB` for lung masks
- Files: `COVID-19-CT-Seg_20cases.zip`, `Lung_Mask.zip`
- Cases: 20 labeled full CT scans

Labels:

- `0`: background
- `1`: lung

The source dataset includes left-lung, right-lung, and infection annotations. This branch converts the lung masks into a binary lung target because that is the navigation-prep class with the strongest signal and the most stable validation behavior.

To download and prepare only the dataset:

```bash
python3 scripts/download_zenodo_lung.py \
  --raw-dir data/real/zenodo_lung/raw \
  --cases-dir data/real/zenodo_lung/cases \
  --target-depth 96 \
  --target-height 128 \
  --target-width 128
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
LUNGNAV_RENDER_PREVIEW=1 python3 scripts/run_zenodo_lung_pipeline.py --workspace outputs/zenodo_lung
```

## Notes

- The default `--epochs 8` command is still a compact engineering run, not a clinically meaningful model.
- This project is for research and engineering demonstration only. It is not validated for clinical decision-making, diagnosis, or robotic bronchoscopy guidance.
- The pipeline uses MONAI for training and inference, SimpleITK for spatial image IO and preprocessing, and VTK for surface extraction and mesh export.
