# LungVolSeg

Real-data lung CT segmentation and surface export pipeline for full-volume lung volumes:

1. download a public chest CT segmentation dataset
2. prepare compact 3D medical volumes with SimpleITK
3. train or run a 3D MONAI segmentation model
4. extract 3D surfaces with VTK
5. export meshes as `.stl` and `.vtp`
6. compute validation metrics
7. generate a short model card

This branch is intentionally narrow: it supports one working dataset and one complete pipeline.

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

The same working model code is also mirrored under `code/`:

- `code/lungvolseg/`: model, training, inference, metrics, mesh export, and dataset pipeline modules
- `code/scripts/`: runnable downloader and full pipeline scripts

Run the mirrored copy directly with:

```bash
PYTHONPATH=code python3 code/scripts/run_zenodo_lung_pipeline.py \
  --workspace outputs/zenodo_lung \
  --epochs 8
```

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

Using the mirrored root code:

```bash
PYTHONPATH=code python3 code/scripts/download_zenodo_lung.py \
  --raw-dir data/real/zenodo_lung/raw \
  --cases-dir data/real/zenodo_lung/cases
```

Downloaded data and generated outputs are ignored by Git.

## Current Results

A completed 25-epoch full run on all 20 prepared cases is available under `outputs/zenodo_lung_full_e25/`.

- Mean lung Dice: `0.9243`
- Mean lung HD95: `10.5909`
- Best validation Dice during training: `0.9539`
- Current internal split: `13` train / `7` validation cases

Full-run command:

```bash
python3 scripts/run_zenodo_lung_pipeline.py \
  --workspace outputs/zenodo_lung_full_e25 \
  --epochs 25 \
  --target-depth 96 \
  --target-height 128 \
  --target-width 128
```

Primary result files:

- `outputs/zenodo_lung_full_e25/metrics.json`
- `outputs/zenodo_lung_full_e25/checkpoints/training_summary.json`
- `outputs/zenodo_lung_full_e25/model_card.md`

## Mesh Preview

Mesh files are always exported as `.stl` and `.vtp`. Preview PNG rendering is disabled by default for headless environments. On a workstation with a working X/GL setup, opt in with:

```bash
LUNGNAV_RENDER_PREVIEW=1 python3 scripts/run_zenodo_lung_pipeline.py --workspace outputs/zenodo_lung
```

## GitHub Pages

The repository includes a static results page under `docs/`. It publishes lightweight tracked artifacts from the full 25-epoch run:

- summary JSON
- metrics JSON
- training summary JSON
- training curve figure
- per-case metrics figure
- best and weakest case overlays

Rebuild the published assets from the saved full run with:

```bash
python3 scripts/build_results_site_assets.py
```

To publish on GitHub Pages, set the repository Pages source to the `main` branch and the `/docs` folder.

## Notes

- The default `--epochs 8` command is still a compact engineering run. The stronger documented baseline in this repository comes from the 25-epoch full run above.
- This project is for research and engineering demonstration only. It is not validated for clinical decision-making, diagnosis, or robotic bronchoscopy guidance.
- The pipeline uses MONAI for training and inference, SimpleITK for spatial image IO and preprocessing, and VTK for surface extraction and mesh export.
