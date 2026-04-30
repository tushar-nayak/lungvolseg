# A Reproducible Lung CT Navigation-Prep Pipeline for Full-Volume Lung Segmentation and Surface Export

## Abstract

**Background:** CT-based planning for lung navigation workflows depends on reliable volumetric preprocessing, segmentation, surface extraction, and validation. This project implements a compact, reproducible pipeline for full-volume chest CT lung segmentation using open-source medical imaging tools.

**Objective:** The objective was to build a working engineering pipeline that downloads a public real CT dataset, prepares 3D volumes, trains a 3D segmentation model, exports anatomical surfaces, computes validation metrics, and generates documentation artifacts suitable for navigation-prep experimentation.

**Methods:** The pipeline uses the COVID-19 CT Lung and Infection Segmentation Dataset from Zenodo, containing 20 labeled CT scans. Lung masks are converted into a binary lung segmentation target. SimpleITK handles image IO, cropping, normalization, and resampling. MONAI provides a 3D UNet segmentation model and training/inference utilities. VTK converts predicted labels into 3D mesh surfaces exported as `.stl` and `.vtp`. Validation uses Dice similarity and 95th percentile Hausdorff distance.

**Results:** A smoke validation run on 4 CT cases trained for 2 epochs produced a mean lung Dice score of `0.8802` and mean lung HD95 of `8.50` voxels after preprocessing to `64 x 96 x 96` volumes. The pipeline generated checkpoints, predictions, VTK meshes, metrics, and a model card.

**Conclusion:** The project provides a complete real-data lung CT navigation-prep pipeline with reproducible dataset download, model training, mesh export, and validation. The current implementation is intended for research and engineering demonstration, not clinical use.

## Introduction

Robotic bronchoscopy and related lung navigation workflows rely on CT-derived anatomical understanding. Before navigation, a CT scan is commonly used for planning, segmentation, registration, and 3D visualization. A practical navigation-prep pipeline should therefore handle medical image spatial metadata, support 3D segmentation, produce geometric surfaces, and report quantitative validation metrics.

This project focuses on the engineering substrate for that workflow. The goal is not to claim clinical performance, but to demonstrate an end-to-end reproducible pipeline built from commonly used medical imaging tools:

- SimpleITK for medical image IO and spatial preprocessing
- MONAI for 3D medical image segmentation
- VTK for surface extraction and mesh export
- PyTorch for model optimization

The first real-data target is binary lung segmentation from full-volume CT scans. Lung segmentation is a useful navigation-prep target because it provides a stable organ-level anatomical surface and avoids the severe imbalance of small lesion labels.

## Methods

### Dataset

The pipeline uses the **COVID-19 CT Lung and Infection Segmentation Dataset** hosted on Zenodo.

- Zenodo record: `https://zenodo.org/records/3757476`
- DOI: `10.5281/zenodo.3757476`
- License: `CC-BY-4.0`
- Cases: 20 labeled full CT scans
- Downloaded archives: `COVID-19-CT-Seg_20cases.zip` and `Lung_Mask.zip`

The original dataset includes left lung, right lung, and infection annotations. This implementation converts the lung masks into a binary target:

- `0`: background
- `1`: lung

This choice was made because binary lung segmentation is the strongest fit for CT navigation-prep surface generation and provides a stable validation target for a compact engineering run.

### Pipeline Overview

The primary command is:

```bash
python3 scripts/run_zenodo_lung_pipeline.py \
  --workspace outputs/zenodo_lung \
  --epochs 8 \
  --target-depth 96 \
  --target-height 128 \
  --target-width 128
```

The same code is mirrored under root `code/`:

```bash
PYTHONPATH=code python3 code/scripts/run_zenodo_lung_pipeline.py \
  --workspace outputs/zenodo_lung \
  --epochs 8
```

The workflow performs the following steps:

1. Download CT and lung-mask archives from Zenodo.
2. Verify archive MD5 checksums.
3. Extract NIfTI volumes.
4. Pair CT volumes with lung masks.
5. Crop around the lung-mask bounding box.
6. Clip CT intensities to `[-1000, 400]` HU and normalize to `[0, 1]`.
7. Resize each case to a compact 3D target shape.
8. Train a MONAI 3D UNet.
9. Run inference on prepared cases.
10. Export predicted lung surfaces as `.stl` and `.vtp`.
11. Compute Dice and HD95 metrics.
12. Generate a model card.

### Preprocessing

Images are loaded with SimpleITK and converted to NumPy arrays in `z, y, x` order. Lung masks are binarized so that any nonzero lung annotation becomes foreground. The pipeline computes a lung bounding box with a small margin, crops the CT and mask to that region, clips CT intensity values to a lung-appropriate window, normalizes intensities, and resizes volumes to a configured target shape.

The default target shape is:

```text
96 x 128 x 128
```

The smoke-test run used:

```text
64 x 96 x 96
```

### Model

The model is a MONAI 3D UNet with:

- `spatial_dims=3`
- `in_channels=1`
- `out_channels=2`
- channels `(16, 32, 64, 128)`
- strides `(2, 2, 2)`
- two residual units per level
- dropout `0.1`

Training uses Dice plus cross-entropy loss through MONAI's `DiceCELoss`, with Adam optimization and learning rate `1e-3`.

### Inference and Surface Export

Inference uses MONAI sliding-window inference with a window matching the prepared target shape. Predicted class labels are written back as NIfTI volumes with SimpleITK.

VTK extracts mesh surfaces from predicted labels using discrete marching cubes, applies windowed-sinc smoothing, and writes:

- `lung.stl`
- `lung.vtp`

Preview rendering is disabled by default because many server and CI environments do not provide a working X/GL context.

### Validation Metrics

Validation computes:

- Dice similarity coefficient for the lung foreground class
- 95th percentile Hausdorff distance for the lung foreground class

Metrics are saved to:

```text
outputs/zenodo_lung/metrics.json
```

## Results

A verified smoke run was executed with:

```bash
python3 scripts/run_zenodo_lung_pipeline.py \
  --workspace outputs/zenodo_lung_smoke \
  --epochs 2 \
  --target-depth 64 \
  --target-height 96 \
  --target-width 96 \
  --max-cases 4
```

The run completed end to end and generated:

- prepared NIfTI cases
- model checkpoint
- prediction volumes
- VTK lung meshes
- metrics JSON
- model card

### Quantitative Results

| Case | Dice Lung | HD95 Lung |
|---|---:|---:|
| `zenodo_lung_001` | 0.8903 | 8.6023 |
| `zenodo_lung_002` | 0.8770 | 9.2736 |
| `zenodo_lung_003` | 0.8873 | 8.1240 |
| `zenodo_lung_004` | 0.8662 | 8.0000 |
| **Mean** | **0.8802** | **8.5000** |

These results are substantially stronger than the earlier small slice-based lesion-segmentation attempt because the Zenodo dataset contains full CT volumes and the binary lung target has clearer anatomical signal.

## Discussion

The pipeline demonstrates a practical path from real CT data to navigation-prep artifacts. The lung segmentation target provides a useful 3D anatomical boundary that can be exported for visualization, QA, registration experiments, and planning-interface prototypes.

Several engineering choices were made to keep the project reproducible and tractable:

- The dataset is downloaded by script instead of stored in Git.
- MD5 checks verify archive integrity.
- Generated data and outputs are ignored by Git.
- The model uses compact 3D volumes to keep training feasible on limited hardware.
- Mesh export is included in the main pipeline rather than treated as a separate visualization demo.

The smoke-run metrics show that the pipeline is functioning and that the task is learnable with minimal training. For stronger reported performance, the next evaluation should train on all 20 cases with more epochs, use a fixed train/validation split, and report metrics only on held-out validation cases.

## Limitations

This work has several limitations:

- The reported smoke-run metrics use only 4 cases and 2 training epochs.
- The default pipeline performs compact resizing, which sacrifices native scanner resolution.
- The current evaluation reports predictions on prepared cases and should be expanded to a stricter held-out protocol.
- The segmentation target is binary lung, not airway, vessel, lobe, or lesion navigation anatomy.
- The generated meshes are suitable for engineering demonstration but are not validated for procedural planning.
- The project is not a medical device and is not intended for clinical decision-making.

## Reproducibility

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

Run the main pipeline:

```bash
python3 scripts/run_zenodo_lung_pipeline.py \
  --workspace outputs/zenodo_lung \
  --epochs 8 \
  --target-depth 96 \
  --target-height 128 \
  --target-width 128
```

Run the smoke-test configuration:

```bash
python3 scripts/run_zenodo_lung_pipeline.py \
  --workspace outputs/zenodo_lung_smoke \
  --epochs 2 \
  --target-depth 64 \
  --target-height 96 \
  --target-width 96 \
  --max-cases 4
```

Run from the mirrored root code:

```bash
PYTHONPATH=code python3 code/scripts/run_zenodo_lung_pipeline.py \
  --workspace outputs/zenodo_lung \
  --epochs 8
```

## Data and Code Availability

The dataset is publicly available from Zenodo:

```text
https://zenodo.org/records/3757476
```

The working code is available in two layouts:

- installable package: `src/lungnav/`
- root mirror: `code/lungnav/`

Command-line entrypoints are available in:

- `scripts/`
- `code/scripts/`

## Conclusion

This project implements a real-data, full-volume lung CT segmentation pipeline for navigation-prep engineering. It downloads a public CT dataset, prepares 3D volumes, trains a MONAI segmentation model, exports VTK lung surfaces, computes validation metrics, and generates documentation artifacts. The verified smoke run demonstrates end-to-end operation with mean lung Dice of `0.8802`, providing a stronger baseline than the earlier slice-based lesion-segmentation workflow.

## References

1. Ma J, et al. COVID-19 CT Lung and Infection Segmentation Dataset. Zenodo. DOI: `10.5281/zenodo.3757476`.
2. MONAI Consortium. MONAI: Medical Open Network for AI.
3. Lowekamp BC, Chen DT, Ibanez L, Blezek D. The Design of SimpleITK.
4. Schroeder W, Martin K, Lorensen B. The Visualization Toolkit.
