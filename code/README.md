# Code Layout

This directory mirrors the working Zenodo lung segmentation pipeline from the installable package.

- `lungvolseg/`: model, training, inference, metrics, VTK mesh export, model-card generation, and Zenodo dataset workflow
- `scripts/`: command-line entrypoints for downloading/preparing data and running the full pipeline

Run directly from the repository root:

```bash
PYTHONPATH=code python3 code/scripts/run_zenodo_lung_pipeline.py --workspace outputs/zenodo_lung --epochs 8
```

Or use the package layout from the repository root:

```bash
python3 scripts/run_zenodo_lung_pipeline.py --workspace outputs/zenodo_lung --epochs 8
```
