# CLTR Framework

Analysis and reporting framework for the Controlled Laboratory Thermal Response study.

## Role In The CLTR Repository

This folder is the software layer inside the main `cltr` repository.
It preprocesses the dataset, runs cohort and session analysis, and generates the atlas and report outputs.

## Layout

- `cltr_framework/`: Python package
- `docs/`: framework notes and audit material
- `scripts/`: utility scripts
- `logos/`: assets copied into generated report outputs

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
cltr-framework run-all --dataset-root /path/to/dataset --outdir /path/to/results
```

Or from source:

```bash
python -m cltr_framework run-all --dataset-root /path/to/dataset --outdir /path/to/results
```

## Output

Generated artifacts should be written into the parent project results tree, for example `../results/test_dataset/`.
Published atlas files can then be copied into `../docs/atlas/test_dataset/`.
