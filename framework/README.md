# CLTR Framework

Analysis framework for the Controlled Laboratory Thermal Response study.

## Role In The CLTR Repository

This folder is the software layer inside the main `cltr` repository.
It preprocesses the dataset, runs cohort and session analysis, and generates the full results tree, including report artifacts under `../../results/.../reports/` when run from `framework/`.

## Layout

- `cltr_framework/`: Python package
- `docs/`: framework notes and audit material
- `scripts/`: utility scripts
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

Generated artifacts should be written into the sibling `work/results/` tree, for example `../../results/test_dataset/`.

Atlas publication is a separate manual step owned by the main repository layer. Only the atlas web bundle should be published into `../docs/atlas/`, not the full results tree:

```bash
python ../scripts/publish_atlas.py \
  --results-dir ../../results/test_dataset \
  --docs-atlas-dir ../docs/atlas
```

This copies only:

- `reports/work/`
- `reports/cohort/`
- `reports/sessions/`

It excludes review JSON/Markdown files from the published atlas bundle and flattens the public atlas to:

- `docs/atlas/index.html`
- `docs/atlas/cohort/`
- `docs/atlas/sessions/`
