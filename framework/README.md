# CLTR Framework

Analysis and reporting framework for the Controlled Laboratory Thermal Response study.

This package is the execution layer behind the CLTR data-processing workflow. It validates the released dataset structure, preprocesses multimodal streams into aligned analytical tables, runs cohort and session analyses, and renders the report bundles that are later published into the CLTR atlas.

## At A Glance

| Item | Value |
| --- | --- |
| Repository role | Analysis engine inside the main `cltr` repository |
| Package entry point | `cltr-framework` / `python -m cltr_framework` |
| Typical input | Extracted CLTR dataset root |
| Typical output | `work/results/<dataset_name>/` |
| Main products | validation, aligned data, analysis tables, cohort reports, session reports |
| Publication boundary | this framework writes results; the repository publisher promotes atlas pages into `docs/atlas/` |

## What The Framework Does

The framework is designed around one end-to-end contract:

1. read a CLTR dataset release from disk
2. validate the expected study structure and session manifest
3. preprocess raw and semi-processed streams into aligned minute-level analytical views
4. run cohort-level and session-level analytical routines
5. generate report bundles suitable for internal review and atlas publication

In practice this means the framework handles:

- dataset structure checks
- session manifest checks
- time parsing and alignment
- sensor preprocessing and support grading
- cohort summaries and condition comparisons
- session narratives and figure generation
- HTML report generation for cohort and session outputs

## Workflow Map

```text
CLTR dataset release
        |
        v
  validate dataset
        |
        v
 preprocess sessions
        |
        v
 analyze cohort + sessions
        |
        v
 render reports into work/results/<dataset_name>/reports/
        |
        v
 publish atlas subset with ../scripts/publish_atlas.py
```

## Repository Context

This folder lives inside the main CLTR repository and is only one part of the full project layout:

```text
work/cltr/
├── docs/                    # GitHub Pages site and published atlas
├── framework/               # this framework
└── scripts/                 # repo-side publication utilities

work/results/
└── <dataset_name>/          # generated framework outputs
```

The framework should write generated artifacts into `work/results/`, not into the repository source tree itself.

## Framework Layout

```text
framework/
├── cltr_framework/
│   ├── cli.py               # command-line entry points
│   ├── pipeline.py          # orchestration
│   ├── dataset.py           # dataset loading and validation helpers
│   ├── preprocessing.py     # preprocessing and harmonization
│   ├── analysis.py          # cohort/session analysis logic
│   ├── reporting.py         # HTML report rendering and figure generation
│   ├── review.py            # quality-review and scoring helpers
│   ├── config.py            # configuration models
│   └── utils.py             # shared utilities
├── docs/                    # framework notes and audit material
├── scripts/                 # framework-adjacent utility scripts
├── pyproject.toml
└── README.md
```

## Environment Setup

The framework is intended to run in an isolated Python environment.

### Local installation

```bash
cd framework
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Conda-based workflow

If you manage environments with conda, activate the target environment first and then install the framework in editable mode:

```bash
cd framework
conda activate <env_name>
pip install -e .
```

## Command-Line Usage

The CLI exposes the full pipeline as well as stage-specific commands.

### Full pipeline

```bash
cltr-framework run-all \
  --dataset-root /path/to/dataset \
  --outdir /path/to/results
```

Equivalent module invocation:

```bash
python -m cltr_framework run-all \
  --dataset-root /path/to/dataset \
  --outdir /path/to/results
```

### Stage-specific runs

Validation only:

```bash
python -m cltr_framework validate \
  --dataset-root /path/to/dataset \
  --outdir /path/to/results
```

Preprocessing only:

```bash
python -m cltr_framework preprocess \
  --dataset-root /path/to/dataset \
  --outdir /path/to/results
```

Analysis only:

```bash
python -m cltr_framework analyze \
  --dataset-root /path/to/dataset \
  --outdir /path/to/results
```

Report rendering only:

```bash
python -m cltr_framework report \
  --dataset-root /path/to/dataset \
  --outdir /path/to/results
```

## Recommended Local Paths

For the current workspace layout, the cleanest pattern is:

```text
/Volumes/dev/20251106_CLTR/CLTR_root/
├── work/
│   ├── cltr/
│   │   ├── docs/
│   │   ├── framework/
│   │   └── scripts/
│   └── results/
│       └── dataset_full/
└── <dataset release>
```

Typical invocation from `framework/`:

```bash
python -m cltr_framework run-all \
  --dataset-root ../../dataset_full \
  --outdir ../../results/dataset_full
```

## Output Structure

The framework writes a complete result bundle into the output directory you pass through `--outdir`.

Typical structure:

```text
work/results/<dataset_name>/
├── analysis/
├── data/
├── manifests/
├── reports/
│   ├── cohort/
│   ├── sessions/
│   └── work/
├── validation/
└── run_summary.json
```

Important report destinations:

- `reports/cohort/` for study-wide report pages and figures
- `reports/sessions/` for per-session report pages and figures
- `reports/work/index.html` for the atlas landing page generated by the framework

## Publishing The Atlas

Atlas publication is intentionally separate from analytical execution.

The framework generates report artifacts under `work/results/.../reports/`. The repository-layer publisher then copies the website-facing subset into `docs/atlas/`.

Run from `framework/`:

```bash
python ../scripts/publish_atlas.py \
  --results-dir ../../results/<dataset_name> \
  --docs-atlas-dir ../docs/atlas
```

The publisher copies only:

- `reports/work/`
- `reports/cohort/`
- `reports/sessions/`

It does not publish the full raw results tree. That boundary keeps GitHub Pages focused on web-facing report artifacts.

## Typical End-To-End Example

```bash
cd framework
source .venv/bin/activate

python -m cltr_framework run-all \
  --dataset-root ../../dataset_full \
  --outdir ../../results/dataset_full

python ../scripts/publish_atlas.py \
  --results-dir ../../results/dataset_full \
  --docs-atlas-dir ../docs/atlas
```

After that:

- framework outputs live under `work/results/dataset_full/`
- the published atlas lives under `work/cltr/docs/atlas/`

## Notes For Developers

- `reporting.py` is the main rendering layer for the cohort/session HTML outputs.
- `analysis.py` is where most derived metrics, support grading, and comparative analysis live.
- `review.py` contains quality and readiness heuristics used to summarize report quality.
- `scripts/create_test_subset.py` is useful when you need a smaller reproducible subset for development or debugging.

## Common Operating Rules

- keep generated analytical outputs in `work/results/`
- do not manually edit generated report bundles if the same change belongs in the framework source
- republish the atlas after report-generation changes that affect HTML output
- treat `docs/atlas/` as published output, not the source of truth

## Related Paths

- main repository: [`../`](..)
- public site: [`../docs/`](../docs)
- atlas publisher: [`../scripts/publish_atlas.py`](../scripts/publish_atlas.py)
- framework package: [`./cltr_framework/`](./cltr_framework)
- framework audit notes: [`./docs/research_audit.md`](./docs/research_audit.md)
