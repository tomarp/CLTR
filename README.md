# CLTR

Chrono-Light Thermal Response (`CLTR`) is a controlled warm-exposure study and the main repository for its public project site, analysis framework, and atlas publication workflow.

![CLTR](./docs/assets/logos/cltr.png)

This repository serves two roles:

- the main GitHub repository for the CLTR project
- the GitHub Pages source for the public CLTR website under [`docs/`](./docs)

The analysis pipeline lives in [`framework/`](./framework). It generates the canonical result bundles into `work/results/`, outside this repository. The public atlas under [`docs/atlas/`](./docs/atlas) is then published from those generated results.

## Project Scope

CLTR studies thermal responses under moderately warm indoor conditions using a repeated-measures `2 x 2` design:

- light intensity: `Bright` vs `Dim`
- time of day: `Morning` vs `Midday`

Each participant completes four sessions, one for each lighting and time combination. The study combines:

- environmental measurements
- physiological measurements
- repeated thermal comfort and sensation responses
- structured fan-cooling and re-warming phases

The project is intended to support work in:

- thermal comfort
- thermo-physiology
- circadian and lighting effects
- adaptive and intelligent building control

## Repository Roles

The repository is intentionally split into three layers:

1. `work/cltr`
   The source repository and GitHub Pages site.
2. `work/cltr/framework`
   The analysis pipeline code.
3. `work/results`
   Generated outputs from the pipeline, kept outside the git repository.

That boundary is deliberate:

- source code and public pages stay versioned in the repository
- generated outputs stay in `work/results`
- published atlas pages are copied from `work/results` into `work/cltr/docs/atlas`

## Repository Layout

- [`docs/`](./docs): GitHub Pages content for the public CLTR website
- [`docs/index.html`](./docs/index.html): public homepage
- [`docs/exp.html`](./docs/exp.html): experiment overview page
- [`docs/publication.html`](./docs/publication.html): publications page
- [`docs/atlas/`](./docs/atlas): published atlas bundle for GitHub Pages
- [`framework/`](./framework): Python analysis pipeline
- [`scripts/publish_atlas.py`](./scripts/publish_atlas.py): repo-side publisher that copies generated atlas reports from `work/results` into `docs/atlas`

## Dataset

The CLTR dataset is distributed separately from this repository. The current project DOI referenced by the framework and site is:

- Zenodo: `https://doi.org/10.5281/zenodo.17817175`

Recommended usage:

1. download the dataset from Zenodo
2. extract it somewhere under `work/` or another local analysis location
3. point the framework to that extracted dataset root with `--dataset-root`

The pipeline expects the CLTR dataset directory structure rather than a single file archive.

## Pipeline

The framework performs the full local workflow:

1. dataset validation
2. preprocessing and minute-level alignment
3. cohort and session analysis
4. report bundle generation under the result tree

The framework writes a complete run into a dataset-specific directory such as:

```text
work/results/test_dataset/
```

Typical contents include:

- `validation/`
- `manifests/`
- `data/`
- `analysis/`
- `reports/`
- `run_summary.json`

Within `reports/`, the framework writes:

- `reports/work/`
- `reports/cohort/`
- `reports/sessions/`

Those report outputs are treated as generated artifacts. They are not the GitHub Pages site by themselves.

## Pipeline Setup

From [`framework/`](./framework):

```bash
cd framework
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The package requires Python `3.11+`.

## Pipeline Execution

Example run from `work/cltr/framework`:

```bash
python -m cltr_framework run-all \
  --dataset-root ../../test_dataset \
  --outdir ../../results/test_dataset
```

You can also run narrower stages:

```bash
python -m cltr_framework validate --dataset-root ../../test_dataset --outdir ../../results/test_dataset
python -m cltr_framework preprocess --dataset-root ../../test_dataset --outdir ../../results/test_dataset
python -m cltr_framework analyze --dataset-root ../../test_dataset --outdir ../../results/test_dataset
python -m cltr_framework report --dataset-root ../../test_dataset --outdir ../../results/test_dataset
```

## Publishing The Atlas

After generating results, publish only the atlas-facing report subset into `docs/atlas`:

```bash
cd framework
python ../scripts/publish_atlas.py \
  --results-dir ../../results/test_dataset \
  --docs-atlas-dir ../docs/atlas
```

This keeps heavy analytical outputs in `work/results` while publishing only the website-facing atlas bundle.

The publisher currently copies:

- `reports/work/`
- `reports/cohort/`
- `reports/sessions/`

and writes the public atlas entry point to:

- [`docs/atlas/index.html`](./docs/atlas/index.html)

## Why `scripts/publish_atlas.py` Exists

The atlas publisher belongs to the repository layer, not the framework package and not the website content itself.

That is why it lives in [`scripts/`](./scripts) now:

- it is not part of the analytical pipeline package
- it is not page content
- it is a repo-side publication utility that bridges generated results into GitHub Pages

This is the intended contract:

- `framework/` produces results
- `work/results` stores results
- `scripts/publish_atlas.py` publishes the atlas subset
- `docs/` serves the public site

## GitHub Pages

GitHub Pages should be configured to publish from the `docs/` directory of this repository.

The repository therefore contains both:

- hand-maintained public pages in `docs/`
- generated atlas pages copied into `docs/atlas/`

## Recommended Local Paths

With the current workspace layout, the clean local arrangement is:

```text
work/
  cltr/
    docs/
    framework/
    scripts/
  results/
    <dataset_name>/
```

That arrangement should be treated as the canonical local schema for this project.
