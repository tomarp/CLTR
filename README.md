# CLTR

Chrono-Light Thermal Response is a multimodal controlled warm-exposure study and repository.

## Synopsis

The Chrono-Light Thermal Response dataset contains multimodal measurements from a controlled warm-exposure experiment using a `2 x 2` factorial design crossing:

- light intensity: `Bright` vs `Dim`
- time-of-day: `Morning` vs `Midday`

Each participant completed four sessions, one in each lighting-time combination, under moderately warm indoor conditions. The study combines continuous physiological responses, environmental measurements, repeated thermal sensation and comfort ratings, and structured fan-cooling and re-warming phases.

The dataset and project support research on:

- thermal comfort
- thermo-physiology
- light and circadian influences
- dynamic and intelligent building control

## Repository Structure

- `docs/`: GitHub Pages site for the public homepage and published atlas
- `framework/`: CLTR analysis and reporting framework
- `results/`: generated framework outputs
- `docs/assets/`: shared homepage assets

## Published Site

- homepage: `docs/index.html`
- atlas: `docs/atlas/test_dataset/work/index.html`

Enable GitHub Pages from the `docs/` folder when publishing this repository.

## Framework

The framework in `framework/` preprocesses the dataset, runs cohort and session analysis, and generates the atlas and report outputs.

Imported legacy acquisition utilities from the previous repository state are preserved under:

- `framework/scripts/legacy/`

## Current Output

The current clean test build is stored in:

- `results/test_dataset/`
