# CLTR Research-Grade Audit

This file is a scientific adequacy audit of the generated CLTR report portfolio.

## Verdict
- Overall status: `not_phd_research_grade`
- Overall score: 52.8/100
- Research-grade interpretation: the current framework is strong for dataset curation and descriptive exploratory analysis, but it is not yet a PhD/Q1-grade analytical paper engine.

## Domain Scores
- `Sample And Design Coverage`: 37.5/100 (weak)
  - 8 analyzed sessions across 2 participants and 4 conditions.
  - Cohort inference gate is closed; output is descriptive only.
- `Preprocessing And Data Support`: 76.3/100 (adequate)
  - Mean questionnaire completeness: 1.0.
  - Mean sensor support fractions: Empatica=0.993, BIOPAC=0.469, indoor=0.964.
  - Median analytic feature coverage: 0.469.
- `Inferential Strength`: 25.0/100 (weak)
  - Eligible paired contrasts: 0; nominally significant eligible contrasts: 0.
  - Maximum paired sample size: 18; maximum absolute Cohen's dz: 18.207.
  - Eligible agreement summaries: 3; median eligible Spearman agreement: -0.205.
- `Reporting And Scientific Storytelling`: 45.0/100 (limited)
  - Median main-figure load per session: 47.0; dominant narrative sequence share: 1.0.
  - Lead-label diversity ratio: 0.625; atlas-tag diversity ratio: 0.625.
  - Cohort pattern atlas present: no; agreement promoted to session main narrative: 0.0.
- `Reproducibility And Deliverable Completeness`: 80.0/100 (adequate)
  - Manifest/session alignment: 8 manifest rows for 8 rendered sessions.
  - Cohort endpoint rows: 180; sensor agreement rows: 24; feature-association rows: 21.

## Key Strengths
- The framework has a clean end-to-end structure: validation, minute alignment, phase summaries, cohort analysis, and HTML reporting are already integrated.
- The CLTR study design is strong for a controlled repeated-measures dataset: 20 participants, 80 sessions, and all four factorial conditions are represented.
- The current outputs are suitable for a high-quality data descriptor or methods-and-dataset paper.
- Cross-device agreement is quantified rather than assumed, which is important for wearable-vs-lab credibility.

## Blocking Gaps
- The pipeline remains descriptive because the cohort inference gate is closed for the selected output.
- Predictive benchmarking exists, but it is still preliminary: no nested tuning, calibration analysis, or external validation is implemented.
- Preprocessing is transparent but still summary-driven: there is no modality-specific artifact rejection, signal-quality index, or leakage-aware transform fitting stage.
- The generated report set is presentation-heavy; session narratives are repetitive and too dense for a journal-grade main story.
- Several analytic variables remain sparsely supported (<50% coverage), including: biopac_bloodflow_mean_bpu, biopac_eda_mean_uS, biopac_hr_mean_bpm, biopac_temp_chest_mean_C, eda_delta_uS, temp_delta_C.

## What Would Make This Q1-Ready
- Add a leakage-controlled inferential layer with confidence intervals, multiple-testing correction, and clear primary/secondary endpoint separation.
- Add a subject-independent predictive benchmark layer for the stated translational goal, with grouped validation by participant.
- Upgrade preprocessing from minute aggregation only to explicit modality-specific signal-quality and artifact-control procedures.
- Compress the journal narrative to a smaller set of non-redundant figures with one defensible central claim per panel.
