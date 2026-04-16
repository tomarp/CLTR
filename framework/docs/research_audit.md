# CLTR Framework Research Audit

## Verdict

`work/cltr_framework/` is not yet PhD research grade for a Q1 results paper.

It is already a strong dataset-processing and exploratory reporting framework. It is not yet a defensible journal-grade analytical engine.

## What Is Already Strong

- The pipeline is clean and modular: validation, preprocessing, cohort analysis, reporting, and review are separated clearly.
- The dataset structure is strong for repeated-measures science: 20 participants, 80 sessions, 4 within-subject conditions.
- The preprocessing layer produces canonical minute-level tables and phase summaries, which is a good foundation for reproducible downstream work.
- The analysis layer already computes paired contrasts, agreement tables, coverage summaries, and endpoint summaries.
- The reporting layer is broad and produces session, cohort, and atlas-style outputs that are useful for exploratory interpretation.

## Why It Is Not Yet Research Grade

### 1. The old report audit was not a scientific gate

In [review.py](/Volumes/dev/20251106_CLTR/CLTR_root/work/cltr_framework/review.py:1), the audit has now been upgraded. Before this change, the audit only described figure diversity and repetition, and explicitly said it was not a pass/fail gate.

That means the framework could generate polished reports without ever answering the real question: are the outputs statistically defensible and publication-ready?

### 2. Preprocessing is transparent but still shallow for physiology

In [preprocessing.py](/Volumes/dev/20251106_CLTR/CLTR_root/work/cltr_framework/preprocessing.py:1), most high-frequency channels are reduced to minute means, SDs, or sums.

Current limitations:

- no explicit signal-quality indices
- no modality-specific artifact rejection
- no motion-artifact control for wearable physiology
- no morphology-preserving processing for PPG/EDA beyond simple aggregation
- no fold-aware preprocessing stage for future predictive modelling

This is acceptable for a dataset descriptor. It is not enough for a top-tier physiology or multimodal-computing paper.

### 3. Inferential analysis is present, but not publication-safe yet

In [analysis.py](/Volumes/dev/20251106_CLTR/CLTR_root/work/cltr_framework/analysis.py:1), paired contrasts are computed using paired t-tests and effect sizes.

Current limitations:

- no multiple-comparison correction across the many endpoint/phase/comparison tests
- no bootstrap confidence intervals
- no robust mixed-effects modelling for repeated measures
- no explicit primary vs secondary endpoint hierarchy
- no power or sensitivity framing
- association screens remain descriptive correlations

So the framework can produce many nominal `p < 0.05` results, but not yet a journal-safe inferential package.

### 4. There is no predictive validation layer

The methods brief asks for a framework capable of stronger scientific outputs, but there is currently no subject-independent modelling benchmark.

Missing pieces:

- grouped train/test splitting by participant
- leakage-controlled feature scaling and selection
- nested CV for hyperparameter selection
- baseline models versus stronger models
- calibrated predictive metrics
- model interpretation layer tied to validated models

Without this, the framework supports descriptive science, not translational or predictive science.

### 5. Reporting is too dense and too repetitive

The current report system in [reporting.py](/Volumes/dev/20251106_CLTR/CLTR_root/work/cltr_framework/reporting.py:1) is comprehensive, but it leans toward exhaustive portfolio generation.

That creates two problems:

- too many figures dilute the scientific story
- repeated session narratives weaken journal readability

This is useful for lab exploration. It is not the same as a publishable figure strategy.

## Research-Grade Gap Summary

To become Q1-ready, the framework needs four upgrades:

1. A physiology-aware preprocessing layer
2. A publication-safe inferential statistics layer
3. A subject-independent predictive benchmarking layer
4. A smaller, sharper manuscript-oriented report layer

## Revolution Plan

### Stage A. Upgrade signal science

- add wearable signal-quality metrics per minute and per phase
- add artifact burden summaries for PPG, EDA, temperature, and motion
- add physiology-aware feature extraction beyond minute means
- keep all transformations parameterized and exportable

### Stage B. Upgrade statistics

- define a strict primary endpoint set
- run repeated-measures or mixed-effects models per endpoint
- add multiplicity correction
- add confidence intervals and effect sizes to all key contrasts
- separate confirmatory analyses from exploratory analyses

### Stage C. Add predictive science

- build participant-grouped modelling datasets
- benchmark logistic regression, elastic net, random forest, and gradient boosting
- use subject-independent validation only
- export calibration and generalization diagnostics

### Stage D. Rebuild the paper narrative

- limit the main paper to a compact set of figures
- move exploratory breadth to appendix or atlas
- promote only claims that survive support, uncertainty, and multiplicity checks

## Bottom Line

Right now, CLTR is strong enough for:

- a data paper
- a framework paper focused on curation and multimodal alignment
- a preliminary results paper with careful wording

Right now, CLTR is not strong enough for:

- a definitive PhD-grade Q1 results paper claiming robust multimodal causal or predictive conclusions

The foundation is good. The scientific engine still needs another serious layer.
