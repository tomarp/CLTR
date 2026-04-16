from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import CLTRConfig, default_config
from .utils import ensure_dir, to_numeric


def _safe_frame(analyzed: dict | None, key: str) -> pd.DataFrame:
    if not analyzed:
        return pd.DataFrame()
    value = analyzed.get(key, pd.DataFrame())
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _status(score: float) -> str:
    if score >= 85:
        return "strong"
    if score >= 65:
        return "adequate"
    if score >= 45:
        return "limited"
    return "weak"


def _round_or_none(value: float | int | None, digits: int = 3) -> float | int | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return round(float(value), digits)


def _domain(name: str, score: float, evidence: list[str]) -> dict:
    return {"name": name, "score": round(float(score), 1), "status": _status(score), "evidence": evidence}


def evaluate_report_quality(
    outdir: str | Path,
    manifest: pd.DataFrame,
    session_reports: list[dict],
    cohort_report: dict,
    *,
    analyzed: dict | None = None,
    config: CLTRConfig | None = None,
) -> dict:
    cfg = config or default_config()
    work_root = ensure_dir(Path(outdir) / "reports" / "work")
    review_md = work_root / "report_quality_review.md"
    review_json = work_root / "report_quality_review.json"

    session_summary = _safe_frame(analyzed, "session_summary")
    sample_status = _safe_frame(analyzed, "sample_status")
    coverage_summary = _safe_frame(analyzed, "coverage_summary")
    condition_contrasts = _safe_frame(analyzed, "condition_contrasts")
    sensor_agreement = _safe_frame(analyzed, "sensor_agreement")
    agreement_summary = _safe_frame(analyzed, "agreement_summary")
    feature_associations = _safe_frame(analyzed, "feature_associations")
    cohort_primary_endpoints = _safe_frame(analyzed, "cohort_primary_endpoints")
    pattern_summary = _safe_frame(analyzed, "pattern_summary")
    predictive_benchmarks = _safe_frame(analyzed, "predictive_benchmarks")
    mixed_effects_primary = _safe_frame(analyzed, "mixed_effects_primary")

    n_sessions = int(len(session_reports))
    n_manifest_rows = int(len(manifest))
    n_participants = int(manifest["participant_id"].nunique()) if "participant_id" in manifest.columns and not manifest.empty else 0
    conditions = int(manifest["condition_code"].nunique()) if "condition_code" in manifest.columns and not manifest.empty else 0

    main_counts = [len(r.get("narrative_codes", [])) for r in session_reports]
    appendix_counts = [len(r.get("appendix_codes", [])) for r in session_reports]
    code_sequences = ["|".join(r.get("narrative_codes", [])) for r in session_reports]
    lead_labels = [str(r.get("lead_label", "")).strip() for r in session_reports if str(r.get("lead_label", "")).strip()]
    card_tags = ["|".join(r.get("atlas_tags", [])) for r in session_reports]
    cohort_codes = cohort_report.get("narrative_codes", []) + cohort_report.get("appendix_codes", [])

    median_main = float(pd.Series(main_counts).median()) if main_counts else 0.0
    median_appendix = float(pd.Series(appendix_counts).median()) if appendix_counts else 0.0
    dominant_sequence_share = float(pd.Series(code_sequences).value_counts(normalize=True).iloc[0]) if code_sequences else 1.0
    unique_lead_ratio = (len(set(lead_labels)) / len(lead_labels)) if lead_labels else 0.0
    unique_tag_ratio = (len(set(card_tags)) / len(card_tags)) if card_tags else 0.0
    agreement_in_main_share = (
        sum("S09" in r.get("narrative_codes", []) for r in session_reports) / n_sessions if n_sessions else 0.0
    )

    inferential_ok = bool(sample_status["cohort_inference_eligible"].iloc[0]) if not sample_status.empty and "cohort_inference_eligible" in sample_status.columns else False

    mean_questionnaire = float(to_numeric(session_summary.get("questionnaire_completeness", pd.Series(dtype=float))).mean()) if not session_summary.empty else np.nan
    mean_empatica = float(to_numeric(session_summary.get("empatica_fraction", pd.Series(dtype=float))).mean()) if not session_summary.empty else np.nan
    mean_biopac = float(to_numeric(session_summary.get("biopac_fraction", pd.Series(dtype=float))).mean()) if not session_summary.empty else np.nan
    mean_indoor = float(to_numeric(session_summary.get("indoor_fraction", pd.Series(dtype=float))).mean()) if not session_summary.empty else np.nan

    coverage_non_support = coverage_summary.loc[~coverage_summary["feature"].astype(str).str.startswith("support_")].copy() if not coverage_summary.empty else pd.DataFrame()
    median_feature_coverage = float(to_numeric(coverage_non_support.get("coverage_fraction", pd.Series(dtype=float))).median()) if not coverage_non_support.empty else np.nan
    sparse_features = (
        coverage_non_support.loc[to_numeric(coverage_non_support["coverage_fraction"]) < 0.5, "feature"].astype(str).tolist()
        if not coverage_non_support.empty
        else []
    )

    eligible_contrasts = condition_contrasts.loc[condition_contrasts.get("eligible", pd.Series(dtype=int)) == 1].copy() if not condition_contrasts.empty else pd.DataFrame()
    significant_contrasts = eligible_contrasts.loc[to_numeric(eligible_contrasts.get("p_value", pd.Series(dtype=float))) < 0.05].copy() if not eligible_contrasts.empty else pd.DataFrame()
    max_pairs = int(to_numeric(condition_contrasts.get("n_pairs", pd.Series(dtype=float))).max()) if not condition_contrasts.empty and to_numeric(condition_contrasts.get("n_pairs", pd.Series(dtype=float))).notna().any() else 0
    max_abs_dz = float(to_numeric(condition_contrasts.get("cohens_dz", pd.Series(dtype=float))).abs().max()) if not condition_contrasts.empty and to_numeric(condition_contrasts.get("cohens_dz", pd.Series(dtype=float))).notna().any() else np.nan

    eligible_agreement = agreement_summary.loc[agreement_summary.get("summary_status", pd.Series(dtype=str)) == "eligible"].copy() if not agreement_summary.empty else pd.DataFrame()
    median_agreement = float(to_numeric(eligible_agreement.get("median_spearman_r", pd.Series(dtype=float))).median()) if not eligible_agreement.empty else np.nan

    endpoint_cells = int(len(cohort_primary_endpoints))
    inferential_endpoint_cells = int((cohort_primary_endpoints.get("evidence_status", pd.Series(dtype=str)) == "inferential").sum()) if not cohort_primary_endpoints.empty else 0

    domains = []

    sample_score = 25.0
    sample_score += min(25.0, (n_sessions / 80.0) * 15.0)
    sample_score += min(20.0, (n_participants / 20.0) * 10.0)
    sample_score += 20.0 if inferential_ok else 0.0
    sample_score += 10.0 if conditions >= 4 else 0.0
    sample_evidence = [
        f"{n_sessions} analyzed sessions across {n_participants} participants and {conditions} conditions.",
        "Cohort inference gate is open." if inferential_ok else "Cohort inference gate is closed; output is descriptive only.",
    ]
    domains.append(_domain("Sample And Design Coverage", sample_score, sample_evidence))

    preprocessing_score = 30.0
    if np.isfinite(mean_questionnaire):
        preprocessing_score += max(0.0, min(15.0, mean_questionnaire * 15.0))
    if np.isfinite(mean_empatica):
        preprocessing_score += max(0.0, min(10.0, mean_empatica * 10.0))
    if np.isfinite(mean_biopac):
        preprocessing_score += max(0.0, min(10.0, mean_biopac * 10.0))
    if np.isfinite(mean_indoor):
        preprocessing_score += max(0.0, min(10.0, mean_indoor * 10.0))
    if np.isfinite(median_feature_coverage):
        preprocessing_score += max(0.0, min(15.0, median_feature_coverage * 15.0))
    preprocessing_evidence = [
        f"Mean questionnaire completeness: {_round_or_none(mean_questionnaire, 3)}.",
        f"Mean sensor support fractions: Empatica={_round_or_none(mean_empatica, 3)}, BIOPAC={_round_or_none(mean_biopac, 3)}, indoor={_round_or_none(mean_indoor, 3)}.",
        f"Median analytic feature coverage: {_round_or_none(median_feature_coverage, 3)}.",
    ]
    domains.append(_domain("Preprocessing And Data Support", preprocessing_score, preprocessing_evidence))

    inference_score = 15.0
    inference_score += 20.0 if inferential_ok else 0.0
    inference_score += min(20.0, float(len(eligible_contrasts)))
    inference_score += min(10.0, float(len(significant_contrasts)))
    inference_score += 10.0 if np.isfinite(max_abs_dz) and max_abs_dz >= 0.5 else 0.0
    inference_score += 10.0 if inferential_endpoint_cells > 0 else 0.0
    inference_score += 10.0 if np.isfinite(median_agreement) and median_agreement >= 0.4 else 0.0
    inference_evidence = [
        f"Eligible paired contrasts: {int(len(eligible_contrasts))}; nominally significant eligible contrasts: {int(len(significant_contrasts))}.",
        f"Maximum paired sample size: {max_pairs}; maximum absolute Cohen's dz: {_round_or_none(max_abs_dz, 3)}.",
        f"Eligible agreement summaries: {int(len(eligible_agreement))}; median eligible Spearman agreement: {_round_or_none(median_agreement, 3)}.",
    ]
    domains.append(_domain("Inferential Strength", inference_score, inference_evidence))

    reporting_score = 20.0
    reporting_score += 10.0 if median_main <= 12 else 0.0
    reporting_score += 10.0 if dominant_sequence_share <= 0.7 else 0.0
    reporting_score += min(15.0, unique_lead_ratio * 20.0)
    reporting_score += min(15.0, unique_tag_ratio * 20.0)
    reporting_score += 10.0 if "C11" in cohort_codes else 0.0
    reporting_score += 10.0 if agreement_in_main_share >= 0.25 else 0.0
    reporting_evidence = [
        f"Median main-figure load per session: {_round_or_none(median_main, 1)}; dominant narrative sequence share: {_round_or_none(dominant_sequence_share, 3)}.",
        f"Lead-label diversity ratio: {_round_or_none(unique_lead_ratio, 3)}; atlas-tag diversity ratio: {_round_or_none(unique_tag_ratio, 3)}.",
        f"Cohort pattern atlas present: {'yes' if 'C11' in cohort_codes else 'no'}; agreement promoted to session main narrative: {_round_or_none(agreement_in_main_share, 3)}.",
    ]
    domains.append(_domain("Reporting And Scientific Storytelling", reporting_score, reporting_evidence))

    reproducibility_score = 25.0
    reproducibility_score += 15.0 if n_manifest_rows == n_sessions else 0.0
    reproducibility_score += 10.0 if endpoint_cells > 0 else 0.0
    reproducibility_score += 10.0 if not sensor_agreement.empty else 0.0
    reproducibility_score += 10.0 if not feature_associations.empty else 0.0
    reproducibility_score += 10.0 if not pattern_summary.empty else 0.0
    reproducibility_evidence = [
        f"Manifest/session alignment: {n_manifest_rows} manifest rows for {n_sessions} rendered sessions.",
        f"Cohort endpoint rows: {endpoint_cells}; sensor agreement rows: {int(len(sensor_agreement))}; feature-association rows: {int(len(feature_associations))}.",
    ]
    domains.append(_domain("Reproducibility And Deliverable Completeness", reproducibility_score, reproducibility_evidence))

    overall_score = float(np.mean([d["score"] for d in domains])) if domains else 0.0

    blockers = []
    if inferential_ok:
        blockers.append("No multiple-comparison correction is applied to the many paired contrasts and association screens.")
    else:
        blockers.append("The pipeline remains descriptive because the cohort inference gate is closed for the selected output.")
    if not ({"ci_low", "ci_high"} <= set(cohort_primary_endpoints.columns) if not cohort_primary_endpoints.empty else False):
        blockers.append("No uncertainty intervals or bootstrap intervals are reported for the main endpoints.")
    if predictive_benchmarks.empty:
        blockers.append("No subject-independent predictive validation layer is implemented.")
    else:
        blockers.append("Predictive benchmarking exists, but it is still preliminary: no nested tuning, calibration analysis, or external validation is implemented.")
    if mixed_effects_primary.empty and inferential_ok:
        blockers.append("The mixed-effects primary-endpoint layer did not yield usable estimates for the current run.")
    blockers.extend(
        [
            "Preprocessing is transparent but still summary-driven: there is no modality-specific artifact rejection, signal-quality index, or leakage-aware transform fitting stage.",
            "The generated report set is presentation-heavy; session narratives are repetitive and too dense for a journal-grade main story.",
        ]
    )
    if sparse_features:
        blockers.append(f"Several analytic variables remain sparsely supported (<50% coverage), including: {', '.join(sparse_features[:6])}.")

    strengths = [
        "The framework has a clean end-to-end structure: validation, minute alignment, phase summaries, cohort analysis, and HTML reporting are already integrated.",
        "The CLTR study design is strong for a controlled repeated-measures dataset: 20 participants, 80 sessions, and all four factorial conditions are represented.",
        "The current outputs are suitable for a high-quality data descriptor or methods-and-dataset paper.",
    ]
    if inferential_ok and len(eligible_contrasts) > 0:
        strengths.append("Within-subject paired contrasts are already wired into the analysis layer and can be upgraded into a stronger statistical workflow.")
    if not eligible_agreement.empty:
        strengths.append("Cross-device agreement is quantified rather than assumed, which is important for wearable-vs-lab credibility.")

    verdict = "not_phd_research_grade"
    if overall_score >= 80 and inferential_ok and len(eligible_contrasts) >= 8 and len(significant_contrasts) >= 2:
        verdict = "near_q1_ready"
    elif overall_score >= 65:
        verdict = "promising_but_not_yet_q1_ready"

    lines = [
        "# CLTR Research-Grade Audit",
        "",
        "This file is a scientific adequacy audit of the generated CLTR report portfolio.",
        "",
        "## Verdict",
        f"- Overall status: `{verdict}`",
        f"- Overall score: {overall_score:.1f}/100",
        "- Research-grade interpretation: the current framework is strong for dataset curation and descriptive exploratory analysis, but it is not yet a PhD/Q1-grade analytical paper engine.",
        "",
        "## Domain Scores",
    ]
    for domain in domains:
        lines.append(f"- `{domain['name']}`: {domain['score']:.1f}/100 ({domain['status']})")
        for item in domain["evidence"]:
            lines.append(f"  - {item}")
    lines.extend(
        [
            "",
            "## Key Strengths",
        ]
    )
    for item in strengths:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Blocking Gaps",
        ]
    )
    for item in blockers:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## What Would Make This Q1-Ready",
            "- Add a leakage-controlled inferential layer with confidence intervals, multiple-testing correction, and clear primary/secondary endpoint separation.",
            "- Add a subject-independent predictive benchmark layer for the stated translational goal, with grouped validation by participant.",
            "- Upgrade preprocessing from minute aggregation only to explicit modality-specific signal-quality and artifact-control procedures.",
            "- Compress the journal narrative to a smaller set of non-redundant figures with one defensible central claim per panel.",
        ]
    )

    payload = {
        "review_mode": "scientific_audit",
        "overall_status": verdict,
        "overall_score": round(overall_score, 1),
        "n_sessions": n_sessions,
        "n_participants": n_participants,
        "n_conditions": conditions,
        "inferential_ok": inferential_ok,
        "domains": domains,
        "strengths": strengths,
        "blocking_gaps": blockers,
        "summary_metrics": {
            "median_main_figures": _round_or_none(median_main, 1),
            "median_appendix_figures": _round_or_none(median_appendix, 1),
            "dominant_sequence_share": _round_or_none(dominant_sequence_share, 3),
            "unique_lead_ratio": _round_or_none(unique_lead_ratio, 3),
            "unique_tag_ratio": _round_or_none(unique_tag_ratio, 3),
            "agreement_in_main_share": _round_or_none(agreement_in_main_share, 3),
            "mean_questionnaire_completeness": _round_or_none(mean_questionnaire, 3),
            "mean_empatica_fraction": _round_or_none(mean_empatica, 3),
            "mean_biopac_fraction": _round_or_none(mean_biopac, 3),
            "mean_indoor_fraction": _round_or_none(mean_indoor, 3),
            "median_feature_coverage": _round_or_none(median_feature_coverage, 3),
            "eligible_contrasts": int(len(eligible_contrasts)),
            "significant_eligible_contrasts": int(len(significant_contrasts)),
            "max_pairs": max_pairs,
            "max_abs_cohens_dz": _round_or_none(max_abs_dz, 3),
            "eligible_agreement_metrics": int(len(eligible_agreement)),
            "median_eligible_spearman_agreement": _round_or_none(median_agreement, 3),
            "endpoint_cells": endpoint_cells,
            "inferential_endpoint_cells": inferential_endpoint_cells,
        },
    }

    review_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    review_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "overall_status": verdict,
        "overall_score": round(overall_score, 1),
        "markdown_path": str(review_md),
        "json_path": str(review_json),
        "domains": domains,
        "strengths": strengths,
        "blocking_gaps": blockers,
    }
