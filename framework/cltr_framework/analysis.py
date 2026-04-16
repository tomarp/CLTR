from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLTRConfig
from .preprocessing import COMPARISON_BLOCKS, QUESTIONNAIRE_SET_COLUMNS
from .utils import benjamini_hochberg, bootstrap_mean_ci, paired_ttest, to_numeric


ANALYTIC_FEATURES = [
    "thermal_comfort",
    "thermal_sensation",
    "thermal_pleasure",
    "thermal_preference",
    "visual_comfort",
    "room_comfort",
    "master_skin_chest_C",
    "master_hand_C",
    "master_dpg_C",
    "thermal_gradient_C",
    "indoor_air_temp_mean_C",
    "indoor_air_velocity_mean_m_s",
    "fan_current_A",
    "fan_control_au",
    "empatica_hr_mean_bpm",
    "empatica_eda_mean_uS",
    "empatica_temp_mean_C",
    "biopac_hr_mean_bpm",
    "biopac_eda_mean_uS",
    "biopac_temp_chest_mean_C",
    "biopac_bloodflow_mean_bpu",
    "thermal_state_index_C",
    "hr_delta_bpm",
    "eda_delta_uS",
    "temp_delta_C",
]

PRIMARY_ENDPOINTS = [
    "thermal_comfort",
    "thermal_sensation",
    "master_dpg_C",
    "indoor_air_velocity_mean_m_s",
    "empatica_temp_mean_C",
    "biopac_temp_chest_mean_C",
    "empatica_hr_mean_bpm",
    "biopac_hr_mean_bpm",
    "biopac_bloodflow_mean_bpu",
]


class CLTRAnalyzer:
    def __init__(self, config: CLTRConfig):
        self.config = config

    @staticmethod
    def _empty_mixed_effects() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "metric",
                "term",
                "beta",
                "se",
                "p_value",
                "ci_low",
                "ci_high",
                "n_obs",
                "n_participants",
                "converged",
                "p_value_fdr",
                "significant_fdr",
            ]
        )

    @staticmethod
    def _empty_predictive_benchmarks() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "task",
                "model",
                "n_features",
                "n_groups",
                "n_samples",
                "balanced_accuracy_mean",
                "balanced_accuracy_sd",
                "macro_f1_mean",
                "macro_f1_sd",
                "roc_auc_mean",
                "target_levels",
            ]
        )

    def build_cohort_outputs(self, session_minutes: list[pd.DataFrame], phase_summaries: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
        cohort_minute = pd.concat(session_minutes, ignore_index=True) if session_minutes else pd.DataFrame()
        comparison_minute = (
            cohort_minute.loc[cohort_minute["protocol_block"].astype(str).isin(COMPARISON_BLOCKS)].copy()
            if not cohort_minute.empty and "protocol_block" in cohort_minute.columns
            else cohort_minute.copy()
        )
        cohort_phase = pd.concat(phase_summaries, ignore_index=True) if phase_summaries else pd.DataFrame()
        session_summary = self._session_summary(cohort_minute)
        sample_status = self._sample_status(session_summary)
        sensor_agreement = self._sensor_agreement(comparison_minute)
        exploratory_feature_summary = self._exploratory_feature_summary(comparison_minute)
        condition_support_summary = self._condition_support_summary(session_summary)
        phase_pattern_inventory = self._phase_pattern_inventory(cohort_phase)
        session_primary_endpoints = self._session_primary_endpoints(cohort_phase)
        preprocessing_qc_summary = self._preprocessing_qc_summary(comparison_minute)
        condition_contrasts = self._condition_contrasts(cohort_phase, sample_status)
        mixed_effects_primary = self._mixed_effects_primary(cohort_phase, sample_status)
        predictive_benchmarks = self._predictive_benchmarks(session_primary_endpoints)
        return {
            "cohort_minute_features": cohort_minute,
            "cohort_minute_comparison_window": comparison_minute,
            "cohort_phase_summary": cohort_phase,
            "session_summary": session_summary,
            "sample_status": sample_status,
            "coverage_summary": self._coverage_summary(comparison_minute),
            "preprocessing_qc_summary": preprocessing_qc_summary,
            "exploratory_feature_summary": exploratory_feature_summary,
            "condition_support_summary": condition_support_summary,
            "sensor_agreement": sensor_agreement,
            "agreement_summary": self._agreement_summary(sensor_agreement),
            "condition_phase_summary": self._condition_phase_summary(cohort_phase),
            "condition_contrasts": condition_contrasts,
            "feature_associations": self._feature_associations(cohort_phase),
            "phase_pattern_inventory": phase_pattern_inventory,
            "pattern_summary": self._pattern_summary(phase_pattern_inventory),
            "participant_profiles": self._participant_profiles(cohort_phase),
            "session_primary_endpoints": session_primary_endpoints,
            "cohort_primary_endpoints": self._cohort_primary_endpoints(cohort_phase, sample_status),
            "mixed_effects_primary": mixed_effects_primary,
            "predictive_benchmarks": predictive_benchmarks,
        }

    def _sample_status(self, session_summary: pd.DataFrame) -> pd.DataFrame:
        n_sessions = int(session_summary["session_id"].nunique()) if not session_summary.empty else 0
        n_participants = int(session_summary["participant_id"].nunique()) if not session_summary.empty else 0
        inferential_ok = (
            n_sessions >= self.config.runtime.min_cohort_sessions_for_inference
            and n_participants >= self.config.runtime.min_cohort_participants_for_inference
        )
        rows = [
            {
                "n_sessions": n_sessions,
                "n_participants": n_participants,
                "min_sessions_required": self.config.runtime.min_cohort_sessions_for_inference,
                "min_participants_required": self.config.runtime.min_cohort_participants_for_inference,
                "cohort_inference_eligible": int(inferential_ok),
                "status": "eligible" if inferential_ok else "descriptive_only",
            }
        ]
        return pd.DataFrame(rows)

    def _coverage_summary(self, cohort_minute: pd.DataFrame) -> pd.DataFrame:
        if cohort_minute.empty:
            return pd.DataFrame()
        rows = []
        for feature in [f for f in ANALYTIC_FEATURES if f in cohort_minute.columns]:
            present = cohort_minute[feature].notna()
            rows.append(
                {
                    "feature": feature,
                    "n_non_null": int(present.sum()),
                    "coverage_fraction": float(present.mean()),
                    "n_sessions_with_data": int(cohort_minute.groupby("session_id")[feature].apply(lambda s: s.notna().any()).sum()),
                }
            )
        support_cols = [c for c in cohort_minute.columns if c.startswith("support_")]
        for feature in support_cols:
            vals = to_numeric(cohort_minute[feature])
            rows.append(
                {
                    "feature": feature,
                    "n_non_null": int(vals.notna().sum()),
                    "coverage_fraction": float(vals.mean()) if vals.notna().any() else np.nan,
                    "n_sessions_with_data": int(cohort_minute.groupby("session_id")[feature].apply(lambda s: s.notna().any()).sum()),
                }
            )
        return pd.DataFrame(rows).sort_values(["coverage_fraction", "feature"], ascending=[False, True]).reset_index(drop=True)

    def _preprocessing_qc_summary(self, cohort_minute: pd.DataFrame) -> pd.DataFrame:
        if cohort_minute.empty:
            return pd.DataFrame()
        qc_cols = [c for c in cohort_minute.columns if c.startswith("quality_")]
        if not qc_cols:
            return pd.DataFrame()
        rows = []
        for col in qc_cols:
            vals = to_numeric(cohort_minute[col])
            rows.append(
                {
                    "metric": col,
                    "valid_fraction": float(vals.mean()) if vals.notna().any() else np.nan,
                    "n_valid_minutes": int(vals.sum()) if vals.notna().any() else 0,
                    "n_minutes": int(vals.notna().sum()),
                    "n_sessions_with_quality_signal": int(cohort_minute.groupby("session_id")[col].apply(lambda s: s.notna().any()).sum()),
                }
            )
        return pd.DataFrame(rows).sort_values(["valid_fraction", "metric"], ascending=[False, True]).reset_index(drop=True)

    def _feature_domain(self, feature: str) -> str:
        if feature.startswith("support_"):
            return "support"
        if "thermal_" in feature or feature.startswith("master_"):
            return "thermal"
        if feature.startswith("empatica_") or feature.startswith("biopac_") or feature in {"hr_delta_bpm", "eda_delta_uS", "temp_delta_C"}:
            return "physiology"
        if feature.startswith("indoor_") or feature.startswith("outdoor_"):
            return "environment"
        if feature.startswith("fan_"):
            return "behavior"
        if feature in {"room_comfort", "visual_comfort"}:
            return "perception"
        return "other"

    def _exploratory_feature_summary(self, cohort_minute: pd.DataFrame) -> pd.DataFrame:
        if cohort_minute.empty:
            return pd.DataFrame()
        features = [f for f in ANALYTIC_FEATURES if f in cohort_minute.columns]
        features.extend(sorted(c for c in cohort_minute.columns if c.startswith("support_")))
        rows = []
        for feature in features:
            vals = to_numeric(cohort_minute[feature])
            observed = vals.dropna()
            q1 = float(observed.quantile(0.25)) if not observed.empty else np.nan
            q3 = float(observed.quantile(0.75)) if not observed.empty else np.nan
            rows.append(
                {
                    "feature": feature,
                    "domain": self._feature_domain(feature),
                    "n_non_null": int(observed.shape[0]),
                    "coverage_fraction": float(vals.notna().mean()),
                    "n_sessions_with_data": int(cohort_minute.groupby("session_id")[feature].apply(lambda s: s.notna().any()).sum()),
                    "n_participants_with_data": int(cohort_minute.groupby("participant_id")[feature].apply(lambda s: s.notna().any()).sum()),
                    "mean": float(observed.mean()) if not observed.empty else np.nan,
                    "sd": float(observed.std(ddof=1)) if len(observed) > 1 else np.nan,
                    "median": float(observed.median()) if not observed.empty else np.nan,
                    "iqr": float(q3 - q1) if pd.notna(q1) and pd.notna(q3) else np.nan,
                    "min": float(observed.min()) if not observed.empty else np.nan,
                    "max": float(observed.max()) if not observed.empty else np.nan,
                    "skewness": float(observed.skew()) if len(observed) > 2 else np.nan,
                }
            )
        return pd.DataFrame(rows).sort_values(["domain", "coverage_fraction", "feature"], ascending=[True, False, True]).reset_index(drop=True)

    def _session_summary(self, cohort_minute: pd.DataFrame) -> pd.DataFrame:
        if cohort_minute.empty:
            return pd.DataFrame()
        comparison_minute = cohort_minute.loc[cohort_minute["protocol_block"].astype(str).isin(COMPARISON_BLOCKS)].copy()
        questionnaire_design = self._questionnaire_completeness_by_session(comparison_minute)
        rows = []
        for session_id, d in comparison_minute.groupby("session_id"):
            q_stats = questionnaire_design.get(
                str(session_id),
                {
                    "expected_event_count": 0,
                    "observed_event_count": 0,
                    "expected_question_cells": 0,
                    "answered_question_cells": 0,
                    "event_completeness": 0.0,
                    "response_completeness": 0.0,
                    "overall_completeness": 0.0,
                },
            )
            row = {
                "session_id": session_id,
                "participant_id": d["participant_id"].iloc[0],
                "condition_code": d["condition_code"].iloc[0],
                "illuminance_level": d["illuminance_level"].iloc[0],
                "time_of_day": d["time_of_day"].iloc[0],
                "n_minutes": int(len(d)),
                "n_minutes_comparison_window": int(len(d)),
                "questionnaire_expected_events": int(q_stats["expected_event_count"]),
                "questionnaire_observed_events": int(q_stats["observed_event_count"]),
                "questionnaire_expected_cells": int(q_stats["expected_question_cells"]),
                "questionnaire_answered_cells": int(q_stats["answered_question_cells"]),
                "questionnaire_event_completeness": float(q_stats["event_completeness"]),
                "questionnaire_response_completeness": float(q_stats["response_completeness"]),
                "questionnaire_completeness": float(q_stats["overall_completeness"]),
                "fan_minutes": int(d["support_fan"].sum()),
                "empatica_fraction": float(d["support_empatica"].mean()),
                "biopac_fraction": float(d["support_biopac"].mean()),
                "indoor_fraction": float(d["support_indoor"].mean()),
                "outdoor_fraction": float(d["support_outdoor"].mean()),
                "hr_overlap_minutes": int(d["support_core_overlap_hr"].sum()),
                "eda_overlap_minutes": int(d["support_core_overlap_eda"].sum()),
                "temp_overlap_minutes": int(d["support_core_overlap_temp"].sum()),
            }
            for metric in ["thermal_comfort", "empatica_hr_mean_bpm", "biopac_hr_mean_bpm", "indoor_air_velocity_mean_m_s", "biopac_temp_chest_mean_C"]:
                if metric in d.columns:
                    row[f"{metric}__mean"] = float(to_numeric(d[metric]).mean())
            for col in [c for c in d.columns if c.startswith("quality_")]:
                row[f"{col}__fraction"] = float(to_numeric(d[col]).mean())
            rows.append(row)
        return pd.DataFrame(rows).sort_values("session_id").reset_index(drop=True)

    def _condition_support_summary(self, session_summary: pd.DataFrame) -> pd.DataFrame:
        if session_summary.empty:
            return pd.DataFrame()
        rows = []
        for keys, d in session_summary.groupby(["condition_code", "illuminance_level", "time_of_day"]):
            row = {
                "condition_code": keys[0],
                "illuminance_level": keys[1],
                "time_of_day": keys[2],
                "n_sessions": int(d["session_id"].nunique()),
                "n_participants": int(d["participant_id"].nunique()),
            }
            for col in [
                "questionnaire_event_completeness",
                "questionnaire_response_completeness",
                "questionnaire_completeness",
                "fan_minutes",
                "empatica_fraction",
                "biopac_fraction",
                "indoor_fraction",
                "outdoor_fraction",
                "hr_overlap_minutes",
                "eda_overlap_minutes",
                "temp_overlap_minutes",
            ]:
                if col in d.columns:
                    row[f"{col}__mean"] = float(to_numeric(d[col]).mean())
                    row[f"{col}__sd"] = float(to_numeric(d[col]).std(ddof=1)) if len(d) > 1 else np.nan
            rows.append(row)
        return pd.DataFrame(rows).sort_values(["condition_code"]).reset_index(drop=True)

    def _questionnaire_completeness_by_session(self, cohort_minute: pd.DataFrame) -> dict[str, dict[str, float | int]]:
        if cohort_minute.empty or "questionnaire_n" not in cohort_minute.columns:
            return {}
        question_cols = [c for c in QUESTIONNAIRE_SET_COLUMNS if c in cohort_minute.columns]
        if not question_cols:
            return {}
        q_rows = cohort_minute.loc[to_numeric(cohort_minute["questionnaire_n"]).notna()].copy()
        if q_rows.empty:
            return {}
        q_rows["questionnaire_n"] = to_numeric(q_rows["questionnaire_n"]).astype(int)
        expected_events = sorted(q_rows["questionnaire_n"].dropna().unique().tolist())
        expected_question_map: dict[int, list[str]] = {}
        for qn, d in q_rows.groupby("questionnaire_n"):
            expected_question_map[int(qn)] = [col for col in question_cols if float(d[col].notna().mean()) >= 0.95]
        expected_event_count = len(expected_events)
        expected_question_cells = int(sum(len(expected_question_map.get(qn, [])) for qn in expected_events))
        out: dict[str, dict[str, float | int]] = {}
        for session_id, d in cohort_minute.groupby("session_id"):
            sess = d.loc[to_numeric(d["questionnaire_n"]).notna()].copy()
            if sess.empty:
                out[str(session_id)] = {
                    "expected_event_count": expected_event_count,
                    "observed_event_count": 0,
                    "expected_question_cells": expected_question_cells,
                    "answered_question_cells": 0,
                    "event_completeness": 0.0,
                    "response_completeness": 0.0,
                    "overall_completeness": 0.0,
                }
                continue
            sess["questionnaire_n"] = to_numeric(sess["questionnaire_n"]).astype(int)
            sess = sess.loc[sess["questionnaire_n"].isin(expected_events)].copy()
            observed_event_count = int(sess["questionnaire_n"].nunique())
            answered_question_cells = 0
            response_denominator = 0
            event_table = sess.groupby("questionnaire_n", as_index=False)[question_cols].first()
            for row in event_table.itertuples(index=False):
                qn = int(row.questionnaire_n)
                expected_cols = expected_question_map.get(qn, [])
                response_denominator += len(expected_cols)
                answered_question_cells += sum(pd.notna(getattr(row, col)) for col in expected_cols)
            out[str(session_id)] = {
                "expected_event_count": expected_event_count,
                "observed_event_count": observed_event_count,
                "expected_question_cells": expected_question_cells,
                "answered_question_cells": int(answered_question_cells),
                "event_completeness": float(observed_event_count / expected_event_count) if expected_event_count else 0.0,
                "response_completeness": float(answered_question_cells / response_denominator) if response_denominator else 0.0,
                "overall_completeness": float(answered_question_cells / expected_question_cells) if expected_question_cells else 0.0,
            }
        return out

    def _sensor_agreement(self, cohort_minute: pd.DataFrame) -> pd.DataFrame:
        if cohort_minute.empty:
            return pd.DataFrame()
        pairs = [
            ("heart_rate", "empatica_hr_mean_bpm", "biopac_hr_mean_bpm", "support_core_overlap_hr"),
            ("eda", "empatica_eda_mean_uS", "biopac_eda_mean_uS", "support_core_overlap_eda"),
            ("temperature", "empatica_temp_mean_C", "biopac_temp_chest_mean_C", "support_core_overlap_temp"),
        ]
        rows = []
        for session_id, d in cohort_minute.groupby("session_id"):
            for metric, left_col, right_col, support_col in pairs:
                pair = d[[left_col, right_col]].apply(to_numeric).dropna()
                n_overlap = int(d[support_col].sum()) if support_col in d.columns else len(pair)
                eligible = n_overlap >= self.config.runtime.min_sensor_overlap_minutes
                if len(pair) < 3:
                    rows.append(
                        {
                            "session_id": session_id,
                            "participant_id": d["participant_id"].iloc[0],
                            "condition_code": d["condition_code"].iloc[0],
                            "metric": metric,
                            "n_overlap_minutes": n_overlap,
                            "eligible": int(eligible),
                            "status": "insufficient_overlap",
                            "pearson_r": np.nan,
                            "spearman_r": np.nan,
                            "mae": np.nan,
                            "mean_bias": np.nan,
                        }
                    )
                    continue
                diff = pair[left_col] - pair[right_col]
                rows.append(
                    {
                        "session_id": session_id,
                        "participant_id": d["participant_id"].iloc[0],
                        "condition_code": d["condition_code"].iloc[0],
                        "metric": metric,
                        "n_overlap_minutes": n_overlap,
                        "eligible": int(eligible),
                        "status": "eligible" if eligible else "descriptive_only",
                        "pearson_r": float(pair[left_col].corr(pair[right_col], method="pearson")),
                        "spearman_r": float(pair[left_col].corr(pair[right_col], method="spearman")),
                        "mae": float(diff.abs().mean()),
                        "mean_bias": float(diff.mean()),
                    }
                )
        return pd.DataFrame(rows)

    def _agreement_summary(self, agreement: pd.DataFrame) -> pd.DataFrame:
        if agreement.empty:
            return pd.DataFrame()
        rows = []
        for metric, d in agreement.groupby("metric"):
            eligible = d.loc[d["eligible"] == 1]
            source = eligible if not eligible.empty else d
            rows.append(
                {
                    "metric": metric,
                    "n_sessions": int(d["session_id"].nunique()),
                    "n_eligible_sessions": int(eligible["session_id"].nunique()),
                    "median_overlap_minutes": float(source["n_overlap_minutes"].median()) if not source.empty else np.nan,
                    "median_spearman_r": float(source["spearman_r"].median()) if "spearman_r" in source.columns and not source["spearman_r"].dropna().empty else np.nan,
                    "median_mae": float(source["mae"].median()) if "mae" in source.columns and not source["mae"].dropna().empty else np.nan,
                    "summary_status": "eligible" if not eligible.empty else "descriptive_only",
                }
            )
        return pd.DataFrame(rows)

    def _condition_phase_summary(self, cohort_phase: pd.DataFrame) -> pd.DataFrame:
        if cohort_phase.empty:
            return pd.DataFrame()
        features = [f for f in ANALYTIC_FEATURES if f in cohort_phase.columns]
        rows = []
        group_cols = ["condition_code", "illuminance_level", "time_of_day", "protocol_phase"]
        for keys, d in cohort_phase.groupby(group_cols):
            row = {col: keys[idx] for idx, col in enumerate(group_cols)}
            row["n_sessions"] = int(d["session_id"].nunique())
            row["n_participants"] = int(d["participant_id"].nunique())
            row["eligible_for_inference"] = int(
                row["n_sessions"] >= self.config.runtime.min_contrast_pairs and row["n_participants"] >= self.config.runtime.min_cohort_participants_for_inference
            )
            for feature in features:
                vals = to_numeric(d[feature]).dropna()
                row[f"{feature}__mean"] = float(vals.mean()) if not vals.empty else np.nan
                row[f"{feature}__sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def _condition_contrasts(self, cohort_phase: pd.DataFrame, sample_status: pd.DataFrame) -> pd.DataFrame:
        if cohort_phase.empty:
            return pd.DataFrame()
        inferential_ok = bool(sample_status["cohort_inference_eligible"].iloc[0]) if not sample_status.empty else False
        features = [f for f in ANALYTIC_FEATURES if f in cohort_phase.columns]
        comparisons = [
            ("bright_vs_dim_morning", {"fixed": {"time_of_day": "MOR"}, "vary": "illuminance_level", "left": "BRI", "right": "DIM"}),
            ("bright_vs_dim_midday", {"fixed": {"time_of_day": "MID"}, "vary": "illuminance_level", "left": "BRI", "right": "DIM"}),
            ("morning_vs_midday_bright", {"fixed": {"illuminance_level": "BRI"}, "vary": "time_of_day", "left": "MOR", "right": "MID"}),
            ("morning_vs_midday_dim", {"fixed": {"illuminance_level": "DIM"}, "vary": "time_of_day", "left": "MOR", "right": "MID"}),
        ]
        rows = []
        for phase, dp in cohort_phase.groupby("protocol_phase"):
            for feature in features:
                for label, spec in comparisons:
                    left = dp.copy()
                    right = dp.copy()
                    for key, value in spec["fixed"].items():
                        left = left.loc[left[key] == value]
                        right = right.loc[right[key] == value]
                    left = left.loc[left[spec["vary"]] == spec["left"], ["participant_id", feature]].rename(columns={feature: "left_value"})
                    right = right.loc[right[spec["vary"]] == spec["right"], ["participant_id", feature]].rename(columns={feature: "right_value"})
                    pairs = left.merge(right, on="participant_id", how="inner")
                    stats = paired_ttest(pairs["left_value"], pairs["right_value"])
                    n_pairs = int(stats["n_pairs"])
                    if n_pairs == 0:
                        continue
                    diff = to_numeric(pairs["left_value"]) - to_numeric(pairs["right_value"])
                    ci_low, ci_high = bootstrap_mean_ci(diff, n_boot=1500, ci=0.95, seed=42)
                    eligible = inferential_ok and n_pairs >= self.config.runtime.min_contrast_pairs
                    status = "eligible" if eligible else "insufficient_pairs" if n_pairs < self.config.runtime.min_contrast_pairs else "descriptive_only"
                    rows.append(
                        {
                            "protocol_phase": phase,
                            "metric": feature,
                            "comparison": label,
                            "left_label": spec["left"],
                            "right_label": spec["right"],
                            "n_pairs": n_pairs,
                            "eligible": int(eligible),
                            "status": status,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            **stats,
                        }
                    )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["p_value_fdr"] = benjamini_hochberg(out["p_value"])
        out["significant_fdr"] = (
            (to_numeric(out["p_value_fdr"]) < 0.05) & (to_numeric(out["eligible"]).fillna(0) > 0)
        ).astype(int)
        out["inference_label"] = np.where(
            out["eligible"] != 1,
            out["status"],
            np.where(out["significant_fdr"] == 1, "fdr_significant", "eligible_not_significant"),
        )
        return out

    def _participant_profiles(self, cohort_phase: pd.DataFrame) -> pd.DataFrame:
        if cohort_phase.empty:
            return pd.DataFrame()
        features = [f for f in ["thermal_comfort", "empatica_hr_mean_bpm", "biopac_temp_chest_mean_C", "master_dpg_C"] if f in cohort_phase.columns]
        rows = []
        for keys, d in cohort_phase.groupby(["participant_id", "condition_code"]):
            row = {"participant_id": keys[0], "condition_code": keys[1]}
            for feature in features:
                vals = to_numeric(d[feature]).dropna()
                row[feature] = float(vals.mean()) if not vals.empty else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def _phase_direction_consistency(self, d: pd.DataFrame, metric: str, phase_name: str, baseline_value: float) -> tuple[float, int]:
        cur = d.loc[d["protocol_phase"] == phase_name].copy()
        vals = to_numeric(cur[metric]).dropna()
        if vals.empty:
            return 0.0, 0
        signs = np.sign(vals - baseline_value)
        signs = signs.loc[signs != 0]
        if signs.empty:
            return 0.0, 0
        dominant_sign = float(np.sign(signs.sum())) if float(signs.sum()) != 0 else float(signs.iloc[0])
        return float((signs == dominant_sign).mean()), int(len(signs))

    def _phase_pattern_inventory(self, cohort_phase: pd.DataFrame) -> pd.DataFrame:
        if cohort_phase.empty:
            return pd.DataFrame()
        metrics = [m for m in ["thermal_comfort", "master_dpg_C", "thermal_gradient_C", "empatica_hr_mean_bpm", "biopac_hr_mean_bpm", "biopac_temp_chest_mean_C", "empatica_temp_mean_C", "indoor_air_velocity_mean_m_s", "fan_control_au"] if m in cohort_phase.columns]
        rows = []
        for session_id, d in cohort_phase.groupby("session_id"):
            for metric in metrics:
                baseline_phase, baseline_value = self._metric_baseline(d, metric)
                if baseline_phase is None or pd.isna(baseline_value):
                    continue
                phase_means = []
                for phase_name, dp in d.groupby("protocol_phase"):
                    vals = to_numeric(dp[metric]).dropna()
                    if vals.empty or phase_name == baseline_phase:
                        continue
                    mean_value = float(vals.mean())
                    phase_means.append((str(phase_name), mean_value, mean_value - baseline_value))
                if not phase_means:
                    continue
                dominant_phase, dominant_value, dominant_delta = max(phase_means, key=lambda item: abs(item[2]))
                consistency, n_blocks = self._phase_direction_consistency(d, metric, dominant_phase, baseline_value)
                cov_col = f"{metric}__coverage"
                dominant_rows = d.loc[d["protocol_phase"] == dominant_phase]
                coverage_mean = float(to_numeric(dominant_rows[cov_col]).mean()) if cov_col in dominant_rows.columns else 1.0
                rows.append(
                    {
                        "session_id": session_id,
                        "participant_id": d["participant_id"].iloc[0],
                        "condition_code": d["condition_code"].iloc[0],
                        "metric": metric,
                        "baseline_phase": baseline_phase,
                        "baseline_value": baseline_value,
                        "dominant_phase": dominant_phase,
                        "dominant_value": dominant_value,
                        "dominant_delta": dominant_delta,
                        "abs_delta": abs(dominant_delta),
                        "direction": "rise" if dominant_delta > 0 else "drop",
                        "consistency": consistency,
                        "n_blocks": n_blocks,
                        "coverage_mean": coverage_mean,
                        "pattern_strength": float(abs(dominant_delta) * max(consistency, 0.35) * max(coverage_mean, 0.2)),
                    }
                )
        return pd.DataFrame(rows).sort_values(["pattern_strength", "abs_delta"], ascending=[False, False]).reset_index(drop=True)

    def _pattern_summary(self, inventory: pd.DataFrame) -> pd.DataFrame:
        if inventory.empty:
            return pd.DataFrame()
        rows = []
        for keys, d in inventory.groupby(["metric", "dominant_phase", "direction"]):
            denom = inventory.loc[inventory["metric"] == keys[0], "session_id"].nunique()
            rows.append(
                {
                    "metric": keys[0],
                    "dominant_phase": keys[1],
                    "direction": keys[2],
                    "n_sessions": int(d["session_id"].nunique()),
                    "n_participants": int(d["participant_id"].nunique()),
                    "share_within_metric": float(d["session_id"].nunique() / denom) if denom else np.nan,
                    "mean_abs_delta": float(to_numeric(d["abs_delta"]).mean()),
                    "median_abs_delta": float(to_numeric(d["abs_delta"]).median()),
                    "mean_consistency": float(to_numeric(d["consistency"]).mean()),
                    "mean_pattern_strength": float(to_numeric(d["pattern_strength"]).mean()),
                }
            )
        return pd.DataFrame(rows).sort_values(
            ["share_within_metric", "mean_pattern_strength", "mean_abs_delta"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    def _feature_associations(self, cohort_phase: pd.DataFrame) -> pd.DataFrame:
        if cohort_phase.empty:
            return pd.DataFrame()
        from scipy import stats

        metrics = [m for m in ["thermal_comfort", "master_dpg_C", "indoor_air_velocity_mean_m_s", "fan_control_au", "empatica_hr_mean_bpm", "biopac_temp_chest_mean_C", "biopac_bloodflow_mean_bpu"] if m in cohort_phase.columns]
        rows = []
        for idx, left in enumerate(metrics):
            for right in metrics[idx + 1:]:
                pair = cohort_phase[[left, right]].apply(to_numeric).dropna()
                if len(pair) < 4:
                    continue
                pearson_r = float(pair[left].corr(pair[right], method="pearson"))
                spearman_r, spearman_p = stats.spearmanr(pair[left], pair[right], nan_policy="omit")
                rows.append(
                    {
                        "left_metric": left,
                        "right_metric": right,
                        "n_pairs": int(len(pair)),
                        "pearson_r": pearson_r,
                        "spearman_r": float(spearman_r),
                        "spearman_p_value": float(spearman_p) if pd.notna(spearman_p) else np.nan,
                    }
                )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["spearman_p_value_fdr"] = benjamini_hochberg(out["spearman_p_value"])
        out["significant_fdr"] = (to_numeric(out["spearman_p_value_fdr"]) < 0.05).astype(int)
        return out.sort_values(["significant_fdr", "n_pairs", "spearman_r"], ascending=[False, False, False]).reset_index(drop=True)

    def _metric_baseline(self, d: pd.DataFrame, metric: str) -> tuple[str | None, float]:
        cov_col = f"{metric}__coverage"
        temp = d.copy()
        if cov_col in temp.columns:
            temp = temp.loc[to_numeric(temp[cov_col]).fillna(0) > 0].copy()
        else:
            temp = temp.loc[to_numeric(temp[metric]).notna()].copy()
        if temp.empty:
            return None, np.nan
        for phase in ["acclimation", "fan_at_constant_speed", "fan_free_control", "skin_rewarming", "steady_state", "overall_comfort"]:
            vals = to_numeric(temp.loc[temp["protocol_phase"] == phase, metric]).dropna()
            if not vals.empty:
                return phase, float(vals.mean())
        return None, np.nan

    def _session_primary_endpoints(self, cohort_phase: pd.DataFrame) -> pd.DataFrame:
        if cohort_phase.empty:
            return pd.DataFrame()
        phases_of_interest = {"acclimation", "fan_at_constant_speed", "fan_free_control", "skin_rewarming", "steady_state", "overall_comfort"}
        rows = []
        for session_id, d in cohort_phase.loc[cohort_phase["protocol_phase"].isin(phases_of_interest)].groupby("session_id"):
            row = {
                "session_id": session_id,
                "participant_id": d["participant_id"].iloc[0],
                "condition_code": d["condition_code"].iloc[0],
                "n_phase_rows": int(len(d)),
            }
            for metric in [m for m in PRIMARY_ENDPOINTS if m in d.columns]:
                overall = to_numeric(d[metric]).dropna()
                baseline_phase, baseline_value = self._metric_baseline(d, metric)
                row[f"{metric}__session_mean"] = float(overall.mean()) if not overall.empty else np.nan
                row[f"{metric}__baseline_phase"] = baseline_phase
                row[f"{metric}__baseline_mean"] = baseline_value
                row[f"{metric}__acclimation_mean"] = baseline_value if baseline_phase == "acclimation" else np.nan
                row[f"{metric}__delta_vs_baseline"] = float(overall.mean() - baseline_value) if not overall.empty and pd.notna(baseline_value) else np.nan
                row[f"{metric}__delta_vs_acclimation"] = float(overall.mean() - baseline_value) if not overall.empty and baseline_phase == "acclimation" and pd.notna(baseline_value) else np.nan
                row[f"{metric}__coverage"] = float(d.get(f"{metric}__coverage", pd.Series(dtype=float)).mean()) if f"{metric}__coverage" in d.columns else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def _cohort_primary_endpoints(self, cohort_phase: pd.DataFrame, sample_status: pd.DataFrame) -> pd.DataFrame:
        if cohort_phase.empty:
            return pd.DataFrame()
        inferential_ok = bool(sample_status["cohort_inference_eligible"].iloc[0]) if not sample_status.empty else False
        rows = []
        phases_of_interest = [p for p in ["fan_at_constant_speed", "fan_free_control", "skin_rewarming", "steady_state", "overall_comfort"] if p in cohort_phase["protocol_phase"].unique()]
        for metric in [m for m in PRIMARY_ENDPOINTS if m in cohort_phase.columns]:
            for phase in phases_of_interest:
                d = cohort_phase.loc[cohort_phase["protocol_phase"] == phase]
                for condition, dc in d.groupby("condition_code"):
                    vals = to_numeric(dc[metric]).dropna()
                    ci_low, ci_high = bootstrap_mean_ci(vals, n_boot=1500, ci=0.95, seed=42)
                    rows.append(
                        {
                            "metric": metric,
                            "protocol_phase": phase,
                            "condition_code": condition,
                            "n_sessions": int(dc["session_id"].nunique()),
                            "n_participants": int(dc["participant_id"].nunique()),
                            "mean_value": float(vals.mean()) if not vals.empty else np.nan,
                            "sd_value": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "evidence_status": "inferential" if inferential_ok else "descriptive_only",
                        }
                    )
        return pd.DataFrame(rows)

    def _mixed_effects_primary(self, cohort_phase: pd.DataFrame, sample_status: pd.DataFrame) -> pd.DataFrame:
        if cohort_phase.empty:
            return self._empty_mixed_effects()
        inferential_ok = bool(sample_status["cohort_inference_eligible"].iloc[0]) if not sample_status.empty else False
        if not inferential_ok:
            return self._empty_mixed_effects()
        try:
            import statsmodels.formula.api as smf
        except Exception:
            return self._empty_mixed_effects()

        rows = []
        comparison = cohort_phase.loc[cohort_phase["protocol_phase"].astype(str) != "acclimation"].copy()
        if comparison.empty:
            return self._empty_mixed_effects()
        for metric in [m for m in PRIMARY_ENDPOINTS if m in comparison.columns]:
            d = comparison[
                ["participant_id", "condition_code", "illuminance_level", "time_of_day", "protocol_phase", metric]
            ].copy()
            d[metric] = to_numeric(d[metric])
            d = d.dropna()
            if d.empty or d["participant_id"].nunique() < max(4, self.config.runtime.min_cohort_participants_for_inference):
                continue
            if d["illuminance_level"].nunique() < 2 or d["time_of_day"].nunique() < 2 or d["protocol_phase"].nunique() < 2:
                continue
            try:
                model = smf.mixedlm(
                    f"{metric} ~ C(illuminance_level) * C(time_of_day) + C(protocol_phase)",
                    data=d,
                    groups=d["participant_id"],
                )
                fit = model.fit(reml=False, method="lbfgs", disp=False)
            except Exception:
                continue
            params = fit.params
            conf = fit.conf_int()
            pvals = fit.pvalues
            ses = fit.bse
            for term, beta in params.items():
                if term == "Intercept":
                    continue
                rows.append(
                    {
                        "metric": metric,
                        "term": str(term),
                        "beta": float(beta),
                        "se": float(ses.get(term, np.nan)),
                        "p_value": float(pvals.get(term, np.nan)),
                        "ci_low": float(conf.loc[term, 0]) if term in conf.index else np.nan,
                        "ci_high": float(conf.loc[term, 1]) if term in conf.index else np.nan,
                        "n_obs": int(len(d)),
                        "n_participants": int(d["participant_id"].nunique()),
                        "converged": int(bool(getattr(fit, "converged", True))),
                    }
                )
        out = pd.DataFrame(rows)
        if out.empty:
            return self._empty_mixed_effects()
        out["p_value_fdr"] = benjamini_hochberg(out["p_value"])
        out["significant_fdr"] = (to_numeric(out["p_value_fdr"]) < 0.05).astype(int)
        return out.sort_values(["significant_fdr", "metric", "p_value_fdr"], ascending=[False, True, True]).reset_index(drop=True)

    def _predictive_benchmarks(self, session_primary_endpoints: pd.DataFrame) -> pd.DataFrame:
        if session_primary_endpoints.empty:
            return self._empty_predictive_benchmarks()
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
            from sklearn.model_selection import GroupKFold
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except Exception:
            return self._empty_predictive_benchmarks()

        d = session_primary_endpoints.copy()
        if "condition_code" in d.columns:
            d["illuminance_level"] = d["condition_code"].astype(str).str.split("-").str[0]
            d["time_of_day"] = d["condition_code"].astype(str).str.split("-").str[1]
        numeric_candidates = []
        for col in d.columns:
            if col in {"session_id", "participant_id", "condition_code", "illuminance_level", "time_of_day"}:
                continue
            if col.endswith("__baseline_phase"):
                continue
            if d[col].dtype == object:
                continue
            numeric_candidates.append(col)
        feature_cols = [c for c in numeric_candidates if to_numeric(d[c]).notna().sum() >= max(4, len(d) // 3)]
        if not feature_cols or "participant_id" not in d.columns:
            return self._empty_predictive_benchmarks()
        d[feature_cols] = d[feature_cols].apply(to_numeric)
        group_count = int(d["participant_id"].nunique())
        if group_count < 2:
            return self._empty_predictive_benchmarks()
        splitter = GroupKFold(n_splits=min(5, group_count))

        models = {
            "logistic_regression": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
                ]
            ),
            "random_forest": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
                ]
            ),
            "gradient_boosting": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", HistGradientBoostingClassifier(random_state=42, max_depth=4)),
                ]
            ),
        }

        def encode_target(series: pd.Series) -> tuple[pd.Series, dict[int, str]]:
            cats = pd.Categorical(series.astype(str))
            mapping = {idx: str(cat) for idx, cat in enumerate(cats.categories)}
            return pd.Series(cats.codes, index=series.index), mapping

        rows = []
        for task in ["illuminance_level", "time_of_day"]:
            if task not in d.columns or d[task].nunique() < 2:
                continue
            y, mapping = encode_target(d[task])
            X = d[feature_cols].copy()
            groups = d["participant_id"].astype(str)
            for model_name, model in models.items():
                fold_scores = []
                fold_f1 = []
                fold_auc = []
                n_test = 0
                for train_idx, test_idx in splitter.split(X, y, groups):
                    x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    if y_train.nunique() < 2 or y_test.nunique() < 2:
                        continue
                    model.fit(x_train, y_train)
                    pred = pd.Series(model.predict(x_test), index=y_test.index)
                    fold_scores.append(float(balanced_accuracy_score(y_test, pred)))
                    fold_f1.append(float(f1_score(y_test, pred, average="macro")))
                    n_test += int(len(test_idx))
                    try:
                        proba = model.predict_proba(x_test)
                        if proba.shape[1] == 2:
                            fold_auc.append(float(roc_auc_score(y_test, proba[:, 1])))
                    except Exception:
                        pass
                if not fold_scores:
                    continue
                rows.append(
                    {
                        "task": task,
                        "model": model_name,
                        "n_features": int(len(feature_cols)),
                        "n_groups": group_count,
                        "n_samples": int(n_test),
                        "balanced_accuracy_mean": float(np.mean(fold_scores)),
                        "balanced_accuracy_sd": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else np.nan,
                        "macro_f1_mean": float(np.mean(fold_f1)),
                        "macro_f1_sd": float(np.std(fold_f1, ddof=1)) if len(fold_f1) > 1 else np.nan,
                        "roc_auc_mean": float(np.mean(fold_auc)) if fold_auc else np.nan,
                        "target_levels": " | ".join(mapping.values()),
                    }
                )
        out = pd.DataFrame(rows)
        if out.empty:
            return self._empty_predictive_benchmarks()
        return out.sort_values(["task", "balanced_accuracy_mean", "macro_f1_mean"], ascending=[True, False, False]).reset_index(drop=True)
