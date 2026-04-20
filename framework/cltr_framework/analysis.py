from __future__ import annotations

import warnings

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

SESSION_MODEL_FEATURES = [
    "thermal_comfort",
    "thermal_sensation",
    "indoor_air_temp_mean_C",
    "indoor_air_velocity_mean_m_s",
    "fan_current_A",
    "fan_control_au",
    "master_dpg_C",
    "empatica_hr_mean_bpm",
    "empatica_eda_mean_uS",
    "empatica_temp_mean_C",
    "biopac_hr_mean_bpm",
    "biopac_eda_mean_uS",
    "biopac_temp_chest_mean_C",
    "biopac_bloodflow_mean_bpu",
    "outdoor_air_temp_C",
    "outdoor_solar_radiation_W_m2",
]

LAG_WINDOWS_MINUTES = [0, 5, 15, 30, 60]
TEMPORAL_RESPONSE_SPECS = [
    ("indoor_air_temp_mean_C", "thermal_comfort"),
    ("indoor_air_temp_mean_C", "thermal_sensation"),
    ("indoor_air_velocity_mean_m_s", "thermal_comfort"),
    ("indoor_air_velocity_mean_m_s", "biopac_temp_chest_mean_C"),
    ("fan_control_au", "thermal_comfort"),
    ("fan_control_au", "empatica_temp_mean_C"),
    ("outdoor_solar_radiation_W_m2", "biopac_hr_mean_bpm"),
    ("outdoor_air_temp_C", "empatica_temp_mean_C"),
]

ENDPOINT_POLICY_GROUPS = {
    "primary_outcome": [
        "thermal_comfort",
        "thermal_sensation",
    ],
    "primary_physiological": [
        "biopac_hr_mean_bpm",
        "biopac_temp_chest_mean_C",
    ],
    "primary_physiological_qc": [
        "empatica_temp_mean_C",
        "biopac_eda_mean_uS",
        "empatica_eda_mean_uS",
    ],
    "protocol_check": [
        "indoor_air_velocity_mean_m_s",
    ],
    "secondary_mechanistic": [
        "master_dpg_C",
        "biopac_bloodflow_mean_bpu",
    ],
    "audit_only": [
        "empatica_hr_mean_bpm",
    ],
}

ENDPOINT_POLICY = {
    metric: role
    for role, metrics in ENDPOINT_POLICY_GROUPS.items()
    for metric in metrics
}

PRIMARY_ENDPOINT_POLICY_ROLES = {
    "primary_outcome",
    "primary_physiological",
    "primary_physiological_qc",
    "protocol_check",
}

PRIMARY_ENDPOINTS = [
    metric
    for metric in dict.fromkeys(
        metric
        for role in ["primary_outcome", "primary_physiological", "primary_physiological_qc", "protocol_check"]
        for metric in ENDPOINT_POLICY_GROUPS[role]
    )
]

SUPPORT_GRADED_ENDPOINTS = list(
    dict.fromkeys(
        list(ANALYTIC_FEATURES)
        + list(ENDPOINT_POLICY.keys())
    )
)


def endpoint_policy_role(metric: str) -> str:
    return ENDPOINT_POLICY.get(str(metric), "not_endpoint_graded")


def endpoint_is_primary(metric: str) -> bool:
    return endpoint_policy_role(metric) in PRIMARY_ENDPOINT_POLICY_ROLES


LEGACY_PRIMARY_ENDPOINTS = [
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

SENSOR_AUDIT_SPECS = [
    {
        "signal_stream": "empatica_bvp",
        "device": "Empatica",
        "construct": "bvp_source",
        "column": "empatica_bvp_mean",
        "quality_column": "quality_empatica_bvp",
        "bounds": None,
        "agreement_metric": None,
    },
    {
        "signal_stream": "empatica_hr",
        "device": "Empatica",
        "construct": "heart_rate",
        "column": "empatica_hr_mean_bpm",
        "quality_column": "quality_empatica_hr",
        "bounds": (40.0, 180.0),
        "agreement_metric": "heart_rate",
    },
    {
        "signal_stream": "empatica_eda",
        "device": "Empatica",
        "construct": "eda",
        "column": "empatica_eda_mean_uS",
        "quality_column": "quality_empatica_eda",
        "bounds": (0.0, 40.0),
        "agreement_metric": "eda",
    },
    {
        "signal_stream": "biopac_temp",
        "device": "BIOPAC",
        "construct": "temperature",
        "column": "biopac_temp_chest_mean_C",
        "quality_column": "quality_biopac_temp",
        "bounds": (20.0, 42.0),
        "agreement_metric": "temperature",
    },
    {
        "signal_stream": "empatica_temp",
        "device": "Empatica",
        "construct": "temperature",
        "column": "empatica_temp_mean_C",
        "quality_column": "quality_empatica_temp",
        "bounds": (20.0, 40.0),
        "agreement_metric": "temperature",
    },
    {
        "signal_stream": "empatica_acc",
        "device": "Empatica",
        "construct": "motion",
        "column": "empatica_acc_mean_g",
        "quality_column": None,
        "bounds": (0.0, 8.0),
        "agreement_metric": None,
    },
    {
        "signal_stream": "empatica_enmo",
        "device": "Empatica",
        "construct": "motion",
        "column": "empatica_enmo_mean_g",
        "quality_column": None,
        "bounds": (0.0, 8.0),
        "agreement_metric": None,
    },
    {
        "signal_stream": "empatica_steps",
        "device": "Empatica",
        "construct": "activity",
        "column": "empatica_steps",
        "quality_column": None,
        "bounds": (0.0, 200.0),
        "agreement_metric": None,
    },
    {
        "signal_stream": "biopac_hr",
        "device": "BIOPAC",
        "construct": "heart_rate",
        "column": "biopac_hr_mean_bpm",
        "quality_column": "quality_biopac_hr",
        "bounds": (40.0, 180.0),
        "agreement_metric": "heart_rate",
    },
    {
        "signal_stream": "biopac_eda",
        "device": "BIOPAC",
        "construct": "eda",
        "column": "biopac_eda_mean_uS",
        "quality_column": "quality_biopac_eda",
        "bounds": (0.0, 60.0),
        "agreement_metric": "eda",
    },
    {
        "signal_stream": "biopac_temp_thigh",
        "device": "BIOPAC",
        "construct": "temperature_site",
        "column": "biopac_temp_thigh_mean_C",
        "quality_column": None,
        "bounds": (20.0, 42.0),
        "agreement_metric": None,
    },
    {
        "signal_stream": "biopac_temp_arm",
        "device": "BIOPAC",
        "construct": "temperature_site",
        "column": "biopac_temp_arm_mean_C",
        "quality_column": None,
        "bounds": (20.0, 42.0),
        "agreement_metric": None,
    },
    {
        "signal_stream": "biopac_temp_tibia",
        "device": "BIOPAC",
        "construct": "temperature_site",
        "column": "biopac_temp_tibia_mean_C",
        "quality_column": None,
        "bounds": (20.0, 42.0),
        "agreement_metric": None,
    },
    {
        "signal_stream": "biopac_bloodflow",
        "device": "BIOPAC",
        "construct": "bloodflow",
        "column": "biopac_bloodflow_mean_bpu",
        "quality_column": None,
        "bounds": (0.0, 500.0),
        "agreement_metric": None,
    },
    {
        "signal_stream": "biopac_backscatter",
        "device": "BIOPAC",
        "construct": "optical",
        "column": "biopac_backscatter_mean_percent",
        "quality_column": None,
        "bounds": (0.0, 100.0),
        "agreement_metric": None,
    },
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
                "model_spec",
                "p_value_fdr",
                "significant_fdr",
            ]
        )

    @staticmethod
    def _empty_mixed_effects_diagnostics() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "metric",
                "status",
                "model_spec",
                "n_obs",
                "n_participants",
                "n_terms_retained",
                "fit_converged",
                "warning_count",
                "warning_summary",
            ]
        )

    @staticmethod
    def _empty_predictive_benchmarks() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "task",
                "model",
                "feature_set",
                "validation_scheme",
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

    @staticmethod
    def _empty_lag_response_register() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "predictor",
                "target",
                "best_lag_minutes",
                "median_spearman_r",
                "median_abs_spearman_r",
                "same_sign_fraction",
                "n_sessions",
                "median_pairs_per_session",
                "evidence_grade",
                "scientific_reading",
            ]
        )

    @staticmethod
    def _empty_lag_response_profile() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "predictor",
                "target",
                "lag_minutes",
                "median_spearman_r",
                "median_abs_spearman_r",
                "same_sign_fraction",
                "n_sessions",
                "median_pairs_per_session",
                "is_best_lag",
                "evidence_grade",
            ]
        )

    @staticmethod
    def _empty_scientific_decision_register() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "claim_family",
                "predictor",
                "target",
                "recommended_operating_band",
                "response_lag_minutes",
                "evidence_grade",
                "supporting_streams",
                "statistical_basis",
                "practical_reading",
                "control_recommendation",
            ]
        )

    @staticmethod
    def _empty_threshold_response_register() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "predictor",
                "target",
                "threshold_value",
                "slope_below",
                "slope_above",
                "slope_change",
                "rss_improvement_fraction",
                "n_pairs",
                "n_sessions",
                "evidence_grade",
                "scientific_reading",
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
        mixed_effects_primary, mixed_effects_diagnostics = self._mixed_effects_primary(cohort_phase, sample_status)
        predictive_benchmarks = self._predictive_benchmarks(session_primary_endpoints)
        session_signal_audit = self._session_signal_audit(comparison_minute, sensor_agreement)
        signal_audit_summary = self._signal_audit_summary(session_signal_audit)
        lag_response_register, lag_response_profile = self._lag_response_outputs(comparison_minute)
        threshold_response_register = self._threshold_response_register(comparison_minute, lag_response_register)
        scientific_decision_register = self._scientific_decision_register(
            comparison_minute,
            lag_response_register,
            threshold_response_register,
            condition_contrasts,
            mixed_effects_primary,
            predictive_benchmarks,
            signal_audit_summary,
        )
        return {
            "cohort_minute_features": cohort_minute,
            "cohort_minute_comparison_window": comparison_minute,
            "cohort_phase_summary": cohort_phase,
            "session_summary": session_summary,
            "sample_status": sample_status,
            "coverage_summary": self._coverage_summary(comparison_minute),
            "master_table_registry": self._master_table_registry(session_summary),
            "preprocessing_qc_summary": preprocessing_qc_summary,
            "exploratory_feature_summary": exploratory_feature_summary,
            "feature_registry": self._feature_registry(comparison_minute),
            "condition_support_summary": condition_support_summary,
            "sensor_agreement": sensor_agreement,
            "agreement_summary": self._agreement_summary(sensor_agreement),
            "session_signal_audit": session_signal_audit,
            "signal_audit_summary": signal_audit_summary,
            "condition_phase_summary": self._condition_phase_summary(cohort_phase),
            "condition_contrasts": condition_contrasts,
            "feature_associations": self._feature_associations(cohort_phase),
            "phase_pattern_inventory": phase_pattern_inventory,
            "pattern_summary": self._pattern_summary(phase_pattern_inventory),
            "participant_profiles": self._participant_profiles(cohort_phase),
            "session_primary_endpoints": session_primary_endpoints,
            "cohort_primary_endpoints": self._cohort_primary_endpoints(cohort_phase, sample_status),
            "mixed_effects_primary": mixed_effects_primary,
            "mixed_effects_diagnostics": mixed_effects_diagnostics,
            "predictive_benchmarks": predictive_benchmarks,
            "lag_response_register": lag_response_register,
            "lag_response_profile": lag_response_profile,
            "threshold_response_register": threshold_response_register,
            "scientific_decision_register": scientific_decision_register,
        }

    def _master_table_registry(self, session_summary: pd.DataFrame) -> pd.DataFrame:
        if session_summary.empty:
            return pd.DataFrame()
        specs = [
            ("empatica", "source_coverage", "empatica_fraction", "Continuous wearable support retained on the aligned minute grid."),
            ("biopac", "source_coverage", "biopac_fraction", "Clinical-grade physiology retained on the aligned minute grid."),
            ("indoor", "source_coverage", "indoor_fraction", "Indoor environmental sensing retained on the aligned minute grid."),
            ("outdoor", "source_coverage", "outdoor_fraction", "Outdoor contextual forcing retained on the aligned minute grid."),
            ("hr_overlap", "paired_overlap", "hr_overlap_minutes", "Paired heart-rate overlap available for cross-device comparison or sensitivity analysis."),
            ("eda_overlap", "paired_overlap", "eda_overlap_minutes", "Paired electrodermal overlap available for cross-device comparison or sensitivity analysis."),
            ("temp_overlap", "paired_overlap", "temp_overlap_minutes", "Paired temperature overlap available for cross-device comparison or sensitivity analysis."),
        ]
        rows = []
        for layer, gate_type, column, scientific_use in specs:
            if column not in session_summary.columns:
                continue
            vals = to_numeric(session_summary[column]).dropna()
            if vals.empty:
                continue
            if gate_type == "source_coverage":
                unit = "fraction (0-1)"
                threshold = 0.80
                status = "pass" if float(vals.mean()) >= threshold else ("conditional" if float(vals.mean()) >= 0.60 else "limited")
                observed_value = float(vals.mean())
            else:
                unit = "minutes"
                threshold = float(self.config.runtime.min_sensor_overlap_minutes)
                status = "pass" if float(vals.median()) >= threshold else ("conditional" if float(vals.mean()) >= threshold else "limited")
                observed_value = float(vals.median())
            rows.append(
                {
                    "layer": layer,
                    "gate_type": gate_type,
                    "unit": unit,
                    "status": status,
                    "threshold": threshold,
                    "observed_value": observed_value,
                    "mean_value": float(vals.mean()),
                    "median_value": float(vals.median()),
                    "min_value": float(vals.min()),
                    "max_value": float(vals.max()),
                    "n_sessions_supported": int((to_numeric(session_summary[column]).fillna(0) > 0).sum()),
                    "scientific_use": scientific_use,
                }
            )
        return pd.DataFrame(rows)

    def _feature_domain_registry_label(self, feature: str) -> str:
        if feature.startswith("support_"):
            return "support gate"
        if feature.startswith("quality_"):
            return "quality gate"
        return self._feature_domain(feature)

    def _feature_unit(self, feature: str) -> str:
        if feature.startswith("support_") or feature.startswith("quality_"):
            return "fraction"
        if feature in QUESTIONNAIRE_SET_COLUMNS or feature.startswith("thermal_") or feature.endswith("_comfort") or feature.endswith("_sensation") or feature.endswith("_preference") or feature.endswith("_pleasure"):
            return "ordinal scale"
        if feature.endswith("_C"):
            return "C"
        if feature.endswith("_bpm"):
            return "bpm"
        if feature.endswith("_uS"):
            return "uS"
        if feature.endswith("_m_s"):
            return "m/s"
        if feature.endswith("_percent"):
            return "%"
        if feature.endswith("_A"):
            return "A"
        if feature.endswith("_au"):
            return "arbitrary units"
        if feature.endswith("_bpu"):
            return "BPU"
        if feature.endswith("_steps") or feature == "empatica_steps":
            return "count"
        return "derived unit"

    def _feature_registry_role(self, feature: str) -> str:
        if feature.startswith("support_"):
            return "support gate"
        if feature.startswith("quality_"):
            return "quality gate"
        role = endpoint_policy_role(feature)
        if role != "not_endpoint_graded":
            return role
        if feature in ANALYTIC_FEATURES:
            return "analytic covariate"
        return "derived feature"

    def _feature_observation_policy(self, feature: str) -> str:
        if feature in QUESTIONNAIRE_SET_COLUMNS or feature.endswith("_comfort") or feature.endswith("_sensation") or feature.endswith("_preference") or feature.endswith("_pleasure"):
            return "discrete ordinal event observations"
        if feature in {"fan_control_au", "fan_control_secondary_au", "fan_current_A"}:
            return "discrete control-state or step signal"
        if feature.startswith("support_") or feature.startswith("quality_"):
            return "derived support or quality gate"
        return "continuous or near-continuous minute summary"

    def _feature_coverage_reading(self, feature: str, coverage_fraction: float) -> str:
        if feature in QUESTIONNAIRE_SET_COLUMNS or feature.endswith("_comfort") or feature.endswith("_sensation") or feature.endswith("_preference") or feature.endswith("_pleasure"):
            return (
                f"{coverage_fraction:.3f} reflects questionnaire-event occupancy on the aligned minute grid, "
                "not missingness of a continuous signal."
            )
        if feature in {"fan_control_au", "fan_control_secondary_au", "fan_current_A"}:
            return (
                f"{coverage_fraction:.3f} reflects the share of aligned minutes with an observed control state/value, "
                "so sparse support is expected and should not be read as continuous-sensor incompleteness."
            )
        if feature.startswith("support_") or feature.startswith("quality_"):
            return (
                f"{coverage_fraction:.3f} reflects gate availability or validity on the aligned minute grid."
            )
        return (
            f"{coverage_fraction:.3f} reflects the non-null share of aligned minute summaries for this feature."
        )

    def _is_event_prompt_feature(self, feature: str) -> bool:
        return bool(
            feature in QUESTIONNAIRE_SET_COLUMNS
            or feature.endswith("_comfort")
            or feature.endswith("_sensation")
            or feature.endswith("_preference")
            or feature.endswith("_pleasure")
        )

    def _feature_prompt_support(self, cohort_minute: pd.DataFrame, feature: str) -> dict[str, float | int | str]:
        empty = {
            "observed_prompt_count": np.nan,
            "expected_prompt_count": np.nan,
            "prompt_response_fraction": np.nan,
            "prompt_support_reading": "",
        }
        if cohort_minute.empty or feature not in cohort_minute.columns or "questionnaire_n" not in cohort_minute.columns:
            return empty
        if not self._is_event_prompt_feature(feature):
            return empty

        q_rows = cohort_minute.loc[to_numeric(cohort_minute["questionnaire_n"]).notna(), ["session_id", "questionnaire_n", feature]].copy()
        if q_rows.empty:
            return empty
        q_rows["questionnaire_n"] = to_numeric(q_rows["questionnaire_n"]).astype(int)
        expected_qn = sorted(
            q_rows.groupby("questionnaire_n")[feature]
            .apply(lambda s: s.notna().any())
            .loc[lambda s: s]
            .index
            .tolist()
        )
        if not expected_qn:
            return empty
        eligible = q_rows.loc[q_rows["questionnaire_n"].isin(expected_qn)].copy()
        if eligible.empty:
            return empty
        prompt_level = eligible.groupby(["session_id", "questionnaire_n"], as_index=False).agg(
            observed=(feature, lambda s: s.notna().any())
        )
        observed_prompt_count = int(prompt_level["observed"].sum())
        expected_prompt_count = int(len(prompt_level))
        prompt_fraction = float(observed_prompt_count / expected_prompt_count) if expected_prompt_count else np.nan
        return {
            "observed_prompt_count": observed_prompt_count,
            "expected_prompt_count": expected_prompt_count,
            "prompt_response_fraction": prompt_fraction,
            "prompt_support_reading": (
                f"reported {observed_prompt_count}/{expected_prompt_count} expected prompts "
                f"({prompt_fraction:.1%})"
                if expected_prompt_count
                else ""
            ),
        }

    def _feature_registry(self, cohort_minute: pd.DataFrame) -> pd.DataFrame:
        if cohort_minute.empty:
            return pd.DataFrame()
        feature_columns = [f for f in ANALYTIC_FEATURES if f in cohort_minute.columns]
        feature_columns.extend(sorted(c for c in cohort_minute.columns if c.startswith("support_")))
        feature_columns.extend(sorted(c for c in cohort_minute.columns if c.startswith("quality_")))
        rows = []
        for feature in dict.fromkeys(feature_columns):
            vals = to_numeric(cohort_minute[feature]) if feature in cohort_minute.columns else pd.Series(dtype=float)
            observed = vals.dropna()
            registry_role = self._feature_registry_role(feature)
            prompt_support = self._feature_prompt_support(cohort_minute, feature)
            if registry_role == "support gate":
                scientific_use = "Used to qualify whether aligned data windows can enter later result layers."
            elif registry_role == "quality gate":
                scientific_use = "Used to audit preprocessing validity before physiology and environment are interpreted."
            elif endpoint_is_primary(feature):
                scientific_use = "Claim-supporting primary endpoint when support and modality gates are satisfied."
            elif registry_role in ENDPOINT_POLICY_GROUPS:
                scientific_use = "Secondary or QC-qualified endpoint retained for mechanism or policy-aware interpretation."
            else:
                scientific_use = "Analytic covariate or derived feature used for exploratory, contrast, or modeling layers."
            rows.append(
                {
                    "feature": feature,
                    "domain": self._feature_domain_registry_label(feature),
                    "registry_role": registry_role,
                    "unit": self._feature_unit(feature),
                    "observation_policy": self._feature_observation_policy(feature),
                    "summary_grain": "aligned minute summary",
                    "coverage_fraction": float(vals.notna().mean()) if len(vals) else np.nan,
                    "observed_prompt_count": prompt_support["observed_prompt_count"],
                    "expected_prompt_count": prompt_support["expected_prompt_count"],
                    "prompt_response_fraction": prompt_support["prompt_response_fraction"],
                    "prompt_support_reading": prompt_support["prompt_support_reading"],
                    "n_non_null": int(observed.shape[0]),
                    "n_sessions_with_data": int(cohort_minute.groupby("session_id")[feature].apply(lambda s: s.notna().any()).sum()),
                    "n_participants_with_data": int(cohort_minute.groupby("participant_id")[feature].apply(lambda s: s.notna().any()).sum()),
                    "coverage_reading": self._feature_coverage_reading(feature, float(vals.notna().mean()) if len(vals) else np.nan),
                    "scientific_use": scientific_use,
                }
            )
        return pd.DataFrame(rows).sort_values(["domain", "registry_role", "coverage_fraction", "feature"], ascending=[True, True, False, True]).reset_index(drop=True)

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

    @staticmethod
    def _clip01(value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return float(max(0.0, min(1.0, value)))

    def _session_signal_audit(self, cohort_minute: pd.DataFrame, sensor_agreement: pd.DataFrame) -> pd.DataFrame:
        if cohort_minute.empty:
            return pd.DataFrame()
        agreement_lookup: dict[tuple[str, str], pd.Series] = {}
        if not sensor_agreement.empty:
            for _, row in sensor_agreement.iterrows():
                agreement_lookup[(str(row["session_id"]), str(row["metric"]))] = row
        rows = []
        for session_id, d in cohort_minute.groupby("session_id"):
            total_minutes = int(len(d))
            condition_code = str(d["condition_code"].iloc[0]) if "condition_code" in d.columns else ""
            participant_id = str(d["participant_id"].iloc[0]) if "participant_id" in d.columns else ""
            for spec in SENSOR_AUDIT_SPECS:
                col = spec["column"]
                if col not in d.columns:
                    continue
                vals = to_numeric(d[col])
                observed = vals.dropna()
                quality_col = spec.get("quality_column")
                quality_fraction = float(to_numeric(d[quality_col]).mean()) if quality_col and quality_col in d.columns else np.nan
                if spec.get("bounds"):
                    low, high = spec["bounds"]
                    plausible_mask = vals.between(low, high)
                    plausible_fraction = float(plausible_mask.loc[vals.notna()].mean()) if observed.shape[0] else np.nan
                    low_outliers = int((vals < low).sum()) if observed.shape[0] else 0
                    high_outliers = int((vals > high).sum()) if observed.shape[0] else 0
                else:
                    plausible_fraction = quality_fraction if np.isfinite(quality_fraction) else float(observed.notna().mean()) if observed.shape[0] else np.nan
                    low_outliers = 0
                    high_outliers = 0
                dif = observed.diff().abs().dropna()
                agreement_metric = spec.get("agreement_metric")
                agreement_row = agreement_lookup.get((str(session_id), str(agreement_metric))) if agreement_metric else None
                rows.append(
                    {
                        "session_id": str(session_id),
                        "participant_id": participant_id,
                        "condition_code": condition_code,
                        "signal_stream": spec["signal_stream"],
                        "device": spec["device"],
                        "construct": spec["construct"],
                        "n_minutes_total": total_minutes,
                        "n_valid_minutes": int(observed.shape[0]),
                        "coverage_fraction": float(observed.shape[0] / total_minutes) if total_minutes else np.nan,
                        "quality_fraction": quality_fraction,
                        "plausible_fraction": plausible_fraction,
                        "low_outlier_minutes": low_outliers,
                        "high_outlier_minutes": high_outliers,
                        "median_value": float(observed.median()) if not observed.empty else np.nan,
                        "q05_value": float(observed.quantile(0.05)) if not observed.empty else np.nan,
                        "q95_value": float(observed.quantile(0.95)) if not observed.empty else np.nan,
                        "min_value": float(observed.min()) if not observed.empty else np.nan,
                        "max_value": float(observed.max()) if not observed.empty else np.nan,
                        "median_abs_step": float(dif.median()) if not dif.empty else np.nan,
                        "p95_abs_step": float(dif.quantile(0.95)) if not dif.empty else np.nan,
                        "paired_overlap_minutes": int(agreement_row["n_overlap_minutes"]) if agreement_row is not None and pd.notna(agreement_row.get("n_overlap_minutes")) else 0,
                        "paired_eligible": int(agreement_row["eligible"]) if agreement_row is not None and pd.notna(agreement_row.get("eligible")) else 0,
                        "paired_spearman_r": float(agreement_row["spearman_r"]) if agreement_row is not None and pd.notna(agreement_row.get("spearman_r")) else np.nan,
                        "paired_mae": float(agreement_row["mae"]) if agreement_row is not None and pd.notna(agreement_row.get("mae")) else np.nan,
                    }
                )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        concern = []
        for _, row in out.iterrows():
            concern_score = (1.0 - self._clip01(float(row.get("coverage_fraction", np.nan)))) * 45.0
            concern_score += (1.0 - self._clip01(float(row.get("plausible_fraction", np.nan)))) * 30.0
            if np.isfinite(row.get("quality_fraction", np.nan)):
                concern_score += (1.0 - self._clip01(float(row["quality_fraction"]))) * 20.0
            if row.get("paired_eligible", 0) and np.isfinite(row.get("paired_spearman_r", np.nan)):
                concern_score += max(0.0, min(15.0, (0.7 - float(row["paired_spearman_r"])) * 25.0))
            concern.append(float(concern_score))
        out["concern_score"] = concern
        return out.sort_values(["construct", "device", "session_id"]).reset_index(drop=True)

    def _signal_audit_summary(self, session_signal_audit: pd.DataFrame) -> pd.DataFrame:
        if session_signal_audit.empty:
            return pd.DataFrame()
        rows = []
        for (signal_stream, device, construct), d in session_signal_audit.groupby(["signal_stream", "device", "construct"]):
            mean_quality = float(to_numeric(d["quality_fraction"]).mean()) if "quality_fraction" in d.columns and to_numeric(d["quality_fraction"]).notna().any() else np.nan
            mean_plausible = float(to_numeric(d["plausible_fraction"]).mean()) if "plausible_fraction" in d.columns and to_numeric(d["plausible_fraction"]).notna().any() else np.nan
            median_overlap = float(to_numeric(d["paired_overlap_minutes"]).median()) if "paired_overlap_minutes" in d.columns and to_numeric(d["paired_overlap_minutes"]).notna().any() else np.nan
            median_spearman = float(to_numeric(d["paired_spearman_r"]).median()) if "paired_spearman_r" in d.columns and to_numeric(d["paired_spearman_r"]).dropna().any() else np.nan
            median_mae = float(to_numeric(d["paired_mae"]).median()) if "paired_mae" in d.columns and to_numeric(d["paired_mae"]).dropna().any() else np.nan
            mean_coverage = float(to_numeric(d["coverage_fraction"]).mean()) if to_numeric(d["coverage_fraction"]).notna().any() else np.nan
            median_valid = float(to_numeric(d["n_valid_minutes"]).median()) if to_numeric(d["n_valid_minutes"]).notna().any() else np.nan
            support_component = self._clip01(mean_coverage) * 35.0
            plausibility_component = self._clip01(mean_plausible if np.isfinite(mean_plausible) else mean_quality) * 25.0
            quality_component = self._clip01(mean_quality if np.isfinite(mean_quality) else mean_plausible) * 20.0
            continuity_component = self._clip01((median_valid if np.isfinite(median_valid) else 0.0) / 30.0) * 10.0
            agreement_component = 0.0
            if np.isfinite(median_overlap):
                agreement_component += self._clip01(median_overlap / float(self.config.runtime.min_sensor_overlap_minutes)) * 5.0
            if np.isfinite(median_spearman):
                agreement_component += self._clip01((median_spearman - 0.3) / 0.55) * 5.0
            adequacy_score = float(support_component + plausibility_component + quality_component + continuity_component + agreement_component)
            if construct == "heart_rate" and device == "Empatica":
                if adequacy_score >= 70 and np.isfinite(median_overlap) and median_overlap >= 20 and np.isfinite(median_spearman) and median_spearman >= 0.8:
                    adequacy_status = "usable_with_caution"
                    recommended_role = "secondary_validation"
                elif adequacy_score >= 45:
                    adequacy_status = "limited"
                    recommended_role = "subset_only"
                else:
                    adequacy_status = "weak"
                    recommended_role = "not_primary"
            elif adequacy_score >= 75:
                adequacy_status = "strong"
                recommended_role = "primary"
            elif adequacy_score >= 60:
                adequacy_status = "usable_with_caution"
                recommended_role = "primary_with_qc"
            elif adequacy_score >= 45:
                adequacy_status = "limited"
                recommended_role = "secondary_only"
            else:
                adequacy_status = "weak"
                recommended_role = "not_recommended"
            if construct in {"eda", "temperature"} and np.isfinite(median_spearman):
                if median_spearman < 0.25:
                    adequacy_status = "usable_with_caution" if adequacy_score >= 60 else adequacy_status
                    recommended_role = "primary_with_qc" if adequacy_score >= 60 else recommended_role
                elif median_spearman < 0.55 and recommended_role == "primary":
                    adequacy_status = "usable_with_caution"
                    recommended_role = "primary_with_qc"
            concern_sessions = (
                d.sort_values(["concern_score", "session_id"], ascending=[False, True])["session_id"].astype(str).head(3).tolist()
            )
            rows.append(
                {
                    "signal_stream": signal_stream,
                    "device": device,
                    "construct": construct,
                    "n_sessions": int(d["session_id"].nunique()),
                    "n_sessions_with_any_data": int((to_numeric(d["n_valid_minutes"]) > 0).sum()),
                    "mean_valid_minutes": float(to_numeric(d["n_valid_minutes"]).mean()),
                    "median_valid_minutes": median_valid,
                    "mean_coverage_fraction": mean_coverage,
                    "mean_quality_fraction": mean_quality,
                    "mean_plausible_fraction": mean_plausible,
                    "median_overlap_minutes": median_overlap,
                    "median_spearman_r": median_spearman,
                    "median_mae": median_mae,
                    "max_concern_score": float(to_numeric(d["concern_score"]).max()) if to_numeric(d["concern_score"]).notna().any() else np.nan,
                    "flagged_sessions": ", ".join(concern_sessions),
                    "adequacy_score": adequacy_score,
                    "adequacy_status": adequacy_status,
                    "recommended_role": recommended_role,
                    "scientific_reading": self._signal_scientific_reading(
                        signal_stream=signal_stream,
                        device=device,
                        construct=construct,
                        mean_coverage=mean_coverage,
                        mean_plausible=mean_plausible,
                        mean_quality=mean_quality,
                        median_overlap=median_overlap,
                        median_spearman=median_spearman,
                        recommended_role=recommended_role,
                    ),
                }
            )
        order = {
            name: idx
            for idx, name in enumerate(
                [
                    "empatica_bvp",
                    "empatica_hr",
                    "empatica_eda",
                    "empatica_temp",
                    "empatica_acc",
                    "empatica_enmo",
                    "empatica_steps",
                    "biopac_hr",
                    "biopac_eda",
                    "biopac_temp",
                    "biopac_temp_thigh",
                    "biopac_temp_arm",
                    "biopac_temp_tibia",
                    "biopac_bloodflow",
                    "biopac_backscatter",
                ]
            )
        }
        out = pd.DataFrame(rows)
        out["_order"] = out["signal_stream"].map(order).fillna(999)
        out = out.sort_values(["_order", "device"]).drop(columns="_order").reset_index(drop=True)
        return out

    def _signal_scientific_reading(
        self,
        *,
        signal_stream: str,
        device: str,
        construct: str,
        mean_coverage: float,
        mean_plausible: float,
        mean_quality: float,
        median_overlap: float,
        median_spearman: float,
        recommended_role: str,
    ) -> str:
        quality_piece = (
            f"quality support is {mean_quality:.2f}"
            if np.isfinite(mean_quality)
            else f"plausibility support is {mean_plausible:.2f}" if np.isfinite(mean_plausible) else "quality is not estimable"
        )
        if construct == "heart_rate" and device == "Empatica":
            if recommended_role == "secondary_validation":
                return (
                    f"{device} heart rate is physiologically plausible when present and can serve as a secondary validation stream; "
                    f"however, its cohort support is partial (coverage {mean_coverage:.2f}, median overlap {median_overlap:.1f} min, median agreement {median_spearman:.2f})."
                )
            if recommended_role == "subset_only":
                return (
                    f"{device} heart rate is value-plausible but too incomplete for full-session inference; "
                    f"{quality_piece}, median overlap is {median_overlap:.1f} min, and it should be restricted to subset or sensitivity analyses."
                )
            return (
                f"{device} heart rate is not adequate as a primary endpoint in the current release: "
                f"coverage is {mean_coverage:.2f}, {quality_piece}, and comparable overlap is too limited."
            )
        if recommended_role == "primary":
            return f"{device} {construct.replace('_', ' ')} is complete and scientifically usable as a primary stream; coverage is {mean_coverage:.2f} and {quality_piece}."
        if recommended_role == "primary_with_qc" and construct in {"eda", "temperature"} and np.isfinite(median_spearman):
            return (
                f"{device} {construct.replace('_', ' ')} is scientifically usable within-device, but cross-device comparability is limited "
                f"(median agreement {median_spearman:.2f}); coverage is {mean_coverage:.2f} and it should be interpreted as a device-specific stream."
            )
        if recommended_role == "primary_with_qc":
            return f"{device} {construct.replace('_', ' ')} is broadly usable but should be interpreted with session-level QC because coverage is {mean_coverage:.2f} and {quality_piece}."
        if recommended_role == "secondary_only":
            return f"{device} {construct.replace('_', ' ')} is usable mainly as a secondary stream; support is partial and cross-device comparability is limited."
        return f"{device} {construct.replace('_', ' ')} is not yet strong enough for primary scientific interpretation in this release."

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
        from scipy import stats
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
                    median_difference = float(diff.median()) if not diff.empty else np.nan
                    wilcoxon_stat = np.nan
                    wilcoxon_p_value = np.nan
                    if n_pairs >= 2:
                        try:
                            wilcoxon = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided", mode="auto")
                            wilcoxon_stat = float(wilcoxon.statistic)
                            wilcoxon_p_value = float(wilcoxon.pvalue)
                        except Exception:
                            pass
                    is_event_scale = feature in QUESTIONNAIRE_SET_COLUMNS or feature == "fan_control_au"
                    primary_test = "wilcoxon_signed_rank" if is_event_scale and pd.notna(wilcoxon_p_value) else "paired_t_test"
                    primary_p_value = float(wilcoxon_p_value) if primary_test == "wilcoxon_signed_rank" else float(stats["p_value"]) if pd.notna(stats["p_value"]) else np.nan
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
                            "primary_test": primary_test,
                            "primary_p_value": primary_p_value,
                            "t_statistic": stats["statistic"],
                            "t_p_value": stats["p_value"],
                            "wilcoxon_statistic": wilcoxon_stat,
                            "wilcoxon_p_value": wilcoxon_p_value,
                            "median_difference": median_difference,
                            **stats,
                        }
                    )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["p_value_fdr"] = benjamini_hochberg(out["primary_p_value"])
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
                "study_day": d["study_day"].iloc[0] if "study_day" in d.columns else np.nan,
                "condition_code": d["condition_code"].iloc[0],
                "n_phase_rows": int(len(d)),
            }
            metrics = [m for m in dict.fromkeys(PRIMARY_ENDPOINTS + SESSION_MODEL_FEATURES) if m in d.columns]
            for metric in metrics:
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

    def _mixed_effects_primary(self, cohort_phase: pd.DataFrame, sample_status: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if cohort_phase.empty:
            return self._empty_mixed_effects(), self._empty_mixed_effects_diagnostics()
        inferential_ok = bool(sample_status["cohort_inference_eligible"].iloc[0]) if not sample_status.empty else False
        if not inferential_ok:
            return self._empty_mixed_effects(), self._empty_mixed_effects_diagnostics()
        try:
            import statsmodels.formula.api as smf
        except Exception:
            return self._empty_mixed_effects(), self._empty_mixed_effects_diagnostics()

        rows = []
        diagnostics = []
        comparison = cohort_phase.loc[cohort_phase["protocol_phase"].astype(str) != "acclimation"].copy()
        if comparison.empty:
            return self._empty_mixed_effects(), self._empty_mixed_effects_diagnostics()
        for metric in [m for m in PRIMARY_ENDPOINTS if m in comparison.columns]:
            d = comparison[
                ["participant_id", "condition_code", "illuminance_level", "time_of_day", "protocol_phase", metric]
            ].copy()
            d[metric] = to_numeric(d[metric])
            d = d.dropna()
            if d.empty or d["participant_id"].nunique() < max(4, self.config.runtime.min_cohort_participants_for_inference):
                diagnostics.append(
                    {
                        "metric": metric,
                        "status": "skipped_insufficient_participants",
                        "n_obs": int(len(d)),
                        "n_participants": int(d["participant_id"].nunique()) if "participant_id" in d.columns and not d.empty else 0,
                        "n_terms_retained": 0,
                        "fit_converged": 0,
                        "warning_count": 0,
                        "warning_summary": "",
                    }
                )
                continue
            if d["illuminance_level"].nunique() < 2 or d["time_of_day"].nunique() < 2 or d["protocol_phase"].nunique() < 2:
                diagnostics.append(
                    {
                        "metric": metric,
                        "status": "skipped_insufficient_design_variation",
                        "n_obs": int(len(d)),
                        "n_participants": int(d["participant_id"].nunique()),
                        "n_terms_retained": 0,
                        "fit_converged": 0,
                        "warning_count": 0,
                        "warning_summary": "",
                    }
                )
                continue
            fit = None
            model_spec = ""
            warning_summary = ""
            fit_converged = 0
            caught_messages: list[str] = []
            model_attempts = [
                (
                    "condition_time_phase_random_phase_slope",
                    f"{metric} ~ C(illuminance_level) * C(time_of_day) * C(protocol_phase)",
                    "~C(protocol_phase)",
                ),
                (
                    "condition_time_phase_random_intercept",
                    f"{metric} ~ C(illuminance_level) * C(time_of_day) + C(protocol_phase)",
                    None,
                ),
                (
                    "condition_time_random_intercept",
                    f"{metric} ~ C(illuminance_level) * C(time_of_day)",
                    None,
                ),
            ]
            try:
                for spec_name, formula, re_formula in model_attempts:
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            model = smf.mixedlm(
                                formula,
                                data=d,
                                groups=d["participant_id"],
                                re_formula=re_formula,
                            )
                            candidate = model.fit(reml=False, method="lbfgs", disp=False)
                        fit = candidate
                        model_spec = spec_name
                        caught_messages = [str(w.message) for w in caught]
                        fit_converged = int(bool(getattr(candidate, "converged", True)))
                        if fit_converged:
                            break
                    except Exception:
                        continue
            except Exception:
                fit = None
            if fit is None:
                diagnostics.append(
                    {
                        "metric": metric,
                        "status": "fit_failed",
                        "model_spec": "",
                        "n_obs": int(len(d)),
                        "n_participants": int(d["participant_id"].nunique()),
                        "n_terms_retained": 0,
                        "fit_converged": 0,
                        "warning_count": 0,
                        "warning_summary": "",
                    }
                )
                continue
            params = fit.params
            conf = fit.conf_int()
            pvals = fit.pvalues
            ses = fit.bse

            def _is_interpretive_fixed_effect(term: object) -> bool:
                text = str(term)
                if text in {"Intercept", "Group Var"}:
                    return False
                if " Var" in text or " Cov" in text:
                    return False
                return True

            fixed_effect_terms = [term for term in params.index if _is_interpretive_fixed_effect(term)]
            warning_summary = " | ".join(sorted(set(caught_messages))) if caught_messages else ""
            status = "retained" if fit_converged and fixed_effect_terms else "retained_with_fit_issue" if fixed_effect_terms else "no_fixed_effect_terms"
            diagnostics.append(
                {
                    "metric": metric,
                    "status": status,
                    "model_spec": model_spec,
                    "n_obs": int(len(d)),
                    "n_participants": int(d["participant_id"].nunique()),
                    "n_terms_retained": int(len(fixed_effect_terms)),
                    "fit_converged": fit_converged,
                    "warning_count": int(len(caught)),
                    "warning_summary": warning_summary,
                }
            )
            for term, beta in params.items():
                if not _is_interpretive_fixed_effect(term):
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
                        "converged": fit_converged,
                        "model_spec": model_spec,
                    }
                )
        out = pd.DataFrame(rows)
        diagnostics_df = pd.DataFrame(diagnostics)
        if out.empty:
            return self._empty_mixed_effects(), diagnostics_df if not diagnostics_df.empty else self._empty_mixed_effects_diagnostics()
        out["p_value_fdr"] = benjamini_hochberg(out["p_value"])
        out["significant_fdr"] = (to_numeric(out["p_value_fdr"]) < 0.05).astype(int)
        if diagnostics_df.empty:
            diagnostics_df = self._empty_mixed_effects_diagnostics()
        return (
            out.sort_values(["significant_fdr", "metric", "p_value_fdr"], ascending=[False, True, True]).reset_index(drop=True),
            diagnostics_df.sort_values(["status", "metric"]).reset_index(drop=True),
        )

    def _predictive_benchmarks(self, session_primary_endpoints: pd.DataFrame) -> pd.DataFrame:
        if session_primary_endpoints.empty:
            return self._empty_predictive_benchmarks()
        try:
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
        if "participant_id" not in d.columns:
            return self._empty_predictive_benchmarks()
        numeric_candidates = []
        for col in d.columns:
            if col in {"session_id", "participant_id", "study_day", "condition_code", "illuminance_level", "time_of_day"}:
                continue
            if col.endswith("__baseline_phase"):
                continue
            if d[col].dtype == object:
                continue
            numeric_candidates.append(col)
        numeric_candidates = [c for c in numeric_candidates if to_numeric(d[c]).notna().sum() >= max(4, len(d) // 3)]
        if not numeric_candidates:
            return self._empty_predictive_benchmarks()
        d[numeric_candidates] = d[numeric_candidates].apply(to_numeric)

        models = {
            "logistic_regression": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
                ]
            ),
            "random_forest": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=1)),
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

        def tertile_state(series: pd.Series, label: str) -> pd.Series:
            vals = to_numeric(series)
            valid = vals.dropna()
            if valid.nunique() < 3:
                return pd.Series(index=series.index, dtype=object)
            ranks = valid.rank(method="average", pct=True)
            mapped = pd.Series(index=valid.index, dtype=object)
            mapped.loc[ranks <= 1 / 3] = f"low_{label}"
            mapped.loc[(ranks > 1 / 3) & (ranks < 2 / 3)] = f"mid_{label}"
            mapped.loc[ranks >= 2 / 3] = f"high_{label}"
            return mapped.reindex(series.index)

        feature_sets = {
            "environment_only": [
                c for c in numeric_candidates
                if c.startswith(("indoor_", "outdoor_", "fan_")) or c in {"indoor_air_temp_mean_C__session_mean", "indoor_air_velocity_mean_m_s__session_mean", "fan_control_au__session_mean", "fan_current_A__session_mean"}
            ],
            "physiology_only": [
                c for c in numeric_candidates
                if any(token in c for token in ["empatica_", "biopac_", "master_dpg_C", "thermal_gradient_C", "thermal_state_index_C", "hr_delta_bpm", "eda_delta_uS", "temp_delta_C"])
            ],
        }
        feature_sets["fused_multimodal"] = sorted(set(feature_sets["environment_only"] + feature_sets["physiology_only"]))
        validation_groups = {
            "participant_grouped": d["participant_id"].astype(str),
            "study_day_grouped": d["study_day"].astype(str) if "study_day" in d.columns else pd.Series(dtype=object),
            "condition_holdout": d["condition_code"].astype(str) if "condition_code" in d.columns else pd.Series(dtype=object),
        }

        rows = []
        task_specs = {
            "thermal_comfort_state": tertile_state(d.get("thermal_comfort__session_mean", pd.Series(dtype=float)), "comfort"),
            "thermal_sensation_state": tertile_state(d.get("thermal_sensation__session_mean", pd.Series(dtype=float)), "sensation"),
        }
        leakage_prefixes = {
            "thermal_comfort_state": ["thermal_comfort__"],
            "thermal_sensation_state": ["thermal_sensation__"],
        }
        for task, target_series in task_specs.items():
            if target_series.empty:
                continue
            keep_idx = target_series.dropna().index
            if len(keep_idx) < 12:
                continue
            target_series = target_series.loc[keep_idx]
            task_frame = d.loc[keep_idx].copy()
            y, mapping = encode_target(target_series)
            if y.nunique() < 2:
                continue
            for feature_set, cols in feature_sets.items():
                cols = [c for c in cols if c in task_frame.columns]
                banned_prefixes = leakage_prefixes.get(task, [])
                cols = [c for c in cols if not any(c.startswith(prefix) for prefix in banned_prefixes)]
                if len(cols) < 2:
                    continue
                X = task_frame[cols].copy()
                for validation_scheme, groups in validation_groups.items():
                    groups = groups.loc[keep_idx] if not groups.empty else pd.Series(dtype=object)
                    group_count = int(groups.nunique()) if not groups.empty else 0
                    if group_count < 2:
                        continue
                    splitter = GroupKFold(n_splits=min(5, group_count))
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
                                "feature_set": feature_set,
                                "validation_scheme": validation_scheme,
                                "n_features": int(len(cols)),
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
        return out.sort_values(
            ["task", "validation_scheme", "feature_set", "balanced_accuracy_mean", "macro_f1_mean"],
            ascending=[True, True, True, False, False],
        ).reset_index(drop=True)

    def _lag_response_outputs(self, cohort_minute: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if cohort_minute.empty:
            return self._empty_lag_response_register(), self._empty_lag_response_profile()
        from scipy import stats

        rows = []
        profile_rows = []
        minute = cohort_minute.sort_values(["session_id", "minute_index"]).copy()
        for predictor, target in TEMPORAL_RESPONSE_SPECS:
            if predictor not in minute.columns or target not in minute.columns:
                continue
            lag_rows = []
            for lag in LAG_WINDOWS_MINUTES:
                session_stats = []
                for _, d in minute.groupby("session_id"):
                    cur = d[["minute_index", predictor, target]].copy()
                    cur[predictor] = to_numeric(cur[predictor])
                    cur[target] = to_numeric(cur[target])
                    shifted = cur[["minute_index", predictor]].copy()
                    shifted["minute_index"] = shifted["minute_index"] + lag
                    pair = cur[["minute_index", target]].merge(shifted, on="minute_index", how="inner").dropna()
                    if len(pair) < 12:
                        continue
                    spearman_r, _ = stats.spearmanr(pair[predictor], pair[target], nan_policy="omit")
                    if pd.isna(spearman_r):
                        continue
                    session_stats.append(
                        {
                            "spearman_r": float(spearman_r),
                            "n_pairs": int(len(pair)),
                        }
                    )
                if not session_stats:
                    continue
                stats_df = pd.DataFrame(session_stats)
                lag_rows.append(
                    {
                        "lag_minutes": lag,
                        "median_spearman_r": float(to_numeric(stats_df["spearman_r"]).median()),
                        "median_abs_spearman_r": float(to_numeric(stats_df["spearman_r"]).abs().median()),
                        "same_sign_fraction": float((np.sign(to_numeric(stats_df["spearman_r"])) == np.sign(float(to_numeric(stats_df["spearman_r"]).median()))).mean()),
                        "n_sessions": int(len(stats_df)),
                        "median_pairs_per_session": int(to_numeric(stats_df["n_pairs"]).median()),
                    }
                )
            lag_df = pd.DataFrame(lag_rows)
            if lag_df.empty:
                continue
            best = lag_df.sort_values(["median_abs_spearman_r", "same_sign_fraction", "n_sessions"], ascending=[False, False, False]).iloc[0]
            evidence_grade = "C"
            if float(best["median_abs_spearman_r"]) >= 0.30 and float(best["same_sign_fraction"]) >= 0.70 and int(best["n_sessions"]) >= 12:
                evidence_grade = "A"
            elif float(best["median_abs_spearman_r"]) >= 0.20 and float(best["same_sign_fraction"]) >= 0.60 and int(best["n_sessions"]) >= 8:
                evidence_grade = "B"
            lag_df["predictor"] = predictor
            lag_df["target"] = target
            lag_df["is_best_lag"] = (to_numeric(lag_df["lag_minutes"]) == float(best["lag_minutes"])).astype(int)
            lag_df["evidence_grade"] = evidence_grade
            profile_rows.extend(lag_df.to_dict(orient="records"))
            rows.append(
                {
                    "predictor": predictor,
                    "target": target,
                    "best_lag_minutes": int(best["lag_minutes"]),
                    "median_spearman_r": float(best["median_spearman_r"]),
                    "median_abs_spearman_r": float(best["median_abs_spearman_r"]),
                    "same_sign_fraction": float(best["same_sign_fraction"]),
                    "n_sessions": int(best["n_sessions"]),
                    "median_pairs_per_session": int(best["median_pairs_per_session"]),
                    "evidence_grade": evidence_grade,
                    "scientific_reading": (
                        f"{predictor} shows its strongest support-screened association with {target} at {int(best['lag_minutes'])} min lag "
                        f"(median Spearman r={float(best['median_spearman_r']):.2f}, same-sign fraction={float(best['same_sign_fraction']):.2f})."
                    ),
                }
            )
        out = pd.DataFrame(rows)
        if out.empty:
            return self._empty_lag_response_register(), self._empty_lag_response_profile()
        profile = pd.DataFrame(profile_rows)
        if profile.empty:
            profile = self._empty_lag_response_profile()
        else:
            profile = profile.sort_values(["predictor", "target", "lag_minutes"]).reset_index(drop=True)
        return (
            out.sort_values(["evidence_grade", "median_abs_spearman_r", "n_sessions"], ascending=[True, False, False]).reset_index(drop=True),
            profile,
        )

    def _threshold_response_register(self, cohort_minute: pd.DataFrame, lag_response_register: pd.DataFrame) -> pd.DataFrame:
        if cohort_minute.empty or lag_response_register.empty:
            return self._empty_threshold_response_register()
        rows = []
        minute = cohort_minute.sort_values(["session_id", "minute_index"]).copy()
        for lag_row in lag_response_register.itertuples(index=False):
            predictor = str(lag_row.predictor)
            target = str(lag_row.target)
            lag = int(lag_row.best_lag_minutes)
            if predictor not in minute.columns or target not in minute.columns:
                continue
            pooled = []
            supported_sessions = set()
            for session_id, d in minute.groupby("session_id"):
                cur = d[["minute_index", predictor, target]].copy()
                cur[predictor] = to_numeric(cur[predictor])
                cur[target] = to_numeric(cur[target])
                shifted = cur[["minute_index", predictor]].copy()
                shifted["minute_index"] = shifted["minute_index"] + lag
                pair = cur[["minute_index", target]].merge(shifted, on="minute_index", how="inner").dropna()
                if len(pair) < 12 or pair[predictor].nunique() < 8 or pair[target].nunique() < 6:
                    continue
                pair["session_id"] = session_id
                pooled.append(pair)
                supported_sessions.add(str(session_id))
            if not pooled:
                continue
            pair = pd.concat(pooled, ignore_index=True)
            x = to_numeric(pair[predictor]).to_numpy(dtype=float)
            y = to_numeric(pair[target]).to_numpy(dtype=float)
            if len(x) < 64 or np.unique(x).size < 12:
                continue
            finite = np.isfinite(x) & np.isfinite(y)
            x = x[finite]
            y = y[finite]
            if len(x) < 64:
                continue

            base_design = np.column_stack([np.ones(len(x)), x])
            try:
                base_beta, *_ = np.linalg.lstsq(base_design, y, rcond=None)
            except Exception:
                continue
            base_pred = base_design @ base_beta
            base_rss = float(np.square(y - base_pred).sum())
            if not np.isfinite(base_rss) or base_rss <= 0:
                continue

            candidates = np.unique(np.quantile(x, np.linspace(0.2, 0.8, 13)))
            best = None
            for threshold in candidates:
                below_n = int((x <= threshold).sum())
                above_n = int((x > threshold).sum())
                if below_n < 16 or above_n < 16:
                    continue
                hinge = np.maximum(0.0, x - threshold)
                design = np.column_stack([np.ones(len(x)), x, hinge])
                try:
                    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
                except Exception:
                    continue
                pred = design @ beta
                rss = float(np.square(y - pred).sum())
                if not np.isfinite(rss):
                    continue
                slope_below = float(beta[1])
                slope_above = float(beta[1] + beta[2])
                slope_change = float(beta[2])
                improvement = float((base_rss - rss) / base_rss)
                candidate = {
                    "threshold_value": float(threshold),
                    "slope_below": slope_below,
                    "slope_above": slope_above,
                    "slope_change": slope_change,
                    "rss_improvement_fraction": improvement,
                }
                if best is None or (
                    candidate["rss_improvement_fraction"],
                    abs(candidate["slope_change"]),
                ) > (
                    best["rss_improvement_fraction"],
                    abs(best["slope_change"]),
                ):
                    best = candidate
            if best is None:
                continue
            evidence_grade = "C"
            if best["rss_improvement_fraction"] >= 0.15 and abs(best["slope_change"]) >= 0.10 and len(supported_sessions) >= 12:
                evidence_grade = "A"
            elif best["rss_improvement_fraction"] >= 0.08 and abs(best["slope_change"]) >= 0.05 and len(supported_sessions) >= 8:
                evidence_grade = "B"
            rows.append(
                {
                    "predictor": predictor,
                    "target": target,
                    "threshold_value": best["threshold_value"],
                    "slope_below": best["slope_below"],
                    "slope_above": best["slope_above"],
                    "slope_change": best["slope_change"],
                    "rss_improvement_fraction": best["rss_improvement_fraction"],
                    "n_pairs": int(len(x)),
                    "n_sessions": int(len(supported_sessions)),
                    "evidence_grade": evidence_grade,
                    "scientific_reading": (
                        f"{predictor} shows a breakpoint near {best['threshold_value']:.2f} for {target}, "
                        f"with slope changing from {best['slope_below']:.3f} below threshold to {best['slope_above']:.3f} above threshold."
                    ),
                }
            )
        out = pd.DataFrame(rows)
        if out.empty:
            return self._empty_threshold_response_register()
        return out.sort_values(["evidence_grade", "rss_improvement_fraction", "n_sessions"], ascending=[True, False, False]).reset_index(drop=True)

    def _scientific_decision_register(
        self,
        cohort_minute: pd.DataFrame,
        lag_response_register: pd.DataFrame,
        threshold_response_register: pd.DataFrame,
        condition_contrasts: pd.DataFrame,
        mixed_effects_primary: pd.DataFrame,
        predictive_benchmarks: pd.DataFrame,
        signal_audit_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        if cohort_minute.empty or lag_response_register.empty:
            return self._empty_scientific_decision_register()
        minute = cohort_minute.copy()
        minute = minute.sort_values(["session_id", "minute_index"])
        stream_roles = {}
        if not signal_audit_summary.empty:
            for row in signal_audit_summary.itertuples(index=False):
                stream_roles[str(getattr(row, "construct", ""))] = str(getattr(row, "recommended_role", ""))
        mixed_metrics = set(mixed_effects_primary.get("metric", pd.Series(dtype=str)).astype(str)) if not mixed_effects_primary.empty else set()
        contrast_metrics = set(condition_contrasts.loc[to_numeric(condition_contrasts.get("significant_fdr", pd.Series(dtype=float))).fillna(0) == 1, "metric"].astype(str)) if not condition_contrasts.empty else set()
        best_benchmark = pd.DataFrame()
        if not predictive_benchmarks.empty:
            best_benchmark = predictive_benchmarks.sort_values(["balanced_accuracy_mean", "macro_f1_mean"], ascending=[False, False]).head(1)
        threshold_lookup = {}
        if not threshold_response_register.empty:
            threshold_lookup = {
                (str(r.predictor), str(r.target)): r
                for r in threshold_response_register.itertuples(index=False)
            }
        rows = []
        for row in lag_response_register.itertuples(index=False):
            predictor = str(row.predictor)
            target = str(row.target)
            if predictor not in minute.columns or target not in minute.columns:
                continue
            pair = minute[[predictor, target]].apply(to_numeric).dropna()
            if len(pair) < 24:
                continue
            quantiles = to_numeric(pair[predictor]).quantile([0.25, 0.5, 0.75]).to_dict()
            if pair[target].nunique() < 4:
                continue
            low_mask = to_numeric(pair[predictor]) <= quantiles.get(0.25, np.nan)
            high_mask = to_numeric(pair[predictor]) >= quantiles.get(0.75, np.nan)
            if low_mask.sum() < 8 or high_mask.sum() < 8:
                continue
            low_target = float(to_numeric(pair.loc[low_mask, target]).median())
            high_target = float(to_numeric(pair.loc[high_mask, target]).median())
            delta = high_target - low_target
            threshold_row = threshold_lookup.get((predictor, target))
            threshold_value = np.nan
            favorable_band = f"{float(quantiles.get(0.25, np.nan)):.2f} to {float(quantiles.get(0.75, np.nan)):.2f}"
            control_action = "Keep this driver within the empirically supported operating band."
            claim_family = "Operational threshold and lag"
            if threshold_row is not None:
                threshold_value = float(getattr(threshold_row, "threshold_value"))
                slope_below = float(getattr(threshold_row, "slope_below"))
                slope_above = float(getattr(threshold_row, "slope_above"))
                if target in {"thermal_comfort", "thermal_sensation"}:
                    if slope_above < 0 or slope_below < 0:
                        favorable_band = f"<= {threshold_value:.2f}"
                        control_action = "Avoid pushing this driver above the estimated breakpoint when comfort is the priority."
                    else:
                        favorable_band = f">= {threshold_value:.2f}"
                        control_action = "Hold this driver at or above the estimated breakpoint when comfort is the priority."
                else:
                    if abs(slope_above) >= abs(slope_below):
                        favorable_band = f"Below {threshold_value:.2f} if minimizing response amplification"
                    else:
                        favorable_band = f"Around {threshold_value:.2f} or above if the post-threshold slope remains flatter"
                    control_action = "Use the breakpoint as the control caution point where physiology becomes more response-sensitive."
                claim_family = "Estimated breakpoint and lag"
            evidence_grade = str(row.evidence_grade)
            if target in mixed_metrics or target in contrast_metrics:
                evidence_grade = {"C": "B", "B": "A", "A": "A"}.get(evidence_grade, evidence_grade)
            if threshold_row is not None:
                evidence_grade = {"C": "B", "B": "A", "A": "A"}.get(evidence_grade, evidence_grade) if str(getattr(threshold_row, "evidence_grade", "C")) in {"A", "B"} else evidence_grade
            supporting_streams = "environment + physiology" if target.startswith(("biopac_", "empatica_")) else "environment + survey"
            statistical_basis = f"Best lag {int(row.best_lag_minutes)} min; median r={float(row.median_spearman_r):.2f}"
            if threshold_row is not None:
                statistical_basis += (
                    f"; threshold={threshold_value:.2f}; segmented RSS improvement={float(getattr(threshold_row, 'rss_improvement_fraction')):.2f}"
                )
            if not best_benchmark.empty:
                best = best_benchmark.iloc[0]
                statistical_basis += f"; best predictive check={best['feature_set']} / {best['validation_scheme']} ({float(best['balanced_accuracy_mean']):.2f})"
            rows.append(
                {
                    "claim_family": claim_family,
                    "predictor": predictor,
                    "target": target,
                    "recommended_operating_band": favorable_band,
                    "response_lag_minutes": int(row.best_lag_minutes),
                    "evidence_grade": evidence_grade,
                    "supporting_streams": supporting_streams,
                    "statistical_basis": statistical_basis,
                    "practical_reading": (
                        f"When {predictor} moves from its lower to upper exposure band, "
                        f"{target} shifts by approximately {delta:.2f} median units across retained windows."
                    ),
                    "control_recommendation": control_action,
                }
            )
        out = pd.DataFrame(rows)
        if out.empty:
            return self._empty_scientific_decision_register()
        return out.sort_values(["evidence_grade", "response_lag_minutes", "target"], ascending=[True, True, True]).head(12).reset_index(drop=True)
