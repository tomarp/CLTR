from __future__ import annotations

from pathlib import Path
import re
from textwrap import fill

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch, Rectangle
from plotly.offline.offline import get_plotlyjs

plt.rcParams["figure.max_open_warning"] = 0

from .analysis import (
    ANALYTIC_FEATURES,
    PRIMARY_ENDPOINTS,
    SUPPORT_GRADED_ENDPOINTS,
    endpoint_is_primary,
    endpoint_policy_role,
)
from .config import CLTRConfig
from .utils import ensure_dir, html_escape, parse_any_datetime, parse_local_datetime, safe_read_csv, to_numeric


PHASE_ORDER = [
    "acclimation",
    "fan_at_constant_speed",
    "skin_rewarming",
    "fan_free_control",
    "steady_state",
    "overall_comfort",
]
PHASE_ABBR = {
    "acclimation": "ACC",
    "fan_at_constant_speed": "FCS",
    "skin_rewarming": "SR",
    "fan_free_control": "FFC",
    "steady_state": "SS",
    "overall_comfort": "OC",
    "unknown": "UNK",
}
PHASE_ABBR_CAPTION = "<strong>ACC</strong> = acclimation; <strong>FCS</strong> = fan at constant speed; <strong>SR</strong> = skin rewarming; <strong>FFC</strong> = fan free control; <strong>SS</strong> = steady state; <strong>OC</strong> = overall comfort."
ACC_ASSUMPTION_CAPTION = "<strong>ACC*</strong> = assumed acclimation baseline when direct acclimation data are unavailable."
CONDITION_ORDER = ["DIM-MOR", "BRI-MOR", "DIM-MID", "BRI-MID"]
CONDITION_COLORS = {
    "DIM-MOR": "#1d4ed8",
    "BRI-MOR": "#f59e0b",
    "DIM-MID": "#0f766e",
    "BRI-MID": "#b91c1c",
}
FEATURE_LABELS = {
    "thermal_comfort": "Thermal Comfort",
    "thermal_sensation": "Thermal Sensation",
    "thermal_preference": "Thermal Preference",
    "thermal_pleasure": "Thermal Pleasure",
    "visual_sensation": "Visual Sensation",
    "color_sensation": "Color Sensation",
    "visual_comfort": "Visual Comfort",
    "sound_comfort_dbA": "Sound Comfort",
    "air_quality_comfort": "Air Quality Comfort",
    "room_comfort": "Room Comfort",
    "empatica_hr_mean_bpm": "Empatica HR",
    "biopac_hr_mean_bpm": "BIOPAC HR",
    "empatica_eda_mean_uS": "Empatica EDA",
    "biopac_eda_mean_uS": "BIOPAC EDA",
    "empatica_temp_mean_C": "Empatica Temperature",
    "biopac_temp_chest_mean_C": "Chest Temperature",
    "indoor_relative_humidity_percent": "Indoor RH",
    "outdoor_air_temp_C": "Outdoor Air Temperature",
    "outdoor_relative_humidity_percent": "Outdoor Relative Humidity",
    "outdoor_wind_speed_m_s": "Outdoor Wind Speed",
    "outdoor_solar_radiation_W_m2": "Solar Radiation",
    "biopac_bloodflow_mean_bpu": "Blood Flow",
    "indoor_air_velocity_mean_m_s": "Air Velocity",
    "indoor_air_temp_mean_C": "Indoor Air Temperature",
    "fan_control_au": "Fan Control",
    "fan_control_secondary_au": "Secondary Fan Control",
    "fan_current_A": "Fan Current",
    "master_skin_chest_C": "Master Skin Chest Temperature",
    "master_hand_C": "Hand Temperature",
    "master_dpg_C": "Distal-Proximal Gradient",
    "thermal_gradient_C": "Distal-Proximal Gradient",
    "thermal_state_index_C": "Thermal State Index",
    "hr_delta_bpm": "Heart-Rate Delta",
    "eda_delta_uS": "Electrodermal Delta",
    "temp_delta_C": "Temperature Delta",
    "thermal_comfort_state": "Thermal Comfort State",
    "thermal_sensation_state": "Thermal Sensation State",
}
AXIS_LABELS = {
    "thermal_comfort": "Thermal Comfort Response (ordinal scale)",
    "thermal_sensation": "Thermal Sensation Vote (ordinal scale)",
    "thermal_preference": "Thermal Preference Vote (ordinal scale)",
    "thermal_pleasure": "Thermal Pleasure Response (ordinal scale)",
    "visual_sensation": "Visual Sensation Vote (ordinal scale)",
    "color_sensation": "Color Sensation Vote (ordinal scale)",
    "visual_comfort": "Visual Comfort Response (ordinal scale)",
    "sound_comfort_dbA": "Sound Comfort Response (ordinal scale)",
    "air_quality_comfort": "Air-Quality Comfort Response (ordinal scale)",
    "room_comfort": "Room Comfort Response (ordinal scale)",
    "empatica_hr_mean_bpm": "Empatica Heart Rate (bpm)",
    "biopac_hr_mean_bpm": "BIOPAC Heart Rate (bpm)",
    "empatica_eda_mean_uS": "Empatica Electrodermal Activity (uS)",
    "biopac_eda_mean_uS": "BIOPAC Electrodermal Activity (uS)",
    "empatica_temp_mean_C": "Empatica Skin Temperature (C)",
    "biopac_temp_chest_mean_C": "BIOPAC Chest Temperature (C)",
    "biopac_temp_thigh_mean_C": "BIOPAC Thigh Temperature (C)",
    "biopac_temp_arm_mean_C": "BIOPAC Arm Temperature (C)",
    "biopac_temp_tibia_mean_C": "BIOPAC Tibia Temperature (C)",
    "empatica_bvp_mean": "Empatica Blood Volume Pulse (nW)",
    "empatica_acc_mean_g": "Empatica Acceleration (g)",
    "empatica_enmo_mean_g": "Empatica ENMO (g)",
    "empatica_steps": "Empatica Steps (count)",
    "biopac_bloodflow_mean_bpu": "BIOPAC Blood Flow (BPU)",
    "biopac_backscatter_mean_percent": "BIOPAC Backscatter (%)",
    "indoor_air_velocity_mean_m_s": "Indoor Air Velocity (m/s)",
    "indoor_air_temp_mean_C": "Indoor Air Temperature (C)",
    "indoor_relative_humidity_percent": "Indoor Relative Humidity (%)",
    "outdoor_air_temp_C": "Outdoor Air Temperature (C)",
    "outdoor_relative_humidity_percent": "Outdoor Relative Humidity (%)",
    "outdoor_wind_speed_m_s": "Outdoor Wind Speed (m/s)",
    "outdoor_solar_radiation_W_m2": "Solar Radiation (W/m2)",
    "fan_control_au": "Fan Control Setting (a.u.)",
    "fan_control_secondary_au": "Secondary Fan Control Setting (a.u.)",
    "fan_current_A": "Fan Current (A)",
    "master_dpg_C": "Distal-Proximal Gradient (C)",
    "thermal_gradient_C": "Thermal Gradient (C)",
}
REPORT_METRIC_LABELS = {
    "empatica_bvp_mean": "Empatica BVP",
    "empatica_hr_mean_bpm": "Empatica HR",
    "empatica_eda_mean_uS": "Empatica EDA",
    "empatica_temp_mean_C": "Empatica Temperature",
    "empatica_acc_mean_g": "Empatica Acceleration",
    "empatica_enmo_mean_g": "Empatica ENMO",
    "empatica_steps": "Empatica Steps",
    "biopac_hr_mean_bpm": "BIOPAC HR",
    "biopac_eda_mean_uS": "BIOPAC EDA",
    "biopac_temp_chest_mean_C": "BIOPAC Chest Temperature",
    "biopac_temp_thigh_mean_C": "BIOPAC Thigh Temperature",
    "biopac_temp_arm_mean_C": "BIOPAC Arm Temperature",
    "biopac_temp_tibia_mean_C": "BIOPAC Tibia Temperature",
    "biopac_bloodflow_mean_bpu": "BIOPAC Blood Flow",
    "biopac_backscatter_mean_percent": "BIOPAC Backscatter",
    "indoor_air_temp_mean_C": "Indoor Air Temperature",
    "indoor_air_velocity_mean_m_s": "Indoor Air Velocity",
    "indoor_relative_humidity_percent": "Indoor Relative Humidity",
    "outdoor_air_temp_C": "Outdoor Air Temperature",
    "outdoor_relative_humidity_percent": "Outdoor Relative Humidity",
    "outdoor_wind_speed_m_s": "Outdoor Wind Speed",
    "outdoor_solar_radiation_W_m2": "Solar Radiation",
    "fan_control_au": "Fan Control",
    "fan_control_secondary_au": "Secondary Fan Control",
    "fan_current_A": "Fan Current",
    "thermal_sensation": "Thermal Sensation",
    "thermal_comfort": "Thermal Comfort",
    "thermal_preference": "Thermal Preference",
    "thermal_pleasure": "Thermal Pleasure",
    "visual_sensation": "Visual Sensation",
    "color_sensation": "Color Sensation",
    "visual_comfort": "Visual Comfort",
    "sound_comfort_dbA": "Sound Comfort",
    "air_quality_comfort": "Air-Quality Comfort",
    "room_comfort": "Room Comfort",
}
REPORT_METRIC_KINDS = {
    "fan_control_au": "trajectory",
    "fan_control_secondary_au": "trajectory",
    "fan_current_A": "trajectory",
}
WORK_INDEX_TITLE = "CLTR Atlas"
WORK_INDEX_SUBTITLE = "Explore the study-wide summary and every individual session in one place."
WORK_HOME_TITLE = "CLTR"
WORK_HOME_SUBTITLE = "Controlled Laboratory Thermal Response"
SESSION_CTA = "Open session report"
COHORT_CTA = "Open cohort report"
COPYRIGHT_NOTE = "&copy; 2026 Tomar & Elkounni. All rights reserved."
COHORT_LEGACY_INDEX_FILENAME = "cohort_report.html"
PROJECT_GITHUB_URL = "https://github.com/tomarp/cltr"
PROJECT_ZENODO_URL = "https://doi.org/10.5281/zenodo.17817175"
PROJECT_FRAMEWORK_URL = "https://github.com/tomarp/cltr/tree/main/framework"
MAX_SESSION_MAIN_FIGURES = 5
MAX_COHORT_MAIN_FIGURES = 5
BLOCK_PHASE_NARRATIVE_THRESHOLD = 2
COMPARISON_BLOCKS = {"1", "2", "3"}
SECTION_TITLES = {
    "frontmatter": "Overview",
    "subjective_behavioral": "Questionnaire Responses And Fan Behavior",
    "physiological": "Physiological Data",
    "environmental": "Environmental Data",
    "processed_cleaned": "Processed And Cleaned Signals",
    "alignment_support": "Alignment And Support Layer",
    "derived": "Derived Results",
    "agreement_section": "Relationships And Validation",
    "raw": "Measured Trends",
    "analyzed": "Scientific Results",
    "interpretive": "Relationships And Validation",
    "appendix": "Additional Figures",
}
SECTION_ORDER = [
    "frontmatter",
    "subjective_behavioral",
    "physiological",
    "environmental",
    "processed_cleaned",
    "alignment_support",
    "derived",
    "agreement_section",
    "raw",
    "analyzed",
    "interpretive",
    "appendix",
]
TABLE_COLUMN_LABELS = {
    "metric": "Measure",
    "contrast_family": "Contrast Family",
    "value": "Value",
    "layer": "Layer",
    "gate": "Gate",
    "gate_type": "Gate Type",
    "unit": "Unit",
    "threshold": "Threshold",
    "observed_value": "Observed Value",
    "protocol_block": "Block",
    "protocol_phase": "Phase",
    "condition_code": "Condition",
    "comparison": "Comparison",
    "primary_test": "Primary Test",
    "primary_p_value": "Primary P",
    "inference_label": "Inference Status",
    "term_reading": "Interpretation",
    "predictor": "Predictor",
    "target": "Target",
    "threshold_unit": "Unit",
    "best_lag_minutes": "Best Lag (min)",
    "response_lag_minutes": "Response Lag (min)",
    "threshold_value": "Threshold Value",
    "slope_below": "Slope Below",
    "slope_above": "Slope Above",
    "slope_change": "Slope Change",
    "rss_improvement_fraction": "RSS Improvement",
    "median_spearman_r": "Median Spearman r",
    "median_abs_spearman_r": "Median |r|",
    "same_sign_fraction": "Same-Sign Fraction",
    "median_pairs_per_session": "Median Pairs/Session",
    "evidence_grade": "Evidence Grade",
    "scientific_reading": "Scientific Reading",
    "claim_family": "Claim Family",
    "recommended_operating_band": "Recommended Operating Band",
    "supporting_streams": "Supporting Streams",
    "statistical_basis": "Statistical Basis",
    "practical_reading": "Practical Reading",
    "control_recommendation": "Control Recommendation",
    "feature_set": "Feature Set",
    "validation_scheme": "Validation Scheme",
    "n_minutes": "Minutes",
    "n_sessions": "Sessions",
    "n_eligible_sessions": "Comparable sessions",
    "n_participants": "Participants",
    "mean_value": "Average",
    "median_value": "Median",
    "min_value": "Min",
    "max_value": "Max",
    "median": "Median",
    "median_overlap_minutes": "Median overlap (min)",
    "median_spearman_r": "Median correlation",
    "median_mae": "Median average error",
    "coverage_fraction": "Coverage",
    "coverage_reading": "Coverage Reading",
    "minute_occupancy_fraction": "Minute Occupancy",
    "minute_occupancy_reading": "Minute Occupancy Reading",
    "observed_prompt_count": "Observed Prompts",
    "expected_prompt_count": "Expected Prompts",
    "prompt_response_fraction": "Prompt Response",
    "prompt_support_reading": "Prompt Support Reading",
    "prompt_support": "Prompt Support",
    "share_within_metric": "Share within measure",
    "mean_consistency": "Consistency",
    "dominant_phase": "Most pronounced phase",
    "direction": "Direction",
    "domain": "Domain",
    "feature": "Feature",
    "registry_role": "Registry Role",
    "unit": "Unit",
    "observation_policy": "Observation Policy",
    "summary_grain": "Summary Grain",
    "iqr": "IQR",
    "skewness": "Skewness",
    "summary_status": "Summary",
    "evidence_status": "Reading guide",
    "support_grade": "Support Grade",
    "support_basis": "Support Basis",
    "supported_phases": "Supported Phases",
    "supported_conditions": "Supported Conditions",
    "supported_condition_phase_cells": "Supported Condition-Phase Cells",
    "cell_coverage_fraction": "Supported Cell Fraction",
    "median_sessions_per_condition_phase": "Median Sessions Per Cell",
    "total_valid_phase_summaries": "Total Valid Phase Summaries",
    "scientific_reading": "Scientific Reading",
    "row_label": "Endpoint / Condition",
    "reference_phase": "Reference Phase",
    "phase_support": "Phase Support",
    "condition_phase_support": "Phase Session Support",
    "qualified_phases": "Qualified Phases",
    "qualified_conditions": "Qualified Conditions",
    "same_sign_fraction": "Same Sign Fraction",
    "relationship_status": "Relationship Status",
    "phase_support_status": "Support Stability",
    "condition_support_status": "Condition Stability",
    "signal_stream": "Signal Stream",
    "stream_label": "Stream",
    "device": "Device",
    "construct": "Construct",
    "comparison_class": "Comparison Class",
    "n_sessions_with_any_data": "Sessions With Data",
    "n_sessions_supported": "Supported Sessions",
    "n_participants_with_data": "Participants With Data",
    "mean_valid_minutes": "Mean Valid Minutes",
    "median_valid_minutes": "Median Valid Minutes",
    "mean_coverage_fraction": "Mean Coverage",
    "mean_quality_fraction": "Mean Quality",
    "mean_plausible_fraction": "Mean Plausibility",
    "adequacy_score": "Adequacy Score",
    "adequacy_status": "Adequacy",
    "recommended_role": "Recommended Role",
    "flagged_sessions": "Flagged Sessions",
    "max_concern_score": "Max Concern Score",
    "present_in_cohort_table": "Present In Cohort Table",
    "signal_audited": "Signal Audited",
    "cross_device_comparable": "Cross-Device Comparable",
    "analytic_feature": "Direct Analytic Feature",
    "stream_usage": "Stream Usage",
    "scientific_use": "Scientific Use",
    "primary_endpoint": "Primary Endpoint",
    "endpoint_policy_role": "Endpoint Policy Role",
    "scenario": "Scenario",
    "included_streams": "Included Streams",
    "excluded_streams": "Excluded Streams",
    "scientific_use": "Scientific Use",
    "manuscript_use": "Manuscript Use",
    "manuscript_claim": "Manuscript Claim",
    "endpoint": "Endpoint",
    "modality_gate": "Modality Gate",
    "claim_status": "Claim Status",
    "claim_note": "Claim Note",
    "in_cohort_table": "In Cohort Table",
    "source_streams": "Source Streams",
    "pathway_status": "Pathway Status",
    "status": "Status",
    "model_spec": "Model Specification",
    "evidence_basis": "Evidence Basis",
    "scientific_implication": "Scientific Implication",
    "n_terms_retained": "Retained Terms",
    "fit_converged": "Fit Converged",
    "warning_count": "Warnings",
    "warning_summary": "Warning Summary",
    "flagged_session_streams": "Flagged Session-Streams",
    "affected_sessions": "Affected Sessions",
    "top_flagged_sessions": "Top Flagged Sessions",
    "primary_concern_driver": "Primary Concern Driver",
    "concern_profile": "Concern Profile",
}
SESSION_STORY_METRICS = {
    "thermal_comfort": {"label": "comfort", "kind": "subjective", "scale": 1.0},
    "master_dpg_C": {"label": "rewarming", "kind": "thermal", "scale": 0.5},
    "thermal_gradient_C": {"label": "gradient", "kind": "thermal", "scale": 0.5},
    "empatica_hr_mean_bpm": {"label": "heart-rate", "kind": "physiology", "scale": 3.0},
    "biopac_hr_mean_bpm": {"label": "heart-rate", "kind": "physiology", "scale": 3.0},
    "biopac_temp_chest_mean_C": {"label": "chest-temperature", "kind": "thermal", "scale": 0.35},
    "empatica_temp_mean_C": {"label": "skin-temperature", "kind": "thermal", "scale": 1.2},
    "indoor_air_velocity_mean_m_s": {"label": "air-velocity", "kind": "environment", "scale": 0.08},
    "fan_control_au": {"label": "fan-control", "kind": "behavior", "scale": 0.2},
    "biopac_bloodflow_mean_bpu": {"label": "blood-flow", "kind": "physiology", "scale": 0.8},
}
SESSION_DERIVED_ENDPOINTS = [
    "thermal_comfort",
    "thermal_sensation",
    "empatica_hr_mean_bpm",
    "empatica_eda_mean_uS",
    "biopac_temp_chest_mean_C",
    "biopac_bloodflow_mean_bpu",
    "indoor_air_velocity_mean_m_s",
    "indoor_air_temp_mean_C",
]
COHORT_DERIVED_ENDPOINTS = SESSION_DERIVED_ENDPOINTS
SPARSE_OBSERVATION_CHANNELS = {
    "thermal_sensation",
    "thermal_comfort",
    "thermal_preference",
    "thermal_pleasure",
    "visual_sensation",
    "color_sensation",
    "visual_comfort",
    "sound_comfort_dbA",
    "air_quality_comfort",
    "room_comfort",
}
QUESTIONNAIRE_FULL_SCALES = {
    "thermal_sensation": np.arange(-3.0, 4.0, 1.0, dtype=float),
    "thermal_comfort": np.arange(-3.0, 4.0, 1.0, dtype=float),
    "thermal_pleasure": np.arange(-3.0, 4.0, 1.0, dtype=float),
    "thermal_preference": np.arange(-1.0, 2.0, 1.0, dtype=float),
    "visual_sensation": np.arange(-3.0, 4.0, 1.0, dtype=float),
    "color_sensation": np.arange(-1.0, 2.0, 1.0, dtype=float),
    "visual_comfort": np.arange(-3.0, 4.0, 1.0, dtype=float),
    "sound_comfort_dbA": np.arange(-3.0, 4.0, 1.0, dtype=float),
    "air_quality_comfort": np.arange(-2.0, 3.0, 1.0, dtype=float),
    "room_comfort": np.arange(-2.0, 3.0, 1.0, dtype=float),
}
CONTROL_SIGNAL_CHANNELS = {
    "fan_current_A",
    "fan_control_au",
    "fan_control_secondary_au",
}
COHORT_SUPPORT_GRADED_METRICS = list(dict.fromkeys(list(SUPPORT_GRADED_ENDPOINTS) + list(COHORT_DERIVED_ENDPOINTS)))
ACC_ASSUMPTION_CHANNELS = SPARSE_OBSERVATION_CHANNELS | {
    "fan_current_A",
    "fan_control_au",
    "fan_control_secondary_au",
}
REPORT_UI = {
    "page_max_width": "1360px",
    "index_page_max_width": "1100px",
    "page_padding": "28px 32px 60px",
    "index_page_padding": "28px 32px 60px",
    "panel_radius": "22px",
    "card_radius": "16px",
    "image_radius": "14px",
    "panel_shadow": "0 18px 44px rgba(23,32,51,0.08)",
    "panel_border": "1px solid rgba(148,163,184,0.24)",
    "panel_padding": "24px 26px",
    "panel_padding_index": "24px 26px",
    "eyebrow_size": "0.74rem",
    "title_size": "2.4rem",
    "index_title_size": "2.4rem",
    "subtitle_line_height": "1.62",
    "nav_font_size": "0.92rem",
    "section_title_size": "1.28rem",
    "figure_title_size": "1.16rem",
    "hero_gap": "24px",
    "cards_gap": "16px",
    "stack_gap": "36px",
    "table_gap": "24px",
    "report_hero_columns": "1.2fr 0.8fr",
    "index_hero_columns": "1.15fr auto",
    "report_cards_columns": "repeat(3,minmax(0,1fr))",
    "index_grid_columns": "repeat(3,minmax(0,1fr))",
    "mobile_breakpoint": "1000px",
    "index_mobile_breakpoint": "1000px",
}

DEVICE_STREAM_CATALOG = [
    {"signal_stream": "empatica_bvp", "metric": "empatica_bvp_mean", "label": "Empatica BVP", "device": "Empatica", "construct": "bvp_source"},
    {"signal_stream": "empatica_hr", "metric": "empatica_hr_mean_bpm", "label": "Empatica HR", "device": "Empatica", "construct": "heart_rate"},
    {"signal_stream": "empatica_eda", "metric": "empatica_eda_mean_uS", "label": "Empatica EDA", "device": "Empatica", "construct": "eda"},
    {"signal_stream": "empatica_temp", "metric": "empatica_temp_mean_C", "label": "Empatica Temperature", "device": "Empatica", "construct": "temperature"},
    {"signal_stream": "empatica_acc", "metric": "empatica_acc_mean_g", "label": "Empatica Acceleration", "device": "Empatica", "construct": "motion"},
    {"signal_stream": "empatica_enmo", "metric": "empatica_enmo_mean_g", "label": "Empatica ENMO", "device": "Empatica", "construct": "motion"},
    {"signal_stream": "empatica_steps", "metric": "empatica_steps", "label": "Empatica Steps", "device": "Empatica", "construct": "activity"},
    {"signal_stream": "biopac_hr", "metric": "biopac_hr_mean_bpm", "label": "BIOPAC HR", "device": "BIOPAC", "construct": "heart_rate"},
    {"signal_stream": "biopac_eda", "metric": "biopac_eda_mean_uS", "label": "BIOPAC EDA", "device": "BIOPAC", "construct": "eda"},
    {"signal_stream": "biopac_temp", "metric": "biopac_temp_chest_mean_C", "label": "BIOPAC Chest Temperature", "device": "BIOPAC", "construct": "temperature"},
    {"signal_stream": "biopac_temp_thigh", "metric": "biopac_temp_thigh_mean_C", "label": "BIOPAC Thigh Temperature", "device": "BIOPAC", "construct": "temperature_site"},
    {"signal_stream": "biopac_temp_arm", "metric": "biopac_temp_arm_mean_C", "label": "BIOPAC Arm Temperature", "device": "BIOPAC", "construct": "temperature_site"},
    {"signal_stream": "biopac_temp_tibia", "metric": "biopac_temp_tibia_mean_C", "label": "BIOPAC Tibia Temperature", "device": "BIOPAC", "construct": "temperature_site"},
    {"signal_stream": "biopac_bloodflow", "metric": "biopac_bloodflow_mean_bpu", "label": "BIOPAC Blood Flow", "device": "BIOPAC", "construct": "bloodflow"},
    {"signal_stream": "biopac_backscatter", "metric": "biopac_backscatter_mean_percent", "label": "BIOPAC Backscatter", "device": "BIOPAC", "construct": "optical"},
]
FIGURE_SIZE_PRESETS = {
    "timeline": (14.8, 4.6),
    "wide_single": (14.2, 5.8),
    "wide_single_short": (14.8, 5.0),
    "wide_single_tall": (15.2, 6.4),
    "three_panel_row": (14.6, 5.4),
    "three_panel_row_wide": (15.2, 5.8),
    "three_panel_stack": (10.2, 14.0),
    "two_panel_row": (14.6, 6.8),
    "two_panel_row_balanced": (14.2, 6.8),
    "two_by_two": (13.4, 9.4),
    "two_by_two_balanced": (13.2, 9.4),
    "two_by_two_wide": (14.0, 9.2),
    "readiness_grid": (13.8, 8.6),
    "matrix": (13.8, 6.8),
    "matrix_tall": (15.4, 11.2),
    "participant_single": (5.8, 5.2),
}


class ReportWriter:
    def __init__(self, outdir: str | Path, dataset_root: str | Path, config: CLTRConfig):
        self.outdir = Path(outdir)
        self.dataset_root = Path(dataset_root)
        self.config = config
        self.o = config.output
        self._style()

    def write_session_report(self, session_inputs: dict, modalities: list[str] | None = None) -> dict:
        session_id = session_inputs["session_id"]
        root = ensure_dir(self.outdir / self.o.report_dir / self.o.session_dir / session_id)
        figs = ensure_dir(root / self.o.figure_dir)
        narrative_specs, appendix_specs = self._build_session_specs(session_inputs)
        narrative_specs, appendix_specs = self._curate_session_specs(session_inputs, narrative_specs, appendix_specs)
        narrative_specs = self._filter_specs(narrative_specs, modalities)
        appendix_specs = self._filter_specs(appendix_specs, modalities)
        for spec in narrative_specs + appendix_specs:
            spec["display_section"] = spec.get("section", "analyzed")
        saved = self._save_specs(figs, narrative_specs + appendix_specs)
        html_path = root / f"{session_id}_report.html"
        html_path.write_text(self._session_html(session_inputs, narrative_specs, appendix_specs), encoding="utf-8")
        return {
            "session_id": session_id,
            "participant_id": session_inputs["processing_metadata"].get("participant_id"),
            "condition": session_inputs["processing_metadata"].get("condition_code"),
            "html_path": str(html_path),
            "figure_paths": [str(p) for p in saved],
            "figure_specs": [{"code": s["code"], "title": s["title"], "tags": s["tags"], "evidence_score": s["evidence_score"], "section": s.get("section", "analyzed")} for s in narrative_specs + appendix_specs],
            "narrative_codes": [s["code"] for s in narrative_specs],
            "appendix_codes": [s["code"] for s in appendix_specs],
            "lead_label": self._session_story_profile(session_inputs)["lead_label"],
            "headline": self._session_story_profile(session_inputs)["headline"],
            "atlas_tags": self._session_atlas_tags(session_inputs, narrative_specs),
        }

    def write_cohort_report(self, cohort_inputs: dict, modalities: list[str] | None = None) -> dict:
        root = ensure_dir(self.outdir / self.o.report_dir / self.o.cohort_dir)
        figs = ensure_dir(root / self.o.figure_dir)
        narrative_specs, appendix_specs = self._build_cohort_specs(cohort_inputs)
        narrative_specs, appendix_specs = self._curate_cohort_specs(cohort_inputs, narrative_specs, appendix_specs)
        narrative_specs = self._filter_specs(narrative_specs, modalities)
        appendix_specs = self._filter_specs(appendix_specs, modalities)
        for spec in narrative_specs + appendix_specs:
            spec["display_section"] = spec.get("section", "analyzed")
        saved = self._save_specs(figs, narrative_specs + appendix_specs)
        full_html_path = root / "cohort_full_report.html"
        full_html_path.write_text(self._cohort_html(cohort_inputs, narrative_specs, appendix_specs), encoding="utf-8")
        chapter_specs = self._cohort_chapter_specs(cohort_inputs, narrative_specs, appendix_specs)
        chapter_route_map = {chapter["slug"]: self._cohort_chapter_route(chapter) for chapter in chapter_specs}
        chapter_menu_items_html = "".join(
            f"<a href='{html_escape(chapter_route_map[chapter['slug']])}'>{html_escape(chapter['title'].split(':')[-1].strip())}<span>{html_escape(chapter['subtitle'])}</span></a>"
            for chapter in chapter_specs
        )
        chapter_paths: dict[str, str] = {}
        for chapter in chapter_specs:
            chapter_path = root / chapter["filename"]
            canonical_chapter_dir = ensure_dir(root / chapter_route_map[chapter["slug"]].rstrip("/"))
            canonical_chapter_path = canonical_chapter_dir / "index.html"
            canonical_chapter_path.write_text(
                self._cohort_chapter_html(
                    cohort_inputs,
                    chapter["title"],
                    chapter["subtitle"],
                    chapter["specs"],
                    chapter["intro_sections"],
                    chapter["section_intro_map"],
                    chapter_menu_items_html,
                    chapter["chapter_number"],
                    home_href="../../index.html",
                    logo_src="../../../../../cltr/docs/assets/logos/cltr.png",
                    figure_src_prefix="../figures/",
                    sessions_href="../../sessions_report.html",
                ),
                encoding="utf-8",
            )
            chapter_path.write_text(self._redirect_html(f"./{chapter_route_map[chapter['slug']]}"), encoding="utf-8")
            chapter_paths[chapter["slug"]] = str(canonical_chapter_path)
        html_path = root / "index.html"
        html_path.write_text(self._cohort_index_html(cohort_inputs, chapter_specs, chapter_paths, full_html_path), encoding="utf-8")
        (root / COHORT_LEGACY_INDEX_FILENAME).write_text(self._redirect_html("./"), encoding="utf-8")
        return {
            "html_path": str(html_path),
            "figure_paths": [str(p) for p in saved],
            "figure_specs": [{"code": s["code"], "title": s["title"], "tags": s["tags"], "evidence_score": s["evidence_score"], "section": s.get("section", "analyzed")} for s in narrative_specs + appendix_specs],
            "narrative_codes": [s["code"] for s in narrative_specs],
            "appendix_codes": [s["code"] for s in appendix_specs],
            "full_html_path": str(full_html_path),
            "chapter_paths": chapter_paths,
        }

    def write_all_sessions_index(self, manifest: pd.DataFrame, session_reports: list[dict], cohort_report: dict) -> dict:
        root = ensure_dir(self.outdir / self.o.report_dir)
        index_path = root / "index.html"
        sessions_path = root / "sessions_report.html"
        atlas_html = self._atlas_html(manifest, session_reports, cohort_report, sessions_path.name)
        sessions_html = self._all_sessions_html(manifest, session_reports, cohort_report)
        index_path.write_text(atlas_html, encoding="utf-8")
        sessions_path.write_text(sessions_html, encoding="utf-8")
        return {"html_path": str(index_path), "sessions_path": str(sessions_path)}

    def _style(self) -> None:
        plt.rcParams.update(
            {
                "figure.dpi": self.config.reporting.figure_dpi,
                "savefig.facecolor": "white",
                "axes.facecolor": "white",
                "axes.edgecolor": "#cbd5e1",
                "axes.axisbelow": True,
                "axes.grid": True,
                "grid.color": "#eef2f7",
                "grid.linewidth": 0.7,
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.titleweight": "bold",
                "axes.labelsize": 12,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 11,
                "legend.title_fontsize": 11,
                "axes.linewidth": 0.9,
                "xtick.major.size": 5.0,
                "ytick.major.size": 5.0,
                "xtick.major.width": 0.9,
                "ytick.major.width": 0.9,
            }
        )

    def _figsize(self, preset: str) -> tuple[float, float]:
        return FIGURE_SIZE_PRESETS[preset]

    def _shared_report_css(self) -> str:
        ui = REPORT_UI
        return f"""
body {{ margin:0; min-height:100vh; display:flex; flex-direction:column; font-family: Georgia, "Times New Roman", serif; color:#172033; background:radial-gradient(circle at top left,#fff6e8 0%,#eef4ff 52%,#f8fafc 100%); }}
.page {{ width:min(100%, {ui['page_max_width']}); margin:0 auto; padding:24px clamp(16px,2.4vw,28px) 48px; box-sizing:border-box; flex:1 0 auto; }}
.primaryBar {{ position:sticky; top:0; z-index:24; backdrop-filter:blur(16px); background:rgba(248,250,252,0.92); border-bottom:1px solid rgba(148,163,184,0.18); }}
.primaryBarInner {{ width:min(100%, {ui['page_max_width']}); margin:0 auto; padding:12px clamp(16px,2.4vw,28px); display:flex; align-items:center; justify-content:space-between; gap:16px; box-sizing:border-box; }}
.logoLink {{ display:inline-flex; align-items:center; gap:12px; min-height:58px; text-decoration:none; }}
.logoLink:hover {{ transform:translateY(-1px); }}
.logoMark,.logoImage {{ width:58px; height:58px; object-fit:contain; display:block; flex-shrink:0; }}
.logoWordmark {{ display:inline-flex; align-items:center; height:58px; font:700 2.1rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:-0.04em; color:#172033; }}
.secondaryBar {{ position:sticky; top:71px; z-index:23; backdrop-filter:blur(14px); background:rgba(255,255,255,0.78); border-bottom:1px solid rgba(148,163,184,0.16); }}
.secondaryBarInner {{ width:min(100%, {ui['page_max_width']}); margin:0 auto; padding:10px clamp(16px,2.4vw,28px); display:flex; align-items:center; justify-content:space-between; gap:14px; box-sizing:border-box; }}
.secondaryBarMeta {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size:0.84rem; color:#475569; }}
.secondaryBarActions {{ position:relative; display:flex; align-items:center; gap:10px; flex-shrink:0; }}
.secondaryBarType {{ display:inline-flex; align-items:center; gap:8px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#172033; }}
.secondaryBarType::before {{ content:""; width:8px; height:8px; border-radius:999px; background:#1d4ed8; box-shadow:0 0 0 3px rgba(29,78,216,0.12); }}
.reportKind--session .secondaryBarType::before {{ background:#f59e0b; box-shadow:0 0 0 3px rgba(245,158,11,0.14); }}
.reportKind--cohort .secondaryBarType::before {{ background:#06b6d4; box-shadow:0 0 0 3px rgba(6,182,212,0.14); }}
.reportKind--atlas .secondaryBarType::before {{ background:#fb7185; box-shadow:0 0 0 3px rgba(251,113,133,0.14); }}
.reportKind--home .secondaryBarType::before {{ background:#7c3aed; box-shadow:0 0 0 3px rgba(124,58,237,0.14); }}
.secondaryBarText {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.mastheadActions {{ display:flex; align-items:center; justify-content:flex-end; gap:12px; flex:1 1 auto; min-width:0; }}
.menuWrap {{ position:relative; display:flex; align-items:center; }}
.socialLinks {{ display:flex; align-items:center; justify-content:flex-end; gap:10px; flex-wrap:wrap; min-width:0; }}
.socialLink {{ display:inline-flex; align-items:center; justify-content:center; min-height:44px; padding:0 16px; border-radius:999px; text-decoration:none; color:#172033; background:linear-gradient(180deg,rgba(255,255,255,0.96) 0%,rgba(255,247,237,0.96) 100%); border:1px solid rgba(251,146,60,0.28); box-shadow:0 12px 28px rgba(23,32,51,0.08); font:700 0.92rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0.01em; }}
.socialLink:hover {{ background:#ffffff; border-color:#fb923c; box-shadow:0 16px 34px rgba(23,32,51,0.12); transform:translateY(-1px); }}
.socialLink.isDisabled {{ pointer-events:none; opacity:0.58; }}
.menuPanel .socialLinks {{ display:grid; gap:8px; }}
.menuPanel .socialLink {{ width:100%; min-height:40px; justify-content:flex-start; padding:10px 12px; border-radius:14px; font-size:0.82rem; line-height:1.2; box-sizing:border-box; box-shadow:0 10px 20px rgba(23,32,51,0.12); background:linear-gradient(135deg,rgba(255,255,255,0.98) 0%,rgba(255,243,224,0.98) 52%,rgba(255,232,214,0.98) 100%); border:1px solid rgba(251,146,60,0.34); }}
.themeToggle {{ appearance:none; width:44px; height:44px; border-radius:999px; border:1px solid rgba(148,163,184,0.24); background:linear-gradient(180deg,rgba(255,255,255,0.96) 0%,rgba(255,247,237,0.96) 100%); color:#172033; box-shadow:0 12px 28px rgba(23,32,51,0.08); cursor:pointer; display:inline-flex; align-items:center; justify-content:center; flex-shrink:0; }}
.themeToggle:hover {{ background:#ffffff; border-color:#fb923c; box-shadow:0 16px 34px rgba(23,32,51,0.12); transform:translateY(-1px); }}
.themeToggleIconDark,.themeToggleIconLight {{ font-size:1.05rem; line-height:1; }}
.themeToggleIconLight {{ display:none; }}
body.theme-dark {{ color:#e5edf7; background:radial-gradient(circle at top left,#0f172a 0%,#111827 52%,#020617 100%); }}
body.theme-dark .primaryBar {{ background:rgba(15,23,42,0.9); border-bottom-color:rgba(71,85,105,0.42); }}
body.theme-dark .secondaryBar {{ background:rgba(15,23,42,0.82); border-bottom-color:rgba(71,85,105,0.32); }}
body.theme-dark .logoWordmark,body.theme-dark .secondaryBarType,body.theme-dark .title,body.theme-dark .sectionTitle,body.theme-dark th {{ color:#f8fafc; }}
body.theme-dark .secondaryBarText,body.theme-dark .label,body.theme-dark .meta,body.theme-dark .figureMeta,body.theme-dark .caption,body.theme-dark .subtitle,body.theme-dark .takeawayText,body.theme-dark td,body.theme-dark .nav a span {{ color:#cbd5e1; }}
body.theme-dark .panel,body.theme-dark .figurePanel,body.theme-dark .tablePanel,body.theme-dark .card,body.theme-dark .takeawayItem,body.theme-dark .takeawayLead {{ background:rgba(15,23,42,0.88); border-color:rgba(71,85,105,0.38); box-shadow:0 18px 44px rgba(2,6,23,0.38); }}
body.theme-dark .socialLink,body.theme-dark .themeToggle,body.theme-dark .menuButton {{ color:#f8fafc; background:linear-gradient(180deg,rgba(30,41,59,0.96) 0%,rgba(15,23,42,0.96) 100%); border-color:rgba(71,85,105,0.5); }}
body.theme-dark .menuPanel .socialLink {{ background:linear-gradient(135deg,rgba(30,41,59,0.98) 0%,rgba(37,99,235,0.34) 58%,rgba(15,23,42,0.98) 100%); border-color:rgba(96,165,250,0.34); box-shadow:0 10px 22px rgba(2,6,23,0.34); }}
body.theme-dark .nav a {{ color:#f8fafc; background:rgba(30,41,59,0.96); border-color:rgba(71,85,105,0.44); box-shadow:inset 0 0 0 4px rgba(15,23,42,0.75); }}
body.theme-dark table th {{ background:#1e293b; }}
body.theme-dark .figureImage,body.theme-dark .lightbox img {{ background:#e2e8f0; }}
body.theme-dark .themeToggleIconDark {{ display:none; }}
body.theme-dark .themeToggleIconLight {{ display:inline; }}
.menuButton {{ appearance:none; border:1px solid rgba(148,163,184,0.28); background:linear-gradient(180deg,rgba(255,255,255,0.96) 0%,rgba(255,247,237,0.96) 100%); color:#172033; border-radius:999px; min-height:44px; padding:0 14px; display:inline-flex; align-items:center; gap:10px; font:700 0.82rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0.04em; cursor:pointer; box-shadow:0 12px 28px rgba(23,32,51,0.08); }}
.menuButton:hover {{ background:#ffffff; border-color:#fb923c; box-shadow:0 16px 34px rgba(23,32,51,0.12); transform:translateY(-1px); }}
.menuButtonBars {{ display:grid; gap:3px; }}
.menuButtonBars span {{ display:block; width:14px; height:2px; border-radius:999px; background:currentColor; }}
.menuPanel {{ position:absolute; right:0; top:calc(100% + 10px); width:min(220px, calc(100vw - 32px)); padding:0; background:transparent; border:0; border-radius:0; box-shadow:none; backdrop-filter:none; display:none; }}
.menuPanel.open {{ display:grid; gap:10px; }}
.menuTitle {{ display:none; }}
.hero {{ display:grid; grid-template-columns:minmax(0,1.2fr) minmax(320px,0.8fr); gap:{ui['hero_gap']}; align-items:end; }}
.panel,.figurePanel,.tablePanel {{ background:rgba(255,255,255,0.9); border:{ui['panel_border']}; border-radius:{ui['panel_radius']}; box-shadow:{ui['panel_shadow']}; padding:{ui['panel_padding']}; backdrop-filter:blur(8px); }}
.heroLead,.heroSide {{ position:relative; overflow:hidden; }}
.heroLead::before,.heroSide::before {{ content:""; position:absolute; inset:0; opacity:1; z-index:0; }}
.heroLead > *,.heroSide > * {{ position:relative; z-index:1; }}
.reportKind--session .heroLead {{ background:linear-gradient(135deg,#fff0dc 0%,#ffd59e 42%,#f59e0b 100%); border-color:rgba(245,158,11,0.32); }}
.reportKind--session .heroLead::before {{ background:radial-gradient(circle at top right,rgba(255,255,255,0.8) 0%,rgba(255,255,255,0) 48%); }}
.reportKind--session .heroSide {{ background:linear-gradient(135deg,#fff8ef 0%,#ffe6c7 100%); border-color:rgba(251,191,36,0.28); }}
.reportKind--cohort .heroLead {{ background:linear-gradient(135deg,#172033 0%,#1d4ed8 55%,#06b6d4 100%); border-color:rgba(191,219,254,0.45); color:#eff6ff; }}
.reportKind--cohort .heroLead::before {{ background:radial-gradient(circle at top right,rgba(255,255,255,0.24) 0%,rgba(255,255,255,0) 42%); }}
.reportKind--cohort .heroSide {{ background:linear-gradient(135deg,#e0f2fe 0%,#dbeafe 100%); border-color:rgba(147,197,253,0.3); }}
.reportKind--cohort .heroLead .eyebrow {{ color:#dbeafe; }}
.reportKind--cohort .heroLead .title,.reportKind--cohort .heroLead .subtitle,.reportKind--cohort .heroLead .label,.reportKind--cohort .heroLead .value {{ color:#f8fafc; }}
.reportKind--cohort .heroLead .card {{ background:rgba(255,255,255,0.14); border-color:rgba(219,234,254,0.3); }}
.eyebrow {{ font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size:{ui['eyebrow_size']}; letter-spacing:0.18em; text-transform:uppercase; color:#9a3412; margin-bottom:8px; }}
.title {{ font-size:{ui['title_size']}; font-weight:700; letter-spacing:-0.04em; margin:0 0 8px; }}
.subtitle {{ color:#52607a; line-height:{ui['subtitle_line_height']}; margin:0; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:{ui['cards_gap']}; margin-top:18px; }}
.card {{ background:linear-gradient(180deg,#fffdf8 0%,#f8fbff 100%); border:1px solid #e2e8f0; border-radius:{ui['card_radius']}; padding:12px; }}
.label {{ font-size:0.78rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; }}
.value {{ font-size:1.05rem; font-weight:700; margin-top:6px; }}
.takeawayPanel {{ display:grid; gap:14px; }}
.takeawayHeader {{ display:flex; align-items:center; justify-content:space-between; gap:12px; }}
.takeawayHeader h2 {{ margin:0; font-size:1.08rem; }}
.takeawayBadge {{ display:inline-flex; align-items:center; border:1px solid rgba(251,191,36,0.38); background:rgba(255,255,255,0.72); color:#9a3412; border-radius:999px; padding:6px 10px; font:700 0.72rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0.08em; text-transform:uppercase; }}
.takeawayLead {{ padding:16px 18px; border-radius:18px; background:linear-gradient(180deg,rgba(255,255,255,0.92) 0%,rgba(255,247,237,0.92) 100%); border:1px solid rgba(251,191,36,0.26); box-shadow:inset 0 1px 0 rgba(255,255,255,0.7); }}
.takeawayLeadLabel {{ margin:0 0 8px; color:#9a3412; font:700 0.72rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0.12em; text-transform:uppercase; }}
.takeawayLeadText {{ margin:0; font-size:1.02rem; line-height:1.55; color:#172033; }}
.takeawayList {{ display:grid; gap:10px; }}
.takeawayItem {{ display:grid; grid-template-columns:auto minmax(0,1fr); gap:12px; align-items:start; padding:12px 14px; border-radius:16px; background:rgba(255,255,255,0.7); border:1px solid rgba(251,191,36,0.18); }}
.takeawayIndex {{ width:28px; height:28px; border-radius:999px; display:flex; align-items:center; justify-content:center; background:#fff7ed; border:1px solid rgba(251,191,36,0.3); color:#9a3412; font:700 0.78rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.takeawayText {{ margin:1px 0 0; color:#334155; line-height:1.55; }}
.reportShell {{ display:grid; grid-template-columns:minmax(0,1fr); gap:30px; align-items:start; margin-top:28px; }}
.stack {{ display:grid; gap:{ui['stack_gap']}; min-width:0; }}
.nav {{ display:grid; gap:14px; }}
.navTitle {{ margin:0; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size:0.78rem; letter-spacing:0.14em; text-transform:uppercase; color:#64748b; }}
.navList {{ display:grid; grid-template-columns:1fr; gap:10px; }}
.nav a {{ width:100%; min-height:44px; text-decoration:none; color:#172033; background:rgba(255,247,237,0.95); border:1px solid #fed7aa; border-radius:14px; display:grid; grid-template-columns:auto minmax(0,1fr); align-items:start; gap:10px; padding:10px 12px; font-size:0.72rem; font-weight:700; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height:1.2; box-shadow:inset 0 0 0 4px rgba(255,255,255,0.96); box-sizing:border-box; }}
.nav a:hover {{ background:#ffedd5; border-color:#fdba74; }}
.nav a span {{ display:block; font-size:0.78rem; font-weight:500; color:#475569; overflow-wrap:anywhere; }}
.sectionBlock {{ display:grid; gap:22px; padding-top:14px; border-top:2px solid #e2e8f0; width:100%; max-width:100%; min-width:0; box-sizing:border-box; }}
.sectionTitle {{ margin:0; font-size:{ui['section_title_size']}; color:#172033; letter-spacing:0.01em; }}
.figureSection {{ display:grid; gap:12px; margin:12px 0 26px; width:100%; max-width:100%; min-width:0; box-sizing:border-box; }}
.figureSectionTitle {{ font-size:1.02rem; letter-spacing:0.02em; color:#52607a; margin:0 0 2px; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.tableGrid {{ display:grid; grid-template-columns:1fr; gap:{ui['table_gap']}; margin:24px 0; }}
.chapterGrid {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:{ui['table_gap']}; margin:24px 0; }}
.tablePanel {{ max-width:100%; min-width:0; overflow:hidden; }}
.chapterLinkCard {{ display:block; text-decoration:none; color:inherit; }}
.chapterLinkCard:hover {{ transform:translateY(-1px); }}
.chapterLinkCard .tablePanel {{ height:100%; cursor:pointer; transition:transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease; overflow:visible; }}
.chapterLinkCard:hover .tablePanel {{ border-color:#93c5fd; box-shadow:0 22px 48px rgba(23,32,51,0.14); }}
.chapterCardPanel {{ position:relative; display:grid; gap:14px; background:linear-gradient(180deg,rgba(255,255,255,0.98) 0%,rgba(239,246,255,0.96) 100%); }}
.chapterCardPanel::before {{ content:""; position:absolute; inset:0 0 auto 0; height:6px; border-radius:{ui['panel_radius']} {ui['panel_radius']} 0 0; background:linear-gradient(90deg,#172033 0%,#1d4ed8 58%,#06b6d4 100%); }}
.chapterCardHeader {{ display:flex; align-items:start; justify-content:space-between; gap:16px; padding-top:10px; }}
.chapterCardTitleGroup {{ display:grid; gap:6px; }}
.chapterCardHeading {{ margin:0; font-size:1.25rem; line-height:1.2; letter-spacing:-0.02em; }}
.chapterCardKicker {{ display:inline-flex; align-items:center; gap:8px; color:#1d4ed8; font:700 0.75rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0.08em; text-transform:uppercase; }}
.chapterCardKicker::before {{ content:""; width:10px; height:10px; border-radius:999px; background:#06b6d4; box-shadow:0 0 0 4px rgba(6,182,212,0.14); }}
.chapterCardBadge {{ display:inline-flex; align-items:center; min-height:36px; padding:0 12px; border-radius:999px; border:1px solid rgba(29,78,216,0.18); background:rgba(255,255,255,0.84); color:#172033; font:700 0.8rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; white-space:nowrap; }}
.chapterCardDesc {{ margin:0; color:#334155; line-height:1.65; font-size:0.98rem; }}
.chapterMetaGrid {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:10px; }}
.chapterMetaItem {{ border:1px solid rgba(148,163,184,0.18); border-radius:16px; background:rgba(255,255,255,0.72); padding:12px 14px; }}
.chapterMetaLabel {{ color:#64748b; font:700 0.72rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0.08em; text-transform:uppercase; }}
.chapterMetaValue {{ margin-top:6px; color:#172033; line-height:1.45; font-size:0.92rem; }}
.chapterOpenRow {{ display:flex; align-items:center; justify-content:space-between; gap:12px; padding-top:4px; }}
.chapterOpenHint {{ color:#475569; font:600 0.82rem/1.4 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.chapterOpenCta {{ display:inline-flex; align-items:center; gap:10px; color:#172033; font:700 0.9rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.chapterOpenCta::before {{ content:""; width:12px; height:12px; border-radius:999px; border:3px solid #1d4ed8; box-shadow:inset 0 0 0 3px #ffffff; background:#06b6d4; }}
.radioCta {{ display:inline-flex; align-items:center; gap:12px; margin-top:14px; padding:12px 18px; border-radius:999px; border:1px solid rgba(191,219,254,0.54); background:linear-gradient(180deg,rgba(255,255,255,0.24) 0%,rgba(255,255,255,0.12) 100%); color:#f8fafc; text-decoration:none; font:700 0.92rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; box-shadow:0 16px 34px rgba(8,47,73,0.18); }}
.radioCta::before {{ content:""; width:16px; height:16px; border-radius:999px; border:4px solid rgba(255,255,255,0.9); box-shadow:0 0 0 3px rgba(191,219,254,0.34); background:#f59e0b; }}
.radioCta:hover {{ border-color:rgba(255,255,255,0.72); background:linear-gradient(180deg,rgba(255,255,255,0.32) 0%,rgba(255,255,255,0.16) 100%); }}
body.theme-dark .chapterCardPanel {{ background:linear-gradient(180deg,rgba(15,23,42,0.96) 0%,rgba(15,23,42,0.9) 100%); }}
body.theme-dark .chapterCardBadge,body.theme-dark .chapterMetaItem {{ background:rgba(15,23,42,0.84); border-color:rgba(71,85,105,0.38); }}
body.theme-dark .chapterCardDesc,body.theme-dark .chapterMetaValue,body.theme-dark .chapterOpenHint {{ color:#cbd5e1; }}
body.theme-dark .chapterCardKicker {{ color:#93c5fd; }}
body.theme-dark .radioCta {{ border-color:rgba(147,197,253,0.44); }}
.heroActions {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-top:20px; }}
.figurePanel .dataTablePanel {{ width:100%; max-width:100%; margin:8px 0 0; padding:0; border:none; background:transparent; box-shadow:none; box-sizing:border-box; }}
.figurePanel .dataTablePanel h3 {{ margin:0 0 8px; font-size:0.9rem; line-height:1.3; color:#334155; }}
.dataTablePanel .tableScroll {{ display:block; width:100%; max-width:100%; overflow-x:scroll; overflow-y:hidden; scrollbar-gutter:stable; scrollbar-width:auto; }}
.figurePanel .dataTablePanel .tableScroll {{ margin:0; padding:10px 12px; border:1px solid #d7dee9; border-radius:12px; background:#ffffff; box-shadow:inset 0 1px 0 rgba(255,255,255,0.8); max-width:100%; box-sizing:border-box; }}
.dataTablePanel .tableScroll::-webkit-scrollbar {{ height:12px; }}
.dataTablePanel .tableScroll::-webkit-scrollbar-track {{ background:#e2e8f0; border-radius:999px; }}
.dataTablePanel .tableScroll::-webkit-scrollbar-thumb {{ background:#94a3b8; border-radius:999px; border:2px solid #e2e8f0; }}
.dataTablePanel .tableScroll::-webkit-scrollbar-thumb:hover {{ background:#64748b; }}
.dataTablePanel table {{ width:max-content; min-width:100%; max-width:none; border-collapse:collapse; font-size:0.82rem; line-height:1.35; table-layout:auto; }}
.dataTablePanel th,.dataTablePanel td {{ border-bottom:1px solid #e2e8f0; padding:6px 8px; text-align:left; vertical-align:top; white-space:normal; overflow-wrap:break-word; word-break:normal; max-width:16rem; }}
.dataTablePanel th {{ color:#334155; background:#f8fafc; }}
.dataTablePanel th.col-prompt_support,.dataTablePanel td.col-prompt_support {{ width:10rem; min-width:10rem; }}
.dataTablePanel th.col-minute_occupancy_fraction,.dataTablePanel td.col-minute_occupancy_fraction {{ width:7rem; min-width:7rem; }}
.dataTablePanel th.col-minute_occupancy_reading,.dataTablePanel td.col-minute_occupancy_reading,.dataTablePanel th.col-coverage_reading,.dataTablePanel td.col-coverage_reading {{ width:16rem; min-width:16rem; }}
body.theme-dark .figurePanel .dataTablePanel {{ background:transparent; border-color:transparent; }}
body.theme-dark .figurePanel .dataTablePanel h3 {{ color:#e2e8f0; }}
body.theme-dark .figurePanel .dataTablePanel .tableScroll {{ background:rgba(15,23,42,0.72); border-color:rgba(71,85,105,0.42); }}
body.theme-dark .dataTablePanel .tableScroll {{ scrollbar-color:#64748b rgba(30,41,59,0.9); }}
body.theme-dark .dataTablePanel .tableScroll::-webkit-scrollbar-track {{ background:rgba(30,41,59,0.9); }}
body.theme-dark .dataTablePanel .tableScroll::-webkit-scrollbar-thumb {{ background:#64748b; border-color:rgba(30,41,59,0.9); }}
body.theme-dark .dataTablePanel .tableScroll::-webkit-scrollbar-thumb:hover {{ background:#94a3b8; }}
.figurePanel {{ width:100%; max-width:100%; min-width:0; box-sizing:border-box; overflow:hidden; }}
.figurePanel.hidden {{ display:none; }}
.figurePanel h2 {{ margin:0 0 12px; font-size:{ui['figure_title_size']}; line-height:1.25; white-space:normal; overflow:visible; text-overflow:clip; }}
.figureImage {{ width:100%; height:auto; border-radius:{ui['image_radius']}; border:1px solid #dbeafe; background:white; cursor:zoom-in; }}
.responsiveFigure {{ width:100%; max-width:100%; min-width:0; overflow:hidden; }}
.responsiveFigure > * {{ max-width:100% !important; }}
.responsiveFigure .js-plotly-plot,.responsiveFigure .plot-container,.responsiveFigure .svg-container {{ width:100% !important; max-width:100% !important; }}
.meta {{ color:#64748b; font-size:0.86rem; margin-bottom:10px; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.figureMeta {{ margin:10px 0 0; color:#64748b; font-size:0.82rem; line-height:1.45; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.caption {{ margin:10px 0 0; color:#334155; line-height:1.55; font-size:0.95rem; }}
.lightbox {{ position:fixed; inset:0; background:rgba(15,23,42,0.86); display:none; align-items:center; justify-content:center; padding:30px; z-index:30; }}
.lightbox.open {{ display:flex; }}
.lightbox img {{ max-width:95vw; max-height:90vh; background:white; border-radius:{ui['card_radius']}; }}
.copyrightNote {{ width:min(100%, {ui['page_max_width']}); margin:0 auto; padding:0 clamp(16px,2.4vw,28px) 18px; box-sizing:border-box; display:flex; justify-content:center; align-items:center; text-align:center; color:#64748b; font:500 0.84rem/1.5 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
body.theme-dark .copyrightNote {{ color:#94a3b8; }}
@media (max-width:1280px) {{ .reportShell {{ grid-template-columns:1fr; }} }}
@media (max-width:{ui['mobile_breakpoint']}) {{ .primaryBarInner,.secondaryBarInner,.hero,.tableGrid,.chapterGrid,.chapterMetaGrid {{ grid-template-columns:1fr; }} .primaryBarInner,.secondaryBarInner {{ display:grid; padding:12px 20px; }} .mastheadActions,.secondaryBarActions {{ justify-content:space-between; }} .secondaryBarText {{ white-space:normal; }} .page {{ padding:20px 16px 40px; }} .nav a {{ grid-template-columns:auto minmax(0,1fr); }} .takeawayHeader {{ align-items:start; }} .chapterCardHeader,.chapterOpenRow {{ grid-template-columns:1fr; display:grid; }} .figurePanel .dataTablePanel {{ width:100%; margin:6px 0 0; }} .figurePanel .dataTablePanel .tableScroll {{ padding:8px 10px; }} }}
@media (max-width:860px) {{ .primaryBarInner {{ flex-wrap:wrap; padding:12px 20px; }} .mastheadActions {{ width:100%; justify-content:flex-end; }} }}
@media (max-width:640px) {{ .mastheadActions {{ width:auto; }} .menuPanel {{ right:0; left:auto; width:min(280px, calc(100vw - 24px)); }} .logoMark,.logoImage {{ width:52px; height:52px; }} .logoWordmark {{ height:52px; font-size:1.9rem; }} }}
""".strip()

    def _shared_index_css(self) -> str:
        ui = REPORT_UI
        return f"""
body {{ margin:0; min-height:100vh; display:flex; flex-direction:column; font-family: Georgia, "Times New Roman", serif; color:#172033; background:radial-gradient(circle at top left,#fff6e8 0%,#eef4ff 52%,#f8fafc 100%); }}
.page {{ width:min(100%, {ui['page_max_width']}); margin:0 auto; padding:{ui['index_page_padding']}; box-sizing:border-box; flex:1 0 auto; }}
.primaryBar {{ position:sticky; top:0; z-index:24; backdrop-filter:blur(16px); background:rgba(248,250,252,0.92); border-bottom:1px solid rgba(148,163,184,0.18); }}
.primaryBarInner {{ width:min(100%, {ui['page_max_width']}); margin:0 auto; padding:12px 28px; display:flex; align-items:center; justify-content:space-between; gap:16px; box-sizing:border-box; }}
.logoLink {{ display:inline-flex; align-items:center; gap:12px; min-height:58px; text-decoration:none; }}
.logoLink:hover {{ transform:translateY(-1px); }}
.logoMark,.logoImage {{ width:58px; height:58px; object-fit:contain; display:block; flex-shrink:0; }}
.logoWordmark {{ display:inline-flex; align-items:center; height:58px; font:700 2.1rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:-0.04em; color:#172033; }}
.secondaryBar {{ position:sticky; top:71px; z-index:23; backdrop-filter:blur(14px); background:rgba(255,255,255,0.78); border-bottom:1px solid rgba(148,163,184,0.16); }}
.secondaryBarInner {{ width:min(100%, {ui['page_max_width']}); margin:0 auto; padding:10px 28px; display:flex; align-items:center; justify-content:space-between; gap:14px; box-sizing:border-box; }}
.secondaryBarMeta {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size:0.84rem; color:#475569; }}
.secondaryBarActions {{ position:relative; display:flex; align-items:center; gap:10px; flex-shrink:0; }}
.secondaryBarType {{ display:inline-flex; align-items:center; gap:8px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#172033; }}
.secondaryBarType::before {{ content:""; width:8px; height:8px; border-radius:999px; background:#fb7185; box-shadow:0 0 0 3px rgba(251,113,133,0.14); }}
.reportKind--home .secondaryBarType::before {{ background:#7c3aed; box-shadow:0 0 0 3px rgba(124,58,237,0.14); }}
.secondaryBarText {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.socialLinks {{ display:flex; align-items:center; justify-content:flex-end; gap:10px; flex-wrap:wrap; min-width:0; }}
.socialLink {{ display:inline-flex; align-items:center; justify-content:center; min-height:44px; padding:0 16px; border-radius:999px; text-decoration:none; color:#172033; background:linear-gradient(180deg,rgba(255,255,255,0.96) 0%,rgba(255,247,237,0.96) 100%); border:1px solid rgba(251,146,60,0.28); box-shadow:0 12px 28px rgba(23,32,51,0.08); font:700 0.92rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0.01em; }}
.socialLink:hover {{ background:#ffffff; border-color:#fb923c; box-shadow:0 16px 34px rgba(23,32,51,0.12); transform:translateY(-1px); }}
.socialLink.isDisabled {{ pointer-events:none; opacity:0.58; }}
.menuPanel .socialLinks {{ display:grid; gap:8px; }}
.menuPanel .socialLink {{ width:100%; min-height:40px; justify-content:flex-start; padding:10px 12px; border-radius:14px; font-size:0.82rem; line-height:1.2; box-sizing:border-box; box-shadow:0 10px 20px rgba(23,32,51,0.12); background:linear-gradient(135deg,rgba(255,255,255,0.98) 0%,rgba(255,243,224,0.98) 52%,rgba(255,232,214,0.98) 100%); border:1px solid rgba(251,146,60,0.34); }}
.themeToggle {{ appearance:none; width:44px; height:44px; border-radius:999px; border:1px solid rgba(148,163,184,0.24); background:linear-gradient(180deg,rgba(255,255,255,0.96) 0%,rgba(255,247,237,0.96) 100%); color:#172033; box-shadow:0 12px 28px rgba(23,32,51,0.08); cursor:pointer; display:inline-flex; align-items:center; justify-content:center; flex-shrink:0; }}
.themeToggle:hover {{ background:#ffffff; border-color:#fb923c; box-shadow:0 16px 34px rgba(23,32,51,0.12); transform:translateY(-1px); }}
.themeToggleIconDark,.themeToggleIconLight {{ font-size:1.05rem; line-height:1; }}
.themeToggleIconLight {{ display:none; }}
body.theme-dark {{ color:#e5edf7; background:radial-gradient(circle at top left,#0f172a 0%,#111827 52%,#020617 100%); }}
body.theme-dark .primaryBar {{ background:rgba(15,23,42,0.9); border-bottom-color:rgba(71,85,105,0.42); }}
body.theme-dark .secondaryBar {{ background:rgba(15,23,42,0.82); border-bottom-color:rgba(71,85,105,0.32); }}
body.theme-dark .logoWordmark,body.theme-dark .secondaryBarType,body.theme-dark .title,body.theme-dark .heroCta .subtitle,body.theme-dark .heroCta .eyebrow {{ color:#f8fafc; }}
body.theme-dark .secondaryBarText,body.theme-dark .subtitle,body.theme-dark .tagLine,body.theme-dark .heroStatement,body.theme-dark .heroFactValue {{ color:#cbd5e1; }}
body.theme-dark .panel,body.theme-dark .sessionCard,body.theme-dark .heroFact {{ background:rgba(15,23,42,0.88); border-color:rgba(71,85,105,0.38); box-shadow:0 18px 44px rgba(2,6,23,0.38); }}
body.theme-dark .heroIntro,body.theme-dark .heroCta {{ background:linear-gradient(135deg,#0f172a 0%,#1e293b 52%,#334155 100%); border-color:rgba(71,85,105,0.4); }}
body.theme-dark .gatewayCard {{ background:linear-gradient(135deg,#0f172a 0%,#1e293b 52%,#334155 100%); border-color:rgba(71,85,105,0.4); }}
body.theme-dark .socialLink,body.theme-dark .themeToggle,body.theme-dark .menuButton {{ color:#f8fafc; background:linear-gradient(180deg,rgba(30,41,59,0.96) 0%,rgba(15,23,42,0.96) 100%); border-color:rgba(71,85,105,0.5); }}
body.theme-dark .menuPanel .socialLink {{ background:linear-gradient(135deg,rgba(30,41,59,0.98) 0%,rgba(37,99,235,0.34) 58%,rgba(15,23,42,0.98) 100%); border-color:rgba(96,165,250,0.34); box-shadow:0 10px 22px rgba(2,6,23,0.34); }}
body.theme-dark .nav a {{ color:#f8fafc; background:rgba(30,41,59,0.96); border-color:rgba(71,85,105,0.44); box-shadow:inset 0 0 0 4px rgba(15,23,42,0.75); }}
body.theme-dark .themeToggleIconDark {{ display:none; }}
body.theme-dark .themeToggleIconLight {{ display:inline; }}
.mastheadActions {{ display:flex; align-items:center; justify-content:flex-end; gap:12px; flex:1 1 auto; min-width:0; }}
.menuWrap {{ position:relative; display:flex; align-items:center; }}
.menuButton {{ appearance:none; border:1px solid rgba(148,163,184,0.28); background:linear-gradient(180deg,rgba(255,255,255,0.96) 0%,rgba(255,247,237,0.96) 100%); color:#172033; border-radius:999px; min-height:44px; padding:0 14px; display:inline-flex; align-items:center; gap:10px; font:700 0.82rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0.04em; cursor:pointer; box-shadow:0 12px 28px rgba(23,32,51,0.08); }}
.menuButton:hover {{ background:#ffffff; border-color:#fb923c; box-shadow:0 16px 34px rgba(23,32,51,0.12); transform:translateY(-1px); }}
.menuButtonBars {{ display:grid; gap:3px; }}
.menuButtonBars span {{ display:block; width:14px; height:2px; border-radius:999px; background:currentColor; }}
.menuPanel {{ position:absolute; right:0; top:calc(100% + 10px); width:min(220px, calc(100vw - 32px)); padding:0; background:transparent; border:0; border-radius:0; box-shadow:none; backdrop-filter:none; display:none; }}
.menuPanel.open {{ display:grid; gap:10px; }}
.menuTitle {{ display:none; }}
.nav {{ display:grid; gap:14px; }}
.navList {{ display:grid; grid-template-columns:1fr; gap:10px; }}
.nav a {{ width:100%; min-height:44px; text-decoration:none; color:#172033; background:rgba(255,247,237,0.95); border:1px solid #fed7aa; border-radius:14px; display:grid; grid-template-columns:auto minmax(0,1fr); align-items:start; gap:10px; padding:10px 12px; font-size:0.72rem; font-weight:700; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height:1.2; box-shadow:inset 0 0 0 4px rgba(255,255,255,0.96); box-sizing:border-box; }}
.nav a:hover {{ background:#ffedd5; border-color:#fdba74; }}
.nav a span {{ display:block; font-size:0.78rem; font-weight:500; color:#475569; overflow-wrap:anywhere; }}
.hero,.grid {{ display:grid; gap:22px; }}
.hero {{ grid-template-columns:{ui['index_hero_columns']}; align-items:end; }}
.panel,.sessionCard {{ background:rgba(255,255,255,0.9); border:{ui['panel_border']}; border-radius:{ui['panel_radius']}; box-shadow:{ui['panel_shadow']}; padding:{ui['panel_padding_index']}; backdrop-filter:blur(8px); }}
.grid {{ grid-template-columns:{ui['index_grid_columns']}; margin-top:28px; }}
.eyebrow {{ font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size:{ui['eyebrow_size']}; letter-spacing:0.18em; text-transform:uppercase; color:#9a3412; margin-bottom:8px; }}
.title {{ font-size:{ui['index_title_size']}; font-weight:700; letter-spacing:-0.04em; margin:0 0 8px; }}
.subtitle,.tagLine {{ color:#52607a; line-height:1.6; }}
.tagLine {{ font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size:0.88rem; }}
.heroIntro {{ background:linear-gradient(135deg,#fef3c7 0%,#fed7aa 45%,#fb7185 100%); border-color:rgba(251,113,133,0.28); }}
.heroCta {{ justify-self:end; max-width:320px; background:linear-gradient(135deg,#172033 0%,#1d4ed8 55%,#06b6d4 100%); color:#f8fafc; border:1px solid rgba(191,219,254,0.45); }}
.heroCta .eyebrow {{ color:#dbeafe; }}
.heroCta .subtitle {{ color:rgba(239,246,255,0.92); }}
.heroIntro .subtitle {{ color:#7c2d12; }}
.gatewayGrid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:22px; width:100%; margin-top:22px; }}
.gatewayCard {{ min-height:100%; background:linear-gradient(135deg,#172033 0%,#1d4ed8 55%,#06b6d4 100%); color:#f8fafc; border:1px solid rgba(191,219,254,0.45); text-decoration:none; display:grid; gap:18px; align-content:start; }}
.gatewayCard:hover {{ transform:translateY(-2px); box-shadow:0 22px 44px rgba(23,32,51,0.16); }}
.gatewayCard .eyebrow {{ color:#dbeafe; }}
.gatewayCard .subtitle,.gatewayCard p {{ color:rgba(239,246,255,0.92); }}
.gatewayMeta {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
.gatewayFact {{ background:rgba(255,255,255,0.12); border:1px solid rgba(219,234,254,0.24); border-radius:14px; padding:10px 12px; }}
.gatewayFactLabel {{ font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:#dbeafe; }}
.gatewayFactValue {{ margin-top:4px; color:#f8fafc; line-height:1.45; }}
.gatewayCta {{ display:inline-flex; align-items:center; gap:10px; color:#f8fafc; font:700 0.92rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.gatewayCta::before {{ content:""; width:12px; height:12px; border-radius:999px; border:3px solid rgba(255,255,255,0.92); box-shadow:inset 0 0 0 3px rgba(6,182,212,0.9); background:#f59e0b; }}
.heroSticky {{ position:sticky; top:18px; align-self:start; }}
.heroMeta {{ display:grid; gap:14px; margin-top:20px; }}
.heroStatement {{ font-size:1rem; line-height:1.65; color:#4a1d0d; max-width:58ch; }}
.heroFacts {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
.heroFact {{ background:rgba(255,255,255,0.42); border:1px solid rgba(255,255,255,0.45); border-radius:14px; padding:10px 12px; }}
.heroFactLabel {{ font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:#9a3412; }}
.heroFactValue {{ margin-top:4px; color:#4a1d0d; line-height:1.45; }}
.sessionCard {{ position:relative; overflow:hidden; display:grid; gap:12px; align-content:start; }}
.sessionCard::before {{ content:""; position:absolute; inset:0 0 auto 0; height:6px; border-radius:{ui['panel_radius']} {ui['panel_radius']} 0 0; background:linear-gradient(90deg,#172033 0%,#1d4ed8 58%,#06b6d4 100%); }}
.sessionCard > * {{ position:relative; z-index:1; }}
.sessionCard h3 {{ margin:0 0 6px; font-size:1.3rem; }}
.sessionCard p {{ margin:0; }}
.sessionCard .pillLink {{ justify-self:start; margin-top:6px; }}
.sessionTone--bri-mid {{ background:linear-gradient(180deg,rgba(255,251,235,0.96) 0%,rgba(254,240,138,0.54) 100%); border-color:rgba(245,158,11,0.34); }}
.sessionTone--bri-mid::before {{ background:linear-gradient(90deg,#b45309 0%,#f59e0b 55%,#f97316 100%); }}
.sessionTone--bri-mor {{ background:linear-gradient(180deg,rgba(255,247,237,0.98) 0%,rgba(253,186,116,0.5) 100%); border-color:rgba(249,115,22,0.34); }}
.sessionTone--bri-mor::before {{ background:linear-gradient(90deg,#9a3412 0%,#ea580c 55%,#fb7185 100%); }}
.sessionTone--dim-mid {{ background:linear-gradient(180deg,rgba(240,253,250,0.98) 0%,rgba(153,246,228,0.5) 100%); border-color:rgba(13,148,136,0.34); }}
.sessionTone--dim-mid::before {{ background:linear-gradient(90deg,#0f766e 0%,#14b8a6 55%,#22c55e 100%); }}
.sessionTone--dim-mor {{ background:linear-gradient(180deg,rgba(245,243,255,0.98) 0%,rgba(196,181,253,0.52) 100%); border-color:rgba(124,58,237,0.32); }}
.sessionTone--dim-mor::before {{ background:linear-gradient(90deg,#5b21b6 0%,#7c3aed 52%,#2563eb 100%); }}
body.theme-dark .sessionTone--bri-mid {{ background:linear-gradient(180deg,rgba(69,26,3,0.96) 0%,rgba(120,53,15,0.92) 100%); border-color:rgba(251,191,36,0.26); }}
body.theme-dark .sessionTone--bri-mor {{ background:linear-gradient(180deg,rgba(67,20,7,0.96) 0%,rgba(124,45,18,0.92) 100%); border-color:rgba(251,146,60,0.28); }}
body.theme-dark .sessionTone--dim-mid {{ background:linear-gradient(180deg,rgba(4,47,46,0.96) 0%,rgba(17,94,89,0.92) 100%); border-color:rgba(45,212,191,0.28); }}
body.theme-dark .sessionTone--dim-mor {{ background:linear-gradient(180deg,rgba(46,16,101,0.96) 0%,rgba(49,46,129,0.92) 100%); border-color:rgba(167,139,250,0.28); }}
.pillLink {{ display:inline-block; margin-top:12px; padding:10px 14px; background:#172033; color:#f8fafc; text-decoration:none; border-radius:999px; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.heroCta .pillLink {{ background:#f8fafc; color:#172033; }}
.copyrightNote {{ width:min(100%, {ui['page_max_width']}); margin:0 auto; padding:0 clamp(16px,2.4vw,28px) 18px; box-sizing:border-box; display:flex; justify-content:center; align-items:center; text-align:center; color:#64748b; font:500 0.84rem/1.5 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
body.theme-dark .copyrightNote {{ color:#94a3b8; }}
@media (max-width:{ui['index_mobile_breakpoint']}) {{ .primaryBarInner,.secondaryBarInner,.hero,.grid,.heroFacts,.gatewayGrid,.gatewayMeta {{ grid-template-columns:1fr; }} .primaryBarInner,.secondaryBarInner {{ display:grid; padding:12px 20px; }} .mastheadActions,.secondaryBarActions {{ justify-content:space-between; }} .secondaryBarText {{ white-space:normal; }} .heroSticky {{ position:static; }} }}
@media (max-width:860px) {{ .primaryBarInner {{ flex-wrap:wrap; padding:12px 20px; }} .mastheadActions {{ width:100%; justify-content:flex-end; }} }}
@media (max-width:640px) {{ .mastheadActions {{ width:auto; }} .menuPanel {{ right:0; left:auto; width:min(280px, calc(100vw - 24px)); }} .logoMark,.logoImage {{ width:52px; height:52px; }} .logoWordmark {{ height:52px; font-size:1.9rem; }} }}
""".strip()

    def _social_links_html(self) -> str:
        links = [
            ("GitHub", PROJECT_GITHUB_URL),
            ("Zenodo", PROJECT_ZENODO_URL),
            ("Framework", PROJECT_FRAMEWORK_URL),
        ]
        items = []
        for label, url in links:
            href = str(url or "").strip() or "#"
            disabled = href == "#"
            css_class = "socialLink isDisabled" if disabled else "socialLink"
            title = f"{label} link unavailable" if disabled else f"Open {label}"
            if disabled:
                attrs = " aria-disabled='true'"
            else:
                attrs = " target='_blank' rel='noopener noreferrer'"
            items.append(
                f"<a class='{css_class}' href='{html_escape(href)}' title='{html_escape(title)}'{attrs}>"
                f"<span>{html_escape(label)}</span></a>"
            )
        return f"<div class='socialLinks'>{''.join(items)}</div>"

    def _logo_image_html(self, logo_src: str) -> str:
        return f"<img class='logoImage' src='{html_escape(logo_src)}' alt='CLTR logo'/>"

    def _shared_chrome(
        self,
        *,
        home_href: str,
        logo_src: str,
        page_type: str,
        page_meta: str,
        menu_button_id: str,
        menu_panel_id: str,
        menu_label: str,
        menu_title: str,
        menu_items_html: str,
        menu_icon_bars: bool = False,
        show_secondary_bar: bool = True,
        show_menu_button: bool = True,
        secondary_actions_html_before: str = "",
        secondary_actions_html_after: str = "",
    ) -> str:
        menu_html = (
            self._menu_button_html(
                button_id=menu_button_id,
                panel_id=menu_panel_id,
                label=menu_label,
                title=menu_title,
                items_html=menu_items_html,
                icon_bars=menu_icon_bars,
            )
            if show_menu_button
            else ""
        )
        secondary_html = (
            f"<div class='secondaryBar'><div class='secondaryBarInner'><div class='secondaryBarMeta'><span class='secondaryBarType'>{html_escape(page_type)}</span><span class='secondaryBarText'>{html_escape(page_meta)}</span></div><div class='secondaryBarActions'>{secondary_actions_html_before}{menu_html}{secondary_actions_html_after}</div></div></div>"
            if show_secondary_bar
            else ""
        )
        return (
            f"<header class='primaryBar'>"
            f"<div class='primaryBarInner'>"
            f"<a class='logoLink' href='{html_escape(home_href)}' title='Open report index' aria-label='Open report index'>{self._logo_image_html(logo_src)}<span class='logoWordmark'>CLTR</span></a>"
            f"<div class='mastheadActions'>"
            f"<div class='menuWrap'>"
            f"<button class='menuButton' id='siteMenuButton' type='button' aria-expanded='false' aria-controls='siteMenuPanel' aria-label='Open site menu'>"
            f"<span class='menuButtonBars' aria-hidden='true'><span></span><span></span><span></span></span>"
            f"<span>Menu</span>"
            f"</button>"
            f"<div class='menuPanel' id='siteMenuPanel' role='menu' aria-label='Site navigation'>{self._social_links_html()}</div>"
            f"</div>"
            f"<button class='themeToggle' id='themeToggle' type='button' aria-label='Toggle dark mode'><span class='themeToggleIconDark' aria-hidden='true'>◐</span><span class='themeToggleIconLight' aria-hidden='true'>◑</span></button>"
            f"</div>"
            f"</div>"
            f"</header>"
            f"{secondary_html}"
        )

    def _menu_button_html(
        self,
        *,
        button_id: str,
        panel_id: str,
        label: str,
        title: str,
        items_html: str,
        icon_bars: bool = False,
    ) -> str:
        menu_prefix = "<span class='menuButtonBars'><span></span><span></span><span></span></span>" if icon_bars else ""
        return (
            f"<button id='{html_escape(button_id)}' class='menuButton' type='button' aria-expanded='false' "
            f"aria-controls='{html_escape(panel_id)}' aria-label='{html_escape(title)}'>{menu_prefix}<span>{html_escape(label)}</span></button>"
            f"<div id='{html_escape(panel_id)}' class='menuPanel'><h2 class='menuTitle'>{html_escape(title)}</h2>"
            f"<nav class='nav'><div class='navList'>{items_html}</div></nav></div>"
        )

    def _menu_script(self, *, button_id: str, panel_id: str, var_prefix: str) -> str:
        return (
            f"const {var_prefix}Button=document.getElementById('{html_escape(button_id)}'); const {var_prefix}Panel=document.getElementById('{html_escape(panel_id)}');\n"
            f"const close{var_prefix.capitalize()}Menu=()=>{{ if(!{var_prefix}Panel||!{var_prefix}Button) return; {var_prefix}Panel.classList.remove('open'); {var_prefix}Button.setAttribute('aria-expanded','false'); }};\n"
            f"const toggle{var_prefix.capitalize()}Menu=()=>{{ if(!{var_prefix}Panel||!{var_prefix}Button) return; const open={var_prefix}Panel.classList.toggle('open'); {var_prefix}Button.setAttribute('aria-expanded', open ? 'true' : 'false'); }};\n"
            f"if({var_prefix}Button&&{var_prefix}Panel){{ {var_prefix}Button.addEventListener('click',(event)=>{{ event.stopPropagation(); toggle{var_prefix.capitalize()}Menu(); }}); {var_prefix}Panel.querySelectorAll('a').forEach(link=>link.addEventListener('click', close{var_prefix.capitalize()}Menu)); document.addEventListener('click',(event)=>{{ if(!{var_prefix}Panel.contains(event.target) && !{var_prefix}Button.contains(event.target)) close{var_prefix.capitalize()}Menu(); }}); document.addEventListener('keydown',(event)=>{{ if(event.key==='Escape') close{var_prefix.capitalize()}Menu(); }}); }}\n"
        )

    def _redirect_html(self, target: str) -> str:
        safe_target = html_escape(target)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url={safe_target}">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CLTR Redirect</title>
</head>
<body>
  <p>Redirecting to <a href="{safe_target}">{safe_target}</a>.</p>
</body>
</html>
"""

    def _canonical_cohort_href(self) -> str:
        return "cohort/index.html"

    def _cohort_chapter_route(self, chapter: dict) -> str:
        return f"ch{int(chapter['chapter_number']):02d}/index.html"

    def _cohort_chapter_menu_items_html(self, prefix: str) -> str:
        chapters = [
            ("index.html", "Cohort Report", "Chapter index for the cohort audit and result suite"),
            ("ch01/index.html", "Chapter 1", "Study overview and audit registers"),
            ("ch02/index.html", "Chapter 2", "Subjective and behavioral data"),
            ("ch03/index.html", "Chapter 3", "Physiological data"),
            ("ch04/index.html", "Chapter 4", "Environmental data"),
            ("ch05/index.html", "Chapter 5", "Derived results and audit registers"),
            ("ch06/index.html", "Chapter 6", "Relationships and validation"),
            ("cohort_full_report.html", "Full Cohort Report", "Full combined cohort export"),
        ]
        return "".join(
            f"<a href='{html_escape(prefix + route)}'>{html_escape(label)}<span>{html_escape(desc)}</span></a>"
            for route, label, desc in chapters
        )

    def _theme_toggle_script(self) -> str:
        return """const themeToggle=document.getElementById('themeToggle');
const siteMenuButton=document.getElementById('siteMenuButton');
const siteMenuPanel=document.getElementById('siteMenuPanel');
const storedTheme=window.localStorage.getItem('cltr-theme');
if(storedTheme==='dark'){document.body.classList.add('theme-dark');}
const syncThemeIcon=()=>{if(themeToggle){themeToggle.setAttribute('aria-pressed', document.body.classList.contains('theme-dark') ? 'true' : 'false');}};
const closeSiteMenu=()=>{if(!siteMenuPanel||!siteMenuButton)return;siteMenuPanel.classList.remove('open');siteMenuButton.setAttribute('aria-expanded','false');};
const toggleSiteMenu=()=>{if(!siteMenuPanel||!siteMenuButton)return;const open=siteMenuPanel.classList.toggle('open');siteMenuButton.setAttribute('aria-expanded',open?'true':'false');};
syncThemeIcon();
if(themeToggle){themeToggle.addEventListener('click',()=>{document.body.classList.toggle('theme-dark');window.localStorage.setItem('cltr-theme', document.body.classList.contains('theme-dark') ? 'dark' : 'light');syncThemeIcon();});}
if(siteMenuButton&&siteMenuPanel){siteMenuButton.addEventListener('click',(event)=>{event.stopPropagation();toggleSiteMenu();});siteMenuPanel.querySelectorAll('a').forEach(link=>link.addEventListener('click',closeSiteMenu));document.addEventListener('click',(event)=>{if(!siteMenuPanel.contains(event.target)&&!siteMenuButton.contains(event.target))closeSiteMenu();});document.addEventListener('keydown',(event)=>{if(event.key==='Escape')closeSiteMenu();});}"""

    def _session_card_tone_class(self, condition_code: str) -> str:
        code = str(condition_code or "").strip().upper()
        if code.startswith("BRI") and "MID" in code:
            return "sessionTone--bri-mid"
        if code.startswith("BRI") and "MOR" in code:
            return "sessionTone--bri-mor"
        if code.startswith("DIM") and "MID" in code:
            return "sessionTone--dim-mid"
        if code.startswith("DIM") and "MOR" in code:
            return "sessionTone--dim-mor"
        return "sessionTone--bri-mid"

    def _home_page_css(self) -> str:
        ui = REPORT_UI
        return f"""
{self._shared_report_css()}
.page {{ padding:{ui['page_padding']}; }}
.landing {{ min-height:calc(100vh - 160px); display:grid; place-items:center; }}
.hero {{ width:min(100%, 1120px); grid-template-columns:minmax(0,1fr); justify-items:center; gap:18px; text-align:center; }}
.heroLead {{ width:min(100%, 960px); background:linear-gradient(135deg,#172033 0%,#1d4ed8 55%,#06b6d4 100%); border-color:rgba(191,219,254,0.45); color:#eff6ff; }}
.heroLead::before {{ background:radial-gradient(circle at top right,rgba(255,255,255,0.24) 0%,rgba(255,255,255,0) 42%); }}
.heroLead .eyebrow {{ color:#dbeafe; }}
.heroLead .title,.heroLead .subtitle {{ color:#f8fafc; }}
.heroLead .subtitle {{ max-width:58ch; margin:0 auto; }}
.heroVisual {{ width:min(100%, 860px); border-radius:28px; border:1px solid rgba(219,234,254,0.34); background:rgba(255,255,255,0.14); box-shadow:inset 0 1px 0 rgba(255,255,255,0.2); padding:24px; margin-top:20px; display:grid; justify-items:center; gap:14px; }}
.heroVisual .logoMark {{ width:min(220px, 42vw); height:auto; }}
.heroVisualText {{ margin:0; max-width:46ch; font:500 1rem/1.6 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:#e2e8f0; }}
.copyrightNote {{ width:min(100%, {ui['page_max_width']}); margin:0 auto; padding:0 clamp(16px,2.4vw,28px) 18px; box-sizing:border-box; display:flex; justify-content:center; align-items:center; text-align:center; color:#64748b; font:500 0.84rem/1.5 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
body.theme-dark .copyrightNote {{ color:#94a3b8; }}
@media (max-width:{ui['mobile_breakpoint']}) {{ .hero {{ gap:16px; }} .heroVisual {{ padding:16px; }} }}
""".strip()

    def _home_html(self, manifest: pd.DataFrame, session_reports: list[dict], cohort_report: dict) -> str:
        chrome = self._shared_chrome(
            home_href="index.html",
            logo_src="../../../cltr/docs/assets/logos/cltr.png",
            page_type="Home",
            page_meta="Report index for the CLTR study",
            menu_button_id="homeMenuButton",
            menu_panel_id="homeMenuPanel",
            menu_label="Navigate",
            menu_title="CLTR Destinations",
            menu_items_html=(
                f"<a href='index.html' title='Open atlas'>Atlas<span>Study-wide hub and session index</span></a>"
                f"<a href='{html_escape(self._canonical_cohort_href())}' title='Open cohort report'>Cohort<span>Study-wide summary report</span></a>"
                f"<a href='sessions/{html_escape(session_reports[0]['session_id'])}/{html_escape(Path(session_reports[0]['html_path']).name)}' title='Open first session report'>Sessions<span>Session-level analytical reports</span></a>"
                if session_reports else f"<a href='index.html' title='Open atlas'>Atlas<span>Study-wide hub</span></a>"
            ),
            show_secondary_bar=False,
            show_menu_button=False,
        )
        return f"""<!doctype html><html><head><meta charset='utf-8'><title>{WORK_HOME_TITLE}</title>
<style>
{self._home_page_css()}
</style></head><body class='reportKind--home'>{chrome}<main class='page'><section class='landing'><section class='hero'><section class='panel heroLead'><div class='eyebrow'>Report Index</div><h1 class='title'>{WORK_HOME_TITLE}</h1><p class='subtitle'>{WORK_HOME_SUBTITLE}. Framework outputs are generated here first, and the atlas bundle can then be published separately from these report artifacts.</p><div class='heroVisual'>{self._logo_image_html('../../../cltr/docs/assets/logos/cltr.png')}<p class='heroVisualText'>This workspace contains the generated CLTR report bundle only. Public-site pages and publication assets are maintained separately under <code>work/cltr/docs</code>.</p></div></section></section></section></main><div class='copyrightNote'>{COPYRIGHT_NOTE}</div><script>
</script></body></html>"""

    def _spec(self, *, code: str, stem: str, title: str, summary: str, fig, tags: list[str], evidence_score: int, evidence_label: str, gating_note: str = "", phase_focus: str = "all", section: str = "results") -> dict:
        normalized_code = re.sub(r"(?<=\d)([A-Z])$", lambda m: m.group(1).lower(), str(code))
        panel_notes = list(getattr(fig, "_cltr_panel_notes", [])) if fig is not None else []
        caption_note = self._figure_caption_note(title=title, summary=summary, tags=tags)
        return {
            "code": normalized_code,
            "stem": stem,
            "title": title,
            "summary": summary,
            "caption_note": caption_note,
            "panel_notes": panel_notes,
            "fig": fig,
            "tags": sorted(set(tags)),
            "evidence_score": int(evidence_score),
            "evidence_label": evidence_label,
            "gating_note": gating_note,
            "phase_focus": phase_focus,
            "section": section,
        }

    def _html_spec(self, *, code: str, stem: str, title: str, summary: str, html_fragment: str, tags: list[str], evidence_score: int, evidence_label: str, gating_note: str = "", phase_focus: str = "all", section: str = "results") -> dict:
        spec = self._spec(
            code=code,
            stem=stem,
            title=title,
            summary=summary,
            fig=None,
            tags=tags,
            evidence_score=evidence_score,
            evidence_label=evidence_label,
            gating_note=gating_note,
            phase_focus=phase_focus,
            section=section,
        )
        spec["html_fragment"] = html_fragment
        return spec

    def _figure_caption_note(self, *, title: str, summary: str, tags: list[str]) -> str:
        title_text = str(title)
        summary_text = str(summary)
        title_lower = title_text.lower()
        summary_lower = summary_text.lower()
        notes: list[str] = []
        phase_abbr_pattern = re.compile(r"\b(?:ACC|FCS|SR|FFC|SS|OC)\b")
        uses_phase_abbr = bool(phase_abbr_pattern.search(title_text) or phase_abbr_pattern.search(summary_text))
        if uses_phase_abbr:
            notes.append(PHASE_ABBR_CAPTION)
        uses_acc_assumption = bool(re.search(r"\bACC\*\b", summary_text)) or "acc-assumed" in summary_lower
        if uses_acc_assumption:
            notes.append(ACC_ASSUMPTION_CAPTION)
        return " ".join(notes)

    def _phase_segments(self, df: pd.DataFrame) -> list[tuple[float, float, str]]:
        if df.empty:
            return []
        temp = df[["minute_index", "protocol_phase"]].dropna().copy()
        if temp.empty:
            return []
        temp["minute_index"] = to_numeric(temp["minute_index"])
        temp = temp.dropna(subset=["minute_index"]).copy()
        if temp.empty:
            return []
        # In cohort figures, multiple sessions can contribute different phase labels
        # at the same aligned minute. Use the dominant phase at each minute so the
        # ribbon follows the study-wide protocol structure instead of an arbitrary row.
        temp = (
            temp.groupby("minute_index")["protocol_phase"]
            .agg(lambda s: s.astype(str).value_counts().index[0] if not s.empty else np.nan)
            .reset_index()
            .sort_values("minute_index")
        )
        if temp.empty:
            return []
        spans = []
        start = float(temp.iloc[0]["minute_index"])
        prev = start
        phase = str(temp.iloc[0]["protocol_phase"])
        for _, row in temp.iloc[1:].iterrows():
            minute = float(row["minute_index"])
            cur_phase = str(row["protocol_phase"])
            if cur_phase != phase or minute != prev + 1:
                spans.append((start, prev + 1.0, phase))
                start = minute
                phase = cur_phase
            prev = minute
        spans.append((start, prev + 1.0, phase))
        return spans

    def _add_phase_spans(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        colors = ["#f8fafc", "#eff6ff", "#fef3c7", "#dcfce7", "#fee2e2", "#ede9fe"]
        for idx, (start, end, phase) in enumerate(self._phase_segments(df)):
            ax.axvspan(start, end, color=colors[idx % len(colors)], alpha=0.45, lw=0)
            width = end - start
            if width >= 7:
                ax.text((start + end) / 2.0, 1.01, PHASE_ABBR.get(phase, phase[:3].upper()), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10, color="#475569")

    def _place_condition_legend(self, ax: plt.Axes, handles=None) -> None:
        labels = []
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        else:
            labels = [getattr(handle, "get_label", lambda: "")().strip() for handle in handles]
        labels = [label for label in labels if label]
        if not labels:
            return
        ax.legend(
            handles=handles,
            frameon=False,
            ncol=len(labels),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.16),
            columnspacing=0.95,
            handletextpad=0.42,
            borderaxespad=0.0,
            fontsize=10,
        )

    def _place_topbar_legend(self, ax: plt.Axes, handles=None, *, y: float = 1.14) -> None:
        labels = []
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        else:
            labels = [getattr(handle, "get_label", lambda: "")().strip() for handle in handles]
        labels = [label for label in labels if label]
        if not labels:
            return
        ax.legend(
            handles=handles,
            frameon=False,
            ncol=len(labels),
            loc="upper center",
            bbox_to_anchor=(0.5, y),
            columnspacing=0.95,
            handletextpad=0.42,
            borderaxespad=0.0,
        )

    def _normalize_legend_layout(self, ax: plt.Axes) -> None:
        legend = ax.get_legend()
        if legend is None:
            return
        labels = [text.get_text().strip() for text in legend.get_texts() if text.get_text().strip()]
        if len(labels) <= 1:
            return
        if hasattr(legend, "set_ncols"):
            legend.set_ncols(len(labels))
        else:
            legend._ncols = len(labels)
        legend.set_frame_on(False)
        try:
            legend.set_bbox_to_anchor((0.5, 1.14), transform=ax.transAxes)
        except Exception:
            legend.set_bbox_to_anchor((0.5, 1.14))
        try:
            legend._loc = 9
        except Exception:
            pass
        legend.set_alignment("center")


    def _session_window_utc(self, minute: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if minute.empty or "minute_utc" not in minute.columns:
            return None, None
        ts = pd.to_datetime(minute["minute_utc"], utc=True, errors="coerce").dropna()
        if ts.empty:
            return None, None
        return ts.min(), ts.max() + pd.Timedelta(minutes=1)

    def _raw_phase_spans(self, minute: pd.DataFrame, session_start_utc: pd.Timestamp) -> list[tuple[float, float, str]]:
        if minute.empty or "minute_utc" not in minute.columns or "protocol_phase" not in minute.columns:
            return []
        temp = minute[["minute_utc", "protocol_phase"]].dropna().copy()
        if temp.empty:
            return []
        temp["minute_utc"] = pd.to_datetime(temp["minute_utc"], utc=True, errors="coerce")
        temp = temp.dropna(subset=["minute_utc"]).sort_values("minute_utc")
        spans = []
        colorspan_start = temp.iloc[0]["minute_utc"]
        prev = colorspan_start
        phase = str(temp.iloc[0]["protocol_phase"])
        for _, row in temp.iloc[1:].iterrows():
            current = row["minute_utc"]
            current_phase = str(row["protocol_phase"])
            if current_phase != phase or current != prev + pd.Timedelta(minutes=1):
                spans.append((((colorspan_start - session_start_utc).total_seconds() / 60.0), ((prev + pd.Timedelta(minutes=1) - session_start_utc).total_seconds() / 60.0), phase))
                colorspan_start = current
                phase = current_phase
            prev = current
        spans.append((((colorspan_start - session_start_utc).total_seconds() / 60.0), ((prev + pd.Timedelta(minutes=1) - session_start_utc).total_seconds() / 60.0), phase))
        return spans

    def _add_raw_phase_spans(
        self,
        ax: plt.Axes,
        minute: pd.DataFrame,
        session_start_utc: pd.Timestamp,
        visible_start: float | None = None,
        visible_end: float | None = None,
    ) -> None:
        colors = ["#f8fafc", "#eff6ff", "#fef3c7", "#dcfce7", "#fee2e2", "#ede9fe"]
        for idx, (start, end, phase) in enumerate(self._raw_phase_spans(minute, session_start_utc)):
            draw_start = start
            draw_end = end
            if visible_start is not None:
                draw_start = max(draw_start, visible_start)
            if visible_end is not None:
                draw_end = min(draw_end, visible_end)
            if draw_end <= draw_start:
                continue
            ax.axvspan(draw_start, draw_end, color=colors[idx % len(colors)], alpha=0.45, lw=0, zorder=0)
            original_width = end - start
            visible_width = draw_end - draw_start
            if visible_width >= 3 or (visible_start is not None and visible_end is not None and original_width >= 7):
                ax.text((draw_start + draw_end) / 2.0, 1.01, PHASE_ABBR.get(phase, phase[:3].upper()), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10, color="#475569", clip_on=False)

    def _clip_raw_source_window(self, df: pd.DataFrame, ts_col: str, minute: pd.DataFrame) -> pd.DataFrame:
        if df.empty or ts_col not in df.columns:
            return df.iloc[0:0].copy()
        start_utc, end_utc = self._session_window_utc(minute)
        out = df.copy()
        out[ts_col] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
        out = out.dropna(subset=[ts_col]).sort_values(ts_col)
        if start_utc is None or end_utc is None:
            return out
        return out.loc[(out[ts_col] >= start_utc) & (out[ts_col] <= end_utc)].copy()

    def _downsample_raw_df(self, df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
        if df.empty or ts_col not in df.columns:
            return df
        max_points = int(self.config.reporting.max_trace_points)
        if len(df) <= max_points:
            return df
        step = max(1, int(np.ceil(len(df) / max_points)))
        return df.iloc[::step].copy()

    def _raw_elapsed_minutes(self, series: pd.Series, session_start_utc: pd.Timestamp) -> pd.Series:
        ts = pd.to_datetime(series, utc=True, errors="coerce")
        return (ts - session_start_utc).dt.total_seconds() / 60.0

    def _raw_line_figure(
        self,
        minute: pd.DataFrame,
        df: pd.DataFrame,
        ts_col: str,
        series_specs: list[dict],
        *,
        ylabel: str,
        figsize: tuple[float, float] | None = None,
        step: bool = False,
        markers_only: bool = False,
        trim_to_support: bool = False,
    ):
        if df.empty or ts_col not in df.columns:
            return None
        start_utc, _ = self._session_window_utc(minute)
        if start_utc is None:
            return None
        plot = self._clip_raw_source_window(df, ts_col, minute)
        plot = self._downsample_raw_df(plot, ts_col)
        if plot.empty:
            return None
        x = self._raw_elapsed_minutes(plot[ts_col], start_utc)
        fig, ax = plt.subplots(figsize=figsize or self._figsize("timeline"))
        any_trace = False
        support_x_min = None
        support_x_max = None
        for spec in series_specs:
            col = spec["column"]
            if col not in plot.columns:
                continue
            y = to_numeric(plot[col])
            if y.notna().sum() == 0:
                continue
            any_trace = True
            mask = y.notna()
            cur_x = x.loc[mask]
            if not cur_x.empty:
                cur_min = float(cur_x.min())
                cur_max = float(cur_x.max())
                support_x_min = cur_min if support_x_min is None else min(support_x_min, cur_min)
                support_x_max = cur_max if support_x_max is None else max(support_x_max, cur_max)
            if markers_only:
                ax.scatter(x.loc[mask], y.loc[mask], color=spec["color"], s=20, label=spec["label"], alpha=0.95)
            elif step:
                ax.step(x, y, where="post", color=spec["color"], lw=1.6, label=spec["label"])
            else:
                ax.plot(x, y, color=spec["color"], lw=1.3, label=spec["label"], alpha=0.95)
        if not any_trace:
            plt.close(fig)
            return None
        ax.set_xlabel(self._time_axis_label())
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper right")
        if trim_to_support and support_x_min is not None and support_x_max is not None:
            pad = max((support_x_max - support_x_min) * 0.02, 0.1)
            visible_start = max(0.0, support_x_min - pad)
            visible_end = support_x_max + pad
            ax.set_xlim(visible_start, visible_end)
            self._add_raw_phase_spans(ax, minute, start_utc, visible_start=visible_start, visible_end=visible_end)
        else:
            self._add_raw_phase_spans(ax, minute, start_utc)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        return fig

    def _raw_peak_raster(self, minute: pd.DataFrame, peaks: pd.DataFrame, ts_col: str = "datetime"):
        if peaks.empty or ts_col not in peaks.columns:
            return None
        start_utc, _ = self._session_window_utc(minute)
        if start_utc is None:
            return None
        plot = self._clip_raw_source_window(peaks, ts_col, minute)
        plot = self._downsample_raw_df(plot, ts_col)
        if plot.empty:
            return None
        x = self._raw_elapsed_minutes(plot[ts_col], start_utc)
        fig, ax = plt.subplots(figsize=self._figsize("wide_single_short"))
        ax.vlines(x, 0, 1, color="#b91c1c", lw=0.7, alpha=0.8)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel(self._time_axis_label())
        ax.set_ylabel("Peaks")
        if not x.empty:
            pad = max((float(x.max()) - float(x.min())) * 0.02, 0.1)
            visible_start = max(0.0, float(x.min()) - pad)
            visible_end = float(x.max()) + pad
            ax.set_xlim(visible_start, visible_end)
            self._add_raw_phase_spans(ax, minute, start_utc, visible_start=visible_start, visible_end=visible_end)
        else:
            self._add_raw_phase_spans(ax, minute, start_utc)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        return fig

    def _raw_segment_windows(self, minute: pd.DataFrame, segments: pd.DataFrame):
        if segments.empty or "segment_start_utc" not in segments.columns or "segment_end_utc" not in segments.columns:
            return None
        start_utc, end_utc = self._session_window_utc(minute)
        if start_utc is None or end_utc is None:
            return None
        plot = segments.copy()
        plot["segment_start_utc"] = pd.to_datetime(plot["segment_start_utc"], utc=True, errors="coerce")
        plot["segment_end_utc"] = pd.to_datetime(plot["segment_end_utc"], utc=True, errors="coerce")
        plot = plot.dropna(subset=["segment_start_utc", "segment_end_utc"])
        plot = plot.loc[(plot["segment_end_utc"] >= start_utc) & (plot["segment_start_utc"] <= end_utc)].copy()
        if plot.empty:
            return None
        fig, ax = plt.subplots(figsize=self._figsize("wide_single_short"))
        visible_start = None
        visible_end = None
        for idx, row in enumerate(plot.itertuples()):
            seg_start = max(row.segment_start_utc, start_utc)
            seg_end = min(row.segment_end_utc, end_utc)
            left = (seg_start - start_utc).total_seconds() / 60.0
            width = max((seg_end - seg_start).total_seconds() / 60.0, 0.05)
            ax.barh(idx, width, left=left, color="#2563eb", alpha=0.85)
            visible_start = left if visible_start is None else min(visible_start, left)
            visible_end = left + width if visible_end is None else max(visible_end, left + width)
        ax.set_yticks(range(len(plot)))
        ax.set_yticklabels([f"Segment {i+1}" for i in range(len(plot))])
        ax.set_xlabel(self._time_axis_label())
        ax.set_ylabel("Empatica segments")
        if visible_start is not None and visible_end is not None:
            pad = max((visible_end - visible_start) * 0.02, 0.1)
            visible_start = max(0.0, visible_start - pad)
            visible_end = visible_end + pad
            ax.set_xlim(visible_start, visible_end)
            self._add_raw_phase_spans(ax, minute, start_utc, visible_start=visible_start, visible_end=visible_end)
        else:
            self._add_raw_phase_spans(ax, minute, start_utc)
        fig.tight_layout()
        return fig

    def _phase_metric_baseline(self, phase: pd.DataFrame, metric: str, exclude_acclimation: bool = False) -> dict | None:
        cov_col = f"{metric}__coverage"
        if metric not in phase.columns:
            return None
        d = phase.copy()
        if cov_col in d.columns:
            d = d.loc[to_numeric(d[cov_col]).fillna(0) > 0].copy()
        else:
            d = d.loc[to_numeric(d[metric]).notna()].copy()
        if d.empty:
            return None
        phase_order = self._comparison_phase_sequence(d["protocol_phase"].astype(str).unique()) if exclude_acclimation else PHASE_ORDER
        for preferred in phase_order:
            cur = d.loc[d["protocol_phase"] == preferred, metric]
            cur = to_numeric(cur).dropna()
            if not cur.empty:
                assumed = bool(preferred != "acclimation" and self._uses_acc_assumption(metric))
                return {
                    "phase": "acclimation" if assumed else preferred,
                    "source_phase": preferred,
                    "assumed": assumed,
                    "value": float(cur.mean()),
                    "coverage_col": cov_col if cov_col in phase.columns else None,
                }
        if exclude_acclimation:
            return self._phase_metric_baseline(phase, metric, exclude_acclimation=False)
        return None

    def _uses_acc_assumption(self, metric: str) -> bool:
        return metric in ACC_ASSUMPTION_CHANNELS

    def _baseline_phase_abbr(self, baseline: dict | None, *, include_assumption_marker: bool = True) -> str:
        if not baseline:
            return ""
        phase_name = str(baseline.get("phase", ""))
        abbr = PHASE_ABBR.get(phase_name, phase_name[:3].upper())
        if include_assumption_marker and bool(baseline.get("assumed")):
            return f"{abbr}*"
        return abbr

    def _baseline_phase_text(self, baseline: dict | None) -> str:
        if not baseline:
            return ""
        abbr = self._baseline_phase_abbr(baseline)
        if bool(baseline.get("assumed")):
            source_phase = str(baseline.get("source_phase", ""))
            source_abbr = PHASE_ABBR.get(source_phase, source_phase[:3].upper())
            return f"{abbr} assumed from first supported {source_abbr}"
        return abbr

    def _baseline_note(self, baseline: dict | None) -> str:
        if not baseline or not bool(baseline.get("assumed")):
            return ""
        source_phase = str(baseline.get("source_phase", ""))
        source_abbr = PHASE_ABBR.get(source_phase, source_phase[:3].upper())
        return f"ACC* denotes an assumed acclimation baseline proxied by first supported {source_abbr}."

    def _discrete_tick_values(self, values, metric: str) -> np.ndarray | None:
        if metric not in SPARSE_OBSERVATION_CHANNELS:
            return None
        predefined = QUESTIONNAIRE_FULL_SCALES.get(metric)
        if predefined is not None:
            return predefined
        finite = to_numeric(pd.Series(values)).replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            return None
        unique = np.sort(np.unique(np.round(finite.to_numpy(dtype=float), 6)))
        if unique.size == 0:
            return None
        if metric in SPARSE_OBSERVATION_CHANNELS:
            lo = float(np.floor(unique.min()))
            hi = float(np.ceil(unique.max()))
            return np.arange(lo, hi + 1.0, 1.0, dtype=float)
        if unique.size == 1:
            return unique
        diffs = np.diff(unique)
        positive = diffs[diffs > 1e-6]
        if positive.size == 0:
            return unique
        step = float(np.min(positive))
        lo = float(unique.min())
        hi = float(unique.max())
        n_steps = int(round((hi - lo) / step))
        if n_steps <= 0 or n_steps > 12:
            return unique
        ticks = lo + np.arange(n_steps + 1, dtype=float) * step
        return np.round(ticks, 6)

    def _apply_discrete_y_axis_matplotlib(self, ax: plt.Axes, values, metric: str, *, pad_steps: float = 0.35) -> None:
        ticks = self._discrete_tick_values(values, metric)
        if ticks is None or ticks.size == 0:
            return
        ax.set_yticks(ticks)
        if ticks.size >= 2:
            step = float(np.min(np.diff(ticks)))
            ax.set_ylim(float(ticks[0] - step * pad_steps), float(ticks[-1] + step * pad_steps))

    def _apply_discrete_y_axis_plotly(self, fig: go.Figure, values, metric: str) -> None:
        ticks = self._discrete_tick_values(values, metric)
        if ticks is None or ticks.size == 0:
            return
        tick_text = [str(int(v)) if float(v).is_integer() else f"{v:g}" for v in ticks]
        fig.update_yaxes(tickmode="array", tickvals=ticks.tolist(), ticktext=tick_text)
        if ticks.size >= 2:
            step = float(np.min(np.diff(ticks)))
            fig.update_yaxes(range=[float(ticks[0] - step * 0.35), float(ticks[-1] + step * 0.35)])

    def _phase_start_summary(self, phase: pd.DataFrame, metrics: list[str]) -> str:
        parts = []
        for metric in metrics:
            base = self._phase_metric_baseline(phase, metric)
            if not base:
                continue
            label = FEATURE_LABELS.get(metric, metric)
            parts.append(f"{label}: {self._baseline_phase_text(base)}")
        return " | ".join(parts)

    def _support_note(self, minute: pd.DataFrame, metrics: list[str]) -> str:
        parts = []
        phase = self._phase_summary_from_minute(minute, metrics)
        for metric in metrics:
            base = self._phase_metric_baseline(phase, metric)
            if not base:
                continue
            parts.append(f"{FEATURE_LABELS.get(metric, metric)} starts at {self._baseline_phase_text(base)}")
        return " | ".join(parts[:4])

    def _channel_display_window(self, minute: pd.DataFrame, column: str) -> tuple[pd.DataFrame, str]:
        if minute.empty or "minute_index" not in minute.columns or column not in minute.columns:
            return minute.iloc[0:0].copy(), ""
        values = to_numeric(minute[column])
        supported = minute.loc[values.notna()].copy()
        if supported.empty:
            return minute.iloc[0:0].copy(), ""
        start_minute = float(to_numeric(supported["minute_index"]).min())
        end_minute = float(to_numeric(supported["minute_index"]).max())
        window = minute.loc[(to_numeric(minute["minute_index"]) >= start_minute) & (to_numeric(minute["minute_index"]) <= end_minute)].copy()
        start_phase = str(supported.iloc[0]["protocol_phase"]) if "protocol_phase" in supported.columns else ""
        end_phase = str(supported.iloc[-1]["protocol_phase"]) if "protocol_phase" in supported.columns else ""
        start_label = PHASE_ABBR.get(start_phase, start_phase[:3].upper()) if start_phase else ""
        end_label = PHASE_ABBR.get(end_phase, end_phase[:3].upper()) if end_phase else ""
        if self._uses_acc_assumption(column) and start_phase != "acclimation":
            note = f"ACC* is assumed as the baseline for this modality; first observed support begins in {start_label}."
        elif start_phase == "acclimation":
            note = f"Display window starts in {start_label} because this modality has supported acclimation data."
        else:
            note = f"Display window starts in {start_label}; earlier phases are omitted because this modality has no supported data there."
        if end_label:
            note += f" The displayed support extends through {end_label}."
        return window, note

    def _is_sparse_observation_channel(self, column: str) -> bool:
        return column in SPARSE_OBSERVATION_CHANNELS

    def _is_control_signal_channel(self, column: str) -> bool:
        return column in CONTROL_SIGNAL_CHANNELS

    def _axis_label(self, metric: str) -> str:
        return AXIS_LABELS.get(str(metric), FEATURE_LABELS.get(str(metric), str(metric)))

    def _compact_axis_label(self, metric: str) -> str:
        label = self._axis_label(metric)
        if "(ordinal scale)" in label:
            return "Ordinal score"
        match = re.search(r"\(([^)]+)\)\s*$", label)
        if match:
            return match.group(1)
        return label

    @staticmethod
    def _time_axis_label() -> str:
        return "Time in Session (min)"

    @staticmethod
    def _phase_axis_label() -> str:
        return "Protocol Phase"

    @staticmethod
    def _condition_axis_label() -> str:
        return "Experimental Condition"

    def _cohort_questionnaire_caption(self, column: str, *, aggregated: bool = False) -> str:
        if aggregated:
            caption_map = {
                "thermal_sensation": (
                    "This panel summarizes session-level Thermal Sensation responses by protocol phase and condition. "
                    "Each condition trace shows the median of session-level phase averages with an interquartile band, "
                    "which makes cross-condition response patterns easier to compare than the raw event cloud."
                ),
                "thermal_comfort": (
                    "This panel summarizes session-level Thermal Comfort responses by protocol phase and condition. "
                    "Each condition trace shows the median of session-level phase averages with an interquartile band, "
                    "so it is appropriate for comparing condition patterns rather than raw questionnaire-event density."
                ),
                "thermal_preference": (
                    "This panel summarizes session-level Thermal Preference responses by protocol phase and condition. "
                    "Because the underlying scale reflects directional preference choices, the condition summaries should be read "
                    "as comparative preference patterns rather than as continuous intensities."
                ),
                "visual_sensation": (
                    "This panel compares session-level Visual Sensation responses across conditions at the overall-comfort stage only. "
                    "The single-phase display is intentional because Visual Sensation is collected as a global end-of-session appraisal."
                ),
                "color_sensation": (
                    "This panel compares session-level Color Sensation responses across conditions at the overall-comfort stage only. "
                    "The single-phase display is intentional because Color Sensation is collected as a global end-of-session appraisal."
                ),
                "thermal_pleasure": (
                    "This panel summarizes session-level Thermal Pleasure responses by protocol phase and condition. "
                    "The missing steady-state phase is a questionnaire-design feature, so the condition traces should be compared "
                    "across FCS, SR, limited FFC, and OC only."
                ),
                "room_comfort": (
                    "This panel compares session-level Room Comfort responses across conditions at the overall-comfort stage only. "
                    "There is no earlier phase structure because Room Comfort is collected as an end-of-session global room appraisal."
                ),
                "visual_comfort": (
                    "This panel compares session-level Visual Comfort responses across conditions at the overall-comfort stage only. "
                    "The single-phase display is intentional because Visual Comfort is collected as a global end-of-session appraisal."
                ),
                "sound_comfort_dbA": (
                    "This panel compares session-level Sound Comfort responses across conditions at the overall-comfort stage only. "
                    "The single-phase display is intentional because Sound Comfort is collected as a global end-of-session appraisal."
                ),
                "air_quality_comfort": (
                    "This panel compares session-level Air-Quality Comfort responses across conditions at the overall-comfort stage only. "
                    "The single-phase display is intentional because this item is asked only in the overall-comfort questionnaire."
                ),
            }
            return caption_map.get(
                column,
                f"This panel summarizes session-level {FEATURE_LABELS.get(column, column)} responses by phase and condition, using condition medians with interquartile bands instead of raw event points.",
            )
        caption_map = {
            "thermal_sensation": (
                "Thermal Sensation is shown as discrete questionnaire observations across FCS, SR, SS, and OC, with only limited "
                "coverage in FFC. These are ordinal responses collected at questionnaire events, so they should be read as sparse "
                "phase-specific observations rather than a continuous trajectory."
            ),
            "thermal_comfort": (
                "Thermal Comfort is shown as discrete questionnaire observations across FCS, SR, SS, and OC, with only limited "
                "coverage in FFC. Because the measure is collected at event times rather than every minute, the panel should be read "
                "as a sparse ordinal response map, not as a continuous comfort trace."
            ),
            "thermal_preference": (
                "Thermal Preference is shown as discrete event-time responses across FCS, SR, SS, and OC, with only limited FFC "
                "coverage. The values reflect directional preference choices rather than a continuous intensity scale."
            ),
            "visual_sensation": (
                "Visual Sensation is shown only in the overall-comfort phase because it is collected as a global end-of-session "
                "visual appraisal rather than a repeated observation during the earlier protocol phases."
            ),
            "color_sensation": (
                "Color Sensation is shown only in the overall-comfort phase because it is collected as a global end-of-session "
                "appraisal, not as a repeated observation during the thermal protocol phases."
            ),
            "room_comfort": (
                "Room Comfort is shown only in the overall-comfort phase because this questionnaire item is collected as an "
                "end-of-session global room appraisal rather than a repeated phase-by-phase observation."
            ),
            "thermal_pleasure": (
                "Thermal Pleasure is shown as discrete questionnaire observations in FCS, SR, limited FFC, and OC. It is absent "
                "from SS in the questionnaire design, so the missing SS phase reflects instrument design rather than data loss."
            ),
            "visual_comfort": (
                "Visual Comfort is shown only in the overall-comfort phase because it is collected as a global end-of-session "
                "appraisal, not as a repeated observation during the thermal protocol phases."
            ),
            "sound_comfort_dbA": (
                "Sound Comfort is shown only in the overall-comfort phase because it is collected as a global end-of-session "
                "sound-environment appraisal, not as a repeated observation during the earlier protocol phases."
            ),
            "air_quality_comfort": (
                "Air-Quality Comfort is shown only in the overall-comfort phase because it is collected as a global end-of-session "
                "appraisal, not as a repeated observation during the earlier protocol phases."
            ),
        }
        return caption_map.get(
            column,
            f"{FEATURE_LABELS.get(column, column)} is shown as phase-wise condition distributions with raw observation points; questionnaire responses are discrete event-time observations, not continuous trajectories.",
        )

    def _display_series(self, values: pd.Series, column: str) -> pd.Series:
        series = to_numeric(values)
        if not self._is_control_signal_channel(column):
            return series
        return series.rolling(window=3, center=True, min_periods=1).median()

    def _phase_summary_from_minute(self, minute: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
        if minute.empty or "protocol_phase" not in minute.columns:
            return pd.DataFrame()
        rows = []
        for phase_name, d in minute.groupby("protocol_phase"):
            row = {"protocol_phase": phase_name}
            for metric in metrics:
                if metric in d.columns:
                    vals = to_numeric(d[metric]).dropna()
                    row[metric] = float(vals.mean()) if not vals.empty else np.nan
                    row[f"{metric}__coverage"] = float(to_numeric(d[metric]).notna().mean())
            rows.append(row)
        return pd.DataFrame(rows)

    def _comparison_phase_sequence(self, available: list[str] | pd.Series | None = None) -> list[str]:
        phases = [p for p in PHASE_ORDER if p != "acclimation"]
        if available is None:
            return phases
        values = {str(x) for x in list(available)}
        return [p for p in phases if p in values]

    def _overlap_start_phase(self, minute: pd.DataFrame, support_col: str) -> str | None:
        if support_col not in minute.columns:
            return None
        d = minute.loc[to_numeric(minute[support_col]).fillna(0) > 0, "protocol_phase"].dropna()
        return str(d.iloc[0]) if not d.empty else None

    def _block_phase_order(self, phase: pd.DataFrame) -> list[tuple[str, str]]:
        if phase.empty or "protocol_block" not in phase.columns or "protocol_phase" not in phase.columns:
            return []
        temp = phase.copy()
        temp["_block_num"] = pd.to_numeric(temp["protocol_block"], errors="coerce")
        temp["_phase_idx"] = temp["protocol_phase"].map({name: idx for idx, name in enumerate(PHASE_ORDER)}).fillna(len(PHASE_ORDER))
        pairs = []
        for _, row in temp.sort_values(["_block_num", "_phase_idx", "protocol_phase"]).iterrows():
            pairs.append((str(row["protocol_block"]), str(row["protocol_phase"])))
        ordered = []
        seen = set()
        for pair in pairs:
            if pair not in seen:
                ordered.append(pair)
                seen.add(pair)
        return ordered

    def _block_phase_label(self, block: str, phase_name: str) -> str:
        return f"B{block}-{PHASE_ABBR.get(phase_name, phase_name[:3].upper())}"

    def _phase_level_summary(self, phase: pd.DataFrame, metric: str) -> pd.DataFrame:
        if phase.empty or metric not in phase.columns or "protocol_phase" not in phase.columns:
            return pd.DataFrame()
        cov_col = f"{metric}__coverage"
        rows = []
        for phase_name in [p for p in PHASE_ORDER if p in phase["protocol_phase"].astype(str).unique()]:
            cur = phase.loc[phase["protocol_phase"] == phase_name].copy()
            vals = to_numeric(cur[metric]).dropna()
            if vals.empty:
                continue
            row = {
                "protocol_phase": phase_name,
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "n_rows": int(len(vals)),
            }
            if cov_col in cur.columns:
                row["coverage_mean"] = float(to_numeric(cur[cov_col]).mean())
            rows.append(row)
        return pd.DataFrame(rows)

    def _phase_baseline_delta_summary(self, phase: pd.DataFrame, metric: str, exclude_acclimation: bool = False) -> tuple[pd.DataFrame, dict | None]:
        summary = self._phase_level_summary(phase, metric)
        baseline = self._phase_metric_baseline(phase, metric, exclude_acclimation=exclude_acclimation)
        if summary.empty:
            return summary, baseline
        if exclude_acclimation:
            summary = summary.loc[summary["protocol_phase"].astype(str) != "acclimation"].copy()
        if baseline and pd.notna(baseline["value"]):
            summary["delta"] = summary["mean"] - float(baseline["value"])
        else:
            summary["delta"] = summary["mean"]
        return summary, baseline

    def _block_phase_deltas(self, phase: pd.DataFrame, metric: str, exclude_acclimation: bool = False) -> tuple[pd.DataFrame, dict | None]:
        baseline = self._phase_metric_baseline(phase, metric, exclude_acclimation=exclude_acclimation)
        if phase.empty or metric not in phase.columns or "protocol_block" not in phase.columns or "protocol_phase" not in phase.columns:
            return pd.DataFrame(), baseline
        cov_col = f"{metric}__coverage"
        temp = phase.copy()
        if cov_col in temp.columns:
            temp = temp.loc[to_numeric(temp[cov_col]).fillna(0) > 0].copy()
        temp[metric] = to_numeric(temp[metric])
        temp = temp.dropna(subset=[metric])
        if exclude_acclimation:
            temp = temp.loc[temp["protocol_phase"].astype(str) != "acclimation"].copy()
        if temp.empty:
            return pd.DataFrame(), baseline
        rows = []
        order = self._block_phase_order(temp)
        order_index = {pair: idx for idx, pair in enumerate(order)}
        baseline_value = baseline["value"] if baseline else np.nan
        for (block, phase_name), d in temp.groupby(["protocol_block", "protocol_phase"]):
            mean_value = float(to_numeric(d[metric]).mean())
            rows.append(
                {
                    "protocol_block": str(block),
                    "protocol_phase": str(phase_name),
                    "block_phase": self._block_phase_label(str(block), str(phase_name)),
                    "mean": mean_value,
                    "delta": mean_value - baseline_value if pd.notna(baseline_value) else mean_value,
                    "coverage_mean": float(to_numeric(d[cov_col]).mean()) if cov_col in d.columns else 1.0,
                    "order_idx": order_index.get((str(block), str(phase_name)), 0),
                }
            )
        out = pd.DataFrame(rows).sort_values("order_idx")
        return out, baseline

    def _phase_repeat_consistency(self, phase: pd.DataFrame, metric: str) -> dict:
        block_deltas, baseline = self._block_phase_deltas(phase, metric)
        if block_deltas.empty:
            return {"consistency": 0.0, "n_blocks": 0, "dominant_phase": None, "dominant_direction": None}
        non_baseline = block_deltas.copy()
        if baseline:
            non_baseline = non_baseline.loc[non_baseline["protocol_phase"] != baseline["phase"]]
        if non_baseline.empty:
            return {"consistency": 0.0, "n_blocks": 0, "dominant_phase": None, "dominant_direction": None}
        rows = []
        for phase_name, d in non_baseline.groupby("protocol_phase"):
            signs = np.sign(to_numeric(d["delta"]).fillna(0))
            strong = signs.loc[signs != 0]
            if strong.empty:
                continue
            dominant_sign = float(np.sign(strong.sum())) if float(strong.sum()) != 0 else float(strong.iloc[0])
            aligned = float((strong == dominant_sign).mean())
            rows.append(
                {
                    "protocol_phase": str(phase_name),
                    "consistency": aligned,
                    "n_blocks": int(len(strong)),
                    "abs_delta_mean": float(to_numeric(d["delta"]).abs().mean()),
                    "direction": "rise" if dominant_sign > 0 else "drop",
                }
            )
        if not rows:
            return {"consistency": 0.0, "n_blocks": 0, "dominant_phase": None, "dominant_direction": None}
        best = pd.DataFrame(rows).sort_values(["consistency", "abs_delta_mean", "n_blocks"], ascending=False).iloc[0]
        return {
            "consistency": float(best["consistency"]),
            "n_blocks": int(best["n_blocks"]),
            "dominant_phase": str(best["protocol_phase"]),
            "dominant_direction": str(best["direction"]),
        }

    def _session_derived_endpoints(self, phase: pd.DataFrame) -> list[str]:
        if phase.empty:
            return []
        return [
            metric
            for metric in SESSION_DERIVED_ENDPOINTS
            if metric in phase.columns and to_numeric(phase[metric]).notna().sum() > 0
        ]

    def _is_questionnaire_endpoint(self, metric: str) -> bool:
        return metric in SPARSE_OBSERVATION_CHANNELS

    def _session_endpoint_support_profile(self, phase: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
        rows = []
        for metric in metrics:
            summary = self._session_phase_median_summary(phase, metric)
            if summary.empty:
                continue
            phase_counts = summary["n_block_phase"].astype(int).tolist()
            supported_phases = int(len(phase_counts))
            min_blocks = int(min(phase_counts)) if phase_counts else 0
            max_blocks = int(max(phase_counts)) if phase_counts else 0
            total_block_phase = int(sum(phase_counts))
            total_minutes = int(summary["total_minutes"].sum()) if "total_minutes" in summary.columns else 0
            total_valid_units = int(summary["total_valid_units"].sum()) if "total_valid_units" in summary.columns else total_block_phase
            basis = "questionnaire events" if self._is_questionnaire_endpoint(metric) else "processed minute summaries"
            if supported_phases >= 4 and min_blocks >= 2 and total_block_phase >= 10:
                grade = "strong"
                reason = "Adequate repeated-block support across most comparison phases."
            elif supported_phases >= 3 and max_blocks >= 2 and total_block_phase >= 6:
                grade = "partial"
                reason = "Usable in parts of the comparison window, but repeated support is incomplete."
            else:
                grade = "insufficient"
                reason = "Too sparse or too late-starting for a session-wide primary result."
            rows.append(
                {
                    "metric": metric,
                    "endpoint": FEATURE_LABELS.get(metric, metric),
                    "support_grade": grade.title(),
                    "support_basis": basis,
                    "supported_phases": supported_phases,
                    "min_block_repeats": min_blocks,
                    "max_block_repeats": max_blocks,
                    "total_block_phase_summaries": total_block_phase,
                    "total_minutes": total_minutes,
                    "total_valid_units": total_valid_units,
                    "scientific_reading": reason,
                }
            )
        return pd.DataFrame(rows)

    def _session_response_matrix(self, phase: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
        if phase.empty or not metrics:
            return pd.DataFrame()
        phase_order = self._comparison_phase_sequence(phase["protocol_phase"].astype(str).unique())
        rows = []
        for metric in metrics:
            summary = self._session_phase_median_summary(phase, metric)
            if summary.empty:
                continue
            row = {
                "endpoint": FEATURE_LABELS.get(metric, metric),
                "support_basis": str(summary["support_basis"].iloc[0]) if "support_basis" in summary.columns and not summary.empty else "",
            }
            total_rows = 0
            total_minutes = 0
            total_valid_units = 0
            support_parts = []
            for phase_name in phase_order:
                phase_row = summary.loc[summary["protocol_phase"].astype(str) == phase_name]
                abbr = PHASE_ABBR.get(phase_name, phase_name[:3].upper())
                if phase_row.empty:
                    row[abbr] = np.nan
                    continue
                phase_row = phase_row.iloc[0]
                row[abbr] = float(phase_row["median"])
                total_rows += int(phase_row["n_block_phase"])
                total_minutes += int(phase_row["total_minutes"])
                total_valid_units += int(phase_row.get("total_valid_units", 0))
                support_parts.append(f"{abbr}:{int(phase_row['n_block_phase'])}/{int(phase_row.get('total_valid_units', 0))}")
            row["n_block_phase_summaries"] = total_rows
            row["total_minutes"] = total_minutes
            row["total_valid_units"] = total_valid_units
            row["phase_support"] = " | ".join(support_parts)
            rows.append(row)
        return pd.DataFrame(rows)

    def _session_phase_contrast_matrix(self, phase: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
        if phase.empty or not metrics:
            return pd.DataFrame()
        phase_order = self._comparison_phase_sequence(phase["protocol_phase"].astype(str).unique())
        rows = []
        for metric in metrics:
            summary, baseline = self._session_phase_contrast_summary(phase, metric)
            if summary.empty:
                continue
            row = {
                "endpoint": FEATURE_LABELS.get(metric, metric),
                "reference_phase": PHASE_ABBR.get(str(baseline.get("phase", "")), str(baseline.get("phase", ""))[:3].upper()) if baseline else "",
            }
            for phase_name in phase_order:
                phase_row = summary.loc[summary["protocol_phase"].astype(str) == phase_name]
                abbr = PHASE_ABBR.get(phase_name, phase_name[:3].upper())
                row[abbr] = float(phase_row.iloc[0]["delta"]) if not phase_row.empty else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def _session_repeatability_matrix(self, phase: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
        if phase.empty or not metrics:
            return pd.DataFrame()
        phase_order = self._comparison_phase_sequence(phase["protocol_phase"].astype(str).unique())
        rows = []
        for metric in metrics:
            agreement_df = self._session_phase_sign_agreement(phase, metric)
            if agreement_df.empty:
                continue
            row = {"endpoint": FEATURE_LABELS.get(metric, metric)}
            support_parts = []
            any_supported = False
            for phase_name in phase_order:
                phase_row = agreement_df.loc[agreement_df["protocol_phase"].astype(str) == phase_name]
                abbr = PHASE_ABBR.get(phase_name, phase_name[:3].upper())
                if phase_row.empty:
                    row[abbr] = np.nan
                    continue
                phase_row = phase_row.iloc[0]
                row[abbr] = float(phase_row["sign_agreement"])
                support_parts.append(f"{abbr}:{int(phase_row['n_blocks'])}")
                any_supported = True
            if not any_supported:
                continue
            row["phase_block_support"] = " | ".join(support_parts)
            rows.append(row)
        return pd.DataFrame(rows)

    def _session_response_fingerprint_matrix(self, phase: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
        if phase.empty or not metrics:
            return pd.DataFrame()
        rows = []
        for metric in metrics:
            contrast_df, baseline = self._session_phase_contrast_summary(phase, metric)
            if contrast_df.empty or baseline is None:
                continue
            non_reference = contrast_df.loc[contrast_df["protocol_phase"].astype(str) != str(baseline["phase"])].copy()
            non_reference["abs_delta"] = non_reference["delta"].abs()
            non_reference = non_reference.dropna(subset=["abs_delta"])
            if non_reference.empty:
                continue
            top = non_reference.sort_values(["abs_delta", "protocol_phase"], ascending=[False, True]).iloc[0]
            agreement_df = self._session_phase_sign_agreement(phase, metric)
            agreement = np.nan
            n_blocks = np.nan
            if not agreement_df.empty:
                match = agreement_df.loc[agreement_df["protocol_phase"].astype(str) == str(top["protocol_phase"])]
                if not match.empty:
                    agreement = float(match.iloc[0]["sign_agreement"])
                    n_blocks = int(match.iloc[0]["n_blocks"])
            summary_df = self._session_phase_median_summary(phase, metric)
            valid_units = int(summary_df["total_valid_units"].sum()) if not summary_df.empty and "total_valid_units" in summary_df.columns else np.nan
            support_basis = str(summary_df["support_basis"].iloc[0]) if not summary_df.empty and "support_basis" in summary_df.columns else ""
            rows.append(
                {
                    "endpoint": FEATURE_LABELS.get(metric, metric),
                    "reference_phase": PHASE_ABBR.get(str(baseline["phase"]), str(baseline["phase"])[:3].upper()),
                    "dominant_phase": PHASE_ABBR.get(str(top["protocol_phase"]), str(top["protocol_phase"])[:3].upper()),
                    "direction": "Rise" if float(top["delta"]) > 0 else "Drop",
                    "dominant_delta": float(top["delta"]),
                    "dominant_agreement": agreement,
                    "dominant_phase_repeats": n_blocks,
                    "support_basis": support_basis,
                    "total_valid_units": valid_units,
                }
            )
        return pd.DataFrame(rows)

    def _matrix_panel_html(self, title: str, df: pd.DataFrame, columns: list[str], n: int = 24) -> str:
        return self._render_table(df, title, columns, n=n)

    def _feature_registry_display(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        view = df.copy()
        observed = to_numeric(view.get("observed_prompt_count", pd.Series(dtype=float)))
        expected = to_numeric(view.get("expected_prompt_count", pd.Series(dtype=float)))
        fraction = to_numeric(view.get("prompt_response_fraction", pd.Series(dtype=float)))

        def format_prompt_support(idx: int) -> str:
            obs = observed.iloc[idx] if idx < len(observed) else np.nan
            exp = expected.iloc[idx] if idx < len(expected) else np.nan
            frac = fraction.iloc[idx] if idx < len(fraction) else np.nan
            if pd.notna(obs) and pd.notna(exp) and exp > 0:
                frac_text = f" ({float(frac):.1%})" if pd.notna(frac) else ""
                return f"{int(obs)}/{int(exp)}{frac_text}"
            return ""

        view["prompt_support"] = [format_prompt_support(i) for i in range(len(view))]
        view["minute_occupancy_fraction"] = to_numeric(view.get("coverage_fraction", pd.Series(dtype=float)))
        view["minute_occupancy_reading"] = view.get("coverage_reading", pd.Series(dtype=object)).fillna("")
        return view

    def _metric_unit(self, metric: str) -> str:
        label = self._axis_label(metric)
        match = re.search(r"\(([^)]+)\)\s*$", str(label))
        return match.group(1) if match else ""

    def _humanize_register_text(self, text: object, predictor: str = "", target: str = "") -> str:
        out = str(text or "")
        replacements = []
        if predictor:
            replacements.append((predictor, FEATURE_LABELS.get(predictor, predictor.replace("_", " "))))
        if target:
            replacements.append((target, FEATURE_LABELS.get(target, target.replace("_", " "))))
        for raw, label in replacements:
            out = out.replace(str(raw), str(label))
        return out

    def _threshold_response_register_display(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        view = df.copy()
        view["threshold_unit"] = view.get("predictor", pd.Series(dtype=object)).astype(str).map(self._metric_unit)
        if "scientific_reading" in view.columns:
            view["scientific_reading"] = [
                self._humanize_register_text(text, predictor=str(pred), target=str(tgt))
                for text, pred, tgt in zip(
                    view["scientific_reading"],
                    view.get("predictor", pd.Series(dtype=object)),
                    view.get("target", pd.Series(dtype=object)),
                )
            ]
        return view

    def _scientific_decision_register_display(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        view = df.copy()
        view["threshold_unit"] = view.get("predictor", pd.Series(dtype=object)).astype(str).map(self._metric_unit)
        if "practical_reading" in view.columns:
            view["practical_reading"] = [
                self._humanize_register_text(text, predictor=str(pred), target=str(tgt))
                for text, pred, tgt in zip(
                    view["practical_reading"],
                    view.get("predictor", pd.Series(dtype=object)),
                    view.get("target", pd.Series(dtype=object)),
                )
            ]
        return view

    def _fig_endpoint_support_grades(self, support_df: pd.DataFrame):
        if support_df.empty:
            return None
        order = {"Strong": 0, "Partial": 1, "Insufficient": 2}
        plot_df = support_df.copy()
        plot_df["_order"] = plot_df["support_grade"].map(order).fillna(9)
        plot_df = plot_df.sort_values(["_order", "supported_phases", "total_block_phase_summaries"], ascending=[True, False, False])
        color_map = {"Strong": "#059669", "Partial": "#d97706", "Insufficient": "#b91c1c"}
        fig, ax = plt.subplots(figsize=self._figsize("wide_single_tall"))
        y = np.arange(len(plot_df))
        ax.barh(y, plot_df["supported_phases"], color=[color_map.get(x, "#64748b") for x in plot_df["support_grade"]])
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["endpoint"].tolist())
        ax.invert_yaxis()
        ax.set_xlabel("Supported comparison phases")
        ax.set_xlim(0, 5)
        for idx, row in enumerate(plot_df.itertuples()):
            ax.text(min(float(row.supported_phases) + 0.08, 4.95), idx, f"{row.support_grade} | min repeats={row.min_block_repeats} | summaries={row.total_block_phase_summaries}", va="center", fontsize=10, color="#172033")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_session_response_fingerprint(self, fingerprint_df: pd.DataFrame):
        if fingerprint_df.empty:
            return None
        phase_cols = [p for p in ["FCS", "SR", "FFC", "SS", "OC"]]
        phase_index = {phase: idx for idx, phase in enumerate(phase_cols)}
        plot_df = fingerprint_df.copy()
        plot_df = plot_df.sort_values(["dominant_phase", "endpoint"], key=lambda s: s.map(phase_index) if s.name == "dominant_phase" else s)
        fig, ax = plt.subplots(figsize=self._figsize("wide_single_tall"))
        y = np.arange(len(plot_df))
        x = plot_df["dominant_phase"].map(phase_index).astype(float)
        sizes = []
        colors = []
        for row in plot_df.itertuples():
            agreement = row.dominant_agreement if pd.notna(row.dominant_agreement) else 0.35
            sizes.append(90 + 180 * float(agreement))
            colors.append("#2563eb" if row.direction == "Rise" else "#dc2626")
        ax.scatter(x, y, s=sizes, c=colors, alpha=0.85, edgecolors="#172033", linewidths=0.6)
        ax.set_xticks(range(len(phase_cols)))
        ax.set_xticklabels(phase_cols)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["endpoint"].tolist())
        ax.invert_yaxis()
        ax.set_xlabel("Dominant response phase")
        for idx, row in enumerate(plot_df.itertuples()):
            agreement_txt = f"agreement={row.dominant_agreement:.2f}" if pd.notna(row.dominant_agreement) else "agreement=n/a"
            ax.text(float(x.iloc[idx]) + 0.08, idx, f"{row.direction.lower()} | {agreement_txt}", va="center", fontsize=10, color="#172033")
        ax.grid(True, axis="x", alpha=0.2)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _session_phase_median_summary(self, phase: pd.DataFrame, metric: str) -> pd.DataFrame:
        if phase.empty or metric not in phase.columns or "protocol_phase" not in phase.columns:
            return pd.DataFrame()
        cov_col = f"{metric}__coverage"
        valid_col = f"{metric}__n_valid"
        temp = phase.copy()
        temp = temp.loc[temp["protocol_block"].astype(str).isin(COMPARISON_BLOCKS)].copy()
        if cov_col in temp.columns:
            temp = temp.loc[to_numeric(temp[cov_col]).fillna(0) > 0].copy()
        temp[metric] = to_numeric(temp[metric])
        temp = temp.dropna(subset=[metric])
        if temp.empty:
            return pd.DataFrame()
        rows = []
        for phase_name in self._comparison_phase_sequence(temp["protocol_phase"].astype(str).unique()):
            cur = temp.loc[temp["protocol_phase"].astype(str) == phase_name].copy()
            if cur.empty:
                continue
            vals = to_numeric(cur[metric]).dropna()
            if vals.empty:
                continue
            rows.append(
                {
                    "protocol_phase": phase_name,
                    "median": float(vals.median()),
                    "mean": float(vals.mean()),
                    "n_block_phase": int(len(vals)),
                    "total_minutes": int(pd.to_numeric(cur["n_minutes"], errors="coerce").fillna(0).sum()) if "n_minutes" in cur.columns else int(len(vals)),
                    "total_valid_units": int(pd.to_numeric(cur[valid_col], errors="coerce").fillna(0).sum()) if valid_col in cur.columns else int(len(vals)),
                    "support_basis": "questionnaire responses" if self._is_questionnaire_endpoint(metric) else "valid minute summaries",
                }
            )
        return pd.DataFrame(rows)

    def _session_phase_contrast_summary(self, phase: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict | None]:
        summary = self._session_phase_median_summary(phase, metric)
        if summary.empty:
            return pd.DataFrame(), None
        baseline = None
        for phase_name in self._comparison_phase_sequence(summary["protocol_phase"].astype(str).unique()):
            cur = summary.loc[summary["protocol_phase"].astype(str) == phase_name]
            if not cur.empty and pd.notna(cur.iloc[0]["median"]):
                baseline = {"phase": str(phase_name), "value": float(cur.iloc[0]["median"])}
                break
        if baseline is None:
            return pd.DataFrame(), None
        summary = summary.copy()
        summary["delta"] = summary["median"] - float(baseline["value"])
        return summary, baseline

    def _session_phase_sign_agreement(self, phase: pd.DataFrame, metric: str) -> pd.DataFrame:
        block_deltas, baseline = self._block_phase_deltas(phase, metric, exclude_acclimation=True)
        if block_deltas.empty:
            return pd.DataFrame()
        rows = []
        baseline_phase = str(baseline["phase"]) if baseline else None
        for phase_name in self._comparison_phase_sequence(block_deltas["protocol_phase"].astype(str).unique()):
            if baseline_phase and phase_name == baseline_phase:
                continue
            cur = block_deltas.loc[block_deltas["protocol_phase"].astype(str) == phase_name].copy()
            if cur.empty:
                continue
            signs = np.sign(to_numeric(cur["delta"]).fillna(0))
            strong = signs.loc[signs != 0]
            if len(strong) < 2:
                continue
            dominant_sign = float(np.sign(strong.sum())) if float(strong.sum()) != 0 else float(strong.iloc[0])
            rows.append(
                {
                    "protocol_phase": phase_name,
                    "sign_agreement": float((strong == dominant_sign).mean()),
                    "n_blocks": int(len(strong)),
                }
            )
        return pd.DataFrame(rows)

    def _fig_session_response_heatmap(self, matrix_df: pd.DataFrame):
        if matrix_df.empty:
            return None
        phase_cols = [col for col in ["FCS", "SR", "FFC", "SS", "OC"] if col in matrix_df.columns]
        if not phase_cols:
            return None
        values = matrix_df[phase_cols].apply(pd.to_numeric, errors="coerce")
        if values.dropna(how="all").empty:
            return None
        scaled = values.copy()
        for idx in scaled.index:
            row = scaled.loc[idx]
            valid = row.dropna()
            if valid.empty:
                continue
            lo = float(valid.min())
            hi = float(valid.max())
            scaled.loc[idx, valid.index] = 0.5 if hi <= lo else (valid - lo) / (hi - lo)
        fig, ax = plt.subplots(figsize=self._figsize("matrix"))
        im = ax.imshow(scaled.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(len(phase_cols)))
        ax.set_xticklabels(phase_cols)
        ax.set_yticks(range(len(matrix_df)))
        ax.set_yticklabels(matrix_df["endpoint"].tolist())
        for i in range(len(matrix_df)):
            for j, col in enumerate(phase_cols):
                value = scaled.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10, color="#172033")
        plt.colorbar(im, ax=ax, shrink=0.82, label="Within-endpoint scaled response")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_session_phase_contrast_heatmap(self, matrix_df: pd.DataFrame):
        if matrix_df.empty:
            return None
        phase_cols = [col for col in ["FCS", "SR", "FFC", "SS", "OC"] if col in matrix_df.columns]
        if not phase_cols:
            return None
        values = matrix_df[phase_cols].apply(pd.to_numeric, errors="coerce")
        if values.dropna(how="all").empty:
            return None
        scaled = values.copy()
        for idx in scaled.index:
            row = scaled.loc[idx]
            valid = row.dropna()
            if valid.empty:
                continue
            vmax = float(valid.abs().max())
            scaled.loc[idx, valid.index] = 0.0 if vmax <= 1e-6 else valid / vmax
        fig, ax = plt.subplots(figsize=self._figsize("matrix"))
        im = ax.imshow(scaled.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(len(phase_cols)))
        ax.set_xticklabels(phase_cols)
        ax.set_yticks(range(len(matrix_df)))
        ax.set_yticklabels(matrix_df["endpoint"].tolist())
        for i in range(len(matrix_df)):
            for j, col in enumerate(phase_cols):
                value = scaled.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10, color="#172033")
        plt.colorbar(im, ax=ax, shrink=0.82, label="Within-endpoint signed display scale")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_session_repeatability_summary(self, matrix_df: pd.DataFrame):
        if matrix_df.empty:
            return None
        phase_cols = [col for col in ["FCS", "SR", "FFC", "SS", "OC"] if col in matrix_df.columns]
        if not phase_cols:
            return None
        values = matrix_df[phase_cols].apply(pd.to_numeric, errors="coerce")
        if values.dropna(how="all").empty:
            return None
        fig, ax = plt.subplots(figsize=self._figsize("matrix"))
        im = ax.imshow(values.to_numpy(dtype=float), aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(len(phase_cols)))
        ax.set_xticklabels(phase_cols)
        ax.set_yticks(range(len(matrix_df)))
        ax.set_yticklabels(matrix_df["endpoint"].tolist())
        for i in range(len(matrix_df)):
            for j, col in enumerate(phase_cols):
                value = values.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10, color="#172033")
        plt.colorbar(im, ax=ax, shrink=0.82, label="Directional agreement across repeated blocks")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _relationship_min_paired_n(self, metric_a: str, metric_b: str) -> int:
        if self._is_questionnaire_endpoint(metric_a) or self._is_questionnaire_endpoint(metric_b):
            return 8
        return int(self.config.runtime.min_sensor_overlap_minutes)

    def _session_relationship_matrix(self, minute: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
        if minute.empty or not metrics:
            return pd.DataFrame()
        comparison_minute = minute.loc[minute["protocol_block"].astype(str).isin(COMPARISON_BLOCKS)].copy()
        if comparison_minute.empty:
            return pd.DataFrame()
        rows = []
        for i, source in enumerate(metrics):
            if source not in comparison_minute.columns:
                continue
            for target in metrics[i + 1:]:
                if target not in comparison_minute.columns:
                    continue
                pair = comparison_minute[[source, target]].apply(to_numeric).dropna()
                paired_n = int(len(pair))
                min_n = self._relationship_min_paired_n(source, target)
                eligible = paired_n >= min_n
                overall_r = float(pair[source].corr(pair[target], method="spearman")) if eligible else np.nan
                qualified_phase_signs = []
                for phase_name in self._comparison_phase_sequence(comparison_minute["protocol_phase"].astype(str).unique()):
                    phase_pair = comparison_minute.loc[comparison_minute["protocol_phase"].astype(str) == phase_name, [source, target]].apply(to_numeric).dropna()
                    phase_min_n = max(3, min_n // 4)
                    if len(phase_pair) < phase_min_n:
                        continue
                    phase_r = phase_pair[source].corr(phase_pair[target], method="spearman")
                    if pd.notna(phase_r) and phase_r != 0:
                        qualified_phase_signs.append(float(np.sign(phase_r)))
                same_sign_fraction = np.nan
                if eligible and pd.notna(overall_r) and overall_r != 0 and qualified_phase_signs:
                    overall_sign = float(np.sign(overall_r))
                    same_sign_fraction = float(np.mean(np.array(qualified_phase_signs) == overall_sign))
                qualified_phases = int(len(qualified_phase_signs))
                phase_support_status = (
                    "same-sign across phases"
                    if qualified_phases >= 2 and pd.notna(same_sign_fraction) and same_sign_fraction >= 0.67
                    else ("limited phase support" if eligible else "insufficient phase support")
                )
                rows.append(
                    {
                        "source": FEATURE_LABELS.get(source, source),
                        "target": FEATURE_LABELS.get(target, target),
                        "spearman_r": overall_r,
                        "paired_n": paired_n,
                        "min_required_n": min_n,
                        "qualified_phases": qualified_phases,
                        "same_sign_fraction": same_sign_fraction,
                        "relationship_status": "retained descriptive association" if eligible else "insufficient paired support",
                        "phase_support_status": phase_support_status,
                    }
                )
        return pd.DataFrame(rows)

    def _fig_session_relationship_heatmap(self, relation_df: pd.DataFrame):
        if relation_df.empty:
            return None
        labels = sorted(set(relation_df["source"]).union(set(relation_df["target"])))
        if not labels:
            return None
        pivot = pd.DataFrame(np.nan, index=labels, columns=labels)
        for row in relation_df.itertuples():
            pivot.loc[row.source, row.target] = row.spearman_r
            pivot.loc[row.target, row.source] = row.spearman_r
        fig, ax = plt.subplots(figsize=self._figsize("matrix_tall"))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="equal", cmap="coolwarm", vmin=-1, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10, color="#172033")
        plt.colorbar(im, ax=ax, shrink=0.82, label="Spearman r")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_session_targeted_relationships(self, minute: pd.DataFrame, metrics: list[str]):
        if minute.empty or not metrics or "thermal_comfort" not in metrics:
            return None
        comparison_minute = minute.loc[minute["protocol_block"].astype(str).isin(COMPARISON_BLOCKS)].copy()
        if comparison_minute.empty:
            return None
        target_order = ["indoor_air_velocity_mean_m_s", "indoor_air_temp_mean_C", "empatica_eda_mean_uS"]
        targets = [metric for metric in target_order if metric in metrics]
        if not targets:
            return None
        fig, axes = plt.subplots(1, len(targets), figsize=(4.6 * len(targets), 4.5))
        if len(targets) == 1:
            axes = [axes]
        panel_positions = ["Left", "Center", "Right"]
        panel_notes = []
        comfort_label = FEATURE_LABELS.get("thermal_comfort", "Thermal Comfort")
        for ax, metric, position in zip(axes, targets, panel_positions):
            pair = comparison_minute[[metric, "thermal_comfort"]].apply(to_numeric).dropna()
            min_n = self._relationship_min_paired_n(metric, "thermal_comfort")
            if len(pair) < min_n:
                ax.axis("off")
                panel_notes.append(f"{position} shows no retained relationship for {FEATURE_LABELS.get(metric, metric)} because paired support is below n={min_n}.")
                continue
            x = pair[metric]
            y = pair["thermal_comfort"]
            ax.scatter(x, y, s=26, alpha=0.78, color="#2563eb")
            r = float(x.corr(y, method="spearman"))
            qualified_phase_signs = []
            for phase_name in self._comparison_phase_sequence(comparison_minute["protocol_phase"].astype(str).unique()):
                phase_pair = comparison_minute.loc[comparison_minute["protocol_phase"].astype(str) == phase_name, [metric, "thermal_comfort"]].apply(to_numeric).dropna()
                if len(phase_pair) < 3:
                    continue
                phase_r = phase_pair[metric].corr(phase_pair["thermal_comfort"], method="spearman")
                if pd.notna(phase_r) and phase_r != 0:
                    qualified_phase_signs.append(float(np.sign(phase_r)))
            same_sign_fraction = np.nan
            if qualified_phase_signs and r != 0:
                same_sign_fraction = float(np.mean(np.array(qualified_phase_signs) == float(np.sign(r))))
            ax.set_xlabel(self._axis_label(metric))
            ax.set_ylabel(comfort_label)
            self._apply_discrete_y_axis_matplotlib(ax, y, "thermal_comfort")
            stability_note = f", same-sign fraction across phases = {same_sign_fraction:.2f}" if pd.notna(same_sign_fraction) else ""
            panel_notes.append(f"{position} shows {comfort_label.lower()} versus {FEATURE_LABELS.get(metric, metric)} with Spearman r = {r:.2f}, n = {len(pair)} retained paired observations{stability_note}.")
        fig._cltr_panel_notes = panel_notes
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _cohort_endpoint_support_profile(self, cohort_phase: pd.DataFrame) -> pd.DataFrame:
        if cohort_phase.empty:
            return pd.DataFrame()
        comparison = cohort_phase.loc[cohort_phase["protocol_phase"].isin(self._comparison_phase_sequence(cohort_phase["protocol_phase"]))].copy()
        if comparison.empty:
            return pd.DataFrame()
        phases = self._comparison_phase_sequence(comparison["protocol_phase"])
        conditions = [c for c in CONDITION_ORDER if c in set(comparison["condition_code"].astype(str))]
        if not conditions:
            conditions = sorted(comparison["condition_code"].astype(str).dropna().unique().tolist())
        total_cells = max(1, len(phases) * max(1, len(conditions)))
        rows = []
        for metric in [m for m in COHORT_SUPPORT_GRADED_METRICS if m in comparison.columns]:
            supported = comparison.loc[to_numeric(comparison[metric]).notna()].copy()
            grouped = (
                supported.groupby(["condition_code", "protocol_phase"])
                .agg(n_sessions=("session_id", "nunique"), n_participants=("participant_id", "nunique"))
                .reset_index()
            ) if not supported.empty else pd.DataFrame(columns=["condition_code", "protocol_phase", "n_sessions", "n_participants"])
            supported_phases = int(grouped["protocol_phase"].nunique()) if not grouped.empty else 0
            supported_conditions = int(grouped["condition_code"].nunique()) if not grouped.empty else 0
            supported_cells = int(len(grouped))
            cell_coverage_fraction = float(supported_cells / total_cells)
            median_sessions_per_cell = float(grouped["n_sessions"].median()) if not grouped.empty else 0.0
            total_valid_phase_summaries = int(len(supported))
            if (
                supported_phases >= max(4, len(phases) - 1)
                and supported_conditions >= min(2, max(1, len(conditions)))
                and cell_coverage_fraction >= 0.6
                and median_sessions_per_cell >= 2
            ):
                grade = "strong"
                reading = "Broad repeated support across the available cohort comparison grid."
            elif supported_phases >= 2 and cell_coverage_fraction >= 0.3 and median_sessions_per_cell >= 1:
                grade = "partial"
                reading = "Descriptively visible across the available cohort grid, but repeated cohort support is still limited."
            else:
                grade = "insufficient"
                reading = "Too sparse across the cohort comparison grid for a stable cohort-level reading."
            if self._is_questionnaire_endpoint(metric):
                basis = "questionnaire responses"
            elif self._is_control_signal_channel(metric):
                basis = "valid phase summaries (control/context)"
            elif metric.endswith("_delta_bpm") or metric.endswith("_delta_uS") or metric.endswith("_delta_C"):
                basis = "derived phase summaries"
            else:
                basis = "valid phase summaries"
            rows.append(
                {
                    "metric": metric,
                    "endpoint": FEATURE_LABELS.get(metric, metric),
                    "support_grade": grade,
                    "support_basis": basis,
                    "supported_phases": supported_phases,
                    "supported_conditions": supported_conditions,
                    "supported_condition_phase_cells": supported_cells,
                    "cell_coverage_fraction": cell_coverage_fraction,
                    "median_sessions_per_condition_phase": median_sessions_per_cell,
                    "total_valid_phase_summaries": total_valid_phase_summaries,
                    "scientific_reading": reading,
                }
            )
        return pd.DataFrame(rows)

    def _cohort_primary_metrics(self, support_profile: pd.DataFrame) -> list[str]:
        if support_profile.empty:
            return []
        support_profile = support_profile.loc[support_profile["metric"].isin(PRIMARY_ENDPOINTS)].copy()
        if support_profile.empty:
            return []
        strong = support_profile.loc[support_profile["support_grade"] == "strong", "metric"].tolist()
        if strong:
            return strong
        return support_profile.loc[support_profile["support_grade"] == "partial", "metric"].tolist()

    def _metric_signal_streams(self, metric: str) -> list[str]:
        mapping = {
            "empatica_bvp_mean": ["empatica_bvp"],
            "empatica_hr_mean_bpm": ["empatica_hr"],
            "biopac_hr_mean_bpm": ["biopac_hr"],
            "empatica_eda_mean_uS": ["empatica_eda"],
            "biopac_eda_mean_uS": ["biopac_eda"],
            "empatica_temp_mean_C": ["empatica_temp"],
            "biopac_temp_chest_mean_C": ["biopac_temp"],
            "empatica_acc_mean_g": ["empatica_acc"],
            "empatica_enmo_mean_g": ["empatica_enmo"],
            "empatica_steps": ["empatica_steps"],
            "biopac_temp_thigh_mean_C": ["biopac_temp_thigh"],
            "biopac_temp_arm_mean_C": ["biopac_temp_arm"],
            "biopac_temp_tibia_mean_C": ["biopac_temp_tibia"],
            "biopac_bloodflow_mean_bpu": ["biopac_bloodflow"],
            "biopac_backscatter_mean_percent": ["biopac_backscatter"],
            "hr_delta_bpm": ["empatica_hr", "biopac_hr"],
            "eda_delta_uS": ["empatica_eda", "biopac_eda"],
            "temp_delta_C": ["empatica_temp", "biopac_temp"],
        }
        return mapping.get(str(metric), [])

    def _signal_role_map(self, signal_audit_summary: pd.DataFrame) -> dict[str, str]:
        if signal_audit_summary is None or signal_audit_summary.empty:
            return {}
        out: dict[str, str] = {}
        for _, row in signal_audit_summary.iterrows():
            out[str(row.get("signal_stream", ""))] = str(row.get("recommended_role", ""))
        return out

    def _metric_allowed_in_scenario(self, metric: str, role_map: dict[str, str], scenario: str) -> bool:
        streams = self._metric_signal_streams(metric)
        if not streams:
            return True
        if scenario == "all_sources":
            return True
        if scenario == "valid_only":
            allowed_roles = {"primary", "primary_with_qc"}
            return all(role_map.get(stream, "") in allowed_roles for stream in streams)
        return True

    def _filter_support_profile_for_scenario(
        self,
        support_profile: pd.DataFrame,
        signal_audit_summary: pd.DataFrame,
        scenario: str,
    ) -> pd.DataFrame:
        if support_profile.empty:
            return support_profile
        role_map = self._signal_role_map(signal_audit_summary)
        keep_mask = support_profile["metric"].astype(str).map(lambda metric: self._metric_allowed_in_scenario(metric, role_map, scenario))
        return support_profile.loc[keep_mask].copy().reset_index(drop=True)

    def _scenario_register(self, signal_audit_summary: pd.DataFrame) -> pd.DataFrame:
        role_map = self._signal_role_map(signal_audit_summary)
        catalog_order = [str(item["signal_stream"]) for item in DEVICE_STREAM_CATALOG]
        all_streams = list(catalog_order)
        for stream in sorted(role_map):
            if stream not in all_streams:
                all_streams.append(stream)
        valid_streams = [stream for stream in all_streams if role_map.get(stream, "") in {"primary", "primary_with_qc"}]
        excluded_streams = [stream for stream in all_streams if stream not in valid_streams]
        fmt = lambda items: ", ".join(self._fmt_cell(item) for item in items) if items else "None"
        excluded_notes = []
        for stream in excluded_streams:
            excluded_notes.append(f"{self._fmt_cell(stream)} ({self._fmt_cell(role_map.get(stream, 'not_audited'))})")
        return pd.DataFrame(
            [
                {
                    "scenario": "All-source",
                    "included_streams": fmt(all_streams),
                    "excluded_streams": "None",
                    "scientific_use": "Audit view that preserves every available modality, including limited or subset-only streams.",
                },
                {
                    "scenario": "Valid-only",
                    "included_streams": fmt(valid_streams),
                    "excluded_streams": ", ".join(excluded_notes) if excluded_notes else "None",
                    "scientific_use": "Claim-supporting view restricted to streams judged primary or primary-with-QC in the signal audit.",
                },
            ]
        )

    def _modality_claim_register(self, signal_audit_summary: pd.DataFrame) -> pd.DataFrame:
        if signal_audit_summary is None or signal_audit_summary.empty:
            return pd.DataFrame()
        rows = []
        for _, row in signal_audit_summary.iterrows():
            role = str(row.get("recommended_role", ""))
            if role == "primary":
                manuscript_use = "Claim-supporting"
                manuscript_claim = "May support primary manuscript claims for this construct."
            elif role == "primary_with_qc":
                manuscript_use = "Claim-supporting with caveat"
                manuscript_claim = "May support claims only if interpreted as device-specific and QC-qualified."
            elif role in {"secondary_validation", "secondary_only", "subset_only"}:
                manuscript_use = "Audit or sensitivity only"
                manuscript_claim = "Do not use as standalone primary evidence; restrict to audit, validation, or sensitivity analyses."
            else:
                manuscript_use = "Do not claim"
                manuscript_claim = "Not strong enough for manuscript evidence in the current release."
            rows.append(
                {
                    "signal_stream": row.get("signal_stream", ""),
                    "construct": row.get("construct", ""),
                    "adequacy_status": row.get("adequacy_status", ""),
                    "recommended_role": role,
                    "manuscript_use": manuscript_use,
                    "manuscript_claim": manuscript_claim,
                }
            )
        return pd.DataFrame(rows)

    def _endpoint_claim_register(self, support_profile: pd.DataFrame, signal_audit_summary: pd.DataFrame) -> pd.DataFrame:
        if support_profile is None or support_profile.empty:
            return pd.DataFrame()
        role_map = self._signal_role_map(signal_audit_summary)
        rows = []
        for _, row in support_profile.iterrows():
            metric = str(row.get("metric", ""))
            streams = self._metric_signal_streams(metric)
            roles = [role_map.get(stream, "") for stream in streams if role_map.get(stream, "")]
            if not streams:
                modality_gate = "not modality-gated"
                claim_status = "Claim-supporting" if str(row.get("support_grade", "")) == "strong" else "Descriptive only"
                claim_note = "Endpoint is not directly limited by wearable/lab modality validity; support grade controls its use."
            elif roles and all(role in {"primary", "primary_with_qc"} for role in roles):
                modality_gate = "valid-only eligible"
                claim_status = "Claim-supporting" if str(row.get("support_grade", "")) == "strong" else "Descriptive with modality support"
                claim_note = "Underlying modality streams pass the valid-only screen."
            elif any(role in {"subset_only", "secondary_only", "secondary_validation", "not_primary", "not_recommended"} for role in roles):
                modality_gate = "audit-only"
                claim_status = "Audit or sensitivity only"
                claim_note = "At least one required source stream is not eligible for valid-only interpretation."
            else:
                modality_gate = "unclear"
                claim_status = "Needs review"
                claim_note = "Modality-role mapping needs manual review."
            rows.append(
                {
                    "endpoint": row.get("endpoint", metric),
                    "support_grade": row.get("support_grade", ""),
                    "support_basis": row.get("support_basis", ""),
                    "modality_gate": modality_gate,
                    "claim_status": claim_status,
                    "claim_note": claim_note,
                }
            )
        return pd.DataFrame(rows)

    def _device_stream_inventory_register(self, minute: pd.DataFrame, signal_audit_summary: pd.DataFrame) -> pd.DataFrame:
        role_map = self._signal_role_map(signal_audit_summary)
        audit_lookup = {}
        if signal_audit_summary is not None and not signal_audit_summary.empty:
            for _, row in signal_audit_summary.iterrows():
                audit_lookup[str(row.get("signal_stream", ""))] = row
        minute_cols = set(minute.columns) if minute is not None and not minute.empty else set()
        rows = []
        for item in DEVICE_STREAM_CATALOG:
            signal_stream = str(item["signal_stream"])
            metric = str(item["metric"])
            audit_row = audit_lookup.get(signal_stream)
            audited = "yes" if audit_row is not None else "no"
            comparable = "yes" if signal_stream in {"empatica_hr", "biopac_hr", "empatica_eda", "biopac_eda", "empatica_temp", "biopac_temp"} else "no"
            if comparable == "yes":
                comparison_class = "directly_comparable"
            elif item["construct"] == "temperature_site":
                comparison_class = "same_construct_not_paired"
            elif item["construct"] in {"heart_rate", "eda", "temperature"}:
                comparison_class = "device_specific"
            else:
                comparison_class = "source_only"
            role = str(audit_row.get("recommended_role", "")) if audit_row is not None else ""
            adequacy = str(audit_row.get("adequacy_status", "")) if audit_row is not None else ""
            is_direct_analytic_feature = metric in ANALYTIC_FEATURES
            if comparison_class == "same_construct_not_paired":
                scientific_use = "Site-specific thermal stream; interpret within-device, not as a wearable-lab pair."
            elif role == "primary_with_qc":
                scientific_use = "Retain for analysis with QC caveat; do not treat as a clean interchangeable reference."
            elif is_direct_analytic_feature:
                scientific_use = "Retain as a direct analytic stream in the current cohort feature set."
            else:
                scientific_use = "Retain as an audited/report-only stream; not promoted into the core analytic feature set."
            rows.append(
                {
                    "stream_label": item["label"],
                    "device": item["device"],
                    "construct": item["construct"],
                    "comparison_class": comparison_class,
                    "present_in_cohort_table": "yes" if metric in minute_cols else "no",
                    "signal_audited": audited,
                    "cross_device_comparable": comparable,
                    "analytic_feature": "yes" if is_direct_analytic_feature else "no",
                    "stream_usage": "direct_analytic_feature" if is_direct_analytic_feature else "audit_report_only",
                    "scientific_use": scientific_use,
                    "primary_endpoint": "yes" if endpoint_is_primary(metric) else "no",
                    "endpoint_policy_role": endpoint_policy_role(metric),
                    "recommended_role": role if role else "not_audited",
                    "adequacy_status": adequacy if adequacy else "not_audited",
                }
            )
        return pd.DataFrame(rows)

    def _analysis_pathway_register(self, minute: pd.DataFrame, support_profile: pd.DataFrame, signal_audit_summary: pd.DataFrame) -> pd.DataFrame:
        role_map = self._signal_role_map(signal_audit_summary)
        support_lookup = {}
        if support_profile is not None and not support_profile.empty:
            for _, row in support_profile.iterrows():
                support_lookup[str(row.get("metric", ""))] = row
        minute_cols = set(minute.columns) if minute is not None and not minute.empty else set()
        rows = []
        catalog_metrics = {str(item["metric"]) for item in DEVICE_STREAM_CATALOG}
        all_metrics = sorted(set(COHORT_DERIVED_ENDPOINTS) | set(ANALYTIC_FEATURES) | set(PRIMARY_ENDPOINTS) | catalog_metrics)
        for metric in all_metrics:
            if metric not in minute_cols and metric not in support_lookup:
                continue
            streams = self._metric_signal_streams(metric)
            roles = [role_map.get(stream, "") for stream in streams if role_map.get(stream, "")]
            if not streams:
                pathway = "derived/context endpoint"
            elif all(role in {"primary", "primary_with_qc"} for role in roles):
                pathway = "valid-only eligible"
            elif any(role in {"subset_only", "secondary_only", "secondary_validation", "not_primary", "not_recommended"} for role in roles):
                pathway = "audit-only if included"
            else:
                pathway = "stream-role unclear"
            support_row = support_lookup.get(metric)
            if support_row is not None:
                support_grade = str(support_row.get("support_grade", "")) or "not_scored"
                support_basis = str(support_row.get("support_basis", "")) or "not scored in current cohort support profile"
            elif metric in COHORT_SUPPORT_GRADED_METRICS:
                support_grade = "not_scored"
                support_basis = "support-graded endpoint, but no cohort support row was produced"
            elif metric in catalog_metrics:
                support_grade = "not_scored"
                support_basis = "stream inventory metric; not part of cohort endpoint support grading"
            else:
                support_grade = "not_scored"
                support_basis = "not included in the current cohort support-graded endpoint set"
            rows.append(
                {
                    "endpoint": FEATURE_LABELS.get(metric, metric),
                    "metric": metric,
                    "in_cohort_table": "yes" if metric in minute_cols else "no",
                    "source_streams": ", ".join(self._fmt_cell(stream) for stream in streams) if streams else "Not modality-gated",
                    "support_grade": support_grade,
                    "support_basis": support_basis,
                    "primary_endpoint": "yes" if endpoint_is_primary(metric) else "no",
                    "endpoint_policy_role": endpoint_policy_role(metric),
                    "pathway_status": pathway,
                }
            )
        return pd.DataFrame(rows)

    def _flagged_stream_session_register(self, session_signal_audit: pd.DataFrame) -> pd.DataFrame:
        if session_signal_audit is None or session_signal_audit.empty:
            return pd.DataFrame()
        temp = session_signal_audit.copy()
        temp["concern_score"] = to_numeric(temp["concern_score"])
        temp["coverage_penalty"] = (1.0 - to_numeric(temp.get("coverage_fraction", pd.Series(dtype=float))).fillna(0).clip(0, 1)) * 45.0
        temp["plausibility_penalty"] = (1.0 - to_numeric(temp.get("plausible_fraction", pd.Series(dtype=float))).fillna(0).clip(0, 1)) * 30.0
        quality = to_numeric(temp.get("quality_fraction", pd.Series(dtype=float)))
        temp["quality_penalty"] = 0.0
        quality_mask = quality.notna()
        temp.loc[quality_mask, "quality_penalty"] = (1.0 - quality.loc[quality_mask].clip(0, 1)) * 20.0
        paired_eligible = to_numeric(temp.get("paired_eligible", pd.Series(dtype=float))).fillna(0)
        paired_spearman = to_numeric(temp.get("paired_spearman_r", pd.Series(dtype=float)))
        temp["agreement_penalty"] = 0.0
        agreement_mask = (paired_eligible > 0) & paired_spearman.notna()
        temp.loc[agreement_mask, "agreement_penalty"] = ((0.7 - paired_spearman.loc[agreement_mask]) * 25.0).clip(lower=0, upper=15)
        flagged = temp.loc[temp["concern_score"].fillna(0) > 0].copy()
        if flagged.empty:
            return pd.DataFrame()
        rows = []
        for signal_stream, d in flagged.groupby("signal_stream", sort=False):
            d = d.sort_values(["concern_score", "session_id"], ascending=[False, True]).copy()
            example_bits = []
            for row in d.head(3).itertuples(index=False):
                example_bits.append(
                    f"{self._fmt_cell(getattr(row, 'session_id', ''))} ({float(getattr(row, 'concern_score', np.nan)):.1f})"
                )
            penalty_means = {
                "coverage": float(to_numeric(d["coverage_penalty"]).mean()) if "coverage_penalty" in d.columns else 0.0,
                "plausibility": float(to_numeric(d["plausibility_penalty"]).mean()) if "plausibility_penalty" in d.columns else 0.0,
                "quality": float(to_numeric(d["quality_penalty"]).mean()) if "quality_penalty" in d.columns else 0.0,
                "agreement": float(to_numeric(d["agreement_penalty"]).mean()) if "agreement_penalty" in d.columns else 0.0,
            }
            driver_map = {
                "coverage": "Missingness / coverage",
                "plausibility": "Plausibility / out-of-range values",
                "quality": "Quality flag support",
                "agreement": "Cross-device agreement",
            }
            primary_driver_key = max(penalty_means, key=penalty_means.get)
            concern_profile = (
                f"C:{penalty_means['coverage']:.1f} | "
                f"P:{penalty_means['plausibility']:.1f} | "
                f"Q:{penalty_means['quality']:.1f} | "
                f"A:{penalty_means['agreement']:.1f}"
            )
            rows.append(
                {
                    "signal_stream": signal_stream,
                    "device": d["device"].iloc[0],
                    "construct": d["construct"].iloc[0],
                    "flagged_session_streams": int(len(d)),
                    "affected_sessions": int(d["session_id"].astype(str).nunique()),
                    "primary_concern_driver": driver_map.get(primary_driver_key, primary_driver_key),
                    "concern_profile": concern_profile,
                    "top_flagged_sessions": ", ".join(example_bits),
                    "max_concern_score": float(d["concern_score"].max()) if d["concern_score"].notna().any() else np.nan,
                }
            )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values(["flagged_session_streams", "max_concern_score", "signal_stream"], ascending=[False, False, True]).reset_index(drop=True)

    def _fig_cohort_hr_scenarios(self, cohort_phase: pd.DataFrame, signal_audit_summary: pd.DataFrame):
        if cohort_phase.empty:
            return None
        role_map = self._signal_role_map(signal_audit_summary)
        scenarios = [
            ("All-source HR", [("empatica_hr_mean_bpm", "Empatica HR", "#b91c1c"), ("biopac_hr_mean_bpm", "BIOPAC HR", "#111827")]),
            (
                "Valid-only HR",
                [
                    item
                    for item in [("empatica_hr_mean_bpm", "Empatica HR", "#b91c1c"), ("biopac_hr_mean_bpm", "BIOPAC HR", "#111827")]
                    if self._metric_allowed_in_scenario(item[0], role_map, "valid_only")
                ],
            ),
        ]
        if not any(metric in cohort_phase.columns for _, items in scenarios for metric, _, _ in items):
            return None
        fig, axes = plt.subplots(len(scenarios), 1, figsize=(12.8, 4.2 * len(scenarios)), sharex=True)
        if len(scenarios) == 1:
            axes = [axes]
        panel_notes: list[str] = []
        for ax, (title, items) in zip(axes, scenarios):
            available = [(metric, label, color) for metric, label, color in items if metric in cohort_phase.columns and to_numeric(cohort_phase[metric]).notna().any()]
            if not available:
                ax.axis("off")
                panel_notes.append(f"{title} contains no retained cohort HR stream under the current scenario.")
                continue
            for metric, label, color in available:
                grouped = (
                    cohort_phase.loc[to_numeric(cohort_phase[metric]).notna()]
                    .groupby("protocol_phase")[metric]
                    .agg(q25=lambda s: s.quantile(0.25), median="median", q75=lambda s: s.quantile(0.75))
                    .reindex(self._comparison_phase_sequence(cohort_phase["protocol_phase"]))
                )
                x = np.arange(len(grouped.index))
                median = to_numeric(grouped["median"])
                q25 = to_numeric(grouped["q25"])
                q75 = to_numeric(grouped["q75"])
                valid = median.notna()
                ax.fill_between(x[valid], q25[valid], q75[valid], color=color, alpha=0.16)
                ax.plot(x[valid], median[valid], marker="o", lw=2.2, color=color, label=label)
            ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
            ax.set_ylabel(self._axis_label("biopac_hr_mean_bpm"))
            ax.set_xticks(range(len(self._comparison_phase_sequence(cohort_phase["protocol_phase"]))))
            ax.set_xticklabels([PHASE_ABBR.get(p, p[:3].upper()) for p in self._comparison_phase_sequence(cohort_phase["protocol_phase"])])
            ax.grid(True, axis="y", color="#e2e8f0")
            self._place_topbar_legend(ax)
            panel_notes.append(f"{title} retains {', '.join(label for _, label, _ in available)}.")
        axes[-1].set_xlabel(self._phase_axis_label())
        fig._cltr_panel_notes = panel_notes
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_cohort_modality_scenarios(
        self,
        minute: pd.DataFrame,
        signal_audit_summary: pd.DataFrame,
        construct: str,
    ):
        if minute.empty:
            return None
        construct_map = {
            "heart_rate": [("empatica_hr_mean_bpm", "Empatica HR", "#b91c1c"), ("biopac_hr_mean_bpm", "BIOPAC HR", "#111827")],
            "eda": [("empatica_eda_mean_uS", "Empatica EDA", "#1d4ed8"), ("biopac_eda_mean_uS", "BIOPAC EDA", "#2563eb")],
            "temperature": [("empatica_temp_mean_C", "Empatica Temperature", "#ea580c"), ("biopac_temp_chest_mean_C", "Chest Temperature", "#dc2626")],
        }
        ylabels = {
            "heart_rate": "Heart rate (bpm)",
            "eda": "EDA (uS)",
            "temperature": "Temperature (C)",
        }
        specs = construct_map.get(str(construct), [])
        if not specs:
            return None
        role_map = self._signal_role_map(signal_audit_summary)
        scenarios = [
            ("All-source", specs),
            ("Valid-only", [item for item in specs if self._metric_allowed_in_scenario(item[0], role_map, "valid_only")]),
        ]
        fig, axes = plt.subplots(len(scenarios), 1, figsize=(13.0, 4.0 * len(scenarios)), sharex=True)
        if len(scenarios) == 1:
            axes = [axes]
        phase_template = (
            minute.loc[:, [c for c in ["minute_index", "protocol_phase"] if c in minute.columns]]
            .dropna()
            .sort_values("minute_index")
            .drop_duplicates(subset=["minute_index"], keep="first")
        )
        panel_notes: list[str] = []
        for ax, (scenario_title, scenario_specs) in zip(axes, scenarios):
            available = [(metric, label, color) for metric, label, color in scenario_specs if metric in minute.columns and to_numeric(minute[metric]).notna().any()]
            if not available:
                ax.axis("off")
                panel_notes.append(f"{scenario_title} contains no retained {construct.replace('_', ' ')} stream under the current scenario.")
                continue
            self._add_phase_spans(ax, phase_template)
            for metric, label, color in available:
                grouped = (
                    minute.dropna(subset=["condition_code", metric])
                    .groupby(["condition_code", "minute_index"])[metric]
                    .agg(q25=lambda s: s.quantile(0.25), median="median", q75=lambda s: s.quantile(0.75))
                    .reset_index()
                )
                for cond in [x for x in CONDITION_ORDER if x in grouped["condition_code"].astype(str).unique()]:
                    cur = grouped.loc[grouped["condition_code"].astype(str) == cond].sort_values("minute_index")
                    x = to_numeric(cur["minute_index"])
                    q25 = to_numeric(cur["q25"])
                    median = to_numeric(cur["median"])
                    q75 = to_numeric(cur["q75"])
                    valid = median.notna()
                    ax.fill_between(x[valid], q25[valid], q75[valid], color=color, alpha=0.08)
                    ax.plot(
                        x[valid],
                        median[valid],
                        color=color,
                        lw=2.0,
                        alpha=0.95 if cond in {"DIM-MOR", "BRI-MOR"} else 0.65,
                        ls="-" if cond in {"DIM-MOR", "BRI-MOR"} else "--",
                    )
            ax.set_ylabel(ylabels.get(str(construct), construct))
            ax.grid(True, axis="y", color="#e2e8f0")
            handles = [plt.Line2D([0], [0], color=color, lw=2, label=label) for _, label, color in available]
            self._place_topbar_legend(ax, handles=handles)
            panel_notes.append(
                f"{scenario_title} scenario retains {', '.join(label for _, label, _ in available)} for {construct.replace('_', ' ')}."
            )
        axes[-1].set_xlabel(self._time_axis_label())
        fig._cltr_panel_notes = panel_notes
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _cohort_response_matrix(self, cohort_phase: pd.DataFrame, support_profile: pd.DataFrame) -> pd.DataFrame:
        metrics = self._cohort_primary_metrics(support_profile)
        if cohort_phase.empty or not metrics:
            return pd.DataFrame()
        comparison = cohort_phase.loc[cohort_phase["protocol_phase"].isin(self._comparison_phase_sequence(cohort_phase["protocol_phase"]))].copy()
        phases = self._comparison_phase_sequence(comparison["protocol_phase"])
        conditions = [c for c in CONDITION_ORDER if c in set(comparison["condition_code"].astype(str))]
        if not conditions:
            conditions = sorted(comparison["condition_code"].astype(str).dropna().unique().tolist())
        rows = []
        for metric in metrics:
            basis = support_profile.loc[support_profile["metric"] == metric, "support_basis"].iloc[0]
            for condition in conditions:
                d = comparison.loc[(comparison["condition_code"].astype(str) == condition) & to_numeric(comparison[metric]).notna()].copy()
                if d.empty:
                    continue
                row = {
                    "metric": metric,
                    "endpoint": FEATURE_LABELS.get(metric, metric),
                    "row_label": f"{FEATURE_LABELS.get(metric, metric)} | {condition}",
                    "condition_code": condition,
                    "support_basis": basis,
                    "n_sessions": int(d["session_id"].nunique()),
                    "total_valid_phase_summaries": int(len(d)),
                }
                support_notes = []
                for phase_name in phases:
                    abbr = PHASE_ABBR.get(phase_name, phase_name[:3].upper())
                    vals = to_numeric(d.loc[d["protocol_phase"] == phase_name, metric]).dropna()
                    row[abbr] = float(vals.median()) if not vals.empty else np.nan
                    support_notes.append(f"{abbr}:{int(d.loc[d['protocol_phase'] == phase_name, 'session_id'].nunique())}")
                row["condition_phase_support"] = " | ".join(support_notes)
                rows.append(row)
        return pd.DataFrame(rows)

    def _cohort_response_heatmap(self, matrix_df: pd.DataFrame):
        if matrix_df.empty:
            return None
        phase_cols = [c for c in ["FCS", "SR", "FFC", "SS", "OC"] if c in matrix_df.columns]
        if not phase_cols:
            return None
        scaled = matrix_df[phase_cols].apply(pd.to_numeric, errors="coerce").copy()
        for idx in scaled.index:
            valid = scaled.loc[idx].dropna()
            if valid.empty:
                continue
            vmin = float(valid.min())
            vmax = float(valid.max())
            scaled.loc[idx, valid.index] = 0.0 if abs(vmax - vmin) <= 1e-6 else (valid - vmin) / (vmax - vmin)
        fig, ax = plt.subplots(figsize=self._figsize("matrix_tall"))
        im = ax.imshow(scaled.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(len(phase_cols)))
        ax.set_xticklabels(phase_cols)
        ax.set_yticks(range(len(matrix_df)))
        ax.set_yticklabels(matrix_df["row_label"].tolist())
        for i in range(len(matrix_df)):
            for j, col in enumerate(phase_cols):
                value = scaled.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10, color="#172033")
        plt.colorbar(im, ax=ax, shrink=0.82, label="Within-row display scale")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _cohort_delta_matrix(self, cohort_phase: pd.DataFrame, support_profile: pd.DataFrame) -> pd.DataFrame:
        matrix = self._cohort_response_matrix(cohort_phase, support_profile)
        if matrix.empty:
            return pd.DataFrame()
        phase_cols = [c for c in ["FCS", "SR", "FFC", "SS", "OC"] if c in matrix.columns]
        rows = []
        for row in matrix.itertuples(index=False):
            base_phase = next((col for col in phase_cols if pd.notna(getattr(row, col))), None)
            if not base_phase:
                continue
            base_value = float(getattr(row, base_phase))
            out = {
                "metric": row.metric,
                "endpoint": row.endpoint,
                "row_label": row.row_label,
                "condition_code": row.condition_code,
                "reference_phase": base_phase,
            }
            for col in phase_cols:
                value = getattr(row, col)
                out[col] = float(value - base_value) if pd.notna(value) else np.nan
            rows.append(out)
        return pd.DataFrame(rows)

    def _cohort_delta_heatmap(self, matrix_df: pd.DataFrame):
        if matrix_df.empty:
            return None
        phase_cols = [c for c in ["FCS", "SR", "FFC", "SS", "OC"] if c in matrix_df.columns]
        scaled = matrix_df[phase_cols].apply(pd.to_numeric, errors="coerce").copy()
        for idx in scaled.index:
            valid = scaled.loc[idx].dropna()
            if valid.empty:
                continue
            vmax = float(valid.abs().max())
            scaled.loc[idx, valid.index] = 0.0 if vmax <= 1e-6 else valid / vmax
        fig, ax = plt.subplots(figsize=self._figsize("matrix_tall"))
        im = ax.imshow(scaled.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(len(phase_cols)))
        ax.set_xticklabels(phase_cols)
        ax.set_yticks(range(len(matrix_df)))
        ax.set_yticklabels(matrix_df["row_label"].tolist())
        for i in range(len(matrix_df)):
            for j, col in enumerate(phase_cols):
                value = scaled.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10, color="#172033")
        plt.colorbar(im, ax=ax, shrink=0.82, label="Within-row signed display scale")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _cohort_directional_agreement_matrix(self, cohort_phase: pd.DataFrame, support_profile: pd.DataFrame) -> pd.DataFrame:
        metrics = self._cohort_primary_metrics(support_profile)
        if cohort_phase.empty or not metrics:
            return pd.DataFrame()
        comparison = cohort_phase.loc[cohort_phase["protocol_phase"].isin(self._comparison_phase_sequence(cohort_phase["protocol_phase"]))].copy()
        phases = self._comparison_phase_sequence(comparison["protocol_phase"])
        conditions = [c for c in CONDITION_ORDER if c in set(comparison["condition_code"].astype(str))]
        if not conditions:
            conditions = sorted(comparison["condition_code"].astype(str).dropna().unique().tolist())
        rows = []
        for metric in metrics:
            for condition in conditions:
                d = comparison.loc[(comparison["condition_code"].astype(str) == condition) & to_numeric(comparison[metric]).notna()].copy()
                if d.empty:
                    continue
                signs_by_phase: dict[str, list[float]] = {phase_name: [] for phase_name in phases}
                for _, ds in d.groupby("session_id"):
                    phase_medians = {}
                    for phase_name in phases:
                        vals = to_numeric(ds.loc[ds["protocol_phase"] == phase_name, metric]).dropna()
                        if not vals.empty:
                            phase_medians[phase_name] = float(vals.median())
                    ref_phase = next((phase_name for phase_name in phases if phase_name in phase_medians), None)
                    if not ref_phase:
                        continue
                    ref_value = phase_medians[ref_phase]
                    for phase_name in phases:
                        if phase_name == ref_phase or phase_name not in phase_medians:
                            continue
                        delta = float(phase_medians[phase_name] - ref_value)
                        if delta != 0:
                            signs_by_phase[phase_name].append(float(np.sign(delta)))
                row = {
                    "metric": metric,
                    "endpoint": FEATURE_LABELS.get(metric, metric),
                    "row_label": f"{FEATURE_LABELS.get(metric, metric)} | {condition}",
                    "condition_code": condition,
                }
                notes = []
                for phase_name in phases:
                    abbr = PHASE_ABBR.get(phase_name, phase_name[:3].upper())
                    signs = signs_by_phase[phase_name]
                    row[abbr] = max(signs.count(-1.0), signs.count(1.0)) / len(signs) if len(signs) >= 2 else np.nan
                    notes.append(f"{abbr}:{len(signs)}")
                row["condition_phase_support"] = " | ".join(notes)
                rows.append(row)
        return pd.DataFrame(rows)

    def _cohort_directional_agreement_heatmap(self, matrix_df: pd.DataFrame):
        if matrix_df.empty:
            return None
        phase_cols = [c for c in ["FCS", "SR", "FFC", "SS", "OC"] if c in matrix_df.columns]
        values = matrix_df[phase_cols].apply(pd.to_numeric, errors="coerce")
        if values.dropna(how="all").empty:
            return None
        fig, ax = plt.subplots(figsize=self._figsize("matrix_tall"))
        im = ax.imshow(values.to_numpy(dtype=float), aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(len(phase_cols)))
        ax.set_xticklabels(phase_cols)
        ax.set_yticks(range(len(matrix_df)))
        ax.set_yticklabels(matrix_df["row_label"].tolist())
        for i in range(len(matrix_df)):
            for j, col in enumerate(phase_cols):
                value = values.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10, color="#172033")
        plt.colorbar(im, ax=ax, shrink=0.82, label="Session-sign agreement")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_cohort_endpoint_support_grades(self, support_profile: pd.DataFrame):
        if support_profile.empty:
            return None
        fig, ax = plt.subplots(figsize=self._figsize("wide_single"))
        temp = support_profile.sort_values(["cell_coverage_fraction", "endpoint"], ascending=[True, True]).copy()
        colors = temp["support_grade"].map({"strong": "#0f766e", "partial": "#f59e0b", "insufficient": "#b91c1c"}).fillna("#64748b")
        ax.barh(temp["endpoint"], temp["cell_coverage_fraction"], color=colors)
        for idx, row in enumerate(temp.itertuples()):
            ax.text(float(row.cell_coverage_fraction) + 0.01, idx, f"{float(row.cell_coverage_fraction):.0%}", va="center", fontsize=11, color="#172033")
        ax.set_xlabel("Supported condition-phase cells (%)")
        ax.set_xlim(0, 1.05)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(axis="x", color="#e2e8f0")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _cohort_session_delta_frame(self, cohort_phase: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
        if cohort_phase.empty or not metrics:
            return pd.DataFrame()
        comparison = cohort_phase.loc[cohort_phase["protocol_phase"].isin(self._comparison_phase_sequence(cohort_phase["protocol_phase"]))].copy()
        phases = self._comparison_phase_sequence(comparison["protocol_phase"])
        rows = []
        for session_id, d in comparison.groupby("session_id"):
            row = {
                "session_id": session_id,
                "participant_id": d["participant_id"].iloc[0],
                "condition_code": str(d["condition_code"].iloc[0]),
            }
            for metric in metrics:
                if metric not in d.columns:
                    continue
                phase_medians = {}
                for phase_name in phases:
                    vals = to_numeric(d.loc[d["protocol_phase"] == phase_name, metric]).dropna()
                    if not vals.empty:
                        phase_medians[phase_name] = float(vals.median())
                ref_phase = next((phase_name for phase_name in phases if phase_name in phase_medians), None)
                if not ref_phase:
                    continue
                ref_value = phase_medians[ref_phase]
                overall = float(np.median(list(phase_medians.values()))) if phase_medians else np.nan
                row[f"{metric}__delta"] = float(overall - ref_value) if pd.notna(overall) else np.nan
            rows.append(row)
        return pd.DataFrame(rows)

    def _cohort_relationship_min_n(self, metric_a: str, metric_b: str) -> int:
        if self._is_questionnaire_endpoint(metric_a) or self._is_questionnaire_endpoint(metric_b):
            return max(4, int(self.config.runtime.min_contrast_pairs))
        return max(4, int(self.config.runtime.min_contrast_pairs))

    def _cohort_relationship_matrix(self, cohort_phase: pd.DataFrame, support_profile: pd.DataFrame) -> pd.DataFrame:
        metrics = self._cohort_primary_metrics(support_profile)
        session_delta = self._cohort_session_delta_frame(cohort_phase, metrics)
        if session_delta.empty:
            return pd.DataFrame()
        rows = []
        for i, source in enumerate(metrics):
            scol = f"{source}__delta"
            if scol not in session_delta.columns:
                continue
            for target in metrics[i + 1:]:
                tcol = f"{target}__delta"
                if tcol not in session_delta.columns:
                    continue
                pair = session_delta[[scol, tcol, "condition_code"]].copy()
                pair.columns = ["source_value", "target_value", "condition_code"]
                pair = pair.dropna()
                paired_n = int(len(pair))
                min_n = self._cohort_relationship_min_n(source, target)
                eligible = paired_n >= min_n
                overall_r = float(pair["source_value"].corr(pair["target_value"], method="spearman")) if eligible else np.nan
                qualified_condition_signs = []
                for condition, dc in pair.groupby("condition_code"):
                    if len(dc) < 2:
                        continue
                    r = dc["source_value"].corr(dc["target_value"], method="spearman")
                    if pd.notna(r) and r != 0:
                        qualified_condition_signs.append(float(np.sign(r)))
                same_sign_fraction = np.nan
                if eligible and pd.notna(overall_r) and overall_r != 0 and qualified_condition_signs:
                    same_sign_fraction = float(np.mean(np.array(qualified_condition_signs) == float(np.sign(overall_r))))
                qualified_conditions = int(len(qualified_condition_signs))
                condition_support_status = (
                    "same-sign across conditions"
                    if qualified_conditions >= 2 and pd.notna(same_sign_fraction) and same_sign_fraction >= 0.67
                    else ("limited condition support" if eligible else "insufficient paired support")
                )
                if not eligible:
                    continue
                rows.append(
                    {
                        "source": FEATURE_LABELS.get(source, source),
                        "target": FEATURE_LABELS.get(target, target),
                        "spearman_r": overall_r,
                        "paired_n": paired_n,
                        "min_required_n": min_n,
                        "qualified_conditions": qualified_conditions,
                        "same_sign_fraction": same_sign_fraction,
                        "relationship_status": "retained descriptive association",
                        "condition_support_status": condition_support_status,
                    }
                )
        return pd.DataFrame(rows)

    def _scenario_title_prefix(self, scenario: str) -> str:
        return {
            "all_sources": "All-source",
            "valid_only": "Valid-only",
        }.get(str(scenario), "Scenario")

    def _policy_gate_register(
        self,
        c: dict,
        support_profile: pd.DataFrame,
        signal_audit_summary: pd.DataFrame,
        all_source_response_matrix: pd.DataFrame,
        valid_only_response_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []
        sample_status = c.get("sample_status", pd.DataFrame())
        if not sample_status.empty:
            row = sample_status.iloc[0]
            inferential_ok = bool(row.get("cohort_inference_eligible", 0))
            rows.append(
                {
                    "gate": "Sample adequacy",
                    "status": "pass" if inferential_ok else "descriptive_only",
                    "threshold": f">={int(row.get('min_sessions_required', 0))} sessions and >={int(row.get('min_participants_required', 0))} participants",
                    "observed_value": f"{int(row.get('n_sessions', 0))} sessions / {int(row.get('n_participants', 0))} participants",
                    "evidence_basis": "Cohort sample-status register",
                    "scientific_implication": "Controls whether Chapter 5 may support inferential cohort claims or only descriptive scientific reporting.",
                }
            )
        qc = c.get("preprocessing_qc_summary", pd.DataFrame())
        if not qc.empty:
            mean_valid = float(to_numeric(qc["valid_fraction"]).mean())
            rows.append(
                {
                    "gate": "Minute-level preprocessing QC",
                    "status": "pass" if mean_valid >= 0.80 else ("conditional" if mean_valid >= 0.60 else "limited"),
                    "threshold": "Mean valid fraction >= 0.80 across QC channels",
                    "observed_value": f"{mean_valid:.2f} mean valid fraction across {len(qc)} channels",
                    "evidence_basis": "Preprocessing quality diagnostics",
                    "scientific_implication": "Determines whether physiological and environmental windows can be interpreted as sufficiently quality-controlled.",
                }
            )
        if not support_profile.empty:
            primary_support = support_profile.loc[support_profile["metric"].isin(PRIMARY_ENDPOINTS)].copy()
            strong_primary = int((primary_support.get("support_grade", pd.Series(dtype=str)) == "strong").sum()) if not primary_support.empty else 0
            rows.append(
                {
                    "gate": "Primary endpoint support breadth",
                    "status": "pass" if strong_primary > 0 else "descriptive_only",
                    "threshold": "At least one strong-support primary endpoint",
                    "observed_value": f"{strong_primary} strong primary endpoints",
                    "evidence_basis": "Endpoint support grading matrix",
                    "scientific_implication": "Controls whether the primary scientific result layer is claim-supporting or restricted to descriptive partial-support endpoints.",
                }
            )
        if not signal_audit_summary.empty:
            roles = signal_audit_summary.get("recommended_role", pd.Series(dtype=str)).astype(str)
            primary = int(roles.isin(["primary", "primary_with_qc"]).sum())
            subset_only = int((roles == "subset_only").sum())
            rows.append(
                {
                    "gate": "Modality validity screen",
                    "status": "pass" if primary > 0 else "limited",
                    "threshold": "At least one primary or QC-qualified primary stream family",
                    "observed_value": f"{primary} primary/QC-qualified streams, {subset_only} subset-only streams",
                    "evidence_basis": "Signal audit summary and modality claim register",
                    "scientific_implication": "Ensures Chapter 5 results are anchored in scientifically valid stream families rather than audit-only or subset-only measurements.",
                }
            )
        diagnostics = c.get("mixed_effects_diagnostics", pd.DataFrame())
        if not diagnostics.empty:
            retained = diagnostics.loc[diagnostics["status"].astype(str).isin(["retained", "retained_with_fit_issue"])].copy()
            converged = int(to_numeric(retained.get("fit_converged", pd.Series(dtype=float))).fillna(0).sum()) if not retained.empty else 0
            rows.append(
                {
                    "gate": "Mixed-effects inferential eligibility",
                    "status": "pass" if converged > 0 else "descriptive_only",
                    "threshold": "At least one retained converged mixed-effects endpoint",
                    "observed_value": f"{converged} converged retained fits out of {len(diagnostics)} attempted endpoints",
                    "evidence_basis": "Mixed-effects fit diagnostics",
                    "scientific_implication": "Determines whether inferential fixed-effect estimates can enter the scientific results layer.",
                }
            )
        benchmarks = c.get("predictive_benchmarks", pd.DataFrame())
        if not benchmarks.empty:
            best = benchmarks.sort_values(["balanced_accuracy_mean", "macro_f1_mean"], ascending=[False, False]).iloc[0]
            rows.append(
                {
                    "gate": "Predictive generalization availability",
                    "status": "pass",
                    "threshold": "Holdout benchmark results available across at least one explicit grouping scheme",
                    "observed_value": f"{best['model']} / {best['feature_set']} / {best['validation_scheme']} on {best['task']} with balanced accuracy {float(best['balanced_accuracy_mean']):.2f}",
                    "evidence_basis": "Validation-aware predictive benchmarks",
                    "scientific_implication": "Allows Chapter 5 to report predictive evidence as a generalization check rather than only descriptive patterning.",
                }
            )
        if not all_source_response_matrix.empty or not valid_only_response_matrix.empty:
            rows.append(
                {
                    "gate": "Scenario sensitivity visibility",
                    "status": "pass",
                    "threshold": "All-source and valid-only scenario views both available",
                    "observed_value": f"{len(all_source_response_matrix)} all-source rows / {len(valid_only_response_matrix)} valid-only rows",
                    "evidence_basis": "Scenario-specific response matrices",
                    "scientific_implication": "Makes sensitivity to modality inclusion explicit instead of leaving it implicit in the final result layer.",
                }
            )
        return pd.DataFrame(rows)

    def _robustness_register(
        self,
        c: dict,
        support_profile: pd.DataFrame,
        all_source_response_matrix: pd.DataFrame,
        valid_only_response_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = []
        if not support_profile.empty:
            strong = int((support_profile.get("support_grade", pd.Series(dtype=str)) == "strong").sum())
            partial = int((support_profile.get("support_grade", pd.Series(dtype=str)) == "partial").sum())
            rows.append(
                {
                    "gate": "Endpoint support robustness",
                    "status": "stable" if strong > 0 else ("partial" if partial > 0 else "limited"),
                    "observed_value": f"{strong} strong endpoints, {partial} partial endpoints",
                    "evidence_basis": "Endpoint support grading matrix",
                    "scientific_reading": "Primary results are more defensible when at least one endpoint remains in the strong-support tier.",
                }
            )
        if not all_source_response_matrix.empty or not valid_only_response_matrix.empty:
            shared = sorted(set(all_source_response_matrix.get("row_label", pd.Series(dtype=str))).intersection(set(valid_only_response_matrix.get("row_label", pd.Series(dtype=str)))))
            shared_fraction = float(len(shared) / max(1, len(all_source_response_matrix))) if len(all_source_response_matrix) else np.nan
            rows.append(
                {
                    "gate": "Scenario sensitivity",
                    "status": "stable" if pd.notna(shared_fraction) and shared_fraction >= 0.60 else "conditional",
                    "observed_value": f"{len(shared)} shared rows across all-source and valid-only layers",
                    "evidence_basis": "Scenario-specific response matrices",
                    "scientific_reading": "High overlap indicates that Chapter 5 patterns are not driven only by weaker or subset-only modalities.",
                }
            )
        contrasts = c.get("condition_contrasts", pd.DataFrame())
        if not contrasts.empty and "p_value_fdr" in contrasts.columns:
            significant = int((to_numeric(contrasts["p_value_fdr"]) < 0.05).sum())
            rows.append(
                {
                    "gate": "Multiplicity-controlled contrasts",
                    "status": "stable" if significant > 0 else "null_after_correction",
                    "observed_value": f"{significant} FDR-significant contrasts",
                    "evidence_basis": "Multiplicity-corrected contrast register",
                    "scientific_reading": "Significant corrected contrasts indicate that condition differences persist after multiplicity control.",
                }
            )
        diagnostics = c.get("mixed_effects_diagnostics", pd.DataFrame())
        if not diagnostics.empty:
            retained = diagnostics.loc[diagnostics["status"].astype(str).isin(["retained", "retained_with_fit_issue"])].copy()
            converged_fraction = float(to_numeric(retained.get("fit_converged", pd.Series(dtype=float))).mean()) if not retained.empty else np.nan
            rows.append(
                {
                    "gate": "Model-fit stability",
                    "status": "stable" if pd.notna(converged_fraction) and converged_fraction >= 0.75 else ("conditional" if not retained.empty else "not_available"),
                    "observed_value": f"{converged_fraction:.2f}" if pd.notna(converged_fraction) else "n/a",
                    "evidence_basis": "Mixed-effects fit diagnostics",
                    "scientific_reading": "Higher convergence among retained models supports more reliable inferential interpretation.",
                }
            )
        benchmarks = c.get("predictive_benchmarks", pd.DataFrame())
        if not benchmarks.empty:
            best = benchmarks.sort_values(["balanced_accuracy_mean", "macro_f1_mean"], ascending=[False, False]).iloc[0]
            sd = float(best["balanced_accuracy_sd"]) if pd.notna(best.get("balanced_accuracy_sd", np.nan)) else np.nan
            rows.append(
                {
                    "gate": "Predictive stability",
                    "status": "stable" if pd.notna(sd) and sd <= 0.10 else "conditional",
                    "observed_value": f"{float(best['balanced_accuracy_mean']):.2f} +/- {sd:.2f}" if pd.notna(sd) else f"{float(best['balanced_accuracy_mean']):.2f}",
                    "evidence_basis": "Participant-grouped predictive benchmarks",
                    "scientific_reading": "Lower fold-to-fold variability indicates more stable subject-independent predictive behavior.",
                }
            )
        partial_count = int((support_profile.get("support_grade", pd.Series(dtype=str)) != "strong").sum()) if not support_profile.empty else 0
        rows.append(
            {
                "gate": "Partial-result quarantine",
                "status": "pass" if partial_count >= 0 else "pass",
                "observed_value": f"{partial_count} endpoints held outside the primary result layer",
                "evidence_basis": "Partial-results register",
                "scientific_reading": "Endpoints with incomplete support remain visible but are explicitly quarantined from the primary scientific result layer.",
            }
        )
        return pd.DataFrame(rows)

    def _fig_cohort_relationship_heatmap(self, relation_df: pd.DataFrame):
        if relation_df.empty:
            return None
        labels = sorted(set(relation_df["source"]).union(set(relation_df["target"])))
        pivot = pd.DataFrame(np.nan, index=labels, columns=labels)
        for row in relation_df.itertuples():
            pivot.loc[row.source, row.target] = row.spearman_r
            pivot.loc[row.target, row.source] = row.spearman_r
        fig, ax = plt.subplots(figsize=self._figsize("matrix"))
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="equal", cmap="coolwarm", vmin=-1, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10, color="#172033")
        plt.colorbar(im, ax=ax, shrink=0.82, label="Spearman r")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_cohort_targeted_relationships(self, cohort_phase: pd.DataFrame, support_profile: pd.DataFrame):
        metrics = self._cohort_primary_metrics(support_profile)
        if "thermal_comfort" not in metrics:
            return None
        session_delta = self._cohort_session_delta_frame(cohort_phase, metrics)
        if session_delta.empty:
            return None
        target_order = ["indoor_air_velocity_mean_m_s", "indoor_air_temp_mean_C", "empatica_eda_mean_uS"]
        targets = [metric for metric in target_order if metric in metrics and f"{metric}__delta" in session_delta.columns]
        if not targets:
            return None
        retained_targets = []
        for metric in targets:
            pair = session_delta[[f"{metric}__delta", "thermal_comfort__delta"]].dropna()
            if len(pair) >= self._cohort_relationship_min_n(metric, "thermal_comfort"):
                retained_targets.append(metric)
        if not retained_targets:
            return None
        fig, axes = plt.subplots(1, len(retained_targets), figsize=(4.6 * len(retained_targets), 4.5))
        if len(retained_targets) == 1:
            axes = [axes]
        panel_positions = ["Left", "Center", "Right"]
        panel_notes = []
        for ax, metric, position in zip(axes, retained_targets, panel_positions):
            pair = session_delta[[f"{metric}__delta", "thermal_comfort__delta", "condition_code"]].copy().dropna()
            x = pair[f"{metric}__delta"]
            y = pair["thermal_comfort__delta"]
            ax.scatter(x, y, s=28, alpha=0.8, color="#2563eb")
            r = float(x.corr(y, method="spearman"))
            qualified_condition_signs = []
            for _, dc in pair.groupby("condition_code"):
                if len(dc) < 2:
                    continue
                dc_r = dc[f"{metric}__delta"].corr(dc["thermal_comfort__delta"], method="spearman")
                if pd.notna(dc_r) and dc_r != 0:
                    qualified_condition_signs.append(float(np.sign(dc_r)))
            same_sign_fraction = np.nan
            if qualified_condition_signs and r != 0:
                same_sign_fraction = float(np.mean(np.array(qualified_condition_signs) == float(np.sign(r))))
            ax.set_xlabel(f"{self._axis_label(metric)} Change")
            ax.set_ylabel("Thermal Comfort Change (ordinal scale)")
            stability_note = f", same-sign fraction across conditions = {same_sign_fraction:.2f}" if pd.notna(same_sign_fraction) else ""
            panel_notes.append(f"{position} shows thermal comfort delta versus {FEATURE_LABELS.get(metric, metric)} delta with Spearman r = {r:.2f}, n = {len(pair)} retained sessions{stability_note}.")
        fig._cltr_panel_notes = panel_notes
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _modality_start_rows(self, minute: pd.DataFrame) -> pd.DataFrame:
        mapping = [
            ("Empatica HR", "empatica_hr_mean_bpm"),
            ("BIOPAC HR", "biopac_hr_mean_bpm"),
            ("Empatica EDA", "empatica_eda_mean_uS"),
            ("BIOPAC EDA", "biopac_eda_mean_uS"),
            ("Empatica Temp", "empatica_temp_mean_C"),
            ("Chest Temp", "biopac_temp_chest_mean_C"),
            ("Blood Flow", "biopac_bloodflow_mean_bpu"),
        ]
        phase = self._phase_summary_from_minute(minute, [col for _, col in mapping])
        rows = []
        for label, metric in mapping:
            base = self._phase_metric_baseline(phase, metric)
            if base:
                rows.append({"label": label, "phase": base["phase"], "phase_abbr": PHASE_ABBR.get(base["phase"], base["phase"][:3].upper())})
        return pd.DataFrame(rows)

    def _support_segment_rows(self, minute: pd.DataFrame, mapping: list[tuple[str, str, str]]) -> pd.DataFrame:
        rows = []
        if minute.empty or "minute_index" not in minute.columns:
            return pd.DataFrame(rows)
        ordered = minute.sort_values("minute_index").copy()
        minute_values = to_numeric(ordered["minute_index"])
        for label, col, color in mapping:
            if col not in ordered.columns:
                continue
            vals = to_numeric(ordered[col])
            support_mask = vals.notna() if not col.startswith("support_") else vals.fillna(0) > 0
            if not bool(support_mask.any()):
                continue
            support_fraction = float(support_mask.mean())
            supported_minutes = minute_values.loc[support_mask].reset_index(drop=True)
            if supported_minutes.empty:
                continue
            start = float(supported_minutes.iloc[0])
            prev = start
            segment_index = 1
            for minute_value in supported_minutes.iloc[1:]:
                cur = float(minute_value)
                if cur != prev + 1:
                    rows.append(
                        {
                            "label": label,
                            "segment_label": label if segment_index == 1 else f"{label} ({segment_index})",
                            "start_minute": start,
                            "end_minute": prev,
                            "support_fraction": support_fraction,
                            "color": color,
                        }
                    )
                    start = cur
                    segment_index += 1
                prev = cur
            rows.append(
                {
                    "label": label,
                    "segment_label": label if segment_index == 1 else f"{label} ({segment_index})",
                    "start_minute": start,
                    "end_minute": prev,
                    "support_fraction": support_fraction,
                    "color": color,
                }
            )
        return pd.DataFrame(rows)

    def _story_focus_metric(self, s: dict) -> str:
        story = self._session_story_profile(s)
        archetype = story.get("archetype", "")
        if archetype == "comfort-drop":
            return "thermal_comfort"
        if archetype.startswith("thermal-shift") or archetype == "rewarming-shift":
            return "biopac_temp_chest_mean_C" if "biopac_temp_chest_mean_C" in s["phase_df"].columns else "empatica_temp_mean_C"
        if archetype == "heart-rate-shift":
            return "empatica_hr_mean_bpm" if "empatica_hr_mean_bpm" in s["phase_df"].columns else "biopac_hr_mean_bpm"
        return "thermal_comfort" if "thermal_comfort" in s["phase_df"].columns else "biopac_temp_chest_mean_C"

    def _plotly_protocol_trace(self, minute: pd.DataFrame, specs: list[dict], title: str, y_title: str, footer: str = ""):
        if minute.empty or "minute_index" not in minute.columns:
            return None
        fig = go.Figure()
        colors = ["rgba(248,250,252,0.65)", "rgba(239,246,255,0.65)", "rgba(254,243,199,0.65)", "rgba(220,252,231,0.65)", "rgba(254,226,226,0.65)", "rgba(237,233,254,0.65)"]
        for idx, (start, end, phase_name) in enumerate(self._phase_segments(minute)):
            fig.add_vrect(x0=start, x1=end, fillcolor=colors[idx % len(colors)], opacity=0.5, line_width=0, layer="below")
            if end - start >= 7:
                fig.add_annotation(x=(start + end) / 2.0, y=1.06, yref="paper", text=PHASE_ABBR.get(phase_name, phase_name[:3].upper()), showarrow=False, font={"size": 10, "color": "#475569"})
        custom = np.column_stack(
            [
                minute["protocol_block"].astype(str).to_numpy() if "protocol_block" in minute.columns else np.array([""] * len(minute)),
                minute["protocol_phase"].astype(str).to_numpy() if "protocol_phase" in minute.columns else np.array([""] * len(minute)),
            ]
        )
        x = to_numeric(minute["minute_index"])
        any_trace = False
        for spec in specs:
            col = spec["column"]
            if col not in minute.columns:
                continue
            y = to_numeric(minute[col])
            if y.notna().sum() == 0:
                continue
            any_trace = True
            is_sparse = self._is_sparse_observation_channel(col)
            y_display = self._display_series(y, col)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_display,
                    mode="markers" if is_sparse else "lines+markers",
                    connectgaps=False,
                    name=spec["label"],
                    line={
                        "color": spec["color"],
                        "width": 2,
                        "dash": spec.get("dash", "solid"),
                        "shape": "hv" if self._is_control_signal_channel(col) else "linear",
                    },
                    marker={"size": 8 if is_sparse else (4 if self._is_control_signal_channel(col) else 6)},
                    customdata=custom,
                    hovertemplate="Minute %{x}<br>Block %{customdata[0]}<br>Phase %{customdata[1]}<br>%{fullData.name}: %{y:.3f}<extra></extra>",
                )
            )
        if not any_trace:
            return None
        fig.update_layout(
            template="plotly_white",
            height=420,
            margin={"l": 60, "r": 30, "t": 70, "b": 85},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
            hovermode="x unified",
        )
        fig.update_xaxes(title_text=self._time_axis_label(), showgrid=True, gridcolor="#eef2f7", zeroline=False)
        fig.update_yaxes(title_text=y_title, showgrid=True, gridcolor="#eef2f7", zeroline=False)
        return fig

    def _matplotlib_protocol_trace(
        self,
        minute: pd.DataFrame,
        specs: list[dict],
        title: str,
        footer: str = "",
        overlay: bool = False,
    ):
        if minute.empty or "minute_index" not in minute.columns:
            return None
        available = []
        for spec in specs:
            col = spec["column"]
            if col in minute.columns and to_numeric(minute[col]).notna().sum() > 0:
                available.append(spec)
        if not available:
            return None
        phase_df = minute.drop_duplicates(subset=["minute_index", "protocol_phase"])
        x = to_numeric(minute["minute_index"])
        if overlay:
            fig, ax = plt.subplots(figsize=self._figsize("timeline"))
            self._add_phase_spans(ax, phase_df)
            for spec in available:
                y = to_numeric(minute[spec["column"]])
                y_display = self._display_series(y, spec["column"])
                mask = y.notna()
                if self._is_sparse_observation_channel(spec["column"]):
                    ax.vlines(x[mask], 0, y[mask], color=spec["color"], lw=0.9, alpha=0.18)
                    ax.scatter(x[mask], y[mask], color=spec["color"], s=18, alpha=0.82, zorder=3, label=spec["label"])
                elif self._is_control_signal_channel(spec["column"]):
                    ax.step(x[mask], y_display[mask], where="mid", color=spec["color"], lw=1.8, label=spec["label"])
                    ax.scatter(x[mask], y[mask], color=spec["color"], s=10, alpha=0.28, zorder=3)
                else:
                    ax.plot(x[mask], y[mask], color=spec["color"], lw=1.8, label=spec["label"])
                    ax.scatter(x[mask], y[mask], color=spec["color"], s=8, alpha=0.45)
            ax.set_ylabel("Temperature (C)")
            ax.set_xlabel(self._time_axis_label())
            ax.grid(True, axis="y")
            ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=11)
        else:
            fig, axes = plt.subplots(len(available), 1, figsize=(13.2, 2.35 * len(available) + 0.8), sharex=True)
            if len(available) == 1:
                axes = [axes]
            for ax, spec in zip(axes, available):
                self._add_phase_spans(ax, phase_df)
                y = to_numeric(minute[spec["column"]])
                y_display = self._display_series(y, spec["column"])
                mask = y.notna()
                if self._is_sparse_observation_channel(spec["column"]):
                    ax.vlines(x[mask], 0, y[mask], color=spec["color"], lw=1.0, alpha=0.4)
                    ax.scatter(x[mask], y[mask], color=spec["color"], s=24, alpha=0.85, zorder=3)
                elif self._is_control_signal_channel(spec["column"]):
                    ax.step(x[mask], y_display[mask], where="mid", color=spec["color"], lw=1.8)
                    ax.scatter(x[mask], y[mask], color=spec["color"], s=12, alpha=0.35, zorder=3)
                else:
                    ax.plot(x[mask], y[mask], color=spec["color"], lw=1.8)
                    ax.scatter(x[mask], y[mask], color=spec["color"], s=9, alpha=0.7)
                ax.set_ylabel(spec["label"])
                ax.grid(True, axis="y")
            axes[-1].set_xlabel(self._time_axis_label())
        fig.tight_layout()
        return fig

    def _fig_session_single_channel_raw(self, minute: pd.DataFrame, column: str, color: str):
        if minute.empty or "minute_index" not in minute.columns or column not in minute.columns:
            return None
        display_minute, display_note = self._channel_display_window(minute, column)
        if display_minute.empty:
            return None
        y = to_numeric(display_minute[column])
        mask = y.notna()
        if not bool(mask.any()):
            return None
        fig, ax = plt.subplots(figsize=self._figsize("timeline"))
        phase_df = display_minute.drop_duplicates(subset=["minute_index", "protocol_phase"])
        self._add_phase_spans(ax, phase_df)
        x = to_numeric(display_minute["minute_index"])
        if self._is_sparse_observation_channel(column):
            ax.vlines(x[mask], 0, y[mask], color=color, lw=1.1, alpha=0.5)
            ax.scatter(x[mask], y[mask], color=color, s=28, alpha=0.88, zorder=3)
        elif self._is_control_signal_channel(column):
            y_display = self._display_series(y, column)
            ax.step(x[mask], y_display[mask], where="mid", color=color, lw=2.0)
            ax.scatter(x[mask], y[mask], color=color, s=16, alpha=0.38, zorder=3)
        else:
            ax.plot(x[mask], y[mask], color=color, lw=1.9)
            ax.scatter(x[mask], y[mask], color=color, s=11, alpha=0.72)
        ax.set_ylabel(self._axis_label(column))
        ax.set_xlabel(self._time_axis_label())
        ax.grid(True, axis="y")
        self._apply_discrete_y_axis_matplotlib(ax, y[mask], column)
        note = self._support_note(minute, [column])
        footer = " ".join(part for part in [note, display_note] if part)
        if self._is_sparse_observation_channel(column):
            footer = " ".join(part for part in ["Sparse questionnaire observations are shown as discrete points, not a connected line.", footer] if part)
        elif self._is_control_signal_channel(column):
            footer = " ".join(part for part in ["Control channels are rendered as a 3-minute rolling-median step trace with raw points retained to suppress minute-to-minute actuator jitter.", footer] if part)
        baseline_note = self._baseline_note(self._phase_metric_baseline(self._phase_summary_from_minute(minute, [column]), column))
        footer = " ".join(part for part in [footer, baseline_note] if part)
        fig.tight_layout()
        return fig

    def _plotly_phase_distribution(self, phase: pd.DataFrame, metric: str, title: str, footer: str = ""):
        if phase.empty or metric not in phase.columns or "protocol_phase" not in phase.columns:
            return None
        fig = go.Figure()
        added = False
        for phase_name in [p for p in PHASE_ORDER if p in phase["protocol_phase"].astype(str).unique()]:
            cur = phase.loc[phase["protocol_phase"] == phase_name].copy()
            values = to_numeric(cur[metric]).dropna()
            if values.empty:
                continue
            xvals = cur.loc[to_numeric(cur[metric]).notna(), "protocol_block"].astype(str).radd("B").to_list() if "protocol_block" in cur.columns else [phase_name] * len(values)
            fig.add_trace(
                go.Box(
                    x=xvals,
                    y=values,
                    name=PHASE_ABBR.get(phase_name, phase_name[:3].upper()),
                    boxpoints="all",
                    jitter=0.25,
                    pointpos=0,
                    marker={"size": 7},
                    hovertemplate=f"{FEATURE_LABELS.get(metric, metric)}: %{{y:.3f}}<br>Block %{{x}}<extra>{PHASE_ABBR.get(phase_name, phase_name[:3].upper())}</extra>",
                )
            )
            added = True
        if not added:
            return None
        fig.update_layout(
            template="plotly_white",
            height=420,
            margin={"l": 60, "r": 30, "t": 70, "b": 85},
            boxmode="group",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        )
        fig.update_xaxes(title_text="Protocol block")
        fig.update_yaxes(title_text=self._axis_label(metric), showgrid=True, gridcolor="#eef2f7", zeroline=False)
        self._apply_discrete_y_axis_plotly(fig, phase[metric], metric)
        return fig

    def _session_evidence(self, minute: pd.DataFrame, meta: dict) -> dict:
        overlap = meta.get("sensor_overlap_minutes", {})
        support = meta.get("support", {})
        evidence_score = 100
        evidence_score -= 30 if overlap.get("heart_rate", 0) < self.config.runtime.min_sensor_overlap_minutes else 0
        evidence_score -= 20 if overlap.get("eda", 0) < self.config.runtime.min_sensor_overlap_minutes else 0
        evidence_score -= 20 if overlap.get("temperature", 0) < self.config.runtime.min_sensor_overlap_minutes else 0
        evidence_score -= 10 if support.get("questionnaire_completeness", 0.0) < 0.8 else 0
        evidence_score = max(5, evidence_score)
        label = "strong" if evidence_score >= 75 else "moderate" if evidence_score >= 50 else "weak"
        notes = []
        if overlap.get("heart_rate", 0) < self.config.runtime.min_sensor_overlap_minutes:
            notes.append("Heart-rate agreement below overlap threshold")
        if overlap.get("eda", 0) < self.config.runtime.min_sensor_overlap_minutes:
            notes.append("EDA agreement below overlap threshold")
        if overlap.get("temperature", 0) < self.config.runtime.min_sensor_overlap_minutes:
            notes.append("Temperature agreement below overlap threshold")
        if support.get("questionnaire_completeness", 0.0) < 0.8:
            notes.append("Incomplete questionnaire responses in Blocks 1 to 3")
        return {"score": evidence_score, "label": label, "note": "; ".join(notes)}

    def _agreement_materiality(self, s: dict) -> dict:
        minute = s["aligned_df"]
        meta = s["processing_metadata"]
        pairs = [
            ("heart_rate", "empatica_hr_mean_bpm", "biopac_hr_mean_bpm"),
            ("eda", "empatica_eda_mean_uS", "biopac_eda_mean_uS"),
            ("temperature", "empatica_temp_mean_C", "biopac_temp_chest_mean_C"),
        ]
        rows = []
        for metric, left, right in pairs:
            if left not in minute.columns or right not in minute.columns:
                continue
            pair = minute[[left, right]].apply(to_numeric).dropna()
            overlap = int(meta.get("sensor_overlap_minutes", {}).get(metric, 0))
            if len(pair) < 3:
                continue
            spearman = float(pair[left].corr(pair[right], method="spearman"))
            diff = pair[left] - pair[right]
            mae = float(diff.abs().mean())
            materially_interpretable = overlap >= self.config.runtime.min_sensor_overlap_minutes
            materially_discordant = materially_interpretable and (
                (pd.notna(spearman) and spearman < 0.45)
                or (
                    metric == "heart_rate" and mae > 8.0
                )
                or (
                    metric == "eda" and mae > 1.5
                )
                or (
                    metric == "temperature" and mae > 1.0
                )
            )
            rows.append(
                {
                    "metric": metric,
                    "overlap": overlap,
                    "spearman": spearman,
                    "mae": mae,
                    "materially_interpretable": materially_interpretable,
                    "materially_discordant": materially_discordant,
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return {"include_in_main": False, "summary": "Device agreement is shown as supporting context."}
        include = bool(df["materially_discordant"].any())
        if not include:
            include = bool((df["materially_interpretable"]).sum() >= 2 and df["spearman"].dropna().median() < 0.6)
        if include:
            top = df.sort_values(["materially_discordant", "mae"], ascending=[False, False]).iloc[0]
            summary = f"Device comparison is highlighted because {top['metric']} shows {int(top['overlap'])} overlapping minutes, correlation {top['spearman']:.2f}, and average error {top['mae']:.2f}."
        else:
            summary = "Device comparison remains supporting context because it does not substantially change the main session summary."
        return {"include_in_main": include, "summary": summary, "table": df}

    def _session_priority_codes(self, s: dict) -> list[str]:
        story = self._session_story_profile(s)
        agreement = self._agreement_materiality(s)
        codes: list[str] = []
        for code in story["priority_codes"]:
            if code == "S09" and not agreement["include_in_main"]:
                continue
            if code not in codes:
                codes.append(code)
        base_fallback = ["S01", "S02", "S07", "S08", "S06", "S05", "S03", "S04", "S10", "S09"]
        for code in base_fallback:
            if code == "S09" and not agreement["include_in_main"]:
                continue
            if code not in codes:
                codes.append(code)
        return codes

    def _cohort_evidence(self, sample_status: pd.DataFrame) -> dict:
        if sample_status.empty:
            return {"score": 5, "label": "weak", "note": "No cohort sample"}
        row = sample_status.iloc[0]
        eligible = bool(row["cohort_inference_eligible"])
        if eligible:
            return {"score": 85, "label": "strong", "note": ""}
        score = min(45, 10 + 4 * int(row["n_sessions"]) + 3 * int(row["n_participants"]))
        return {
            "score": score,
            "label": "descriptive_only",
            "note": f"Inferential cohort reporting disabled: {int(row['n_sessions'])} sessions / {int(row['n_participants'])} participants.",
        }

    def _is_tiny_cohort(self, sample_status: pd.DataFrame) -> bool:
        if sample_status.empty:
            return True
        row = sample_status.iloc[0]
        return int(row.get("n_participants", 0)) <= 1 or int(row.get("n_sessions", 0)) <= 4

    def _build_session_specs(self, s: dict) -> tuple[list[dict], list[dict]]:
        minute = s["aligned_df"].copy().sort_values("minute_index")
        phase = s["phase_df"].copy()
        ev = self._session_evidence(minute, s["processing_metadata"])
        narrative = [
            self._spec(
                code="S01",
                stem=f"{s['session_id']}_readiness",
                title="Comparison-window coverage and readiness",
                summary="Coverage, questionnaire completeness, and paired-device overlap are summarized over Blocks 1 to 3 so the comparable part of the session is explicit before later summaries are read.",
                fig=self._fig_session_readiness(minute, s["processing_metadata"], ev),
                tags=["overview", "qc", "support"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="alignment_support",
            ),
            self._spec(
                code="S02",
                stem=f"{s['session_id']}_preprocessing_burden",
                title="Retention after alignment and preprocessing",
                summary="Usable support after alignment is quantified directly, separating modality retention from paired-overlap retention and making sparse questionnaire sampling visible instead of implicit.",
                fig=self._fig_session_preprocessing_burden(minute, s["processing_metadata"]),
                tags=["qc", "support", "preprocessing"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="alignment_support",
            ),
            self._spec(
                code="S03",
                stem=f"{s['session_id']}_sync_audit",
                title="Shared-timeline synchronization and overlap audit",
                summary="Support windows are placed on the same session timeline so delayed starts, shortened overlap, and modality-specific acquisition windows are visible before the main results are interpreted.",
                fig=self._fig_session_sync_audit(minute, s["processing_metadata"]),
                tags=["qc", "support", "agreement", "preprocessing"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="alignment_support",
            ),
            self._spec(
                code="S04",
                stem=f"{s['session_id']}_coverage",
                title="Aligned availability and overlap map by minute",
                summary="The aligned minute-level availability and overlap map separates missing support from absent response and shows where questionnaire, source presence, and paired non-null overlap are actually available.",
                fig=self._fig_session_coverage(minute, s["processing_metadata"]),
                tags=["qc", "support", "preprocessing"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="alignment_support",
            ),
        ]
        narrative.extend(self._build_session_source_raw_specs(s, ev))
        narrative.extend(self._build_session_processed_signal_specs(s, ev))
        derived_metrics = self._session_derived_endpoints(phase)
        support_profile = self._session_endpoint_support_profile(phase, derived_metrics)
        primary_metrics = support_profile.loc[support_profile["support_grade"] == "Strong", "metric"].astype(str).tolist() if not support_profile.empty else []
        partial_profile = support_profile.loc[support_profile["support_grade"] != "Strong"].copy() if not support_profile.empty else pd.DataFrame()
        relationship_matrix = self._session_relationship_matrix(minute, primary_metrics)
        fingerprint_matrix = self._session_response_fingerprint_matrix(phase, primary_metrics)
        response_matrix = self._session_response_matrix(phase, primary_metrics)
        contrast_matrix = self._session_phase_contrast_matrix(phase, primary_metrics)
        repeatability_matrix = self._session_repeatability_matrix(phase, primary_metrics)
        narrative.extend([
            self._html_spec(
                code="S10Z",
                stem=f"{s['session_id']}_endpoint_support_grading",
                title="Endpoint support grading matrix",
                summary="This matrix grades each candidate endpoint for scientific use in the session results section. Only endpoints graded strong are carried into the primary result matrices and heatmaps.",
                html_fragment=self._matrix_panel_html(
                    "Endpoint Support Grading Matrix",
                    support_profile,
                    ["endpoint", "support_grade", "support_basis", "supported_phases", "min_block_repeats", "total_block_phase_summaries", "total_valid_units", "scientific_reading"],
                    n=24,
                ),
                tags=["phase", "matrix", "support", "statistics"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="derived",
            ),
            self._spec(
                code="S10Y",
                stem=f"{s['session_id']}_endpoint_support_grades",
                title="Endpoint support grading summary",
                summary="This figure summarizes how much repeated phase support each candidate endpoint has and whether it is retained as a primary result, downgraded to partial evidence, or excluded.",
                fig=self._fig_endpoint_support_grades(support_profile),
                tags=["phase", "support", "statistics"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="derived",
            ),
            self._html_spec(
                code="S10X",
                stem=f"{s['session_id']}_response_fingerprint_matrix",
                title="Primary-result response fingerprint matrix",
                summary="This matrix condenses each primary endpoint to its dominant response phase, direction of departure from the reference phase, raw dominant delta, and repeated-block directional agreement at that phase.",
                html_fragment=self._matrix_panel_html(
                    "Primary-Result Response Fingerprint Matrix",
                    fingerprint_matrix,
                    ["endpoint", "reference_phase", "dominant_phase", "direction", "dominant_delta", "dominant_agreement", "dominant_phase_repeats", "support_basis", "total_valid_units"],
                    n=24,
                ),
                tags=["phase", "matrix", "statistics", "support"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="derived",
            ),
            self._spec(
                code="S10W",
                stem=f"{s['session_id']}_response_fingerprint",
                title="Primary-result dominant response phase",
                summary="This figure shows where each primary endpoint has its strongest descriptive departure from the reference phase. Dot color indicates rise or drop, and dot size reflects repeated-block directional agreement when available.",
                fig=self._fig_session_response_fingerprint(fingerprint_matrix),
                tags=["phase", "statistics", "support"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="derived",
            ),
            self._html_spec(
                code="S11A",
                stem=f"{s['session_id']}_response_matrix",
                title="Primary-result phase-level median matrix",
                summary="This matrix reports phase-level medians for the strong-support endpoints only. Endpoints with partial or insufficient support are intentionally excluded from the primary result layer.",
                html_fragment=self._matrix_panel_html("Primary-Result Phase-Level Median Matrix", response_matrix, ["endpoint", "support_basis", "FCS", "SR", "FFC", "SS", "OC", "n_block_phase_summaries", "total_valid_units", "phase_support"], n=16),
                tags=["phase", "matrix", "statistics", "comfort", "temperature", "heart_rate"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="derived",
            ),
            self._spec(
                code="S11B",
                stem=f"{s['session_id']}_response_heatmap",
                title="Primary-result phase-level median heatmap",
                summary="The heatmap shows the same phase-level medians as the primary-result matrix, but colors are scaled within each endpoint to a 0 to 1 display range for visual comparison only. Exact raw values should be read from the matrix.",
                fig=self._fig_session_response_heatmap(response_matrix),
                tags=["phase", "heatmap", "statistics", "comfort", "temperature", "heart_rate"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="derived",
            ),
            self._html_spec(
                code="S12A",
                stem=f"{s['session_id']}_phase_contrast_matrix",
                title="Primary-result reference-phase delta matrix",
                summary="This matrix reports phase medians minus the earliest supported comparison-phase median for the strong-support endpoints only. It is a descriptive contrast table, not an inferential effect estimate.",
                html_fragment=self._matrix_panel_html("Primary-Result Reference-Phase Delta Matrix", contrast_matrix, ["endpoint", "reference_phase", "FCS", "SR", "FFC", "SS", "OC"], n=16),
                tags=["phase", "matrix", "statistics", "contrast"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="derived",
            ),
            self._spec(
                code="S12B",
                stem=f"{s['session_id']}_phase_contrast_heatmap",
                title="Primary-result reference-phase delta heatmap",
                summary="The heatmap visualizes the same reference-phase delta pattern as the primary-result matrix, but colors and in-cell values are scaled within each endpoint to a signed display range from -1 to 1. Exact raw deltas should be read from the matrix.",
                fig=self._fig_session_phase_contrast_heatmap(contrast_matrix),
                tags=["phase", "heatmap", "statistics", "contrast"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="derived",
            ),
            self._html_spec(
                code="S13A",
                stem=f"{s['session_id']}_repeatability_matrix",
                title="Primary-result phase-by-phase directional agreement matrix",
                summary="This matrix reports the fraction of repeated blocks that share the dominant direction of change for each phase and strong-support endpoint. Blank cells indicate insufficient repeated-block support.",
                html_fragment=self._matrix_panel_html("Primary-Result Phase-By-Phase Directional Agreement Matrix", repeatability_matrix, ["endpoint", "FCS", "SR", "FFC", "SS", "OC", "phase_block_support"], n=16),
                tags=["phase", "matrix", "statistics", "repeatability"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="derived",
            ),
            self._spec(
                code="S13B",
                stem=f"{s['session_id']}_repeatability_summary",
                title="Primary-result phase-by-phase directional agreement heatmap",
                summary="The heatmap visualizes the same repeated-block directional agreement values as the primary-result matrix on a 0 to 1 agreement scale. It should be interpreted only where repeated-block support is present.",
                fig=self._fig_session_repeatability_summary(repeatability_matrix),
                tags=["phase", "statistics", "repeatability"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="derived",
            ),
            self._html_spec(
                code="S13C",
                stem=f"{s['session_id']}_partial_results_register",
                title="Partial-results register",
                summary="This register lists endpoints that are not used as primary session results because their repeated support is incomplete or too sparse for a session-wide scientific reading.",
                html_fragment=self._matrix_panel_html(
                    "Partial-Results Register",
                    partial_profile,
                    ["endpoint", "support_grade", "support_basis", "supported_phases", "min_block_repeats", "total_block_phase_summaries", "total_valid_units", "scientific_reading"],
                    n=24,
                ),
                tags=["matrix", "support", "statistics"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="derived",
            ),
            self._html_spec(
                code="S14A",
                stem=f"{s['session_id']}_relationship_matrix",
                title="Primary-result relationship matrix",
                summary="This matrix reports unique support-gated pairwise Spearman associations among the strong endpoints only. It also shows how many phases independently support the same direction of association, so retained rows are not interpreted from paired count alone.",
                html_fragment=self._matrix_panel_html(
                    "Primary-Result Relationship Matrix",
                    relationship_matrix,
                    ["source", "target", "spearman_r", "paired_n", "min_required_n", "qualified_phases", "same_sign_fraction", "relationship_status", "phase_support_status"],
                    n=48,
                ),
                tags=["matrix", "relationships", "statistics"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="agreement_section",
            ),
            self._spec(
                code="S14B",
                stem=f"{s['session_id']}_relationship_heatmap",
                title="Primary-result relationship heatmap",
                summary="The heatmap visualizes the same retained descriptive associations as the relationship matrix. It is a pattern screen only: exact support thresholds, qualified phase counts, and same-sign fractions should be read from the matrix.",
                fig=self._fig_session_relationship_heatmap(relationship_matrix),
                tags=["heatmap", "relationships", "statistics"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="agreement_section",
            ),
            self._spec(
                code="S14C",
                stem=f"{s['session_id']}_targeted_relationships",
                title="Targeted relationships with thermal comfort",
                summary="These scatter panels focus on the most interpretable support-gated relationships between thermal comfort and retained environmental or physiological endpoints. They show ordinal comfort observations without fitted linear trend lines, and the caption reports same-sign stability across phases where available.",
                fig=self._fig_session_targeted_relationships(minute, primary_metrics),
                tags=["relationships", "statistics", "comfort"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="agreement_section",
            ),
            self._spec(
                code="S14",
                stem=f"{s['session_id']}_agreement",
                title="How closely the devices agree",
                summary="These panels show how closely paired devices track one another, while also showing how much overlapping data is available for the comparison.",
                fig=self._fig_session_agreement(minute, s["processing_metadata"]),
                tags=["agreement", "heart_rate", "eda", "temperature"],
                evidence_score=min(ev["score"], 60 if ev["label"] == "weak" else ev["score"]),
                evidence_label="moderate" if ev["label"] != "weak" else "weak",
                gating_note=ev["note"],
                section="agreement_section",
            ),
            self._spec(
                code="S15",
                stem=f"{s['session_id']}_bland_altman",
                title="Agreement bias versus mean level",
                summary="Bland-Altman style panels expose whether cross-sensor disagreement stays centered or drifts with signal magnitude, which is more informative than correlation alone.",
                fig=self._fig_session_bland_altman(minute, s["processing_metadata"]),
                tags=["agreement", "heart_rate", "eda", "temperature"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="agreement_section",
            ),
        ])
        return [x for x in narrative if x["fig"] is not None or x.get("html_fragment")], []

    def _build_cohort_specs(self, c: dict) -> tuple[list[dict], list[dict]]:
        sample_status = c["sample_status"]
        ev = self._cohort_evidence(sample_status)
        support_profile = self._cohort_endpoint_support_profile(c.get("cohort_phase_summary", pd.DataFrame()))
        scenario_register = self._scenario_register(c.get("signal_audit_summary", pd.DataFrame()))
        modality_claim_register = self._modality_claim_register(c.get("signal_audit_summary", pd.DataFrame()))
        endpoint_claim_register = self._endpoint_claim_register(support_profile, c.get("signal_audit_summary", pd.DataFrame()))
        device_stream_inventory = self._device_stream_inventory_register(c.get("cohort_minute_features", pd.DataFrame()), c.get("signal_audit_summary", pd.DataFrame()))
        analysis_pathway_register = self._analysis_pathway_register(c.get("cohort_minute_features", pd.DataFrame()), support_profile, c.get("signal_audit_summary", pd.DataFrame()))
        all_source_support = self._filter_support_profile_for_scenario(support_profile, c.get("signal_audit_summary", pd.DataFrame()), "all_sources")
        valid_only_support = self._filter_support_profile_for_scenario(support_profile, c.get("signal_audit_summary", pd.DataFrame()), "valid_only")
        has_strong = bool((support_profile.get("support_grade", pd.Series(dtype=str)) == "strong").any()) if not support_profile.empty else False
        signal_audit_summary = c.get("signal_audit_summary", pd.DataFrame())
        role_counts = signal_audit_summary.get("recommended_role", pd.Series(dtype=str)).astype(str).value_counts() if not signal_audit_summary.empty else pd.Series(dtype=int)
        primary_stream_count = int(role_counts.get("primary", 0))
        qc_primary_stream_count = int(role_counts.get("primary_with_qc", 0))
        subset_only_stream_count = int(role_counts.get("subset_only", 0))
        result_prefix = "Primary-result" if has_strong else "Partial-result descriptive"
        result_summary_suffix = (
            "Only strong-support endpoints are carried into this result layer."
            if has_strong
            else "No endpoint reaches strong cohort support in the current sample, so this layer is shown descriptively from partial-support endpoints only."
        )
        response_matrix = self._cohort_response_matrix(c.get("cohort_phase_summary", pd.DataFrame()), support_profile)
        all_source_response_matrix = self._cohort_response_matrix(c.get("cohort_phase_summary", pd.DataFrame()), all_source_support)
        valid_only_response_matrix = self._cohort_response_matrix(c.get("cohort_phase_summary", pd.DataFrame()), valid_only_support)
        delta_matrix = self._cohort_delta_matrix(c.get("cohort_phase_summary", pd.DataFrame()), support_profile)
        agreement_matrix = self._cohort_directional_agreement_matrix(c.get("cohort_phase_summary", pd.DataFrame()), support_profile)
        relationship_matrix = self._cohort_relationship_matrix(c.get("cohort_phase_summary", pd.DataFrame()), support_profile)
        all_source_relationship_matrix = self._cohort_relationship_matrix(c.get("cohort_phase_summary", pd.DataFrame()), all_source_support)
        valid_only_relationship_matrix = self._cohort_relationship_matrix(c.get("cohort_phase_summary", pd.DataFrame()), valid_only_support)
        policy_gate_register = self._policy_gate_register(c, support_profile, signal_audit_summary, all_source_response_matrix, valid_only_response_matrix)
        robustness_register = self._robustness_register(c, support_profile, all_source_response_matrix, valid_only_response_matrix)

        narrative = [
            self._spec(
                code="C02",
                stem="cohort_design_support",
                title="Session-type cohort synopsis",
                summary="The cohort is summarized at the session-type level so condition balance, factor balance, and support across the main acquisition streams can be read before inferential results are interpreted.",
                fig=self._fig_cohort_design(c),
                tags=["overview", "support", "qc"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="frontmatter",
            ),
            self._spec(
                code="C03",
                stem="cohort_window_validation",
                title="Audited device-stream landscape",
                summary=f"All audited Empatica and BIOPAC streams are shown together so adequacy, role assignment, and raw validity components can be read across the full modality inventory rather than only through the directly comparable subset. In the current audit, the stream mix comprises {primary_stream_count} primary streams, {qc_primary_stream_count} QC-qualified primary streams, and {subset_only_stream_count} subset-only stream; this figure should therefore be read as a full modality landscape rather than as a summary of only the directly comparable device pairs.",
                fig=self._fig_cohort_window_validation(c),
                tags=["overview", "qc", "support", "agreement"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="frontmatter",
            ),
            self._spec(
                code="C03A",
                stem="cohort_comparable_validation",
                title="Comparable-family validation summary",
                summary="The directly comparable Empatica/BIOPAC families are summarized separately so overlap, validation readiness, and agreement strength for heart rate, electrodermal activity, and temperature are visible without collapsing the full audit into only those families.",
                fig=self._fig_cohort_comparable_validation_summary(c),
                tags=["overview", "qc", "support", "agreement"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="frontmatter",
            ),
            self._spec(
                code="C04",
                stem="cohort_support_map",
                title="Phase-annotated cohort availability and overlap map",
                summary="Availability and paired non-null overlap are aggregated by condition and minute across the full protocol timeline so acclimation, intervention, and terminal phases remain visible before audited validity and derived summaries are interpreted.",
                fig=self._fig_cohort_support_map(c.get("cohort_minute_features", c.get("cohort_minute_comparison_window", pd.DataFrame()))),
                tags=["overview", "qc", "support", "phase"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="frontmatter",
            ),
        ]
        narrative.extend(self._build_cohort_burst_specs(c["cohort_minute_features"], ev))
        narrative.extend(self._build_cohort_condition_trace_specs(c["cohort_minute_features"], ev))
        narrative.extend(
            [
                self._spec(
                    code="C06E",
                    stem="cohort_hr_scenario_trends",
                    title="Heart-rate measured-trend scenarios",
                    summary="These measured-trend panels compare the all-source and valid-only heart-rate scenarios so the effect of excluding subset-only Empatica heart rate is visible at the trajectory level.",
                    fig=self._fig_cohort_modality_scenarios(c.get("cohort_minute_features", pd.DataFrame()), c.get("signal_audit_summary", pd.DataFrame()), "heart_rate"),
                    tags=["heart_rate", "scenario", "phase", "qc"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="physiological",
                ),
                self._spec(
                    code="C06F",
                    stem="cohort_eda_scenario_trends",
                    title="Electrodermal measured-trend scenarios",
                    summary="These measured-trend panels compare all-source and valid-only electrodermal views so device-specific inclusion and exclusion rules remain explicit when EDA is shown in the manuscript.",
                    fig=self._fig_cohort_modality_scenarios(c.get("cohort_minute_features", pd.DataFrame()), c.get("signal_audit_summary", pd.DataFrame()), "eda"),
                    tags=["eda", "scenario", "phase", "qc"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="physiological",
                ),
                self._spec(
                    code="C06G",
                    stem="cohort_temperature_scenario_trends",
                    title="Temperature measured-trend scenarios",
                    summary="These measured-trend panels compare all-source and valid-only temperature views so wearable and laboratory thermal signals can be inspected under the same scenario rules used later in the result layer.",
                    fig=self._fig_cohort_modality_scenarios(c.get("cohort_minute_features", pd.DataFrame()), c.get("signal_audit_summary", pd.DataFrame()), "temperature"),
                    tags=["temperature", "scenario", "phase", "qc"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="physiological",
                ),
                self._html_spec(
                    code="C07AA",
                    stem="cohort_master_table_registry",
                    title="Aligned master-table readiness register",
                    summary="This register reports the support status of the aligned minute-level master table across the major modality layers and paired-overlap gates. `Source coverage` rows are reported as fractions on a 0-1 scale, while `paired overlap` rows are reported in minutes. Threshold, observed, mean, and median values use the unit shown in the `Unit` column. It is the operational entry point for the Chapter 5 scientific result layer.",
                    html_fragment=self._matrix_panel_html(
                        "Aligned Master-Table Readiness Register",
                        c.get("master_table_registry", pd.DataFrame()),
                        ["layer", "gate_type", "unit", "status", "threshold", "observed_value", "mean_value", "median_value", "n_sessions_supported", "scientific_use"],
                        n=24,
                    ),
                    tags=["matrix", "qc", "support", "pipeline"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C07AB",
                    stem="cohort_feature_registry",
                    title="Derived feature registry",
                    summary="This registry documents the aligned feature space used by the Chapter 5 scientific result layer, separating primary endpoints, QC gates, support gates, and analytic covariates. For questionnaire-derived responses, the scientific denominator is expected prompts rather than elapsed time, so `Prompt Support` is the primary completeness reading. `Minute Occupancy` remains a technical aligned-minute audit field and should not be treated as the main completeness measure for discrete responses. Use `Observation Policy`, `Prompt Support Reading`, and `Minute Occupancy Reading` together.",
                    html_fragment=self._matrix_panel_html(
                        "Derived Feature Registry",
                        self._feature_registry_display(c.get("feature_registry", pd.DataFrame())),
                        ["feature", "domain", "registry_role", "unit", "observation_policy", "summary_grain", "prompt_support", "prompt_support_reading", "minute_occupancy_fraction", "minute_occupancy_reading", "n_sessions_with_data", "n_participants_with_data", "scientific_use"],
                        n=36,
                    ),
                    tags=["matrix", "registry", "pipeline", "statistics"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C07AC",
                    stem="cohort_policy_gate_register",
                    title="Scientific result gate register",
                    summary="This register translates the Section 5 scientific policy into operational gates for Chapter 5: sample adequacy, preprocessing QC, endpoint support, modality validity, inferential eligibility, predictive generalization, and scenario sensitivity.",
                    html_fragment=self._matrix_panel_html(
                        "Scientific Result Gate Register",
                        policy_gate_register,
                        ["gate", "status", "threshold", "observed_value", "evidence_basis", "scientific_implication"],
                        n=24,
                    ),
                    tags=["matrix", "policy", "qc", "pipeline"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C07AD",
                    stem="cohort_robustness_register",
                    title="Robustness and sensitivity register",
                    summary="This register condenses the main robustness checks for Chapter 5, including endpoint support stability, scenario sensitivity, multiplicity-controlled contrasts, model-fit stability, predictive variability, and the quarantine of partial-support endpoints.",
                    html_fragment=self._matrix_panel_html(
                        "Robustness And Sensitivity Register",
                        robustness_register,
                        ["gate", "status", "observed_value", "evidence_basis", "scientific_reading"],
                        n=24,
                    ),
                    tags=["matrix", "robustness", "sensitivity", "statistics"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C07A",
                    stem="cohort_endpoint_support_grading",
                    title="Endpoint support grading matrix",
                    summary="This matrix grades each candidate cohort endpoint by how completely it spans the available condition-by-phase comparison grid. It is a support screen, not an inferential result.",
                    html_fragment=self._matrix_panel_html(
                        "Cohort Endpoint Support Grading Matrix",
                        support_profile,
                        [
                            "endpoint",
                            "support_grade",
                            "support_basis",
                            "supported_phases",
                            "supported_conditions",
                            "supported_condition_phase_cells",
                            "cell_coverage_fraction",
                            "median_sessions_per_condition_phase",
                            "total_valid_phase_summaries",
                            "scientific_reading",
                        ],
                        n=24,
                    ),
                    tags=["matrix", "support", "statistics"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._spec(
                    code="C07B",
                    stem="cohort_endpoint_support_grades",
                    title="Endpoint support grading summary",
                    summary="This figure summarizes how many condition-phase cells are supported for each endpoint before the cohort result layer is interpreted. For questionnaire and control/context endpoints, this reflects supported event- or state-derived condition-phase cells rather than continuous timeline completeness.",
                    fig=self._fig_cohort_endpoint_support_grades(support_profile),
                    tags=["support", "statistics"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._spec(
                    code="C12A",
                    stem="cohort_preprocessing_qc",
                    title="Preprocessing quality diagnostics",
                    summary="This panel summarizes minute-level quality retention for the major wearable and laboratory channels after preprocessing. It is intended to show whether later inferential and predictive layers rest on sufficiently valid signal support.",
                    fig=self._fig_preprocessing_qc_summary(c.get("preprocessing_qc_summary", pd.DataFrame())),
                    tags=["qc", "support", "statistics"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._spec(
                    code="C12AA",
                    stem="cohort_derived_feature_landscape",
                    title="Derived-feature support, variability, and condition-balance overview",
                    summary="This figure surveys the derived feature layer to show which measures have the strongest aligned support, which supported features show the largest robust variation, how support is distributed across feature domains, and how support balance differs across questionnaire, wearable, and indoor streams by condition.",
                    fig=self._fig_cohort_exploratory_landscape(
                        c.get("exploratory_feature_summary", pd.DataFrame()),
                        c.get("condition_support_summary", pd.DataFrame()),
                        c.get("feature_registry", pd.DataFrame()),
                    ),
                    tags=["statistics", "coverage", "exploratory", "landscape"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._spec(
                    code="C12AB",
                    stem="cohort_primary_endpoint_mean_atlas_questionnaire_environment",
                    title="Primary endpoint mean atlas: questionnaire and environment",
                    summary="This atlas summarizes cohort-level means for the questionnaire and environmental primary endpoints across protocol phases and conditions before contrast models are applied. It is a descriptive view of the higher-level comfort and context layer.",
                    fig=self._fig_cohort_primary_endpoints_raw(
                        c.get("cohort_primary_endpoints", pd.DataFrame()),
                        metrics_override=[
                            "thermal_comfort",
                            "thermal_sensation",
                            "indoor_air_velocity_mean_m_s",
                            "empatica_eda_mean_uS",
                        ],
                    ),
                    tags=["statistics", "endpoints", "atlas", "phase"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._spec(
                    code="C12ABA",
                    stem="cohort_primary_endpoint_mean_atlas_physiology",
                    title="Primary endpoint mean atlas: physiology",
                    summary="This atlas summarizes cohort-level means for the physiological primary endpoints across protocol phases and conditions before contrast models are applied. It is a descriptive view of the retained physiology layer.",
                    fig=self._fig_cohort_primary_endpoints_raw(
                        c.get("cohort_primary_endpoints", pd.DataFrame()),
                        metrics_override=[
                            "empatica_temp_mean_C",
                            "biopac_hr_mean_bpm",
                            "biopac_eda_mean_uS",
                            "biopac_temp_chest_mean_C",
                        ],
                    ),
                    tags=["statistics", "endpoints", "atlas", "phase"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._spec(
                    code="C12AC",
                    stem="cohort_primary_endpoint_standardized_atlas",
                    title="Primary endpoint standardized atlas",
                    summary="This heatmap standardizes the primary endpoint layer within each metric so cross-metric pattern structure can be compared without conflating units or scale ranges.",
                    fig=self._fig_cohort_primary_endpoints(c.get("cohort_primary_endpoints", pd.DataFrame())),
                    tags=["statistics", "endpoints", "atlas", "heatmap"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C12B",
                    stem="cohort_corrected_contrast_register",
                    title="Multiplicity-corrected contrast register",
                    summary="This register lists the globally strongest eligible paired condition contrasts after Benjamini-Hochberg correction. It is intentionally rank-ordered by corrected significance and may therefore be dominated by one endpoint family when one family carries the smallest corrected p-values. Confidence intervals are bootstrap intervals on the matched mean difference. Use Table 5.7 to read breadth across endpoint families and Table 5.8 to read model-based fixed-effect evidence for the same primary endpoint layer. How to read it: this is the strongest-hit contrast list, so each row is a phase-specific paired condition comparison rather than a whole-model coefficient.",
                    html_fragment=self._matrix_panel_html(
                        "Corrected Condition Contrast Register",
                        self._cohort_top_contrast_register(c.get("condition_contrasts", pd.DataFrame())),
                        ["metric", "protocol_phase", "comparison", "primary_test", "n_pairs", "mean_difference", "ci_low", "ci_high", "primary_p_value", "p_value_fdr", "inference_label"],
                        n=24,
                    ),
                    tags=["statistics", "contrast", "phase"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C12BA",
                    stem="cohort_balanced_contrast_register",
                    title="Balanced multiplicity-corrected contrast register",
                    summary="This companion register keeps the same Benjamini-Hochberg-corrected contrast layer but balances representation across endpoint families before filling remaining slots by corrected significance. It should be used to interpret the breadth of the contrast layer, whereas Table 5.6 should be read as the strongest-hit list. Confidence intervals are bootstrap intervals on the matched mean difference. Use Table 5.8 when you need model-based fixed-effect evidence instead of paired contrast evidence. How to read it: this is the breadth-aware contrast summary, so rows are still paired condition contrasts, but family balancing prevents one endpoint family from monopolizing the table.",
                    html_fragment=self._matrix_panel_html(
                        "Balanced Corrected Condition Contrast Register",
                        self._cohort_balanced_contrast_register(c.get("condition_contrasts", pd.DataFrame())),
                        ["contrast_family", "metric", "protocol_phase", "comparison", "primary_test", "n_pairs", "mean_difference", "ci_low", "ci_high", "primary_p_value", "p_value_fdr", "inference_label"],
                        n=24,
                    ),
                    tags=["statistics", "contrast", "phase", "balanced_register"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C12C",
                    stem="cohort_mixed_effects_register",
                    title="Mixed-effects primary-endpoint register",
                    summary="This register reports participant-level mixed-effects fixed-effect estimates for the endpoint-policy primary set, screening terms by Benjamini-Hochberg-corrected p-values across the retained mixed-model term layer. Only interpretive fixed effects are shown here; variance and covariance parameters are excluded and fit warnings should be checked in Table 5.9. Reference levels are the bright condition (`BRI`), midday (`MID`), and fan-at-constant-speed phase (`FCS`). Use Tables 5.6 and 5.7 for paired contrast summaries and use this register when you need whole-model repeated-measures evidence. How to read it: each row is a model coefficient relative to those reference levels, so beta gives the direction and magnitude of the modeled shift for that term rather than a pairwise condition contrast.",
                    html_fragment=self._matrix_panel_html(
                        "Mixed-Effects Primary-Endpoint Register",
                        self._mixed_effects_register(c.get("mixed_effects_primary", pd.DataFrame())),
                        ["metric", "term", "term_reading", "beta", "ci_low", "ci_high", "p_value", "p_value_fdr", "significant_fdr", "n_obs", "n_participants"],
                        n=30,
                    ),
                    tags=["statistics", "mixed_model", "phase"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C12CA",
                    stem="cohort_mixed_effects_diagnostics",
                    title="Mixed-effects fit diagnostics",
                    summary="This register shows which endpoint-policy primary metrics were retained, skipped, or fit with warnings in the mixed-effects layer, so model eligibility is explicit rather than inferred from terminal warnings. `Model specification` names the richest converged model that was retained for that metric, and warning summaries indicate boundary, singular-covariance, or Hessian issues that can weaken coefficient trust. How to read it: read `Status`, `Fit Converged`, and `Warnings` first to judge model reliability, then use `Retained Terms` and `Model Specification` to understand how much interpretable fixed-effect structure survived.",
                    html_fragment=self._matrix_panel_html(
                        "Mixed-Effects Fit Diagnostics",
                        self._mixed_effects_diagnostics_register(c.get("mixed_effects_diagnostics", pd.DataFrame())),
                        ["metric", "status", "model_spec", "n_obs", "n_participants", "n_terms_retained", "fit_converged", "warning_count", "warning_summary"],
                        n=24,
                    ),
                    tags=["statistics", "mixed_model", "qc"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._spec(
                    code="C12CB",
                    stem="cohort_modeling_overview",
                    title="Statistical and advanced modeling overview",
                    summary="This figure condenses the statistical evidence layer into two views: the top panel summarizes corrected contrast evidence across endpoint families, and the middle panel shows which endpoints retain phase, light, time, or interaction evidence in the mixed-effects layer. Use Table 5.6 for the strongest-hit contrast list, Table 5.7 for the breadth-aware contrast summary, Table 5.8 for endpoint-level mixed-effects coefficients, and Table 5.9 for fit warnings. How to read it: the top panel shows family-level contrast burden and breadth, and the middle panel shows endpoint-level mixed-model evidence structure.",
                    fig=self._fig_cohort_modeling_overview(
                        c.get("condition_contrasts", pd.DataFrame()),
                        c.get("mixed_effects_primary", pd.DataFrame()),
                        c.get("predictive_benchmarks", pd.DataFrame()),
                    ),
                    tags=["statistics", "mixed_model", "prediction", "overview"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._spec(
                    code="C12CC",
                    stem="cohort_lag_response_register",
                    title="Lag-response evidence register",
                    summary="This register identifies the strongest support-screened lag for each pre-registered driver-response pair, while also showing the tested lag profile behind each retained result so delayed thermal and physiological response timing is explicit in the Chapter 5 evidence layer.",
                    fig=self._fig_lag_response_register(
                        c.get("lag_response_register", pd.DataFrame()),
                        c.get("lag_response_profile", pd.DataFrame()),
                    ),
                    tags=["statistics", "lag", "temporal", "evidence"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C12CD",
                    stem="cohort_threshold_response_register",
                    title="Threshold-response register",
                    summary="This register reports estimated breakpoints for retained driver-response pairs using segmented fits at each pair's strongest lag, so Chapter 5 decisions can cite explicit threshold evidence rather than only quartile bands. Threshold values use the predictor unit shown in the `Unit` column. Simpler scientific interpretation: this table asks whether a response appears to change behavior around a specific driver level, and if so, where that possible breakpoint sits. How to read it: read `Evidence Grade`, `RSS Improvement`, and `Scientific Reading` together before treating a breakpoint as operationally meaningful.",
                    html_fragment=self._matrix_panel_html(
                        "Threshold-Response Register",
                        self._threshold_response_register_display(c.get("threshold_response_register", pd.DataFrame())),
                        ["predictor", "target", "threshold_unit", "threshold_value", "slope_below", "slope_above", "slope_change", "rss_improvement_fraction", "n_pairs", "n_sessions", "evidence_grade", "scientific_reading"],
                        n=18,
                    ),
                    tags=["statistics", "threshold", "segmented", "lag"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C12CE",
                    stem="cohort_scientific_decision_register",
                    title="Scientific decision register",
                    summary="This final decision layer translates validated Chapter 5 findings into estimated breakpoints or operating bands, response lags, evidence grades, and control-facing scientific readings. Operating bands use the predictor unit shown in the `Unit` column. Simpler scientific interpretation: this table turns the modeled lag and breakpoint evidence into a practical statement about which driver range is more favorable and how quickly the outcome tends to respond. How to read it: use `Statistical Basis` for the evidential basis, `Practical Reading` for the size and direction of the observed shift, and `Control Recommendation` for the decision-facing interpretation.",
                    html_fragment=self._matrix_panel_html(
                        "Scientific Decision Register",
                        self._scientific_decision_register_display(c.get("scientific_decision_register", pd.DataFrame())),
                        ["claim_family", "predictor", "target", "threshold_unit", "recommended_operating_band", "response_lag_minutes", "evidence_grade", "supporting_streams", "statistical_basis", "practical_reading", "control_recommendation"],
                        n=18,
                    ),
                    tags=["statistics", "decision", "lag", "control"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._spec(
                    code="C12D",
                    stem="cohort_predictive_benchmarks",
                    title="Validation-aware predictive benchmarks",
                    summary="This panel compares environment-only, physiology-only, and fused models for comfort-state prediction across participant, study-day, and condition holdout schemes, so multimodal gain and generalization risk remain explicit.",
                    fig=self._fig_predictive_benchmarks(c.get("predictive_benchmarks", pd.DataFrame())),
                    tags=["statistics", "prediction", "benchmark"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._spec(
                    code="C12E",
                    stem="cohort_pattern_atlas",
                    title="Derived pattern atlas",
                    summary="This figure summarizes recurring within-session motifs in the derived endpoint layer, showing which phases dominate repeated response patterns and which session-level motifs are strongest across the cohort.",
                    fig=self._fig_cohort_pattern_atlas(c.get("pattern_summary", pd.DataFrame()), c.get("phase_pattern_inventory", pd.DataFrame())),
                    tags=["statistics", "patterns", "atlas", "phase"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._spec(
                    code="C12F",
                    stem="cohort_participant_profile_atlas",
                    title="Participant heterogeneity atlas",
                    summary="This atlas shows participant-by-condition heterogeneity across the main subjective and physiological summary endpoints, making between-participant variation explicit before group summaries are generalized.",
                    fig=self._fig_cohort_participant_profile_atlas(c.get("participant_profiles", pd.DataFrame())),
                    tags=["statistics", "participants", "heterogeneity", "atlas"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C07C",
                    stem="cohort_modality_scenario_register",
                    title="Modality scenario register",
                    summary="This register defines the modality combinations used in the cohort report so audit views with all sources remain distinct from claim-supporting views restricted to scientifically valid streams.",
                    html_fragment=self._matrix_panel_html(
                        "Cohort Modality Scenario Register",
                        scenario_register,
                        ["scenario", "included_streams", "excluded_streams", "scientific_use"],
                        n=8,
                    ),
                    tags=["matrix", "support", "qc", "scenario"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="frontmatter",
                ),
                self._html_spec(
                    code="C07CA",
                    stem="cohort_modality_claim_register",
                    title="Modality manuscript-claim register",
                    summary="This register states what each signal stream may support in the manuscript: primary claim, QC-qualified claim, audit-only use, or no claim.",
                    html_fragment=self._matrix_panel_html(
                        "Cohort Modality Manuscript-Claim Register",
                        modality_claim_register,
                        ["signal_stream", "construct", "adequacy_status", "recommended_role", "manuscript_use", "manuscript_claim"],
                        n=16,
                    ),
                    tags=["matrix", "qc", "support", "scenario", "manuscript"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="frontmatter",
                ),
                self._html_spec(
                    code="C07CB",
                    stem="cohort_endpoint_claim_register",
                    title="Endpoint manuscript-claim register",
                    summary="This register translates endpoint support grades and modality gates into manuscript-use categories, separating claim-supporting endpoints from audit-only or descriptive endpoints.",
                    html_fragment=self._matrix_panel_html(
                        "Cohort Endpoint Manuscript-Claim Register",
                        endpoint_claim_register,
                        ["endpoint", "support_grade", "support_basis", "modality_gate", "claim_status", "claim_note"],
                        n=24,
                    ),
                    tags=["matrix", "support", "scenario", "manuscript"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="frontmatter",
                ),
                self._html_spec(
                    code="C07CC",
                    stem="cohort_device_stream_inventory",
                    title="Empatica and BIOPAC stream inventory",
                    summary="This register lists all major Empatica and BIOPAC streams in the cohort export, showing whether each stream is present, audited, cross-device comparable, used as a direct analytic feature or only as an audited/reported stream, and linked to an explicit endpoint-policy role.",
                    html_fragment=self._matrix_panel_html(
                        "Cohort Device Stream Inventory",
                        device_stream_inventory,
                        ["stream_label", "device", "construct", "comparison_class", "present_in_cohort_table", "signal_audited", "cross_device_comparable", "analytic_feature", "stream_usage", "endpoint_policy_role", "recommended_role", "adequacy_status"],
                        n=24,
                    ),
                    tags=["matrix", "inventory", "qc", "scenario", "manuscript"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="frontmatter",
                ),
                self._html_spec(
                    code="C07CD",
                    stem="cohort_analysis_pathway_register",
                    title="Endpoint analysis pathway register",
                    summary="This register shows how endpoints move from available cohort tables into support scoring, valid-only eligibility, and endpoint-policy interpretation. It makes explicit what is primary, QC-qualified, secondary, audit-only, or outside the endpoint policy.",
                    html_fragment=self._matrix_panel_html(
                        "Cohort Endpoint Analysis Pathway Register",
                        analysis_pathway_register,
                        ["endpoint", "metric", "source_streams", "in_cohort_table", "support_grade", "support_basis", "endpoint_policy_role", "pathway_status"],
                        n=36,
                    ),
                    tags=["matrix", "inventory", "support", "scenario", "manuscript"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="frontmatter",
                ),
                self._spec(
                    code="C07D",
                    stem="cohort_hr_scenario_comparison",
                    title="Heart-rate scenario comparison",
                    summary="Heart-rate summaries are shown under both all-source and valid-only scenarios so the effect of excluding Empatica heart rate from claim-supporting interpretation is visible rather than implicit.",
                    fig=self._fig_cohort_hr_scenarios(c.get("cohort_phase_summary", pd.DataFrame()), c.get("signal_audit_summary", pd.DataFrame())),
                    tags=["heart_rate", "scenario", "qc", "support"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C08AA",
                    stem="cohort_all_source_response_matrix",
                    title="All-source support-screened condition-phase median matrix",
                    summary="This audit matrix retains every support-screened endpoint, including streams that remain descriptive, subset-only, or device-limited. It should be used to inspect how modality inclusion changes the cohort picture, not as the default claim-supporting figure.",
                    html_fragment=self._matrix_panel_html(
                        "All-Source Condition-Phase Median Matrix",
                        all_source_response_matrix,
                        ["row_label", "support_basis", "FCS", "SR", "FFC", "SS", "OC", "n_sessions", "total_valid_phase_summaries", "condition_phase_support"],
                        n=36,
                    ),
                    tags=["matrix", "phase", "statistics", "scenario"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C08AB",
                    stem="cohort_valid_only_response_matrix",
                    title="Valid-only support-screened condition-phase median matrix",
                    summary="This claim-supporting matrix excludes streams that fail the modality-validity screen. It is the defensible cohort view when only primary and primary-with-QC modalities are retained.",
                    html_fragment=self._matrix_panel_html(
                        "Valid-Only Condition-Phase Median Matrix",
                        valid_only_response_matrix,
                        ["row_label", "support_basis", "FCS", "SR", "FFC", "SS", "OC", "n_sessions", "total_valid_phase_summaries", "condition_phase_support"],
                        n=36,
                    ),
                    tags=["matrix", "phase", "statistics", "scenario"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._spec(
                    code="C12AE",
                    stem="cohort_contrast_overview",
                    title="Condition-contrast overview",
                    summary="This figure summarizes matched condition differences across the main endpoint families, separating condition contrasts that meet the paired inferential screen from descriptive-only contrasts.",
                    fig=self._fig_cohort_contrasts(c.get("condition_contrasts", pd.DataFrame()), ev),
                    tags=["statistics", "contrast", "phase", "overview"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C08A",
                    stem="cohort_response_matrix",
                    title=f"{result_prefix} condition-phase median matrix",
                    summary=f"This matrix reports cohort condition-phase medians for endpoints retained by the support screen. Values remain descriptive and should be read alongside session counts and phase support. {result_summary_suffix}",
                    html_fragment=self._matrix_panel_html(
                        f"{result_prefix.title()} Condition-Phase Median Matrix",
                        response_matrix,
                        ["row_label", "support_basis", "FCS", "SR", "FFC", "SS", "OC", "n_sessions", "total_valid_phase_summaries", "condition_phase_support"],
                        n=36,
                    ),
                    tags=["matrix", "phase", "statistics"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._spec(
                    code="C08B",
                    stem="cohort_response_heatmap",
                    title=f"{result_prefix} condition-phase median heatmap",
                    summary=f"The heatmap visualizes the same condition-phase medians as the matrix, scaled within each endpoint-condition row for display only. Exact raw values should be read from the matrix. {result_summary_suffix}",
                    fig=self._cohort_response_heatmap(response_matrix),
                    tags=["heatmap", "phase", "statistics"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C09A",
                    stem="cohort_delta_matrix",
                    title=f"{result_prefix} reference-phase delta matrix",
                    summary=f"This matrix reports condition-phase medians minus the earliest supported phase median within the same condition. It is a descriptive contrast table, not an inferential effect estimate. {result_summary_suffix}",
                    html_fragment=self._matrix_panel_html(
                        f"{result_prefix.title()} Reference-Phase Delta Matrix",
                        delta_matrix,
                        ["row_label", "reference_phase", "FCS", "SR", "FFC", "SS", "OC"],
                        n=36,
                    ),
                    tags=["matrix", "phase", "statistics"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._spec(
                    code="C09B",
                    stem="cohort_delta_heatmap",
                    title=f"{result_prefix} reference-phase delta heatmap",
                    summary=f"The heatmap visualizes the same reference-phase deltas as the matrix, scaled within each row to a signed display range from -1 to 1. Exact raw deltas should be read from the matrix. {result_summary_suffix}",
                    fig=self._cohort_delta_heatmap(delta_matrix),
                    tags=["heatmap", "phase", "statistics"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C10A",
                    stem="cohort_directional_agreement_matrix",
                    title=f"{result_prefix} phase-by-phase session-sign agreement matrix",
                    summary=f"This matrix reports how often sessions within the same condition share the same direction of change relative to their own reference phase. Blank cells indicate too few sessions for a stable agreement read. {result_summary_suffix}",
                    html_fragment=self._matrix_panel_html(
                        f"{result_prefix.title()} Phase-By-Phase Session-Sign Agreement Matrix",
                        agreement_matrix,
                        ["row_label", "FCS", "SR", "FFC", "SS", "OC", "condition_phase_support"],
                        n=36,
                    ),
                    tags=["matrix", "statistics", "repeatability"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._spec(
                    code="C10B",
                    stem="cohort_directional_agreement_heatmap",
                    title=f"{result_prefix} phase-by-phase session-sign agreement heatmap",
                    summary=f"The heatmap visualizes the same session-sign agreement values as the matrix on a 0 to 1 scale. It should be interpreted only where the matrix shows enough contributing sessions. {result_summary_suffix}",
                    fig=self._cohort_directional_agreement_heatmap(agreement_matrix),
                    tags=["heatmap", "statistics", "repeatability"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C10C",
                    stem="cohort_partial_results_register",
                    title="Partial-results register",
                    summary="This register lists endpoints that are not used as primary cohort results because their condition-phase support is incomplete or too sparse for a stable cohort-level reading.",
                    html_fragment=self._matrix_panel_html(
                        "Cohort Partial-Results Register",
                        support_profile.loc[support_profile["support_grade"] != "strong"].copy() if not support_profile.empty else pd.DataFrame(),
                        [
                            "endpoint",
                            "support_grade",
                            "support_basis",
                            "supported_phases",
                            "supported_conditions",
                            "supported_condition_phase_cells",
                            "cell_coverage_fraction",
                            "total_valid_phase_summaries",
                            "scientific_reading",
                        ],
                        n=24,
                    ),
                    tags=["matrix", "support", "statistics"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
            ]
        )
        appendix = [
            self._html_spec(
                code="C11AA",
                stem="cohort_all_source_relationship_matrix",
                title="All-source relationship matrix",
                summary="This audit matrix keeps relationships from every retained all-source endpoint, including device-limited or subset-only modalities. It is intended to show how relationship structure changes when weaker streams are included.",
                html_fragment=self._matrix_panel_html(
                    "Cohort All-Source Relationship Matrix",
                    all_source_relationship_matrix,
                    ["source", "target", "spearman_r", "paired_n", "min_required_n", "qualified_conditions", "same_sign_fraction", "relationship_status", "condition_support_status"],
                    n=64,
                ),
                tags=["matrix", "relationships", "statistics", "scenario"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="interpretive",
            ),
            self._spec(
                code="C11AB",
                stem="cohort_all_source_relationship_heatmap",
                title="All-source relationship heatmap",
                summary="This audit heatmap visualizes the all-source relationship matrix and may include associations carried by subset-only or device-limited streams.",
                fig=self._fig_cohort_relationship_heatmap(all_source_relationship_matrix),
                tags=["heatmap", "relationships", "statistics", "scenario"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="interpretive",
            ),
            self._html_spec(
                code="C11AC",
                stem="cohort_valid_only_relationship_matrix",
                title="Valid-only relationship matrix",
                summary="This claim-supporting matrix retains only relationships among endpoints supported by scientifically valid modalities, excluding subset-only streams such as Empatica heart rate in the current release.",
                html_fragment=self._matrix_panel_html(
                    "Cohort Valid-Only Relationship Matrix",
                    valid_only_relationship_matrix,
                    ["source", "target", "spearman_r", "paired_n", "min_required_n", "qualified_conditions", "same_sign_fraction", "relationship_status", "condition_support_status"],
                    n=64,
                ),
                tags=["matrix", "relationships", "statistics", "scenario"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="interpretive",
            ),
            self._spec(
                code="C11AD",
                stem="cohort_valid_only_relationship_heatmap",
                title="Valid-only relationship heatmap",
                summary="This claim-supporting heatmap visualizes only relationships that remain after restricting the cohort result layer to scientifically valid modalities.",
                fig=self._fig_cohort_relationship_heatmap(valid_only_relationship_matrix),
                tags=["heatmap", "relationships", "statistics", "scenario"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="interpretive",
            ),
            self._html_spec(
                code="C11A",
                stem="cohort_relationship_matrix",
                title=f"{result_prefix} relationship matrix",
                summary=f"This matrix reports unique session-level delta associations among the retained cohort endpoints only. It also shows how many conditions independently support the same association sign, so paired count is not read in isolation. {result_summary_suffix}",
                html_fragment=self._matrix_panel_html(
                    f"Cohort {result_prefix.title()} Relationship Matrix",
                    relationship_matrix,
                    ["source", "target", "spearman_r", "paired_n", "min_required_n", "qualified_conditions", "same_sign_fraction", "relationship_status", "condition_support_status"],
                    n=48,
                ),
                tags=["matrix", "relationships", "statistics"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="interpretive",
            ),
            self._spec(
                code="C11B",
                stem="cohort_relationship_heatmap",
                title=f"{result_prefix} relationship heatmap",
                summary=f"The heatmap visualizes the same retained descriptive associations as the matrix. It is a pattern screen only: exact support thresholds, qualified condition counts, and same-sign fractions should be read from the matrix. {result_summary_suffix}",
                fig=self._fig_cohort_relationship_heatmap(relationship_matrix),
                tags=["heatmap", "relationships", "statistics"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="interpretive",
            ),
            self._spec(
                code="C11AE",
                stem="cohort_all_source_targeted_relationships",
                title="All-source targeted relationships with thermal comfort",
                summary="These scatter panels keep the full all-source endpoint set so the audit view can show whether subset-only or device-limited modalities materially alter the apparent comfort relationships.",
                fig=self._fig_cohort_targeted_relationships(c.get("cohort_phase_summary", pd.DataFrame()), all_source_support),
                tags=["relationships", "statistics", "comfort", "scenario"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="interpretive",
            ),
            self._spec(
                code="C11AF",
                stem="cohort_valid_only_targeted_relationships",
                title="Valid-only targeted relationships with thermal comfort",
                summary="These scatter panels restrict the relationship view to endpoints supported by scientifically valid modalities and should be preferred when the figure is used to support manuscript claims.",
                fig=self._fig_cohort_targeted_relationships(c.get("cohort_phase_summary", pd.DataFrame()), valid_only_support),
                tags=["relationships", "statistics", "comfort", "scenario"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="interpretive",
            ),
            self._spec(
                code="C11C",
                stem="cohort_targeted_relationships",
                title="Targeted relationships with thermal comfort",
                summary=f"These scatter panels focus on session-level thermal-comfort deltas versus the most interpretable retained environmental or physiological deltas. No fitted linear trend lines are added to avoid overstating a small-cohort association screen. {result_summary_suffix}",
                fig=self._fig_cohort_targeted_relationships(c.get("cohort_phase_summary", pd.DataFrame()), support_profile),
                tags=["relationships", "statistics", "comfort"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="interpretive",
            ),
            self._spec(
                code="C05",
                stem="cohort_agreement",
                title="How closely directly comparable device pairs agree across sessions",
                summary="This view is limited to the directly comparable Empatica/BIOPAC constructs and shows where paired devices align across sessions with enough overlapping data.",
                fig=self._fig_cohort_agreement(c["sensor_agreement"]),
                tags=["agreement", "heart_rate", "eda", "temperature"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="interpretive",
            ),
            self._spec(
                code="C10",
                stem="cohort_agreement_summary",
                title="Agreement summary for directly comparable modalities",
                summary="This summary compares overlap, correlation, and error across the directly comparable cross-device constructs only; it is not a summary of the full modality inventory.",
                fig=self._fig_cohort_agreement_summary(c.get("agreement_summary", pd.DataFrame())),
                tags=["appendix", "agreement", "statistics"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="interpretive",
            ),
        ]
        return [x for x in narrative if x["fig"] is not None or x.get("html_fragment")], [x for x in appendix if x["fig"] is not None or x.get("html_fragment")]

    def _build_cohort_condition_trace_specs(self, minute: pd.DataFrame, ev: dict) -> list[dict]:
        metric_specs = [
            ("C06A", "thermal_comfort", "Condition-stratified raw Thermal Comfort observations", ["appendix", "phase", "comfort", "raw"], True),
            ("C06B", "thermal_sensation", self._series_title("thermal_sensation", scope="Condition-stratified", kind="distributions"), ["appendix", "phase", "comfort", "summary"], False),
            ("C06C", "thermal_comfort", self._series_title("thermal_comfort", scope="Condition-stratified", kind="distributions"), ["appendix", "phase", "comfort", "summary"], False),
            ("C06D", "thermal_preference", self._series_title("thermal_preference", scope="Condition-stratified", kind="distributions"), ["appendix", "phase", "comfort", "summary"], False),
            ("C06E", "thermal_pleasure", self._series_title("thermal_pleasure", scope="Condition-stratified", kind="distributions"), ["appendix", "phase", "comfort", "summary"], False),
            ("C06F", "visual_sensation", self._series_title("visual_sensation", scope="Condition-comparison", kind="distributions"), ["appendix", "phase", "comfort", "summary"], False),
            ("C06G", "color_sensation", self._series_title("color_sensation", scope="Condition-comparison", kind="distributions"), ["appendix", "phase", "comfort", "summary"], False),
            ("C06H", "room_comfort", self._series_title("room_comfort", scope="Condition-comparison", kind="distributions"), ["appendix", "phase", "comfort", "summary"], False),
            ("C06I", "visual_comfort", self._series_title("visual_comfort", scope="Condition-comparison", kind="distributions"), ["appendix", "phase", "comfort", "summary"], False),
            ("C06J", "sound_comfort_dbA", self._series_title("sound_comfort_dbA", scope="Condition-comparison", kind="distributions"), ["appendix", "phase", "comfort", "summary"], False),
            ("C06K", "air_quality_comfort", self._series_title("air_quality_comfort", scope="Condition-comparison", kind="distributions"), ["appendix", "phase", "comfort", "summary"], False),
            ("C06L", "empatica_hr_mean_bpm", self._series_title("empatica_hr_mean_bpm", scope="Condition-stratified", kind="trajectories"), ["appendix", "phase", "heart_rate"], False),
            ("C06M", "biopac_temp_chest_mean_C", self._series_title("biopac_temp_chest_mean_C", scope="Condition-stratified", kind="trajectories"), ["appendix", "phase", "temperature"], False),
            ("C06N", "indoor_air_velocity_mean_m_s", self._series_title("indoor_air_velocity_mean_m_s", scope="Condition-stratified", kind="trajectories"), ["appendix", "phase", "environment"], False),
            ("C06O", "outdoor_air_temp_C", self._series_title("outdoor_air_temp_C", scope="Condition-stratified", kind="trajectories"), ["appendix", "phase", "environment"], False),
            ("C06P", "outdoor_wind_speed_m_s", self._series_title("outdoor_wind_speed_m_s", scope="Condition-stratified", kind="trajectories"), ["appendix", "phase", "environment"], False),
            ("C06Q", "outdoor_relative_humidity_percent", self._series_title("outdoor_relative_humidity_percent", scope="Condition-stratified", kind="trajectories"), ["appendix", "phase", "environment"], False),
            ("C06R", "outdoor_solar_radiation_W_m2", self._series_title("outdoor_solar_radiation_W_m2", scope="Condition-stratified", kind="trajectories"), ["appendix", "phase", "environment"], False),
        ]
        specs: list[dict] = []
        for code, metric, title, tags, use_raw_sparse in metric_specs:
            fig = self._fig_sparse_phase_distribution(minute, metric) if use_raw_sparse else self._fig_cohort_condition_trace(minute, metric)
            stem = f"cohort_condition_trace_{metric}_raw" if use_raw_sparse else f"cohort_condition_trace_{metric}"
            specs.append(
                self._spec(
                    code=code,
                    stem=stem,
                    title=title,
                    summary=(
                        (
                            "This condition-stratified raw view keeps the questionnaire-event observations visible by condition and phase, so sparse support and uneven event density remain auditable before reading the summarized condition pattern panel that follows."
                            if use_raw_sparse
                            else self._cohort_questionnaire_caption(metric, aggregated=True)
                        )
                        if self._is_sparse_observation_channel(metric)
                        else "Condition trajectories are shown as standalone plots so each modality can be read without subplot compression or cross-axis crowding."
                    ),
                    fig=fig,
                    tags=tags,
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section=self._cohort_metric_section(metric),
                )
            )
        return specs

    def _build_cohort_burst_specs(self, minute: pd.DataFrame, ev: dict) -> list[dict]:
        channel_specs = [
            ("C04B", "empatica_bvp_mean", "#7c3aed", self._series_title("empatica_bvp_mean", scope="Cohort", kind="bursts"), ["bvp", "exploratory"]),
            ("C04C", "empatica_hr_mean_bpm", "#b91c1c", self._series_title("empatica_hr_mean_bpm", scope="Cohort", kind="bursts"), ["heart_rate", "exploratory"]),
            ("C04D", "empatica_eda_mean_uS", "#1d4ed8", self._series_title("empatica_eda_mean_uS", scope="Cohort", kind="bursts"), ["eda", "exploratory"]),
            ("C04E", "empatica_temp_mean_C", "#ea580c", self._series_title("empatica_temp_mean_C", scope="Cohort", kind="bursts"), ["temperature", "exploratory"]),
            ("C04W", "empatica_acc_mean_g", "#b91c1c", self._series_title("empatica_acc_mean_g", scope="Cohort", kind="bursts"), ["motion", "exploratory"]),
            ("C04X", "empatica_enmo_mean_g", "#2563eb", self._series_title("empatica_enmo_mean_g", scope="Cohort", kind="bursts"), ["motion", "exploratory"]),
            ("C04Y", "empatica_steps", "#0f766e", self._series_title("empatica_steps", scope="Cohort", kind="bursts"), ["activity", "exploratory"]),
            ("C04F", "biopac_hr_mean_bpm", "#111827", self._series_title("biopac_hr_mean_bpm", scope="Cohort", kind="bursts"), ["heart_rate", "exploratory"]),
            ("C04G", "biopac_eda_mean_uS", "#2563eb", self._series_title("biopac_eda_mean_uS", scope="Cohort", kind="bursts"), ["eda", "exploratory"]),
            ("C04H", "biopac_temp_chest_mean_C", "#ea580c", self._series_title("biopac_temp_chest_mean_C", scope="Cohort", kind="bursts"), ["temperature", "exploratory"]),
            ("C04Z", "biopac_temp_thigh_mean_C", "#f59e0b", self._series_title("biopac_temp_thigh_mean_C", scope="Cohort", kind="bursts"), ["temperature", "exploratory"]),
            ("C04ZA", "biopac_temp_arm_mean_C", "#dc2626", self._series_title("biopac_temp_arm_mean_C", scope="Cohort", kind="bursts"), ["temperature", "exploratory"]),
            ("C04ZB", "biopac_temp_tibia_mean_C", "#7c3aed", self._series_title("biopac_temp_tibia_mean_C", scope="Cohort", kind="bursts"), ["temperature", "exploratory"]),
            ("C04I", "biopac_bloodflow_mean_bpu", "#7c3aed", self._series_title("biopac_bloodflow_mean_bpu", scope="Cohort", kind="bursts"), ["bloodflow", "exploratory"]),
            ("C04ZC", "biopac_backscatter_mean_percent", "#64748b", self._series_title("biopac_backscatter_mean_percent", scope="Cohort", kind="bursts"), ["optical", "exploratory"]),
            ("C04J", "indoor_air_temp_mean_C", "#ea580c", self._series_title("indoor_air_temp_mean_C", scope="Cohort", kind="bursts"), ["environment", "exploratory"]),
            ("C04K", "indoor_air_velocity_mean_m_s", "#0f766e", self._series_title("indoor_air_velocity_mean_m_s", scope="Cohort", kind="bursts"), ["environment", "exploratory"]),
            ("C04L", "indoor_relative_humidity_percent", "#2563eb", self._series_title("indoor_relative_humidity_percent", scope="Cohort", kind="bursts"), ["environment", "exploratory"]),
            ("C04LD", "outdoor_air_temp_C", "#ea580c", self._series_title("outdoor_air_temp_C", scope="Cohort", kind="bursts"), ["environment", "exploratory"]),
            ("C04LE", "outdoor_relative_humidity_percent", "#2563eb", self._series_title("outdoor_relative_humidity_percent", scope="Cohort", kind="bursts"), ["environment", "exploratory"]),
            ("C04LF", "outdoor_wind_speed_m_s", "#0f766e", self._series_title("outdoor_wind_speed_m_s", scope="Cohort", kind="bursts"), ["environment", "exploratory"]),
            ("C04LG", "outdoor_solar_radiation_W_m2", "#f59e0b", self._series_title("outdoor_solar_radiation_W_m2", scope="Cohort", kind="bursts"), ["environment", "exploratory"]),
            ("C04M", "fan_control_au", "#111827", self._series_title("fan_control_au", scope="Cohort", kind="bursts"), ["fan", "exploratory"]),
            ("C04R", "fan_current_A", "#111827", self._series_title("fan_current_A", scope="Cohort", kind="bursts"), ["fan", "exploratory"]),
            ("C04S", "fan_control_secondary_au", "#7c3aed", self._series_title("fan_control_secondary_au", scope="Cohort", kind="bursts"), ["fan", "exploratory"]),
            ("C04N", "thermal_sensation", "#b91c1c", self._series_title("thermal_sensation", scope="Cohort", kind="observations"), ["comfort", "exploratory"]),
            ("C04O", "thermal_comfort", "#0f172a", self._series_title("thermal_comfort", scope="Cohort", kind="observations"), ["comfort", "exploratory"]),
            ("C04P", "thermal_preference", "#2563eb", self._series_title("thermal_preference", scope="Cohort", kind="observations"), ["comfort", "exploratory"]),
            ("C04PA", "visual_sensation", "#0f766e", self._series_title("visual_sensation", scope="Cohort", kind="observations"), ["comfort", "exploratory"]),
            ("C04PB", "color_sensation", "#b45309", self._series_title("color_sensation", scope="Cohort", kind="observations"), ["comfort", "exploratory"]),
            ("C04Q", "room_comfort", "#7c3aed", self._series_title("room_comfort", scope="Cohort", kind="observations"), ["comfort", "exploratory"]),
            ("C04T", "thermal_pleasure", "#ea580c", self._series_title("thermal_pleasure", scope="Cohort", kind="observations"), ["comfort", "exploratory"]),
            ("C04U", "visual_comfort", "#0f766e", self._series_title("visual_comfort", scope="Cohort", kind="observations"), ["comfort", "exploratory"]),
            ("C04UA", "sound_comfort_dbA", "#475569", self._series_title("sound_comfort_dbA", scope="Cohort", kind="observations"), ["comfort", "exploratory"]),
            ("C04V", "air_quality_comfort", "#2563eb", self._series_title("air_quality_comfort", scope="Cohort", kind="observations"), ["comfort", "exploratory"]),
        ]
        specs: list[dict] = []
        for code, column, color, title, tags in channel_specs:
            fig = self._fig_cohort_single_channel_burst(minute, column, color)
            specs.append(
                self._spec(
                    code=code,
                    stem=f"cohort_{column}_bursts",
                    title=title,
                    summary=(
                        "This cohort figure shows a minute-level control state over the shared session timeline; step-like changes indicate control adjustments at particular time points rather than a continuous physiological waveform."
                        if column in {"fan_control_au", "fan_control_secondary_au", "fan_current_A"}
                        else (
                            self._cohort_questionnaire_caption(column)
                            if self._is_sparse_observation_channel(column)
                            else "Each cohort figure is dedicated to a single audited or recorded signal so modality-specific support timing, condition balance, and signal shape remain interpretable before endpoint reduction."
                        )
                    ),
                    fig=fig,
                    tags=tags + ["phase"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section=self._cohort_metric_section(column),
                )
            )
        return specs

    def _load_session_questionnaire_raw(self, session_id: str) -> pd.DataFrame:
        path = self.dataset_root / self.config.dataset.master_dir / self.config.dataset.questionnaire_file
        df = safe_read_csv(path)
        df = df.rename(columns={
            "Session ID": "session_id",
            "q n": "questionnaire_n",
            "thermal sensation": "thermal_sensation",
            "thermal comfort": "thermal_comfort",
            "thermal pleasure": "thermal_pleasure",
            "thermal preference": "thermal_preference",
            "visual comfort": "visual_comfort",
            "sound comfort (dB(A))": "sound_comfort_dbA",
            "airQuality comfort": "air_quality_comfort",
            "room comfort": "room_comfort",
        })
        df = df.loc[df["session_id"].astype(str) == str(session_id)].copy()
        if df.empty:
            return df
        df["datetime"] = parse_local_datetime(df["datetime"], self.config.runtime.timeline_timezone).dt.tz_convert("UTC")
        return df

    def _load_session_fan_raw(self, session_id: str) -> pd.DataFrame:
        path = self.dataset_root / self.config.dataset.master_dir / self.config.dataset.fan_behavior_file
        df = safe_read_csv(path)
        df = df.rename(columns={
            "Session ID": "session_id",
            "fan current (A)": "fan_current_A",
            "fan control (a.u.)": "fan_control_au",
            "fan control (a.u.).1": "fan_control_secondary_au",
        })
        df = df.loc[df["session_id"].astype(str) == str(session_id)].copy()
        if df.empty:
            return df
        df["datetime"] = parse_local_datetime(df["datetime"], self.config.runtime.timeline_timezone).dt.tz_convert("UTC")
        return df

    def _load_session_indoor_raw(self, session_id: str) -> pd.DataFrame:
        path = self.dataset_root / self.config.dataset.env_dir / self.config.dataset.indoor_file
        df = safe_read_csv(path)
        df = df.rename(columns={"Session ID": "session_id"})
        df = df.loc[df["session_id"].astype(str) == str(session_id)].copy()
        if df.empty:
            return df
        df["datetime"] = parse_local_datetime(df["datetime"], self.config.runtime.timeline_timezone).dt.tz_convert("UTC")
        return df

    def _load_session_biopac_raw(self, session_id: str) -> pd.DataFrame:
        path = self.dataset_root / self.config.dataset.biopac_dir / session_id / "biopac.csv"
        if not path.exists():
            return pd.DataFrame()
        df = safe_read_csv(path)
        df["Datetime"] = parse_any_datetime(df["Datetime"], self.config.runtime.timeline_timezone)
        return df

    def _load_session_empatica_raw(self, session_id: str, filename: str) -> pd.DataFrame:
        path = self.dataset_root / self.config.dataset.empatica_dir / session_id / filename
        if not path.exists():
            return pd.DataFrame()
        return safe_read_csv(path)

    def _build_session_source_raw_specs(self, s: dict, ev: dict) -> list[dict]:
        session_id = s["session_id"]
        minute = s["aligned_df"]
        specs: list[dict] = []

        questionnaire = self._load_session_questionnaire_raw(session_id)
        q_fig = self._raw_line_figure(
            minute,
            questionnaire,
            "datetime",
            [
                {"column": "thermal_sensation", "label": "Thermal sensation", "color": "#b91c1c"},
                {"column": "thermal_comfort", "label": "Thermal comfort", "color": "#0f172a"},
                {"column": "thermal_pleasure", "label": "Thermal pleasure", "color": "#ea580c"},
                {"column": "thermal_preference", "label": "Thermal preference", "color": "#2563eb"},
            ],
            ylabel="Ordinal response",
            figsize=self._figsize("wide_single_short"),
            markers_only=True,
            trim_to_support=True,
        )
        specs.append(self._spec(code="S09A", stem=f"{session_id}_questionnaire_raw", title="Questionnaire event responses", summary="Raw questionnaire events are shown at their recorded event times so subjective responses remain discrete observations rather than continuous traces.", fig=q_fig, tags=["comfort", "phase", "exploratory"], evidence_score=ev["score"], evidence_label=ev["label"], gating_note=ev["note"], section="subjective_behavioral"))

        fan = self._load_session_fan_raw(session_id)
        fan_fig = self._raw_line_figure(
            minute,
            fan,
            "datetime",
            [
                {"column": "fan_current_A", "label": "Fan current", "color": "#111827"},
                {"column": "fan_control_au", "label": "Fan control", "color": "#2563eb"},
                {"column": "fan_control_secondary_au", "label": "Secondary fan control", "color": "#7c3aed"},
            ],
            ylabel="Raw fan signal",
            figsize=self._figsize("wide_single_short"),
            step=True,
            trim_to_support=True,
        )
        specs.append(self._spec(code="S10A", stem=f"{session_id}_fan_behavior_raw", title="Fan behavior channels", summary="Fan behavior is shown from the recorded fan telemetry so control changes can be inspected before any aligned summaries are introduced.", fig=fan_fig, tags=["fan", "phase", "exploratory"], evidence_score=ev["score"], evidence_label=ev["label"], gating_note=ev["note"], section="subjective_behavioral"))

        emp_acc = self._load_session_empatica_raw(session_id, "accelerometer.csv")
        emp_acc["datetime"] = pd.to_datetime(emp_acc.get("datetime"), utc=True, errors="coerce")
        specs.append(self._spec(code="S06A", stem=f"{session_id}_empatica_acc_raw", title="Empatica accelerometer", summary="Raw Empatica accelerometer axes are downsampled only for plotting density, preserving the recorded signal shape.", fig=self._raw_line_figure(minute, emp_acc, "datetime", [{"column": "x_g", "label": "X", "color": "#b91c1c"}, {"column": "y_g", "label": "Y", "color": "#2563eb"}, {"column": "z_g", "label": "Z", "color": "#059669"}], ylabel="Acceleration (g)", trim_to_support=True), tags=["physiology", "phase", "exploratory"], evidence_score=ev["score"], evidence_label=ev["label"], gating_note=ev["note"], section="physiological"))

        for code, filename, column, title, color, ylabel, tags in [
            ("S06B", "bvp.csv", "bvp_nW", "Empatica BVP", "#7c3aed", "BVP (nW)", ["physiology", "phase", "exploratory"]),
            ("S06C", "eda.csv", "eda_uS", "Empatica EDA", "#1d4ed8", "EDA (uS)", ["eda", "phase", "exploratory"]),
            ("S06D", "temperature.csv", "temperature_C", "Empatica temperature", "#ea580c", "Temperature (C)", ["temperature", "phase", "exploratory"]),
            ("S06E", "steps.csv", "steps", "Empatica steps", "#0f766e", "Steps", ["physiology", "phase", "exploratory"]),
        ]:
            df = self._load_session_empatica_raw(session_id, filename)
            if not df.empty:
                df["datetime"] = pd.to_datetime(df.get("datetime"), utc=True, errors="coerce")
            fig = self._raw_line_figure(minute, df, "datetime", [{"column": column, "label": title, "color": color}], ylabel=ylabel, step=(column == "steps"), trim_to_support=True)
            specs.append(self._spec(code=code, stem=f"{session_id}_{column}_source_raw", title=title, summary="This raw Empatica channel is shown directly from the recorded source file with plotting-only downsampling.", fig=fig, tags=tags, evidence_score=ev["score"], evidence_label=ev["label"], gating_note=ev["note"], section="physiological"))

        peaks = self._load_session_empatica_raw(session_id, "systolic_peaks.csv")
        if not peaks.empty:
            peaks["datetime"] = pd.to_datetime(peaks.get("datetime"), utc=True, errors="coerce")
        specs.append(self._spec(code="S06F", stem=f"{session_id}_empatica_peaks_raw", title="Empatica systolic peaks", summary="Systolic peaks are shown as recorded peak events rather than being converted into derived heart-rate summaries.", fig=self._raw_peak_raster(minute, peaks), tags=["heart_rate", "phase", "exploratory"], evidence_score=ev["score"], evidence_label=ev["label"], gating_note=ev["note"], section="physiological"))

        segments = self._load_session_empatica_raw(session_id, "segments_used.csv")
        specs.append(self._spec(code="S06G", stem=f"{session_id}_empatica_segments_raw", title="Empatica segment windows", summary="Empatica segment windows show which recorded source segments were available for the session, without collapsing them into minute-level support.", fig=self._raw_segment_windows(minute, segments), tags=["physiology", "phase", "support"], evidence_score=ev["score"], evidence_label=ev["label"], gating_note=ev["note"], section="physiological"))

        biopac = self._load_session_biopac_raw(session_id)
        specs.append(self._spec(code="S07A", stem=f"{session_id}_biopac_hr_raw", title="BIOPAC heart rate", summary="Raw BIOPAC heart rate is shown from the recorded high-frequency stream with plotting-only downsampling.", fig=self._raw_line_figure(minute, biopac, "Datetime", [{"column": "HR [BPM]", "label": "HR", "color": "#111827"}], ylabel="Heart rate (BPM)", trim_to_support=True), tags=["heart_rate", "phase", "exploratory"], evidence_score=ev["score"], evidence_label=ev["label"], gating_note=ev["note"], section="physiological"))
        specs.append(self._spec(code="S07B", stem=f"{session_id}_biopac_eda_raw", title="BIOPAC EDA", summary="Raw BIOPAC EDA is shown from the recorded high-frequency stream with plotting-only downsampling.", fig=self._raw_line_figure(minute, biopac, "Datetime", [{"column": "EDA [microsiemens]", "label": "EDA", "color": "#2563eb"}], ylabel="EDA (microsiemens)", trim_to_support=True), tags=["eda", "phase", "exploratory"], evidence_score=ev["score"], evidence_label=ev["label"], gating_note=ev["note"], section="physiological"))
        specs.append(self._spec(code="S07C", stem=f"{session_id}_biopac_temp_raw", title="BIOPAC temperature channels", summary="Raw BIOPAC temperature channels are shown as recorded so site-specific thermal patterns can be inspected before summarization.", fig=self._raw_line_figure(minute, biopac, "Datetime", [{"column": "Chest [deg. C]", "label": "Chest", "color": "#ea580c"}, {"column": "Thigh [deg. C]", "label": "Thigh", "color": "#f59e0b"}, {"column": "UpperArm [deg. C]", "label": "Upper arm", "color": "#dc2626"}, {"column": "Tibia [deg. C]", "label": "Tibia", "color": "#7c3aed"}], ylabel="Temperature (C)", trim_to_support=True), tags=["temperature", "phase", "exploratory"], evidence_score=ev["score"], evidence_label=ev["label"], gating_note=ev["note"], section="physiological"))
        specs.append(self._spec(code="S07D", stem=f"{session_id}_biopac_flow_ppg_raw", title="BIOPAC perfusion and optical channels", summary="Blood flow, PPG, and backscatter are shown from the source BIOPAC stream to preserve the recorded perfusion context.", fig=self._raw_line_figure(minute, biopac, "Datetime", [{"column": "BloodFlow [BPU]", "label": "Blood flow", "color": "#7c3aed"}, {"column": "PPG [Volts]", "label": "PPG", "color": "#0f766e"}, {"column": "Backscatter [%]", "label": "Backscatter", "color": "#64748b"}], ylabel="Raw BIOPAC signal", trim_to_support=True), tags=["physiology", "phase", "exploratory"], evidence_score=ev["score"], evidence_label=ev["label"], gating_note=ev["note"], section="physiological"))

        indoor = self._load_session_indoor_raw(session_id)
        temp_cols = [c for c in indoor.columns if "air temperature" in str(c).lower()]
        vel_cols = [c for c in indoor.columns if "air velocity" in str(c).lower()]
        specs.append(self._spec(code="S08A", stem=f"{session_id}_indoor_temperature_raw", title="Indoor air-temperature probes", summary="Indoor air-temperature probes are shown as recorded to preserve the spatial spread of the environmental sensors.", fig=self._raw_line_figure(minute, indoor, "datetime", [{"column": col, "label": f"T{i+1}", "color": plt.cm.Oranges(0.3 + 0.6 * (i / max(len(temp_cols), 1)))} for i, col in enumerate(temp_cols[:6])], ylabel="Temperature (C)", trim_to_support=True), tags=["environment", "phase", "exploratory"], evidence_score=ev["score"], evidence_label=ev["label"], gating_note=ev["note"], section="environmental"))
        specs.append(self._spec(code="S08B", stem=f"{session_id}_indoor_velocity_raw", title="Indoor air-velocity probes", summary="Indoor air-velocity probes are shown directly from the recorded environmental file without further aggregation.", fig=self._raw_line_figure(minute, indoor, "datetime", [{"column": col, "label": f"V{i+1}", "color": plt.cm.Greens(0.3 + 0.6 * (i / max(len(vel_cols), 1)))} for i, col in enumerate(vel_cols[:6])], ylabel="Velocity (m/s)", trim_to_support=True), tags=["environment", "phase", "exploratory"], evidence_score=ev["score"], evidence_label=ev["label"], gating_note=ev["note"], section="environmental"))
        ambient_cols = [
            ("relative humidity  1034 (%RH)", "Relative humidity", "#2563eb"),
            ("illuminance  1035 (lux)", "Illuminance", "#f59e0b"),
            ("sound level  1037 (dB(A))", "Sound level", "#64748b"),
            ("CO2 (ppm)", "CO2", "#0f766e"),
            ("PMV fanger", "PMV", "#b91c1c"),
        ]
        specs.append(self._spec(code="S08C", stem=f"{session_id}_indoor_ambient_raw", title="Indoor ambient channels", summary="Humidity, light, sound, CO2, and PMV are shown from the recorded indoor environmental stream to preserve the native ambient context.", fig=self._raw_line_figure(minute, indoor, "datetime", [{"column": col, "label": label, "color": color} for col, label, color in ambient_cols if col in indoor.columns], ylabel="Raw ambient value", trim_to_support=True), tags=["environment", "phase", "exploratory"], evidence_score=ev["score"], evidence_label=ev["label"], gating_note=ev["note"], section="environmental"))

        return [spec for spec in specs if spec["fig"] is not None]

    def _build_session_processed_signal_specs(self, s: dict, ev: dict) -> list[dict]:
        session_id = s["session_id"]
        minute = s["aligned_df"].copy().sort_values("minute_index")
        specs: list[dict] = []

        processed_specs = [
            (
                "S16A",
                f"{session_id}_questionnaire_processed",
                "Processed questionnaire responses",
                "Questionnaire responses are shown from the processed session table after event parsing and minute-level placement, while keeping the responses as discrete observations.",
                [
                    {"column": "thermal_sensation", "label": "Thermal sensation", "color": "#b91c1c"},
                    {"column": "thermal_comfort", "label": "Thermal comfort", "color": "#0f172a"},
                    {"column": "thermal_preference", "label": "Thermal preference", "color": "#2563eb"},
                    {"column": "room_comfort", "label": "Room comfort", "color": "#7c3aed"},
                ],
                ["comfort", "phase", "preprocessing"],
            ),
            (
                "S16B",
                f"{session_id}_fan_processed",
                "Processed fan signals",
                "Fan channels are shown after processing and minute-level retention so retained control changes can be compared on the common session timeline.",
                [
                    {"column": "fan_current_A", "label": "Fan current", "color": "#111827"},
                    {"column": "fan_control_au", "label": "Fan control", "color": "#2563eb"},
                    {"column": "fan_control_secondary_au", "label": "Secondary fan control", "color": "#7c3aed"},
                ],
                ["fan", "phase", "preprocessing"],
            ),
            (
                "S16C",
                f"{session_id}_bvp_processed",
                "Processed blood-volume-pulse signals",
                "Empatica BVP summaries are shown from the cleaned minute-level table so the retained pulse waveform intensity can be inspected after aggregation.",
                [
                    {"column": "empatica_bvp_mean", "label": "Empatica BVP mean", "color": "#7c3aed"},
                    {"column": "empatica_bvp_sd", "label": "Empatica BVP SD", "color": "#a855f7"},
                ],
                ["physiology", "phase", "preprocessing"],
            ),
            (
                "S16D",
                f"{session_id}_activity_processed",
                "Processed activity and motion signals",
                "Minute-level Empatica motion summaries are shown after processing so retained movement intensity and step counts can be checked on the shared session timeline.",
                [
                    {"column": "empatica_acc_mean_g", "label": "Acceleration magnitude", "color": "#b91c1c"},
                    {"column": "empatica_enmo_mean_g", "label": "ENMO", "color": "#2563eb"},
                    {"column": "empatica_steps", "label": "Steps", "color": "#0f766e"},
                ],
                ["physiology", "phase", "preprocessing"],
            ),
            (
                "S16E",
                f"{session_id}_heart_rate_processed",
                "Processed heart-rate signals",
                "Empatica and BIOPAC heart-rate signals are shown after cleaning and minute-level alignment so retained support and delayed starts are visible before agreement analysis.",
                [
                    {"column": "empatica_hr_mean_bpm", "label": "Empatica HR", "color": "#b91c1c"},
                    {"column": "biopac_hr_mean_bpm", "label": "BIOPAC HR", "color": "#111827"},
                ],
                ["heart_rate", "phase", "preprocessing"],
            ),
            (
                "S16F",
                f"{session_id}_eda_processed",
                "Processed electrodermal signals",
                "Empatica and BIOPAC EDA are shown from the cleaned minute-level table so retained overlap and modality-specific dropout are visible directly.",
                [
                    {"column": "empatica_eda_mean_uS", "label": "Empatica EDA", "color": "#1d4ed8"},
                    {"column": "biopac_eda_mean_uS", "label": "BIOPAC EDA", "color": "#2563eb"},
                ],
                ["eda", "phase", "preprocessing"],
            ),
            (
                "S16G",
                f"{session_id}_skin_temperature_processed",
                "Processed skin-temperature signals",
                "Empatica and BIOPAC temperature channels are shown after cleaning and minute-level aggregation so retained site-specific thermal patterns can be compared on the common session index.",
                [
                    {"column": "empatica_temp_mean_C", "label": "Empatica temperature", "color": "#ea580c"},
                    {"column": "biopac_temp_chest_mean_C", "label": "Chest temperature", "color": "#dc2626"},
                    {"column": "biopac_temp_thigh_mean_C", "label": "Thigh temperature", "color": "#f59e0b"},
                    {"column": "biopac_temp_arm_mean_C", "label": "Arm temperature", "color": "#fb7185"},
                    {"column": "biopac_temp_tibia_mean_C", "label": "Tibia temperature", "color": "#7c3aed"},
                ],
                ["temperature", "phase", "preprocessing"],
            ),
            (
                "S16H",
                f"{session_id}_temperature_perfusion_processed",
                "Processed perfusion and optical signals",
                "Perfusion and optical channels are shown after cleaning and minute-level aggregation so retained circulatory patterns can be inspected before agreement and phase summaries.",
                [
                    {"column": "biopac_bloodflow_mean_bpu", "label": "Blood flow", "color": "#7c3aed"},
                    {"column": "biopac_backscatter_mean_percent", "label": "Backscatter", "color": "#64748b"},
                ],
                ["bloodflow", "physiology", "phase", "preprocessing"],
            ),
            (
                "S16I",
                f"{session_id}_environment_processed",
                "Processed indoor thermal and airflow signals",
                "Indoor temperature, air velocity, and humidity are shown from the processed session table so the retained environmental context can be checked before phase summaries are read.",
                [
                    {"column": "indoor_air_temp_mean_C", "label": "Indoor air temperature", "color": "#ea580c"},
                    {"column": "indoor_air_velocity_mean_m_s", "label": "Air velocity", "color": "#0f766e"},
                    {"column": "indoor_relative_humidity_percent", "label": "Relative humidity", "color": "#2563eb"},
                ],
                ["environment", "phase", "preprocessing"],
            ),
            (
                "S16J",
                f"{session_id}_indoor_ambient_processed",
                "Processed indoor ambient signals",
                "Indoor ambient channels are shown from the processed minute-level table so retained light, sound, air-quality, and comfort context can be inspected before higher-level summaries.",
                [
                    {"column": "indoor_illuminance_lux", "label": "Illuminance", "color": "#f59e0b"},
                    {"column": "indoor_sound_dbA", "label": "Sound level", "color": "#64748b"},
                    {"column": "indoor_co2_ppm", "label": "CO2", "color": "#0f766e"},
                    {"column": "indoor_pmv_fanger", "label": "PMV", "color": "#b91c1c"},
                ],
                ["environment", "phase", "preprocessing"],
            ),
            (
                "S16K",
                f"{session_id}_outdoor_processed",
                "Processed outdoor context signals",
                "Outdoor temperature, humidity, wind, and solar context are shown after nearest-minute matching so the retained external conditions are visible alongside the session timeline.",
                [
                    {"column": "outdoor_air_temp_C", "label": "Outdoor temperature", "color": "#ea580c"},
                    {"column": "outdoor_relative_humidity_percent", "label": "Outdoor humidity", "color": "#2563eb"},
                    {"column": "outdoor_wind_speed_m_s", "label": "Wind speed", "color": "#0f766e"},
                    {"column": "outdoor_solar_radiation_W_m2", "label": "Solar radiation", "color": "#f59e0b"},
                ],
                ["environment", "phase", "preprocessing"],
            ),
        ]

        for code, stem, title, summary, trace_specs, tags in processed_specs:
            fig = self._matplotlib_protocol_trace(
                minute,
                trace_specs,
                title,
                overlay=(code == "S16G"),
            )
            if fig is None:
                continue
            specs.append(
                self._spec(
                    code=code,
                    stem=stem,
                    title=title,
                    summary=summary,
                    fig=fig,
                    tags=tags,
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="processed_cleaned",
                )
            )
        return specs

    def _build_session_raw_channel_specs(self, session_id: str, minute: pd.DataFrame, ev: dict) -> list[dict]:
        channel_specs = [
            ("S09A", "thermal_sensation", "#b91c1c", "Raw thermal-sensation observations", ["comfort", "exploratory"], "subjective_behavioral"),
            ("S09B", "thermal_comfort", "#0f172a", "Raw thermal-comfort observations", ["comfort", "exploratory"], "subjective_behavioral"),
            ("S09C", "thermal_preference", "#2563eb", "Raw thermal-preference observations", ["comfort", "exploratory"], "subjective_behavioral"),
            ("S09D", "room_comfort", "#7c3aed", "Raw room-comfort observations", ["comfort", "exploratory"], "subjective_behavioral"),
            ("S10A", "fan_current_A", "#111827", "Raw fan-current trajectory", ["fan", "exploratory"], "subjective_behavioral"),
            ("S10B", "fan_control_au", "#2563eb", "Raw fan-control trajectory", ["fan", "exploratory"], "subjective_behavioral"),
            ("S10C", "fan_control_secondary_au", "#7c3aed", "Raw secondary-fan-control trajectory", ["fan", "exploratory"], "subjective_behavioral"),
            ("S06A", "empatica_hr_mean_bpm", "#b91c1c", "Raw Empatica HR trajectory", ["heart_rate", "exploratory"], "physiological"),
            ("S06B", "empatica_eda_mean_uS", "#1d4ed8", "Raw Empatica EDA trajectory", ["eda", "exploratory"], "physiological"),
            ("S06C", "empatica_temp_mean_C", "#ea580c", "Raw Empatica temperature trajectory", ["temperature", "exploratory"], "physiological"),
            ("S07A", "biopac_hr_mean_bpm", "#111827", "Raw BIOPAC HR trajectory", ["heart_rate", "exploratory"], "physiological"),
            ("S07B", "biopac_eda_mean_uS", "#2563eb", "Raw BIOPAC EDA trajectory", ["eda", "exploratory"], "physiological"),
            ("S07C", "biopac_temp_chest_mean_C", "#ea580c", "Raw chest-temperature trajectory", ["temperature", "exploratory"], "physiological"),
            ("S07D", "biopac_bloodflow_mean_bpu", "#7c3aed", "Raw blood-flow trajectory", ["bloodflow", "exploratory"], "physiological"),
            ("S08A", "indoor_air_temp_mean_C", "#ea580c", "Raw indoor-air-temperature trajectory", ["environment", "exploratory"], "environmental"),
            ("S08B", "indoor_air_velocity_mean_m_s", "#0f766e", "Raw air-velocity trajectory", ["environment", "exploratory"], "environmental"),
            ("S08C", "indoor_relative_humidity_percent", "#2563eb", "Raw indoor-relative-humidity trajectory", ["environment", "exploratory"], "environmental"),
        ]
        specs: list[dict] = []
        for code, column, color, title, tags, section in channel_specs:
            fig = self._fig_session_single_channel_raw(minute, column, color)
            specs.append(
                self._spec(
                    code=code,
                    stem=f"{session_id}_{column}_raw",
                    title=title,
                    summary=(
                        "This panel shows a minute-level control state over the session timeline; step-like changes indicate control adjustments at particular time points rather than a continuous physiological waveform."
                        if column in {"fan_current_A", "fan_control_au", "fan_control_secondary_au"}
                        else "Each modality is shown in its own panel with a renderer matched to its data structure, so sparse questionnaire observations are not misrepresented as continuous signals."
                    ),
                    fig=fig,
                    tags=tags + ["phase"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section=section,
                )
            )
        return specs

    def _fig_session_readiness(self, minute: pd.DataFrame, meta: dict, ev: dict):
        fig, axes = plt.subplots(2, 2, figsize=self._figsize("readiness_grid"))
        comparison_minute = minute.loc[minute["protocol_block"].astype(str).isin(COMPARISON_BLOCKS)].copy()
        fig._cltr_panel_notes = [
            "Top left shows the session snapshot cards for confidence, questionnaire completeness, and overlap minutes within Blocks 1 to 3.",
            "Top right shows comparison-window coverage by source.",
            "Bottom left shows minutes captured by block and phase within Blocks 1 to 3.",
            "Bottom right shows overlapping minutes across paired devices within Blocks 1 to 3.",
        ]
        cards_ax = axes[0, 0]
        cards_ax.axis("off")
        cards = [
            f"Evidence\n{ev['label'].upper()} ({ev['score']})",
            f"Questionnaire completeness\n{meta['support']['questionnaire_completeness']:.1%}",
            f"HR overlap\n{meta['sensor_overlap_minutes']['heart_rate']} min",
            f"EDA overlap\n{meta['sensor_overlap_minutes']['eda']} min",
        ]
        for i, txt in enumerate(cards):
            x = 0.05 + (i % 2) * 0.47
            y = 0.78 - (i // 2) * 0.42
            cards_ax.text(x, y, txt, transform=cards_ax.transAxes, va="center", ha="left", fontsize=12, fontweight="bold", bbox={"boxstyle": "round,pad=0.6", "fc": "#eff6ff", "ec": "#bfdbfe"})
        cov_ax = axes[0, 1]
        support = meta["support"]
        cov_ax.barh(
            ["Questionnaire", "Empatica", "BIOPAC", "Indoor", "Outdoor"],
            [support["questionnaire_completeness"], support["empatica_fraction"], support["biopac_fraction"], support["indoor_fraction"], support["outdoor_fraction"]],
            color=["#0f172a", "#2563eb", "#dc2626", "#059669", "#7c3aed"],
        )
        cov_ax.set_xlim(0, 1)
        phase_ax = axes[1, 0]
        block_phase = comparison_minute.groupby(["protocol_block", "protocol_phase"]).size().reset_index(name="n_minutes")
        if not block_phase.empty:
            def _block_sort_key(value: object) -> tuple[int, float | str]:
                numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                return (0, float(numeric)) if pd.notna(numeric) else (1, str(value))

            block_order = sorted(block_phase["protocol_block"].astype(str).unique(), key=_block_sort_key)
            pivot = block_phase.assign(protocol_block=block_phase["protocol_block"].astype(str)).pivot(index="protocol_block", columns="protocol_phase", values="n_minutes").reindex(index=block_order, columns=[p for p in PHASE_ORDER if p in block_phase["protocol_phase"].astype(str).unique()]).fillna(0)
            im = phase_ax.imshow(pivot.values, aspect="equal", cmap="Blues")
            phase_ax.grid(False)
            phase_ax.set_yticks(range(len(pivot.index)))
            phase_ax.set_yticklabels([f"B{x}" for x in pivot.index])
            phase_ax.set_xticks(range(len(pivot.columns)))
            phase_ax.set_xticklabels([PHASE_ABBR.get(x, x[:3].upper()) for x in pivot.columns])
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    if pivot.values[i, j] > 0:
                        phase_ax.text(j, i, int(pivot.values[i, j]), ha="center", va="center", fontsize=10, color="#172033")
            plt.colorbar(im, ax=phase_ax, shrink=0.75, label="Minutes")
        else:
            phase_ax.axis("off")
        overlap_ax = axes[1, 1]
        overlap_labels = ["Heart rate", "EDA", "Temperature"]
        overlap_values = [
            meta["sensor_overlap_minutes"]["heart_rate"],
            meta["sensor_overlap_minutes"]["eda"],
            meta["sensor_overlap_minutes"]["temperature"],
        ]
        bars = overlap_ax.bar(overlap_labels, overlap_values, color=["#b91c1c", "#1d4ed8", "#ea580c"])
        overlap_ax.axhline(self.config.runtime.min_sensor_overlap_minutes, color="#cbd5e1", lw=0.9, ls="--", zorder=0, label="Eligibility threshold")
        overlap_ax.legend(frameon=False, loc="upper left")
        overlap_ax.set_ylabel("Minutes")
        overlap_ax.tick_params(axis="x", labelrotation=0)
        ymax = max(overlap_values + [self.config.runtime.min_sensor_overlap_minutes, 1])
        overlap_ax.set_ylim(0, ymax * 1.18)
        for bar, value in zip(bars, overlap_values):
            overlap_ax.text(bar.get_x() + bar.get_width() / 2.0, value + ymax * 0.03, f"{int(value)}", ha="center", va="bottom", fontsize=11, color="#172033")
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        return fig

    def _fig_session_preprocessing_burden(self, minute: pd.DataFrame, meta: dict):
        if minute.empty:
            return None
        fig, axes = plt.subplots(1, 3, figsize=self._figsize("three_panel_row_wide"))
        fig._cltr_panel_notes = [
            "Left|Retention by source|Minute-level retention by source.",
            "Center|Overlap burden|Overlap burden as a share of the session.",
            "Right|Phase-wise coverage|Phase-wise modality coverage across Blocks 1 to 3.",
        ]
        support = meta.get("support", {})
        session_len = max(int(meta.get("n_minutes_comparison_window", 0)), 1)
        comparison_minute = minute.loc[minute["protocol_block"].astype(str).isin(COMPARISON_BLOCKS)].copy()
        support_rows = [
            ("Questionnaire", float(support.get("questionnaire_completeness", 0.0)), "#0f172a"),
            ("Empatica", float(support.get("empatica_fraction", 0.0)), "#2563eb"),
            ("BIOPAC", float(support.get("biopac_fraction", 0.0)), "#dc2626"),
            ("Indoor", float(support.get("indoor_fraction", 0.0)), "#059669"),
            ("Outdoor", float(support.get("outdoor_fraction", 0.0)), "#7c3aed"),
        ]
        axes[0].barh([x[0] for x in support_rows][::-1], [x[1] for x in support_rows][::-1], color=[x[2] for x in support_rows][::-1])
        axes[0].set_xlim(0, 1)
        overlap_rows = [
            ("HR overlap", float(meta["sensor_overlap_minutes"]["heart_rate"]) / session_len, "#b91c1c"),
            ("EDA overlap", float(meta["sensor_overlap_minutes"]["eda"]) / session_len, "#1d4ed8"),
            ("Temp overlap", float(meta["sensor_overlap_minutes"]["temperature"]) / session_len, "#ea580c"),
        ]
        axes[1].barh([x[0] for x in overlap_rows][::-1], [x[1] for x in overlap_rows][::-1], color=[x[2] for x in overlap_rows][::-1])
        axes[1].set_xlim(0, 1)
        phase_map = [
            ("Questionnaire", "support_questionnaire"),
            ("Empatica", "support_empatica"),
            ("BIOPAC", "support_biopac"),
            ("Indoor", "support_indoor"),
            ("Outdoor", "support_outdoor"),
        ]
        if not comparison_minute.empty and "protocol_phase" in comparison_minute.columns:
            rows = []
            phase_order = self._comparison_phase_sequence(comparison_minute["protocol_phase"].astype(str).unique())
            for label, col in phase_map:
                if col not in comparison_minute.columns:
                    continue
                for phase_name in phase_order:
                    cur = comparison_minute.loc[comparison_minute["protocol_phase"].astype(str) == phase_name]
                    if cur.empty:
                        continue
                    vals = to_numeric(cur[col]).fillna(0)
                    rows.append(
                        {
                            "label": label,
                            "phase": phase_name,
                            "coverage": float(vals.mean()),
                        }
                    )
            coverage = pd.DataFrame(rows)
            if not coverage.empty and phase_order:
                pivot = (
                    coverage.pivot(index="label", columns="phase", values="coverage")
                    .reindex(index=[label for label, _ in phase_map if label in coverage["label"].unique()])
                    .reindex(columns=phase_order)
                    .fillna(0.0)
                )
                im = axes[2].imshow(pivot.values, aspect="equal", cmap="Blues", vmin=0, vmax=1)
                axes[2].grid(False)
                axes[2].set_xticks(range(len(pivot.columns)))
                axes[2].set_xticklabels([PHASE_ABBR.get(x, x[:3].upper()) for x in pivot.columns], fontsize=10)
                axes[2].set_yticks(range(len(pivot.index)))
                axes[2].set_yticklabels(list(pivot.index))
                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        value = float(pivot.iloc[i, j])
                        if value > 0:
                            axes[2].text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=10, color="#172033")
                plt.colorbar(im, ax=axes[2], shrink=0.82, label="Coverage")
            else:
                axes[2].axis("off")
        else:
            axes[2].axis("off")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_session_sync_audit(self, minute: pd.DataFrame, meta: dict):
        comparison_minute = minute.loc[minute["protocol_block"].astype(str).isin(COMPARISON_BLOCKS)].copy()
        modality_mapping = [
            ("Fan", "support_fan", "#2563eb"),
            ("Empatica", "support_empatica", "#2563eb"),
            ("BIOPAC", "support_biopac", "#dc2626"),
            ("Indoor", "support_indoor", "#059669"),
            ("Outdoor", "support_outdoor", "#7c3aed"),
        ]
        overlap_mapping = [
            ("HR overlap", "support_core_overlap_hr", "#7c3aed"),
            ("EDA overlap", "support_core_overlap_eda", "#1d4ed8"),
            ("Temp overlap", "support_core_overlap_temp", "#ea580c"),
        ]
        left = self._support_segment_rows(comparison_minute, modality_mapping)
        right = self._support_segment_rows(comparison_minute, overlap_mapping)
        if left.empty and right.empty:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9), gridspec_kw={"width_ratios": [1.2, 0.8]})
        fig._cltr_panel_notes = [
            "Left|Support segments|Contiguous support segments by modality across Blocks 1 to 3.",
            "Right|Paired-device overlap segments|Contiguous overlap segments between paired devices across Blocks 1 to 3.",
        ]
        for ax, data in [(axes[0], left), (axes[1], right)]:
            if data.empty:
                ax.axis("off")
                continue
            for idx, row in enumerate(data.itertuples()):
                ax.barh(idx, row.end_minute - row.start_minute + 1, left=row.start_minute, color=row.color, alpha=0.85)
                ax.text(row.end_minute + 1.5, idx, f"{row.support_fraction:.0%}", va="center", fontsize=10, color="#475569")
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels(list(data["segment_label"]))
            ax.set_xlabel("Timeline minute")
            ax.set_xlim(float(to_numeric(comparison_minute["minute_index"]).min()), float(to_numeric(comparison_minute["minute_index"]).max()) + 6)
            for _, end, phase_name in self._phase_segments(comparison_minute):
                ax.axvline(end, color="#dbe4ee", lw=0.8, zorder=0)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_session_endpoints(self, phase: pd.DataFrame):
        if phase.empty:
            return None
        metrics = [m for m in ["thermal_comfort", "master_dpg_C", "indoor_air_velocity_mean_m_s", "biopac_temp_chest_mean_C", "empatica_hr_mean_bpm"] if m in phase.columns]
        if not metrics:
            return None
        summaries = []
        ylabels = []
        order = self._comparison_phase_sequence(phase["protocol_phase"].astype(str).unique())
        for metric in metrics:
            summary, baseline_info = self._phase_baseline_delta_summary(phase, metric, exclude_acclimation=True)
            if summary.empty:
                continue
            summary = summary.set_index("protocol_phase").reindex(order)
            summaries.append(summary["delta"].to_numpy())
            suffix = f" vs {PHASE_ABBR.get(baseline_info['phase'], baseline_info['phase'][:3].upper())}" if baseline_info else " raw mean"
            ylabels.append(FEATURE_LABELS.get(metric, metric) + suffix)
        if not summaries:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), gridspec_kw={"width_ratios": [1.3, 0.9]})
        fig._cltr_panel_notes = [
            "Left shows phase-level departures from each metric's support-aware baseline.",
            "Right shows repeat consistency across blocks for the same metrics.",
        ]
        ax = axes[0]
        im = ax.imshow(np.array(summaries), aspect="equal", cmap="coolwarm")
        ax.grid(False)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([PHASE_ABBR.get(p, p[:3].upper()) for p in order])
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Delta vs support-aware baseline")
        cons_ax = axes[1]
        consistency_rows = []
        for metric in metrics:
            c = self._phase_repeat_consistency(phase, metric)
            if c["dominant_phase"] is None:
                continue
            consistency_rows.append((FEATURE_LABELS.get(metric, metric), c["consistency"], c["n_blocks"], c["dominant_phase"], c["dominant_direction"]))
        if consistency_rows:
            labels = [r[0] for r in consistency_rows]
            vals = [r[1] for r in consistency_rows]
            cons_ax.barh(labels, vals, color="#2563eb")
            cons_ax.set_xlim(0, 1)
            for idx, row in enumerate(consistency_rows):
                cons_ax.text(min(row[1] + 0.02, 0.98), idx, f"{PHASE_ABBR.get(row[3], row[3][:3].upper())} | {row[4]} | n={row[2]}", va="center", fontsize=10, color="#172033")
        else:
            cons_ax.axis("off")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_session_hr_trace(self, minute: pd.DataFrame):
        footer = self._support_note(minute, ["empatica_hr_mean_bpm", "biopac_hr_mean_bpm"])
        return self._plotly_protocol_trace(
            minute,
            [
                {"column": "empatica_hr_mean_bpm", "label": "Empatica HR", "color": "#b91c1c"},
                {"column": "biopac_hr_mean_bpm", "label": "BIOPAC HR", "color": "#111827", "dash": "dash"},
            ],
            "Protocol-aligned heart-rate traces",
            "Heart rate (BPM)",
            footer=footer or "BIOPAC HR starts later than Empatica in this dataset; gaps represent missing support rather than zero physiological response.",
        )

    def _fig_session_eda_trace(self, minute: pd.DataFrame):
        footer = self._support_note(minute, ["empatica_eda_mean_uS", "biopac_eda_mean_uS"])
        return self._plotly_protocol_trace(
            minute,
            [
                {"column": "empatica_eda_mean_uS", "label": "Empatica EDA", "color": "#1d4ed8"},
                {"column": "biopac_eda_mean_uS", "label": "BIOPAC EDA", "color": "#111827", "dash": "dash"},
            ],
            "Protocol-aligned EDA traces",
            "EDA (uS)",
            footer=footer or "Empatica spans the full session; BIOPAC EDA begins later. Plot gaps reflect acquisition support.",
        )

    def _fig_session_temp_trace(self, minute: pd.DataFrame):
        footer = self._support_note(minute, ["empatica_temp_mean_C", "biopac_temp_chest_mean_C", "biopac_bloodflow_mean_bpu"])
        return self._plotly_protocol_trace(
            minute,
            [
                {"column": "empatica_temp_mean_C", "label": "Empatica Temperature", "color": "#ea580c"},
                {"column": "biopac_temp_chest_mean_C", "label": "Chest Temperature", "color": "#111827", "dash": "dash"},
                {"column": "biopac_bloodflow_mean_bpu", "label": "Blood Flow", "color": "#7c3aed"},
            ],
            "Protocol-aligned thermal and perfusion traces",
            "Signal value",
            footer=footer or "Chest temperature and blood-flow start after acclimation in many sessions; interpretation should follow support windows, not a forced shared baseline.",
        )

    def _fig_session_perception(self, minute: pd.DataFrame):
        return self._plotly_protocol_trace(
            minute,
            [
                {"column": "indoor_air_velocity_mean_m_s", "label": "Air Velocity", "color": "#0f766e"},
                {"column": "fan_control_au", "label": "Fan Control", "color": "#2563eb"},
                {"column": "thermal_comfort", "label": "Thermal Comfort", "color": "#0f172a"},
            ],
            "Perception, fan behavior, and environmental forcing",
            "Signal value",
            footer="Thermal comfort is sparse questionnaire data; fan and indoor channels are minute-level support. Keep those support densities in mind when comparing curves.",
        )

    def _fig_session_focus_distribution(self, s: dict):
        metric = self._story_focus_metric(s)
        phase = s["phase_df"]
        baseline = self._phase_metric_baseline(phase, metric)
        footer = f"Distribution is shown separately from the protocol trace. Baseline reference is {self._baseline_phase_text(baseline)}." if baseline else "Distribution is shown separately from the protocol trace."
        baseline_note = self._baseline_note(baseline)
        footer = " ".join(part for part in [footer, baseline_note] if part)
        return self._plotly_phase_distribution(phase, metric, f"Lead-metric distribution by phase and block: {FEATURE_LABELS.get(metric, metric)}", footer=footer)

    def _fig_session_phase_deltas(self, phase: pd.DataFrame):
        if phase.empty or "protocol_phase" not in phase.columns:
            return None
        metrics = [m for m in ["thermal_comfort", "empatica_hr_mean_bpm", "biopac_temp_chest_mean_C", "indoor_air_velocity_mean_m_s"] if m in phase.columns]
        if not metrics:
            return None
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2))
        axes = axes.ravel()
        baseline_notes: list[str] = []
        panel_notes: list[str] = []
        panel_positions = ["Top left", "Top right", "Bottom left", "Bottom right"]
        for ax, metric in zip(axes, metrics):
            block_deltas, baseline_info = self._block_phase_deltas(phase, metric, exclude_acclimation=True)
            if block_deltas.empty:
                ax.axis("off")
                continue
            ax.axhline(0, color="#dbe4ee", lw=0.85, ls="--", zorder=0)
            colors = ["#2563eb" if val >= 0 else "#dc2626" for val in block_deltas["delta"]]
            ax.bar(range(len(block_deltas)), block_deltas["delta"], color=colors)
            ax.set_xticks(range(len(block_deltas)))
            ax.set_xticklabels(block_deltas["block_phase"], rotation=45, ha="right")
            suffix = f" vs {self._baseline_phase_abbr(baseline_info)}" if baseline_info else " raw mean"
            ax.set_ylabel(FEATURE_LABELS.get(metric, metric))
            panel_notes.append(f"{panel_positions[len(panel_notes)]} shows {FEATURE_LABELS.get(metric, metric)} relative to {suffix}.")
            self._apply_discrete_y_axis_matplotlib(ax, block_deltas["delta"], metric)
            consistency = self._phase_repeat_consistency(phase, metric)
            if consistency["dominant_phase"] is not None and consistency["n_blocks"] >= BLOCK_PHASE_NARRATIVE_THRESHOLD:
                ax.text(0.0, 1.02, f"dominant repeat: {PHASE_ABBR.get(consistency['dominant_phase'], consistency['dominant_phase'][:3].upper())} {consistency['dominant_direction']} | consistency={consistency['consistency']:.2f}", transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color="#64748b")
            note = self._baseline_note(baseline_info)
            if note and note not in baseline_notes:
                baseline_notes.append(note)
        for ax in axes[len(metrics):]:
            ax.axis("off")
        fig._cltr_panel_notes = panel_notes
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_session_agreement(self, minute: pd.DataFrame, meta: dict):
        pairs = [
            ("heart_rate", "empatica_hr_mean_bpm", "biopac_hr_mean_bpm"),
            ("eda", "empatica_eda_mean_uS", "biopac_eda_mean_uS"),
            ("temperature", "empatica_temp_mean_C", "biopac_temp_chest_mean_C"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=self._figsize("three_panel_row"))
        panel_notes: list[str] = []
        panel_positions = ["Left", "Center", "Right"]
        for ax, (metric, left, right) in zip(axes, pairs):
            pair = minute[[left, right]].apply(to_numeric).dropna()
            n_overlap = meta["sensor_overlap_minutes"].get(metric, 0)
            eligible = n_overlap >= self.config.runtime.min_sensor_overlap_minutes
            start_phase = self._overlap_start_phase(minute, f"support_core_overlap_{'hr' if metric == 'heart_rate' else metric if metric != 'temperature' else 'temp'}")
            if pair.empty:
                ax.text(0.5, 0.5, "No overlap", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue
            ax.scatter(pair[left], pair[right], s=18, alpha=0.65, color="#2563eb" if eligible else "#94a3b8")
            lo = min(pair[left].min(), pair[right].min())
            hi = max(pair[left].max(), pair[right].max())
            ax.plot([lo, hi], [lo, hi], color="#94a3b8", ls="--", lw=1)
            r = pair[left].corr(pair[right], method="spearman")
            phase_note = f"\nfirst overlap: {PHASE_ABBR.get(start_phase, start_phase[:3].upper())}" if start_phase else ""
            ax.set_xlabel(FEATURE_LABELS.get(left, left))
            ax.set_ylabel(FEATURE_LABELS.get(right, right))
            panel_notes.append(
                f"{panel_positions[len(panel_notes)]} shows {metric.replace('_', ' ')} agreement with Spearman r = {r:.2f}, overlap = {n_overlap} minutes{phase_note.replace(chr(10), ', ')} and a {'comparison-ready' if eligible else 'limited-overlap'} status."
            )
        fig._cltr_panel_notes = panel_notes
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_session_bland_altman(self, minute: pd.DataFrame, meta: dict):
        pairs = [
            ("heart_rate", "empatica_hr_mean_bpm", "biopac_hr_mean_bpm"),
            ("eda", "empatica_eda_mean_uS", "biopac_eda_mean_uS"),
            ("temperature", "empatica_temp_mean_C", "biopac_temp_chest_mean_C"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=self._figsize("three_panel_row"))
        panel_notes: list[str] = []
        panel_positions = ["Left", "Center", "Right"]
        for ax, (metric, left, right) in zip(axes, pairs):
            if left not in minute.columns or right not in minute.columns:
                ax.axis("off")
                continue
            pair = minute[[left, right]].apply(to_numeric).dropna()
            if pair.empty:
                ax.axis("off")
                continue
            mean = (pair[left] + pair[right]) / 2.0
            diff = pair[left] - pair[right]
            md = diff.mean()
            sd = diff.std(ddof=1) if len(diff) > 1 else np.nan
            ax.scatter(mean, diff, s=18, alpha=0.65, color="#2563eb")
            ax.axhline(md, color="#0f172a", lw=1.5)
            if pd.notna(sd):
                ax.axhline(md + 1.96 * sd, color="#dbe4ee", lw=0.85, ls="--", zorder=0)
                ax.axhline(md - 1.96 * sd, color="#dbe4ee", lw=0.85, ls="--", zorder=0)
            start_phase = self._overlap_start_phase(minute, f"support_core_overlap_{'hr' if metric == 'heart_rate' else metric if metric != 'temperature' else 'temp'}")
            start_note = f"\nfirst overlap: {PHASE_ABBR.get(start_phase, start_phase[:3].upper())}" if start_phase else ""
            ax.set_xlabel("Mean of paired sensors")
            ax.set_ylabel(f"{metric.replace('_', ' ').title()} difference")
            panel_notes.append(f"{panel_positions[len(panel_notes)]} shows Bland-Altman bias for {metric.replace('_', ' ')}{start_note.replace(chr(10), ', ')}.")
        fig._cltr_panel_notes = panel_notes
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_session_coverage(self, minute: pd.DataFrame, meta: dict):
        if minute.empty:
            return None
        comparison_minute = minute.loc[minute["protocol_block"].astype(str).isin(COMPARISON_BLOCKS)].copy()
        if comparison_minute.empty:
            return None
        mapping = [
            ("Questionnaire events", "questionnaire_n"),
            ("Fan", "support_fan"),
            ("Empatica", "support_empatica"),
            ("BIOPAC", "support_biopac"),
            ("Indoor", "support_indoor"),
            ("Outdoor", "support_outdoor"),
            ("HR overlap", "support_core_overlap_hr"),
            ("EDA overlap", "support_core_overlap_eda"),
            ("Temp overlap", "support_core_overlap_temp"),
        ]
        rows = []
        ylabels = []
        questionnaire_design = meta.get("questionnaire_design", {})
        expected_events = None
        observed_events = None
        if questionnaire_design:
            expected_events = questionnaire_design.get("expected_event_count")
            observed_events = questionnaire_design.get("observed_event_count")
        for label, col in mapping:
            if col == "questionnaire_n":
                vals = to_numeric(comparison_minute[col]).notna().astype(float).to_numpy() if col in comparison_minute.columns else np.zeros(len(comparison_minute))
                rows.append(vals)
                if expected_events and observed_events is not None:
                    ylabels.append(f"{label} ({int(observed_events)}/{int(expected_events)})")
                else:
                    ylabels.append(label)
                continue
            vals = to_numeric(comparison_minute[col]).fillna(0).to_numpy() if col in comparison_minute.columns else np.zeros(len(comparison_minute))
            rows.append(vals)
            ylabels.append(f"{label} ({vals.mean() * 100:.0f}%)")
        mat = np.vstack(rows)
        fig, ax = plt.subplots(figsize=self._figsize("wide_single_tall"))
        cmap = LinearSegmentedColormap.from_list("support_burden", ["#f8fafc", "#2563eb"])
        ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
        ax.grid(False)
        ax.set_yticks(range(len(mapping)))
        ax.set_yticklabels(ylabels, fontsize=11)
        tick_count = min(8, len(comparison_minute))
        xticks = np.linspace(0, len(comparison_minute) - 1, tick_count, dtype=int) if tick_count > 1 else np.array([0])
        ax.set_xticks(xticks)
        minute_tick_values = to_numeric(comparison_minute.iloc[xticks]["minute_index"]).fillna(0).astype(int).tolist()
        ax.set_xticklabels([str(x) for x in minute_tick_values], fontsize=11)
        ax.set_xlabel(self._time_axis_label())
        for y in np.arange(0.5, len(mapping), 1.0):
            ax.axhline(y, color="#ffffff", lw=1.0, alpha=0.95, zorder=2)
        if "protocol_phase" in comparison_minute.columns:
            phase_series = comparison_minute["protocol_phase"].fillna("").astype(str).reset_index(drop=True)
            start = 0
            spans = []
            for idx in range(1, len(phase_series)):
                if phase_series.iloc[idx] != phase_series.iloc[idx - 1]:
                    spans.append((start, idx - 1, phase_series.iloc[idx - 1]))
                    start = idx
            spans.append((start, len(phase_series) - 1, phase_series.iloc[-1]))
            for start_idx, end_idx, _ in spans[:-1]:
                ax.axvline(end_idx + 0.5, color="#dbe4ee", lw=0.9, zorder=2)
            for start_idx, end_idx, phase_name in spans:
                midpoint = (start_idx + end_idx) / 2
                ax.text(
                    midpoint,
                    -0.82,
                    PHASE_ABBR.get(phase_name, phase_name[:3].upper()),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#64748b",
                    clip_on=False,
                )
        for spine in ax.spines.values():
            spine.set_color("#dbe4ee")
            spine.set_linewidth(0.9)
        fig.tight_layout()
        return fig

    def _fig_session_phase_distributions(self, phase: pd.DataFrame):
        metrics = [m for m in ["thermal_comfort", "empatica_hr_mean_bpm", "biopac_temp_chest_mean_C", "master_dpg_C"] if m in phase.columns]
        if not metrics:
            return None
        fig, axes = plt.subplots(2, 2, figsize=self._figsize("two_by_two_balanced"))
        axes = axes.ravel()
        baseline_notes: list[str] = []
        panel_notes: list[str] = []
        panel_positions = ["Top left", "Top right", "Bottom left", "Bottom right"]
        for ax, metric in zip(axes, metrics):
            data = []
            labels = []
            for p in PHASE_ORDER:
                cur = to_numeric(phase.loc[phase["protocol_phase"] == p, metric]).dropna()
                if not cur.empty:
                    data.append(cur.to_numpy())
                    labels.append(PHASE_ABBR.get(p, p[:3].upper()))
            if not data:
                ax.axis("off")
                continue
            ax.boxplot(data, tick_labels=labels, patch_artist=True, boxprops={"facecolor": "#dbeafe"})
            baseline_info = self._phase_metric_baseline(phase, metric)
            suffix = f"\nbaseline: {self._baseline_phase_abbr(baseline_info)}" if baseline_info else ""
            panel_notes.append(f"{panel_positions[len(panel_notes)]} shows phase distributions for {FEATURE_LABELS.get(metric, metric)}{suffix.replace(chr(10), ', ')}.")
            self._apply_discrete_y_axis_matplotlib(ax, np.concatenate(data) if data else [], metric)
            note = self._baseline_note(baseline_info)
            if note and note not in baseline_notes:
                baseline_notes.append(note)
        for ax in axes[len(metrics):]:
            ax.axis("off")
        fig._cltr_panel_notes = panel_notes
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_cohort_gate(self, c: dict):
        sample = c["sample_status"].iloc[0]
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        ax.axis("off")
        status = "FULL COMPARISONS AVAILABLE" if int(sample["cohort_inference_eligible"]) else "EARLY DIRECTIONAL SUMMARY"
        txt = (
            f"{status}\n\n"
            f"Sessions: {int(sample['n_sessions'])} / required {int(sample['min_sessions_required'])}\n"
            f"Participants: {int(sample['n_participants'])} / required {int(sample['min_participants_required'])}\n\n"
            "This check determines whether the report can support full cross-session comparisons or should focus on directional patterns."
        )
        ax.text(0.5, 0.55, txt, ha="center", va="center", fontsize=14, fontweight="bold", bbox={"boxstyle": "round,pad=0.8", "fc": "#eff6ff", "ec": "#bfdbfe"})
        return fig

    def _fig_cohort_design(self, c: dict):
        session_summary = c["session_summary"]
        if session_summary.empty:
            return None
        cond_order = [x for x in CONDITION_ORDER if x in session_summary["condition_code"].astype(str).unique()]
        if not cond_order:
            cond_order = sorted(session_summary["condition_code"].astype(str).dropna().unique().tolist())
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(17.2, 5.6),
            gridspec_kw={"width_ratios": [1.15, 0.9, 1.08], "wspace": 0.32},
            constrained_layout=True,
        )

        cond_counts = (
            session_summary["condition_code"]
            .astype(str)
            .value_counts()
            .reindex(cond_order)
            .fillna(0)
            .astype(int)
        )
        ypos = np.arange(len(cond_counts), dtype=float)
        axes[0].hlines(ypos, 0, cond_counts.to_numpy(dtype=float), color="#d7e0ea", linewidth=3.2, zorder=1)
        axes[0].scatter(
            cond_counts.to_numpy(dtype=float),
            ypos,
            s=170,
            color=[CONDITION_COLORS.get(x, "#475569") for x in cond_counts.index],
            edgecolors="#172033",
            linewidths=0.7,
            zorder=3,
        )
        for idx, (label, value) in enumerate(cond_counts.items()):
            axes[0].text(
                max(float(value) * 0.48, 2.2),
                idx,
                f"{label} | n={int(value)}",
                ha="center",
                va="center",
                fontsize=10,
                color="#172033",
                bbox={"boxstyle": "round,pad=0.18", "fc": "#ffffff", "ec": "#dbe4ee", "lw": 0.35, "alpha": 0.96},
                zorder=4,
            )
        axes[0].set_yticks(ypos)
        axes[0].set_yticklabels(["" for _ in ypos])
        axes[0].set_xlim(0, max(float(cond_counts.max()) * 1.18, 1.0))
        axes[0].set_xlabel("Sessions")
        axes[0].set_title("Session types")
        axes[0].invert_yaxis()
        axes[0].grid(True, axis="x", alpha=0.2)
        axes[0].grid(False, axis="y")

        factor_palette = {
            "DIM": "#5b6472",
            "BRI": "#2563eb",
            "MOR": "#0b6e4f",
            "MID": "#c96b00",
        }
        design_matrix = (
            session_summary.groupby(["illuminance_level", "time_of_day"]).size().unstack(fill_value=0).reindex(index=["DIM", "BRI"], columns=["MOR", "MID"]).fillna(0)
        )
        condition_lookup = {
            ("DIM", "MOR"): "DIM-MOR",
            ("DIM", "MID"): "DIM-MID",
            ("BRI", "MOR"): "BRI-MOR",
            ("BRI", "MID"): "BRI-MID",
        }
        axes[1].set_title("Design balance")
        axes[1].set_xlabel("Diurnal timing")
        axes[1].set_ylabel("Illuminance")
        axes[1].set_xlim(-0.5, len(design_matrix.columns) - 0.5)
        axes[1].set_ylim(len(design_matrix.index) - 0.5, -0.5)
        axes[1].set_xticks(range(len(design_matrix.columns)))
        axes[1].set_xticklabels([str(col) for col in design_matrix.columns])
        axes[1].set_yticks(range(len(design_matrix.index)))
        axes[1].set_yticklabels([str(idx) for idx in design_matrix.index])
        axes[1].grid(False)
        for i, illum in enumerate(design_matrix.index):
            for j, timing in enumerate(design_matrix.columns):
                cell_condition = condition_lookup.get((str(illum), str(timing)), "")
                face = CONDITION_COLORS.get(cell_condition, "#cbd5e1")
                rect = Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, facecolor=face, edgecolor="white", linewidth=2.2)
                axes[1].add_patch(rect)
                count_val = int(design_matrix.loc[illum, timing])
                axes[1].text(
                    j,
                    i - 0.09,
                    cell_condition,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                    fontweight="bold",
                )
                axes[1].text(
                    j,
                    i + 0.18,
                    f"n={count_val}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                )

        support_cols = [
            ("Questionnaire", "questionnaire_completeness"),
            ("Empatica", "empatica_fraction"),
            ("BIOPAC", "biopac_fraction"),
            ("Indoor", "indoor_fraction"),
        ]
        support_table = (
            session_summary.groupby("condition_code", as_index=True)[[col for _, col in support_cols]]
            .mean()
            .reindex(cond_order)
        )
        mat = support_table.to_numpy(dtype=float)
        cmap = LinearSegmentedColormap.from_list("session_support", ["#fff7ed", "#bfdbfe", "#0b6e4f"])
        im = axes[2].imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        axes[2].grid(False)
        axes[2].set_xticks(range(len(support_cols)))
        axes[2].set_xticklabels([label for label, _ in support_cols], rotation=0, ha="center")
        axes[2].set_yticks(range(len(cond_order)))
        axes[2].set_yticklabels(cond_order)
        axes[2].tick_params(axis="x", labelsize=11, pad=8)
        axes[2].set_title("Mean analytic support")
        for row in range(mat.shape[0]):
            for col in range(mat.shape[1]):
                val = float(mat[row, col])
                axes[2].text(
                    col,
                    row,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color="white" if val >= 0.62 else "#172033",
                    fontweight="bold",
                )
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03, label="Mean support fraction")

        for ax in axes:
            ax.set_axisbelow(True)
            if ax is not axes[2]:
                ax.grid(True, axis="y", alpha=0.18)
        axes[1].grid(False)
        fig._cltr_panel_notes = [
            "Left|Session types|Condition-defined session types shown as ordered lollipop counts so the study composition is readable at a glance.",
            "Middle|Design balance|A 2x2 balance matrix showing how sessions distribute across illuminance and diurnal-timing factors.",
            "Right|Mean analytic support by session type|Mean analytic support by session type for questionnaire, Empatica, BIOPAC, and indoor environmental streams.",
        ]
        return fig

    def _fig_cohort_window_validation(self, c: dict):
        signal_audit = c.get("signal_audit_summary", pd.DataFrame()).copy()
        session_signal_audit = c.get("session_signal_audit", pd.DataFrame()).copy()
        if signal_audit.empty:
            return None
        fig = plt.figure(figsize=(16.2, 10.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.95])
        landscape_ax = fig.add_subplot(gs[0, 0])
        quality_ax = fig.add_subplot(gs[0, 1])

        stream_label_map = {str(item["signal_stream"]): str(item["label"]) for item in DEVICE_STREAM_CATALOG}
        role_label_map = {
            "primary": "Primary",
            "primary_with_qc": "Primary with QC",
            "subset_only": "Subset only",
            "secondary_validation": "Secondary validation",
            "secondary_only": "Secondary only",
            "not_primary": "Not primary",
            "not_recommended": "Not recommended",
        }
        role_color_map = {
            "primary": "#0b6e4f",
            "primary_with_qc": "#c96b00",
            "subset_only": "#a83a32",
            "secondary_validation": "#1d4ed8",
            "secondary_only": "#5b6472",
            "not_primary": "#5b6472",
            "not_recommended": "#7f1d1d",
        }
        adequacy_fill_map = {
            "strong": "#dff3ea",
            "usable_with_caution": "#fef0d9",
            "limited": "#f8ddda",
            "not_audited": "#e2e8f0",
        }
        audit = signal_audit.copy()
        audit["stream_label"] = audit["signal_stream"].astype(str).map(stream_label_map).fillna(audit["signal_stream"].astype(str))
        audit["adequacy_score"] = to_numeric(audit.get("adequacy_score", pd.Series(dtype=float))).fillna(0)
        audit["coverage"] = to_numeric(audit.get("mean_coverage_fraction", pd.Series(dtype=float)))
        audit["quality"] = to_numeric(audit.get("mean_quality_fraction", pd.Series(dtype=float)))
        audit["plausibility"] = to_numeric(audit.get("mean_plausible_fraction", pd.Series(dtype=float)))
        audit["role_label"] = audit["recommended_role"].astype(str).map(role_label_map).fillna(audit["recommended_role"].astype(str))
        audit["role_color"] = audit["recommended_role"].astype(str).map(role_color_map).fillna("#64748b")
        audit["adequacy_fill"] = audit["adequacy_status"].astype(str).map(adequacy_fill_map).fillna("#e2e8f0")

        flagged_streams = self._flagged_stream_session_register(session_signal_audit)
        flagged_small = flagged_streams[["signal_stream", "affected_sessions", "primary_concern_driver"]].copy() if not flagged_streams.empty else pd.DataFrame(columns=["signal_stream", "affected_sessions", "primary_concern_driver"])
        audit = audit.merge(flagged_small, on="signal_stream", how="left")
        audit["affected_sessions"] = to_numeric(audit.get("affected_sessions", pd.Series(dtype=float))).fillna(0)
        concern_short = {
            "Missingness / coverage": "Coverage",
            "Plausibility / out-of-range values": "Plausibility",
            "Quality flag support": "Quality",
            "Cross-device agreement": "Agreement",
        }
        audit["concern_short"] = audit["primary_concern_driver"].astype(str).map(concern_short).fillna("")
        device_order = {"Empatica": 0, "BIOPAC": 1}
        role_order = {"primary": 0, "primary_with_qc": 1, "secondary_validation": 2, "secondary_only": 3, "subset_only": 4, "not_primary": 5, "not_recommended": 6}
        audit = audit.sort_values(
            ["device", "recommended_role", "adequacy_score", "stream_label"],
            ascending=[True, True, False, True],
            key=lambda col: col.map(device_order if col.name == "device" else role_order) if col.name in {"device", "recommended_role"} else col,
        ).reset_index(drop=True)

        y = np.arange(len(audit), dtype=float)
        landscape_ax.axvspan(0, 60, color="#fbefee", zorder=0)
        landscape_ax.axvspan(60, 80, color="#fdf5e8", zorder=0)
        landscape_ax.axvspan(80, 100, color="#edf8f2", zorder=0)
        for idx, row in enumerate(audit.itertuples(index=False)):
            landscape_ax.hlines(idx, 0, float(row.adequacy_score), color="#cbd5e1", linewidth=2.8, zorder=1)
            landscape_ax.scatter(
                float(row.adequacy_score),
                idx,
                s=130,
                color=row.role_color,
                edgecolors="#172033",
                linewidths=0.8,
                zorder=3,
            )
            role_short = {
                "Primary": "Primary",
                "Primary with QC": "QC",
                "Subset only": "Subset",
            }.get(str(row.role_label), str(row.role_label))
            note_bits = [f"{int(round(float(row.adequacy_score)))}", role_short]
            if float(row.affected_sessions) > 0:
                note_bits.append(f"F={int(row.affected_sessions)}")
            score = float(row.adequacy_score)
            label_x = min(max(score * 0.52, 12.0), max(score - 4.0, 12.0))
            landscape_ax.text(
                label_x,
                idx,
                " | ".join(note_bits),
                va="center",
                ha="center",
                fontsize=9.5,
                color="#172033",
                bbox={"boxstyle": "round,pad=0.18", "fc": "#ffffff", "ec": "#dbe4ee", "lw": 0.35, "alpha": 0.96},
                zorder=4,
            )
        landscape_ax.set_yticks(y)
        landscape_ax.set_yticklabels(audit["stream_label"])
        landscape_ax.set_xlim(0, 104)
        landscape_ax.set_xlabel("Adequacy score")
        landscape_ax.set_title("All audited device streams")
        landscape_ax.invert_yaxis()
        landscape_ax.grid(True, axis="x", alpha=0.22)
        landscape_ax.grid(False, axis="y")
        landscape_legend_handles = [
            Patch(facecolor="#0f766e", edgecolor="none", label="Primary"),
            Patch(facecolor="#ea580c", edgecolor="none", label="Primary with QC"),
            Patch(facecolor="#b91c1c", edgecolor="none", label="Subset only"),
        ]
        landscape_ax.legend(
            handles=landscape_legend_handles,
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.085),
            columnspacing=0.95,
            handletextpad=0.42,
            borderaxespad=0.0,
        )

        quality_cols = [
            ("coverage", "Coverage"),
            ("quality", "Quality"),
            ("plausibility", "Plausibility"),
        ]
        quality_matrix = audit[[col for col, _ in quality_cols]].to_numpy(dtype=float)
        quality_im = quality_ax.imshow(np.nan_to_num(quality_matrix, nan=0.0), aspect="auto", cmap="GnBu", vmin=0, vmax=1)
        quality_ax.set_xticks(range(len(quality_cols)))
        quality_ax.set_xticklabels([label for _, label in quality_cols])
        quality_ax.set_yticks(y)
        quality_ax.set_yticklabels(audit["stream_label"])
        quality_ax.set_title("Validity components across audited streams")
        quality_ax.grid(False)
        quality_ax.xaxis.tick_top()
        quality_ax.tick_params(axis="x", pad=8)
        for i in range(quality_matrix.shape[0]):
            for j in range(quality_matrix.shape[1]):
                value = quality_matrix[i, j]
                txt = "n/a" if pd.isna(value) else f"{value:.2f}"
                quality_ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#172033",
                )
        plt.colorbar(quality_im, ax=quality_ax, fraction=0.04, pad=0.015, label="Fraction")

        fig._cltr_panel_notes = [
            "Left|All audited device streams|Each audited Empatica and BIOPAC stream is shown once, with adequacy score, policy role, flagged-session burden, and dominant concern driver kept in a single ordered landscape.",
            "Right|Validity components across audited streams|Coverage, quality support, and plausibility are shown for every audited stream so the full modality inventory can be reviewed without collapsing the audit into only the comparable device pairs.",
        ]
        fig.tight_layout(rect=(0, 0.065, 1, 0.975))
        return fig

    def _fig_cohort_comparable_validation_summary(self, c: dict):
        support = c.get("condition_support_summary", pd.DataFrame()).copy()
        agreement_summary = c.get("agreement_summary", pd.DataFrame()).copy()
        if support.empty and agreement_summary.empty:
            return None
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.6), gridspec_kw={"width_ratios": [1.15, 1.0, 0.92]})
        metrics = ["heart_rate", "eda", "temperature"]
        metric_labels = {"heart_rate": "Heart rate", "eda": "EDA", "temperature": "Temperature"}
        metric_colors = {"heart_rate": "#111827", "eda": "#2563eb", "temperature": "#ea580c"}

        cond_order = [x for x in CONDITION_ORDER if x in support.get("condition_code", pd.Series(dtype=str)).astype(str).unique()]
        if not cond_order and not support.empty:
            cond_order = support["condition_code"].astype(str).tolist()
        temp = support.set_index("condition_code").reindex(cond_order) if not support.empty else pd.DataFrame(index=cond_order)
        x = np.arange(len(cond_order), dtype=float)
        overlap_cols = [
            ("hr_overlap_minutes__mean", "heart_rate"),
            ("eda_overlap_minutes__mean", "eda"),
            ("temp_overlap_minutes__mean", "temperature"),
        ]
        if cond_order:
            for idx, (col, metric) in enumerate(overlap_cols):
                vals = to_numeric(temp[col]).fillna(0).to_numpy(dtype=float) if col in temp.columns else np.zeros(len(cond_order))
                axes[0].bar(x + (idx - 1) * 0.24, vals, width=0.24, color=metric_colors[metric], label=metric_labels[metric])
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(cond_order)
            axes[0].set_ylabel("Mean overlap minutes")
            axes[0].set_title("Comparable-pair overlap by condition")
            self._place_topbar_legend(axes[0], y=1.04)
        else:
            axes[0].axis("off")

        if not agreement_summary.empty:
            agreement_summary = agreement_summary.loc[agreement_summary["metric"].astype(str).isin(metrics)].copy()
            agreement_summary["metric"] = pd.Categorical(agreement_summary["metric"].astype(str), categories=metrics, ordered=True)
            agreement_summary = agreement_summary.sort_values("metric")
            xpos = np.arange(len(agreement_summary), dtype=float)
            total_sessions = to_numeric(agreement_summary.get("n_sessions", pd.Series(dtype=float))).fillna(0).to_numpy(dtype=float)
            eligible_sessions = to_numeric(agreement_summary.get("n_eligible_sessions", pd.Series(dtype=float))).fillna(0).to_numpy(dtype=float)
            total_bars = axes[1].bar(xpos, total_sessions, width=0.58, color="#e2e8f0", edgecolor="#cbd5e1", linewidth=1.0, label="Cohort sessions")
            axes[1].bar(
                xpos,
                eligible_sessions,
                width=0.42,
                color=[metric_colors[str(metric)] for metric in agreement_summary["metric"].astype(str)],
                edgecolor="none",
                label="Validation-eligible",
            )
            axes[1].set_xticks(xpos)
            axes[1].set_xticklabels([metric_labels[str(metric)] for metric in agreement_summary["metric"].astype(str)])
            axes[1].set_ylabel("Sessions")
            axes[1].set_title("Validation readiness against cohort size")
            ymax = max(float(total_sessions.max()) if len(total_sessions) else 0.0, 1.0)
            axes[1].set_ylim(0, ymax * 1.28)
            self._place_topbar_legend(axes[1], y=1.04)
            for idx, row in enumerate(agreement_summary.itertuples(index=False)):
                axes[1].text(
                    idx,
                    float(eligible_sessions[idx]) + ymax * 0.05,
                    f"{int(eligible_sessions[idx])}/{int(total_sessions[idx])}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#172033",
                )

            corr_vals = to_numeric(agreement_summary.get("median_spearman_r", pd.Series(dtype=float))).fillna(np.nan).to_numpy(dtype=float)
            mae_vals = to_numeric(agreement_summary.get("median_mae", pd.Series(dtype=float))).fillna(np.nan).to_numpy(dtype=float)
            overlap_vals = to_numeric(agreement_summary.get("median_overlap_minutes", pd.Series(dtype=float))).fillna(np.nan).to_numpy(dtype=float)
            ypos = np.arange(len(agreement_summary), dtype=float)
            axes[2].hlines(ypos, 0, corr_vals, color="#cbd5e1", linewidth=2.8)
            axes[2].scatter(
                corr_vals,
                ypos,
                s=130,
                color=[metric_colors[str(metric)] for metric in agreement_summary["metric"].astype(str)],
                edgecolors="#172033",
                linewidths=0.8,
                zorder=3,
            )
            axes[2].axvline(0.7, color="#94a3b8", lw=1.0, ls="--")
            axes[2].set_yticks(ypos)
            axes[2].set_yticklabels([metric_labels[str(metric)] for metric in agreement_summary["metric"].astype(str)])
            axes[2].set_xlim(-0.02, 1.04)
            axes[2].set_xlabel("Median Spearman r")
            axes[2].set_title("Agreement strength and error")
            for idx, row in enumerate(agreement_summary.itertuples(index=False)):
                axes[2].text(
                    min(float(corr_vals[idx]) + 0.03, 0.99),
                    idx,
                    f"MAE {float(mae_vals[idx]):.2f} | {float(overlap_vals[idx]):.0f} min",
                    va="center",
                    fontsize=10,
                    color="#172033",
                )
            axes[2].invert_yaxis()
        else:
            axes[1].axis("off")
            axes[2].axis("off")

        for ax in axes:
            if ax.axison:
                ax.set_axisbelow(True)
                ax.grid(True, axis="y", alpha=0.2)
        fig._cltr_panel_notes = [
            "Left|Comparable-pair overlap by condition|Mean overlap minutes by condition for the directly comparable Empatica/BIOPAC heart-rate, electrodermal, and temperature families.",
            "Center|Validation-ready sessions|Eligible-session counts for each comparable family shown against the full cohort session count.",
            "Right|Agreement strength and error|Median correlation is shown directly, with median error and overlap annotated per comparable family.",
        ]
        fig.tight_layout(rect=(0, 0.035, 1, 0.97))
        return fig

    def _fig_cohort_support_map(self, minute: pd.DataFrame):
        if minute.empty or "minute_index" not in minute.columns or "condition_code" not in minute.columns:
            return None
        mapping = [
            ("Questionnaire", "questionnaire_n"),
            ("Fan", "support_fan"),
            ("Empatica", "support_empatica"),
            ("BIOPAC", "support_biopac"),
            ("Indoor", "support_indoor"),
            ("Outdoor", "support_outdoor"),
            ("HR overlap", "support_core_overlap_hr"),
            ("EDA overlap", "support_core_overlap_eda"),
            ("Temp overlap", "support_core_overlap_temp"),
        ]
        cond_order = [x for x in CONDITION_ORDER if x in minute["condition_code"].astype(str).unique()]
        if not cond_order:
            cond_order = sorted(minute["condition_code"].astype(str).dropna().unique().tolist())
        if not cond_order:
            return None
        minute_template = (
            minute[["minute_index", "protocol_phase"]]
            .dropna()
            .sort_values(["minute_index", "protocol_phase"])
            .groupby("minute_index", as_index=False)
            .agg(protocol_phase=("protocol_phase", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]))
        )
        minute_index_values = to_numeric(minute_template["minute_index"]).dropna().astype(int).tolist()
        if not minute_index_values:
            return None
        rows = []
        labels = []
        row_meta = []
        for cond in cond_order:
            dcond = minute.loc[minute["condition_code"].astype(str) == cond].copy()
            if dcond.empty:
                continue
            grouped = dcond.groupby("minute_index")
            for label, col in mapping:
                if col == "questionnaire_n":
                    series = grouped[col].apply(lambda s: float(to_numeric(s).notna().mean()) if col in dcond.columns else 0.0)
                elif col in dcond.columns:
                    series = grouped[col].apply(lambda s: float((to_numeric(s).fillna(0) > 0).mean()))
                else:
                    series = pd.Series(dtype=float)
                series = series.reindex(minute_index_values).fillna(0.0)
                rows.append(series.to_numpy(dtype=float))
                labels.append(label)
                row_meta.append({"condition": cond, "label": label})
        if not rows:
            return None
        mat = np.vstack(rows)
        fig, ax = plt.subplots(figsize=(14.4, max(5.2, 0.18 * len(labels) + 1.7)))
        cmap = LinearSegmentedColormap.from_list("cohort_support", ["#f8fafc", "#dbeafe", "#0b6e4f"])
        minute_min = float(minute_index_values[0]) - 0.5
        minute_max = float(minute_index_values[-1]) + 0.5
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            vmin=0,
            vmax=1,
            extent=(minute_min, minute_max, len(labels) - 0.5, -0.5),
        )
        ax.grid(False)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8.5)
        tick_count = min(8, len(minute_index_values))
        tick_indices = np.linspace(0, len(minute_index_values) - 1, tick_count, dtype=int) if tick_count > 1 else np.array([0])
        tick_values = [minute_index_values[idx] for idx in tick_indices]
        ax.set_xticks(tick_values)
        ax.set_xticklabels([str(value) for value in tick_values], fontsize=11)
        ax.set_xlabel("Timeline minute")
        ax.set_title("Phase-annotated cohort availability and overlap map")
        self._add_phase_spans(ax, minute_template)
        group_size = len(mapping)
        family_breaks = [1, 6]
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for idx, cond in enumerate(cond_order):
            start = idx * group_size
            end = start + group_size - 1
            if start >= len(labels):
                continue
            if idx > 0:
                ax.axhline(start - 0.5, color="#94a3b8", lw=1.2, alpha=0.9)
            ax.text(
                -0.17,
                (start + end) / 2.0,
                cond,
                transform=trans,
                ha="right",
                va="center",
                rotation=90,
                fontsize=10,
                fontweight="bold",
                color=CONDITION_COLORS.get(cond, "#334155"),
                bbox={"boxstyle": "round,pad=0.18", "fc": "#ffffff", "ec": "#cbd5e1", "lw": 0.6, "alpha": 0.96},
            )
            for local_row in range(group_size):
                row_idx = start + local_row
                if row_idx >= len(labels):
                    continue
                if local_row % 2 == 1:
                    ax.axhspan(row_idx - 0.5, row_idx + 0.5, color="#f8fafc", alpha=0.35, zorder=0)
                if local_row in family_breaks:
                    ax.axhline(row_idx - 0.5, color="#cbd5e1", lw=0.9, alpha=0.85)
        plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02, label="Support fraction")
        fig._cltr_panel_notes = [
            "Rows are grouped by condition and then by availability family so questionnaire presence, fan/control support, device or environmental source presence, and paired non-null overlap can be compared against the shared protocol timeline.",
            "Phase annotations mark the shared protocol structure so availability and overlap windows can be read against acclimation, intervention, and terminal phases.",
        ]
        fig.tight_layout(rect=(0.14, 0.02, 1, 0.96))
        return fig

    def _cohort_band(self, d: pd.DataFrame, feature: str, ax: plt.Axes):
        grouped = d.dropna(subset=["condition_code", feature]).groupby(["condition_code", "minute_index"])[feature]
        summary = grouped.agg(q25=lambda s: s.quantile(0.25), median="median", q75=lambda s: s.quantile(0.75)).reset_index()
        for cond in [x for x in CONDITION_ORDER if x in summary["condition_code"].unique()]:
            cur = summary.loc[summary["condition_code"] == cond].sort_values("minute_index")
            x = to_numeric(cur["minute_index"])
            if self._is_sparse_observation_channel(feature):
                y = to_numeric(cur["median"])
                lo = y - to_numeric(cur["q25"])
                hi = to_numeric(cur["q75"]) - y
                valid = y.notna()
                if feature not in {"thermal_pleasure", "visual_comfort", "air_quality_comfort"} and bool(valid.sum() >= 2):
                    ax.plot(
                        x[valid],
                        y[valid],
                        color=CONDITION_COLORS[cond],
                        lw=0.9,
                        alpha=0.28,
                        zorder=1,
                    )
                ax.errorbar(
                    x,
                    y,
                    yerr=np.vstack([lo.to_numpy(dtype=float), hi.to_numpy(dtype=float)]),
                    fmt="o",
                    ms=5,
                    lw=1.0,
                    capsize=2.5,
                    color=CONDITION_COLORS[cond],
                    alpha=0.9,
                    label=cond,
                )
            elif self._is_control_signal_channel(feature):
                q25 = self._display_series(cur["q25"], feature)
                y = self._display_series(cur["median"], feature)
                q75 = self._display_series(cur["q75"], feature)
                ax.fill_between(x, q25, q75, color=CONDITION_COLORS[cond], alpha=0.18, step="mid")
                ax.step(x, y, where="mid", color=CONDITION_COLORS[cond], lw=2, label=cond)
                ax.scatter(x, to_numeric(cur["median"]), s=10, alpha=0.3, color=CONDITION_COLORS[cond], zorder=3)
            else:
                ax.fill_between(x, cur["q25"], cur["q75"], color=CONDITION_COLORS[cond], alpha=0.18)
                ax.plot(x, cur["median"], color=CONDITION_COLORS[cond], lw=2, label=cond)

    def _fig_cohort_single_channel_burst(self, minute: pd.DataFrame, column: str, color: str):
        if minute.empty or "minute_index" not in minute.columns or column not in minute.columns:
            return None
        if self._is_sparse_observation_channel(column):
            return self._fig_sparse_phase_distribution(minute, column)
        display_minute, display_note = self._channel_display_window(minute, column)
        if display_minute.empty:
            return None
        fig, ax = plt.subplots(figsize=self._figsize("wide_single_short"))
        phase_df = display_minute.drop_duplicates(subset=["minute_index", "protocol_phase"])
        self._add_phase_spans(ax, phase_df)
        self._cohort_band(display_minute, column, ax)
        ax.set_ylabel(self._axis_label(column))
        ax.set_xlabel(self._time_axis_label())
        ax.grid(True, axis="y")
        self._apply_discrete_y_axis_matplotlib(ax, display_minute[column], column)
        marker_only = self._is_sparse_observation_channel(column)
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=CONDITION_COLORS[cond],
                lw=0 if marker_only else 2,
                marker="o" if marker_only else None,
                label=cond,
            )
            for cond in CONDITION_ORDER
            if cond in display_minute["condition_code"].astype(str).unique()
        ]
        if handles:
            self._place_condition_legend(ax, handles=handles)
        baseline_note = self._baseline_note(self._phase_metric_baseline(self._phase_summary_from_minute(minute, [column]), column))
        note = " ".join(
            part
            for part in [
                (
                    self._cohort_questionnaire_caption(column)
                    if self._is_sparse_observation_channel(column)
                    else (
                        f"{FEATURE_LABELS.get(column, column)} is shown as condition-stratified rolling-median step trends with interquartile bands; faint markers retain the unsmoothed minute medians."
                        if self._is_control_signal_channel(column)
                        else f"{FEATURE_LABELS.get(column, column)} is shown as condition-stratified cohort medians with interquartile bands."
                    )
                ),
                display_note,
                baseline_note,
            ]
            if part
        )
        fig.tight_layout()
        return fig

    def _fig_sparse_phase_distribution(self, minute: pd.DataFrame, metric: str):
        if minute.empty or metric not in minute.columns:
            return None
        temp = minute.loc[minute["protocol_phase"].astype(str) != "acclimation", ["protocol_phase", "condition_code", metric]].copy()
        temp[metric] = to_numeric(temp[metric])
        temp = temp.dropna()
        if temp.empty:
            return None
        phase_order = [p for p in PHASE_ORDER if p != "acclimation" and p in temp["protocol_phase"].astype(str).unique()]
        if not phase_order:
            return None
        fig, ax = plt.subplots(figsize=self._figsize("wide_single"))
        x = np.arange(len(phase_order), dtype=float)
        offsets = {"DIM-MOR": -0.24, "BRI-MOR": -0.08, "DIM-MID": 0.08, "BRI-MID": 0.24}
        width = 0.13
        legend_handles = []
        lower_annotations: list[tuple[float, int]] = []
        rng = np.random.default_rng(42)
        for cond in [c for c in CONDITION_ORDER if c in temp["condition_code"].astype(str).unique()]:
            grouped = []
            positions = []
            point_x = []
            point_y = []
            counts = []
            cond_color = CONDITION_COLORS[cond]
            for idx, phase in enumerate(phase_order):
                vals = temp.loc[
                    (temp["condition_code"].astype(str) == cond) & (temp["protocol_phase"].astype(str) == phase),
                    metric,
                ].dropna()
                if vals.empty:
                    continue
                arr = vals.to_numpy(dtype=float)
                xpos = float(x[idx] + offsets.get(cond, 0.0))
                grouped.append(arr)
                positions.append(xpos)
                counts.append((xpos, int(len(arr))))
                point_x.extend((xpos + rng.uniform(-0.028, 0.028, size=len(arr))).tolist())
                point_y.extend(arr.tolist())
            if not grouped:
                continue
            # Use a light violin only when the ordinal sample is large enough and
            # has enough distinct support to avoid implying a fake smooth density.
            violin_groups = []
            violin_positions = []
            for arr, xpos in zip(grouped, positions):
                unique_n = int(len(np.unique(np.round(arr, 6))))
                if len(arr) >= 5 and unique_n >= 3:
                    violin_groups.append(arr)
                    violin_positions.append(xpos)
            if violin_groups:
                vp = ax.violinplot(
                    violin_groups,
                    positions=violin_positions,
                    widths=width * 1.35,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for body in vp["bodies"]:
                    body.set_facecolor(cond_color)
                    body.set_edgecolor(cond_color)
                    body.set_alpha(0.12)
            median_y = []
            for arr, xpos in zip(grouped, positions):
                median_y.append((xpos, float(np.median(arr))))
            ax.scatter(point_x, point_y, s=18, alpha=0.38, color=cond_color, edgecolors="none", zorder=3)
            for xpos, y_med in median_y:
                ax.hlines(y_med, xpos - width * 0.38, xpos + width * 0.38, color=cond_color, linewidth=1.6, zorder=4)
            legend_handles.append(plt.Line2D([0], [0], color=cond_color, lw=6, alpha=0.5, label=cond))
            lower_annotations.extend(counts)
        ax.set_xticks(x)
        ax.set_xticklabels([PHASE_ABBR.get(p, p[:3].upper()) for p in phase_order])
        ax.set_ylabel(self._axis_label(metric))
        ax.set_xlabel(self._phase_axis_label())
        ax.grid(True, axis="y")
        self._apply_discrete_y_axis_matplotlib(ax, temp[metric], metric)
        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min if y_max > y_min else 1.0
        y_tail = y_min - 0.08 * y_span
        for xpos, n_obs in lower_annotations:
            ax.text(
                xpos,
                y_tail,
                f"n={n_obs}",
                ha="center",
                va="top",
                fontsize=7,
                color="#475569",
                clip_on=False,
            )
        ax.set_ylim(y_min - 0.16 * y_span, y_max + 0.04 * y_span)
        if legend_handles:
            self._place_condition_legend(ax, handles=legend_handles)
        fig.tight_layout()
        return fig

    def _cohort_condition_phase_summary(self, minute: pd.DataFrame, metric: str) -> pd.DataFrame:
        if minute.empty or metric not in minute.columns or "condition_code" not in minute.columns or "protocol_phase" not in minute.columns:
            return pd.DataFrame()
        cols = ["session_id", "condition_code", "protocol_phase", metric]
        temp = minute.loc[:, [c for c in cols if c in minute.columns]].copy()
        temp[metric] = to_numeric(temp[metric])
        temp = temp.dropna(subset=[metric, "condition_code", "protocol_phase"])
        if temp.empty:
            return pd.DataFrame()
        session_phase = (
            temp.groupby(["session_id", "condition_code", "protocol_phase"], as_index=False)[metric]
            .mean()
        )
        rows = []
        for cond in [c for c in CONDITION_ORDER if c in session_phase["condition_code"].astype(str).unique()]:
            dcond = session_phase.loc[session_phase["condition_code"] == cond]
            for phase_name in [p for p in PHASE_ORDER if p in dcond["protocol_phase"].astype(str).unique()]:
                vals = to_numeric(dcond.loc[dcond["protocol_phase"] == phase_name, metric]).dropna()
                if vals.empty:
                    continue
                rows.append(
                    {
                        "condition_code": cond,
                        "protocol_phase": phase_name,
                        "median": float(vals.median()),
                        "q25": float(vals.quantile(0.25)),
                        "q75": float(vals.quantile(0.75)),
                        "n_sessions": int(vals.shape[0]),
                    }
                )
        return pd.DataFrame(rows)

    def _fig_cohort_condition_comparison_summary(self, minute: pd.DataFrame, metric: str):
        if minute.empty or metric not in minute.columns or "condition_code" not in minute.columns:
            return None
        cols = ["session_id", "condition_code", metric]
        temp = minute.loc[:, [c for c in cols if c in minute.columns]].copy()
        temp[metric] = to_numeric(temp[metric])
        temp = temp.dropna(subset=[metric, "condition_code"])
        if temp.empty:
            return None
        session_values = temp.groupby(["session_id", "condition_code"], as_index=False)[metric].mean()
        condition_order = [c for c in CONDITION_ORDER if c in session_values["condition_code"].astype(str).unique()]
        if not condition_order:
            return None
        fig, ax = plt.subplots(figsize=self._figsize("wide_single_short"))
        x = np.arange(len(condition_order), dtype=float)
        rng = np.random.default_rng(42)
        violin_groups = []
        violin_positions = []
        for idx, cond in enumerate(condition_order):
            vals = to_numeric(session_values.loc[session_values["condition_code"].astype(str) == cond, metric]).dropna()
            if vals.empty:
                continue
            arr = vals.to_numpy(dtype=float)
            unique_n = int(len(np.unique(np.round(arr, 6))))
            if len(arr) >= 5 and unique_n >= 3:
                violin_groups.append(arr)
                violin_positions.append(float(x[idx]))
        if violin_groups:
            vp = ax.violinplot(
                violin_groups,
                positions=violin_positions,
                widths=0.72,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body, xpos in zip(vp["bodies"], violin_positions):
                cond = condition_order[int(round(xpos))]
                cond_color = CONDITION_COLORS.get(cond, "#475569")
                body.set_facecolor(cond_color)
                body.set_edgecolor(cond_color)
                body.set_alpha(0.14)
        count_annotations: list[tuple[float, int]] = []
        for idx, cond in enumerate(condition_order):
            vals = to_numeric(session_values.loc[session_values["condition_code"].astype(str) == cond, metric]).dropna()
            if vals.empty:
                continue
            arr = vals.to_numpy(dtype=float)
            cond_color = CONDITION_COLORS.get(cond, "#475569")
            xpos = float(x[idx])
            jitter = rng.uniform(-0.08, 0.08, size=len(arr))
            ax.scatter(np.full(len(arr), xpos) + jitter, arr, s=20, alpha=0.4, color=cond_color, edgecolors="none", zorder=3)
            q25 = float(np.quantile(arr, 0.25))
            q75 = float(np.quantile(arr, 0.75))
            med = float(np.median(arr))
            ax.vlines(xpos, q25, q75, color=cond_color, linewidth=2.1, zorder=4)
            ax.hlines(med, xpos - 0.17, xpos + 0.17, color=cond_color, linewidth=2.2, zorder=5)
            count_annotations.append((xpos, len(arr)))
        ax.set_xticks(x)
        ax.set_xticklabels(condition_order)
        ax.set_xlabel(self._condition_axis_label())
        ax.set_ylabel(self._axis_label(metric))
        ax.grid(True, axis="y")
        self._apply_discrete_y_axis_matplotlib(ax, session_values[metric], metric)
        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min if y_max > y_min else 1.0
        y_tail = y_min - 0.06 * y_span
        for xpos, n_obs in count_annotations:
            ax.text(xpos, y_tail, f"n={n_obs}", ha="center", va="top", fontsize=8, color="#475569", clip_on=False)
        ax.set_ylim(y_min - 0.12 * y_span, y_max + 0.04 * y_span)
        fig.tight_layout()
        return fig

    def _fig_cohort_condition_trace(self, minute: pd.DataFrame, metric: str):
        if minute.empty or metric not in minute.columns:
            return None
        if self._is_sparse_observation_channel(metric):
            summary = self._cohort_condition_phase_summary(minute, metric)
            if summary.empty:
                return None
            phase_order = [p for p in PHASE_ORDER if p in summary["protocol_phase"].astype(str).unique()]
            if not phase_order:
                return None
            if len(phase_order) == 1:
                return self._fig_cohort_condition_comparison_summary(minute, metric)
            fig, ax = plt.subplots(figsize=self._figsize("wide_single"))
            x = np.arange(len(phase_order), dtype=float)
            for cond in [c for c in CONDITION_ORDER if c in summary["condition_code"].astype(str).unique()]:
                cur = summary.loc[summary["condition_code"].astype(str) == cond].copy()
                cur["protocol_phase"] = cur["protocol_phase"].astype(str)
                cur = cur.set_index("protocol_phase").reindex(phase_order).reset_index()
                y = to_numeric(cur["median"])
                q25 = to_numeric(cur["q25"])
                q75 = to_numeric(cur["q75"])
                valid = y.notna()
                if not valid.any():
                    continue
                xv = x[valid.to_numpy()]
                yv = y.loc[valid].to_numpy(dtype=float)
                q25v = q25.loc[valid].to_numpy(dtype=float)
                q75v = q75.loc[valid].to_numpy(dtype=float)
                ax.fill_between(xv, q25v, q75v, color=CONDITION_COLORS[cond], alpha=0.18)
                ax.plot(xv, yv, color=CONDITION_COLORS[cond], lw=2, marker="o", ms=4, label=cond)
            ax.set_xticks(x)
            ax.set_xticklabels([PHASE_ABBR.get(p, p[:3].upper()) for p in phase_order])
            ax.set_ylabel(self._axis_label(metric))
            ax.set_xlabel(self._phase_axis_label())
            ax.grid(True, axis="y")
            self._apply_discrete_y_axis_matplotlib(ax, minute[metric], metric)
            self._place_condition_legend(ax)
            fig.tight_layout()
            return fig
        fig, ax = plt.subplots(figsize=self._figsize("wide_single"))
        self._add_phase_spans(ax, minute.drop_duplicates(subset=["minute_index", "protocol_phase"]))
        self._cohort_band(minute, metric, ax)
        ax.set_ylabel(self._axis_label(metric))
        ax.set_xlabel(self._time_axis_label())
        self._place_topbar_legend(ax)
        fig.tight_layout()
        return fig

    def _phase_condition_ticklabels(self, values: list[str]) -> list[str]:
        labels = []
        for value in values:
            phase, _, cond = str(value).partition(" | ")
            phase_label = PHASE_ABBR.get(phase, phase[:3].upper())
            labels.append(f"{phase_label}\n{cond}")
        return labels

    def _fig_cohort_contrasts(self, contrasts: pd.DataFrame, ev: dict):
        if contrasts.empty:
            return None
        contrasts = contrasts.loc[contrasts["protocol_phase"].astype(str) != "acclimation"].copy()
        if contrasts.empty:
            return None
        metrics = [m for m in ["thermal_comfort", "empatica_hr_mean_bpm", "biopac_temp_chest_mean_C", "indoor_air_velocity_mean_m_s"] if m in contrasts["metric"].unique()]
        if not metrics:
            return None
        fig, axes = plt.subplots(2, 2, figsize=self._figsize("two_by_two"))
        axes = axes.ravel()
        panel_notes: list[str] = []
        panel_positions = ["Top left", "Top right", "Bottom left", "Bottom right"]
        axis_phases = self._comparison_phase_sequence(contrasts["protocol_phase"].astype(str).unique())
        for ax, metric in zip(axes, metrics):
            d = contrasts.loc[contrasts["metric"] == metric]
            eligible = d.loc[d["eligible"] == 1]
            descriptive = d.loc[d["eligible"] == 0]
            if not descriptive.empty:
                tmp = descriptive.groupby("protocol_phase")["mean_difference"].mean().reindex(axis_phases)
                if self._is_sparse_observation_channel(metric):
                    ax.scatter(range(len(tmp)), tmp.values, color="#94a3b8", s=40, alpha=0.9, label="Descriptive only")
                else:
                    ax.plot(range(len(tmp)), tmp.values, color="#94a3b8", lw=1.5, marker="o", label="Descriptive only")
            if not eligible.empty:
                tmp = eligible.groupby("protocol_phase")["mean_difference"].mean().reindex(axis_phases)
                if self._is_sparse_observation_channel(metric):
                    ax.scatter(range(len(tmp)), tmp.values, color="#2563eb", s=50, alpha=0.95, label="Eligible")
                else:
                    ax.plot(range(len(tmp)), tmp.values, color="#2563eb", lw=2.5, marker="o", label="Eligible")
            ax.axhline(0, color="#dbe4ee", lw=0.85, ls="--", zorder=0)
            ax.set_xticks(range(len(axis_phases)))
            ax.set_xticklabels([PHASE_ABBR.get(x, x[:3].upper()) for x in axis_phases])
            ax.set_ylabel(self._axis_label(metric))
            ax.text(0.01, 0.98, FEATURE_LABELS.get(metric, metric), transform=ax.transAxes, ha="left", va="top", fontsize=10.5, fontweight="bold", color="#172033")
            panel_notes.append(f"{panel_positions[len(panel_notes)]} shows matched condition differences for {FEATURE_LABELS.get(metric, metric)}.")
            self._apply_discrete_y_axis_matplotlib(ax, d["mean_difference"], metric)
            ax.legend(frameon=False, fontsize=10)
        for ax in axes[len(metrics):]:
            ax.axis("off")
        fig._cltr_panel_notes = panel_notes
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _cohort_top_contrast_register(self, contrasts: pd.DataFrame) -> pd.DataFrame:
        if contrasts.empty:
            return pd.DataFrame()
        keep = contrasts.copy()
        if "eligible" in keep.columns:
            keep = keep.loc[keep["eligible"] == 1].copy()
        if keep.empty:
            return pd.DataFrame()
        sort_cols = ["significant_fdr", "p_value_fdr", "primary_p_value", "n_pairs"]
        ascending = [False, True, True, False]
        existing_cols = [c for c in sort_cols if c in keep.columns]
        keep = keep.sort_values(existing_cols, ascending=ascending[: len(existing_cols)])
        return keep.head(24).reset_index(drop=True)

    def _contrast_family(self, metric: object) -> str:
        text = str(metric)
        if text in {
            "thermal_comfort",
            "thermal_sensation",
            "thermal_preference",
            "thermal_pleasure",
            "visual_comfort",
            "room_comfort",
            "visual_sensation",
            "color_sensation",
            "sound_comfort_dbA",
            "air_quality_comfort",
        }:
            return "Questionnaire and perception"
        if text.startswith(("indoor_", "outdoor_")):
            return "Environment"
        if text.startswith("fan_"):
            return "Behavior and control"
        if text.startswith(("empatica_", "biopac_")):
            return "Wearable and physiology"
        if any(token in text for token in ["master_", "thermal_gradient", "thermal_state_index", "_delta_"]):
            return "Derived thermal and deltas"
        return "Other"

    def _cohort_balanced_contrast_register(self, contrasts: pd.DataFrame, *, max_rows: int = 24, per_family: int = 4) -> pd.DataFrame:
        if contrasts.empty:
            return pd.DataFrame()
        keep = contrasts.copy()
        if "eligible" in keep.columns:
            keep = keep.loc[keep["eligible"] == 1].copy()
        if keep.empty:
            return pd.DataFrame()
        keep["contrast_family"] = keep["metric"].map(self._contrast_family)
        sort_cols = [c for c in ["significant_fdr", "p_value_fdr", "primary_p_value", "n_pairs"] if c in keep.columns]
        ascending = [False, True, True, False][: len(sort_cols)]
        keep = keep.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        selected_parts: list[pd.DataFrame] = []
        selected_keys: set[tuple[str, str, str]] = set()
        for family, d in keep.groupby("contrast_family", sort=False):
            family_rows = []
            for metric, dm in d.groupby("metric", sort=False):
                family_rows.append(dm.head(1))
            if family_rows:
                family_pick = pd.concat(family_rows, ignore_index=True).sort_values(sort_cols, ascending=ascending).head(per_family)
                selected_parts.append(family_pick)
                for row in family_pick.itertuples(index=False):
                    selected_keys.add((str(row.metric), str(row.protocol_phase), str(row.comparison)))
        selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else keep.iloc[0:0].copy()
        if len(selected) < max_rows:
            remainder = keep.loc[
                ~keep.apply(lambda r: (str(r["metric"]), str(r["protocol_phase"]), str(r["comparison"])) in selected_keys, axis=1)
            ].copy()
            if not remainder.empty:
                fill = remainder.head(max_rows - len(selected))
                selected = pd.concat([selected, fill], ignore_index=True)
        selected = selected.sort_values(sort_cols, ascending=ascending).head(max_rows).reset_index(drop=True)
        return selected

    def _mixed_effects_register(self, mixed_effects: pd.DataFrame) -> pd.DataFrame:
        if mixed_effects.empty:
            return pd.DataFrame()
        keep = self._mixed_effects_fixed_only(mixed_effects)
        if "term" in keep.columns:
            keep["term_reading"] = keep["term"].map(self._mixed_effect_term_label)
        if "significant_fdr" in keep.columns:
            keep = keep.sort_values(["significant_fdr", "p_value_fdr", "metric"], ascending=[False, True, True])
        return keep.head(30).reset_index(drop=True)

    def _mixed_effects_fixed_only(self, mixed_effects: pd.DataFrame) -> pd.DataFrame:
        keep = mixed_effects.copy()
        if "term" not in keep.columns:
            return keep
        term_text = keep["term"].astype(str)
        return keep.loc[
            ~term_text.str.contains(r"(?:^Intercept$)|(?:^Group Var$)|(?:^| )Var$| Cov$", regex=True)
        ].copy()

    def _mixed_effect_term_label(self, term: object) -> str:
        parts = [self._mixed_effect_component_label(part) for part in str(term).split(":")]
        if not parts:
            return str(term)
        if len(parts) == 1:
            return parts[0]["main"]
        left = parts[0]["contrast"]
        right = parts[1]["context"]
        if len(parts) == 2:
            return f"Interaction: the {left} changes across {right}"
        extra = ", ".join(part["context"] for part in parts[2:])
        return f"Interaction: the {left} changes across {right}, jointly with {extra}"

    def _mixed_effect_component_label(self, term_part: object) -> dict[str, str]:
        text = str(term_part)
        mapping = {
            "C(illuminance_level)[T.DIM]": {
                "main": "Illuminance main effect: DIM vs BRI",
                "contrast": "DIM-BRI illuminance contrast",
                "context": "illuminance level (DIM vs BRI)",
            },
            "C(time_of_day)[T.MOR]": {
                "main": "Time-of-day main effect: MOR vs MID",
                "contrast": "MOR-MID time-of-day contrast",
                "context": "time of day (MOR vs MID)",
            },
            "C(protocol_phase)[T.fan_free_control]": {
                "main": "Phase main effect: FFC vs FCS",
                "contrast": "FFC-FCS phase contrast",
                "context": "phase (FFC vs FCS)",
            },
            "C(protocol_phase)[T.overall_comfort]": {
                "main": "Phase main effect: OC vs FCS",
                "contrast": "OC-FCS phase contrast",
                "context": "phase (OC vs FCS)",
            },
            "C(protocol_phase)[T.skin_rewarming]": {
                "main": "Phase main effect: SR vs FCS",
                "contrast": "SR-FCS phase contrast",
                "context": "phase (SR vs FCS)",
            },
            "C(protocol_phase)[T.steady_state]": {
                "main": "Phase main effect: SS vs FCS",
                "contrast": "SS-FCS phase contrast",
                "context": "phase (SS vs FCS)",
            },
        }
        return mapping.get(
            text,
            {
                "main": text,
                "contrast": text,
                "context": text,
            },
        )

    def _mixed_effect_term_tokens(self, terms: list[str]) -> str:
        phase_map = {
            "fan_free_control": "FFC",
            "overall_comfort": "OC",
            "skin_rewarming": "SR",
            "steady_state": "SS",
        }
        tokens: list[str] = []
        for term in terms:
            parts: list[str] = []
            if "illuminance_level" in term:
                parts.append("DIM")
            if "time_of_day" in term:
                parts.append("MOR")
            for raw, abbr in phase_map.items():
                if raw in term:
                    parts.append(abbr)
            if not parts:
                parts.append(str(term))
            tokens.append("x".join(parts))
        ordered: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
        return fill(", ".join(ordered), width=16)

    def _mixed_effects_diagnostics_register(self, diagnostics: pd.DataFrame) -> pd.DataFrame:
        if diagnostics.empty:
            return pd.DataFrame()
        keep = diagnostics.copy()
        status_map = {
            "retained": "Retained",
            "retained_with_fit_issue": "Retained with fit issue",
            "fit_failed": "Fit failed",
            "skipped_insufficient_participants": "Skipped: insufficient participants",
            "skipped_insufficient_design_variation": "Skipped: insufficient design variation",
            "no_fixed_effect_terms": "No fixed effects retained",
        }
        model_spec_map = {
            "condition_time_phase_random_phase_slope": "Condition x time x phase with participant random phase slopes",
            "condition_time_phase_random_intercept": "Condition x time + phase with participant random intercept",
            "condition_time_random_intercept": "Condition x time with participant random intercept",
        }
        if "status" in keep.columns:
            keep["status"] = keep["status"].astype(str).map(lambda x: status_map.get(x, x.replace("_", " ").title()))
        if "model_spec" in keep.columns:
            keep["model_spec"] = keep["model_spec"].astype(str).map(lambda x: model_spec_map.get(x, x.replace("_", " ").title()) if x else "")
        if "fit_converged" in keep.columns:
            keep["fit_converged"] = keep["fit_converged"].map({1: "Yes", 0: "No"}).fillna(keep["fit_converged"])
        if "warning_summary" in keep.columns:
            keep["warning_summary"] = keep["warning_summary"].fillna("").replace("", "No reported warning")
        if "status" in keep.columns:
            keep["_status_rank"] = keep["status"].astype(str).str.startswith("Retained").map({True: 0, False: 1}).fillna(1)
            keep = keep.sort_values(["_status_rank", "warning_count", "metric"], ascending=[True, False, True]).drop(columns="_status_rank")
        return keep.reset_index(drop=True)

    def _fig_preprocessing_qc_summary(self, qc: pd.DataFrame):
        if qc.empty:
            return None
        top = qc.sort_values("valid_fraction", ascending=True).tail(7).copy()
        fig, ax = plt.subplots(figsize=self._figsize("wide_single_short"))
        y = np.arange(len(top))
        ax.barh(y, top["valid_fraction"], color="#0f766e", alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([str(x).replace("quality_", "").replace("_", " ") for x in top["metric"]])
        ax.set_xlabel("Valid-minute fraction")
        ax.set_xlim(0, 1.08)
        for idx, row in enumerate(top.itertuples(index=False)):
            ax.text(min(float(row.valid_fraction) + 0.02, 1.04), idx, f"{float(row.valid_fraction):.2f}", va="center", fontsize=11)
        fig.tight_layout(rect=(0, 0, 0.98, 0.96))
        return fig

    def _fig_predictive_benchmarks(self, benchmarks: pd.DataFrame):
        if benchmarks.empty:
            return None
        plot_df = benchmarks.copy()
        plot_df["row_label"] = (
            plot_df["task"].astype(str).str.replace("_", " ").str.title()
            + "\n"
            + plot_df["feature_set"].astype(str).str.replace("_", " ").str.title()
            + "\n"
            + plot_df["validation_scheme"].astype(str).str.replace("_", " ").str.title()
        )
        plot_df = plot_df.sort_values("balanced_accuracy_mean", ascending=True).tail(18)
        fig, ax = plt.subplots(figsize=self._figsize("wide_single_tall"))
        palette = {
            "environment_only": "#2563eb",
            "physiology_only": "#b91c1c",
            "fused_multimodal": "#0f766e",
        }
        y = np.arange(len(plot_df))
        ax.barh(y, plot_df["balanced_accuracy_mean"], color=[palette.get(str(x), "#475569") for x in plot_df["feature_set"]], alpha=0.92)
        for idx, row in enumerate(plot_df.itertuples(index=False)):
            spread = f" +/- {float(row.balanced_accuracy_sd):.2f}" if pd.notna(row.balanced_accuracy_sd) else ""
            ax.text(min(float(row.balanced_accuracy_mean) + 0.02, 0.98), idx, f"{float(row.balanced_accuracy_mean):.2f}{spread}", va="center", fontsize=10.5)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["row_label"].astype(str))
        ax.set_xlim(0, 1)
        ax.set_xlabel("Balanced accuracy under explicit holdout validation")
        ax.text(0.01, 0.98, "Comfort-state benchmark landscape", transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold", color="#172033")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_lag_response_register(self, lag_register: pd.DataFrame, lag_profile: pd.DataFrame):
        if lag_register.empty:
            return None
        d = lag_register.copy().head(12)
        d["row_label"] = d["predictor"].astype(str).map(lambda x: FEATURE_LABELS.get(x, x)) + " -> " + d["target"].astype(str).map(lambda x: FEATURE_LABELS.get(x, x))
        d["row_label_wrapped"] = d["predictor"].astype(str).map(lambda x: FEATURE_LABELS.get(x, x)) + "\n-> " + d["target"].astype(str).map(lambda x: FEATURE_LABELS.get(x, x))
        d = d.sort_values(["evidence_grade", "median_abs_spearman_r", "best_lag_minutes"], ascending=[True, False, True])
        fig = plt.figure(figsize=(14.8, 6.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 0.95], wspace=0.15)
        ax_lag = fig.add_subplot(gs[0, 0])
        ax_strength = fig.add_subplot(gs[0, 1], sharey=ax_lag)
        y = np.arange(len(d))
        grade_colors = {"A": "#0f766e", "B": "#2563eb", "C": "#b9770e"}
        sizes = 80 + 4.0 * to_numeric(d["median_pairs_per_session"]).fillna(0).clip(lower=0)

        profile = lag_profile.copy()
        if not profile.empty:
            profile["row_key"] = profile["predictor"].astype(str) + " -> " + profile["target"].astype(str)
        size_legend_vals: list[float] = []
        if not profile.empty:
            row_map = {row.row_label: idx for idx, row in enumerate(d.itertuples(index=False))}
            profile["row_label"] = profile["predictor"].astype(str).map(lambda x: FEATURE_LABELS.get(x, x)) + " -> " + profile["target"].astype(str).map(lambda x: FEATURE_LABELS.get(x, x))
            profile = profile.loc[profile["row_label"].isin(row_map)].copy()
            profile["y"] = profile["row_label"].map(row_map)
            profile = profile.sort_values(["y", "lag_minutes"]).reset_index(drop=True)
            max_abs = max(0.2, float(to_numeric(profile["median_abs_spearman_r"]).max()))
            size_scale = 30 + 280 * (to_numeric(profile["median_abs_spearman_r"]).fillna(0) / max_abs)
            size_legend_vals = [round(max_abs * frac, 2) for frac in [0.33, 0.66, 1.0]]
            for _, grp in profile.groupby("row_label", sort=False):
                ax_lag.plot(
                    to_numeric(grp["lag_minutes"]),
                    grp["y"],
                    color="#cbd5e1",
                    lw=1.0,
                    zorder=1,
                )
            sc = ax_lag.scatter(
                to_numeric(profile["lag_minutes"]).fillna(0),
                profile["y"],
                s=size_scale,
                c=to_numeric(profile["same_sign_fraction"]).fillna(0.0),
                cmap="Blues",
                vmin=0.5,
                vmax=1.0,
                alpha=0.95,
                edgecolors="white",
                linewidths=0.9,
                zorder=2,
            )
            best = profile.loc[to_numeric(profile["is_best_lag"]).fillna(0).astype(int) == 1].copy()
            if not best.empty:
                ax_lag.scatter(
                    to_numeric(best["lag_minutes"]).fillna(0),
                    best["y"],
                    s=(30 + 280 * (to_numeric(best["median_abs_spearman_r"]).fillna(0) / max_abs)) + 60,
                    facecolors="none",
                    edgecolors=[grade_colors.get(str(x), "#475569") for x in best["evidence_grade"]],
                    linewidths=2.2,
                    zorder=4,
                )
            cbar = plt.colorbar(sc, ax=ax_lag, fraction=0.03, pad=0.02)
            cbar.set_label("Same-sign fraction")
        else:
            ax_lag.scatter(
                to_numeric(d["best_lag_minutes"]).fillna(0),
                y,
                s=sizes,
                c=[grade_colors.get(str(x), "#64748b") for x in d["evidence_grade"]],
                alpha=0.92,
                edgecolors="white",
                linewidths=1.1,
                zorder=3,
            )
        ax_lag.set_yticks(y)
        ax_lag.set_yticklabels(d["row_label_wrapped"].astype(str), fontsize=9.3, linespacing=1.0)
        ax_lag.set_xlabel("Tested lag window (min)")
        ax_lag.set_xlim(-3, max(68, float(to_numeric(d["best_lag_minutes"]).max()) + 12))
        ax_lag.grid(True, axis="x", alpha=0.2)
        ax_lag.set_axisbelow(True)
        ax_lag.spines["top"].set_visible(False)
        ax_lag.spines["right"].set_visible(False)
        lag_xmax = float(ax_lag.get_xlim()[1])
        lag_xmin = float(ax_lag.get_xlim()[0])
        for idx, row in enumerate(d.itertuples(index=False)):
            label_x = float(row.best_lag_minutes) + 2.2
            ha = "left"
            if float(row.best_lag_minutes) >= 55 or label_x > lag_xmax - 9.5:
                label_x = max(float(row.best_lag_minutes) - 3.0, lag_xmin + 11.5)
                ha = "right"
            ax_lag.text(
                label_x,
                idx,
                f"best={int(row.best_lag_minutes)} min | grade {row.evidence_grade}",
                ha=ha,
                va="center",
                fontsize=8.9,
                color="#172033",
                bbox={
                    "boxstyle": "round,pad=0.18,rounding_size=0.1",
                    "facecolor": "#ffffff",
                    "edgecolor": "#e2e8f0",
                    "linewidth": 0.7,
                },
            )

        ax_strength.barh(y, d["median_abs_spearman_r"], color=[grade_colors.get(str(x), "#64748b") for x in d["evidence_grade"]], alpha=0.88, height=0.66)
        strength_max = float(to_numeric(d["median_abs_spearman_r"]).max())
        ax_strength.set_xlim(0, max(0.52, strength_max * 2.05))
        ax_strength.set_xlabel("Median |Spearman r|")
        ax_strength.grid(True, axis="x", alpha=0.18)
        ax_strength.set_axisbelow(True)
        ax_strength.spines["top"].set_visible(False)
        ax_strength.spines["right"].set_visible(False)
        ax_strength.spines["left"].set_visible(False)
        ax_strength.tick_params(axis="y", left=False, labelleft=False)
        for idx, row in enumerate(d.itertuples(index=False)):
            ax_strength.text(
                min(float(row.median_abs_spearman_r) + 0.018, ax_strength.get_xlim()[1] - 0.02),
                idx,
                f"r={float(row.median_spearman_r):.2f} | sign={float(row.same_sign_fraction):.2f}\nsess={int(row.n_sessions)}",
                va="center",
                fontsize=8.5,
                color="#475569",
                linespacing=1.04,
                bbox={
                    "boxstyle": "round,pad=0.16,rounding_size=0.1",
                    "facecolor": "#ffffff",
                    "edgecolor": "#e2e8f0",
                    "linewidth": 0.7,
                },
            )

        legend_handles = [
            Patch(facecolor="#0f766e", edgecolor="none", label="Grade A"),
            Patch(facecolor="#2563eb", edgecolor="none", label="Grade B"),
            Patch(facecolor="#b9770e", edgecolor="none", label="Grade C"),
        ]
        self._place_topbar_legend(ax_lag, legend_handles, y=1.045)
        if size_legend_vals:
            size_handles = [
                plt.scatter([], [], s=30 + 280 * (val / max(size_legend_vals)), color="#2563eb", alpha=0.8, edgecolors="white", linewidths=0.8, label=f"|r|={val:.2f}")
                for val in size_legend_vals
            ]
            self._place_topbar_legend(ax_strength, size_handles, y=1.045)
        fig._cltr_panel_notes = [
            "Left|Lag sweep and best-retained timing|Each row shows the tested lag window for one driver-response pair. Marker size reflects median absolute lagged association, marker color shows same-sign fraction across sessions, and the outlined marker identifies the best retained lag with its evidence grade.",
            "Right|Association strength and stability|Bars show median absolute lagged association, while the annotation reports signed median Spearman correlation, same-sign fraction across sessions, and the number of supporting sessions.",
        ]
        fig.subplots_adjust(left=0.28, right=0.985, top=0.87, bottom=0.12, wspace=0.18)
        return fig

    def _fig_cohort_agreement(self, agreement: pd.DataFrame):
        if agreement.empty:
            return None
        metrics = ["heart_rate", "eda", "temperature"]
        fig, axes = plt.subplots(len(metrics), 1, figsize=self._figsize("three_panel_stack"))
        if len(metrics) == 1:
            axes = [axes]
        panel_notes: list[str] = []
        panel_positions = ["Top", "Middle", "Bottom"]
        for ax, metric in zip(axes, metrics):
            d = agreement.loc[agreement["metric"] == metric]
            if d.empty:
                ax.axis("off")
                continue
            colors = d["eligible"].map({1: "#2563eb", 0: "#94a3b8"}).fillna("#94a3b8")
            ax.scatter(d["spearman_r"], d["mae"], c=colors, s=45, alpha=0.8)
            for _, row in d.iterrows():
                ax.text(row["spearman_r"] if pd.notna(row["spearman_r"]) else 0, row["mae"] if pd.notna(row["mae"]) else 0, str(row["session_id"]), fontsize=9)
            ax.axvline(0, color="#dbe4ee", lw=0.85, ls="--", zorder=0)
            ax.set_xlabel("Spearman r")
            ax.set_ylabel(f"{metric.replace('_', ' ').title()} MAE")
            panel_notes.append(f"{panel_positions[len(panel_notes)]} shows agreement across sessions for {metric.replace('_', ' ')}.")
            ax.grid(True, axis="both", alpha=0.25)
        fig._cltr_panel_notes = panel_notes
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_cohort_primary_endpoints_raw(self, endpoints: pd.DataFrame, metrics_override: list[str] | None = None):
        if endpoints.empty:
            return None
        preferred = [
            "thermal_comfort",
            "thermal_sensation",
            "indoor_air_velocity_mean_m_s",
            "empatica_eda_mean_uS",
            "empatica_temp_mean_C",
            "biopac_hr_mean_bpm",
            "biopac_eda_mean_uS",
            "biopac_temp_chest_mean_C",
        ]
        selected = metrics_override or preferred
        metrics = [m for m in selected if m in endpoints["metric"].unique()]
        if not metrics:
            return None
        endpoints = endpoints.loc[endpoints["protocol_phase"].astype(str) != "acclimation"].copy()
        if endpoints.empty:
            return None
        fig, axes = plt.subplots(len(metrics), 1, figsize=(self._figsize("wide_single")[0] + 0.6, 2.05 * len(metrics) + 0.9), sharex=True)
        panel_notes: list[str] = []
        panel_positions = ["Top", "Upper middle", "Center", "Lower middle", "Bottom", "Panel 6", "Panel 7", "Panel 8"]
        if len(metrics) == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            d = endpoints.loc[endpoints["metric"] == metric].copy()
            if d.empty:
                ax.axis("off")
                continue
            d["phase_condition"] = d["protocol_phase"].astype(str).map(lambda x: PHASE_ABBR.get(str(x), str(x)[:3].upper())) + "\n" + d["condition_code"].astype(str)
            ax.bar(range(len(d)), d["mean_value"], color=[CONDITION_COLORS.get(str(x), "#475569") for x in d["condition_code"]])
            ax.set_ylabel("")
            ax.set_title(
                f"{FEATURE_LABELS.get(metric, metric)} ({self._compact_axis_label(metric)})",
                loc="left",
                fontsize=10.5,
                fontweight="bold",
                color="#172033",
                pad=8,
            )
            panel_notes.append(f"{panel_positions[len(panel_notes)]} shows raw means for {FEATURE_LABELS.get(metric, metric)}.")
            ax.axhline(0, color="#dbe4ee", lw=0.85, ls="--", zorder=0)
            ax.grid(True, axis="y")
            self._apply_discrete_y_axis_matplotlib(ax, d["mean_value"], metric)
        axes[-1].set_xticks(range(len(d)))
        axes[-1].set_xticklabels(list(d["phase_condition"]), rotation=90, ha="center", va="top")
        fig._cltr_panel_notes = panel_notes
        fig.tight_layout(rect=(0.03, 0.02, 0.99, 0.98))
        return fig

    def _fig_cohort_primary_endpoints(self, endpoints: pd.DataFrame):
        if endpoints.empty:
            return None
        preferred = [
            "thermal_comfort",
            "thermal_sensation",
            "indoor_air_velocity_mean_m_s",
            "empatica_eda_mean_uS",
            "empatica_temp_mean_C",
            "biopac_hr_mean_bpm",
            "biopac_eda_mean_uS",
            "biopac_temp_chest_mean_C",
        ]
        metrics = [m for m in preferred if m in endpoints["metric"].unique()]
        if not metrics:
            return None
        tmp = endpoints.loc[endpoints["metric"].isin(metrics) & (endpoints["protocol_phase"].astype(str) != "acclimation")].copy()
        if tmp.empty:
            return None
        phase_sequence = [p for p in self._comparison_phase_sequence(tmp["protocol_phase"]) if str(p) != "acclimation"]
        condition_sequence = [c for c in CONDITION_ORDER if c in set(tmp["condition_code"].astype(str))]
        if not condition_sequence:
            condition_sequence = sorted(tmp["condition_code"].astype(str).dropna().unique().tolist())
        phase_order = {str(phase): idx for idx, phase in enumerate(phase_sequence)}
        condition_order = {str(condition): idx for idx, condition in enumerate(condition_sequence)}
        tmp["phase_condition"] = tmp["protocol_phase"].astype(str).map(lambda x: PHASE_ABBR.get(str(x), str(x)[:3].upper())) + "\n" + tmp["condition_code"].astype(str)
        pivot = tmp.pivot(index="metric", columns="phase_condition", values="mean_value")
        pivot = pivot.reindex(metrics)
        ordered_columns = [
            f"{PHASE_ABBR.get(str(phase), str(phase)[:3].upper())}\n{condition}"
            for phase in phase_sequence
            for condition in condition_sequence
            if f"{PHASE_ABBR.get(str(phase), str(phase)[:3].upper())}\n{condition}" in pivot.columns
        ]
        if ordered_columns:
            pivot = pivot.reindex(columns=ordered_columns)
        z = pivot.apply(lambda col: (col - col.mean()) / col.std(ddof=0) if col.notna().sum() > 1 and col.std(ddof=0) > 0 else col * 0, axis=1)
        fig, ax = plt.subplots(figsize=self._figsize("matrix"))
        im = ax.imshow(z.values, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
        ax.grid(False)
        ax.set_yticks(range(len(z.index)))
        ax.set_yticklabels([FEATURE_LABELS.get(x, x) for x in z.index])
        ax.set_xticks(range(len(z.columns)))
        ax.set_xticklabels(list(z.columns), rotation=90, ha="center", va="top")
        ax.set_xlabel("Phase And Condition")
        for i in range(len(z.index)):
            for j in range(len(z.columns)):
                value = z.iloc[i, j]
                if pd.notna(value):
                    text_color = "#f8fafc" if abs(float(value)) >= 1.0 else "#172033"
                    ax.text(j, i, f"{float(value):.1f}", ha="center", va="center", fontsize=8.5, color=text_color)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Within-metric standardized mean")
        fig.tight_layout(rect=(0.03, 0.05, 0.99, 0.96))
        return fig

    def _fig_cohort_exploratory_landscape(self, summary: pd.DataFrame, condition_support: pd.DataFrame, feature_registry: pd.DataFrame | None = None):
        if summary.empty and condition_support.empty:
            return None
        fig, axes = plt.subplots(2, 2, figsize=(self._figsize("three_panel_row")[0], 8.8))
        axes = axes.ravel()
        panel_notes = [
            "Top left shows the derived features with the highest aligned support across the cohort.",
            "Top right shows the most variable supported derived features when variability can be estimated reliably.",
            "Bottom left shows the average aligned support fraction by feature domain.",
            "Bottom right shows condition-level support balance across questionnaire, wearable, and indoor streams.",
        ]
        for ax in axes:
            ax.grid(True, axis="x")
        if not summary.empty:
            scientific = summary.loc[~summary["feature"].astype(str).str.startswith("support_")].copy()
            top_cov = scientific.sort_values(["coverage_fraction", "n_non_null"], ascending=[False, False]).head(10)
            axes[0].barh(
                [FEATURE_LABELS.get(x, x) for x in top_cov["feature"][::-1]],
                top_cov["coverage_fraction"][::-1],
                color="#2563eb",
            )
            axes[0].set_xlim(0, 1)
            spread = scientific.loc[scientific["coverage_fraction"] >= 0.2].copy()
            spread["robust_cv"] = spread["iqr"] / spread["median"].abs().replace(0, np.nan)
            spread = spread.replace([np.inf, -np.inf], np.nan).dropna(subset=["robust_cv"]).sort_values("robust_cv", ascending=False).head(10)
            if not spread.empty:
                axes[1].barh(
                    [FEATURE_LABELS.get(x, x) for x in spread["feature"][::-1]],
                    spread["robust_cv"][::-1],
                    color="#b91c1c",
                )
            else:
                axes[1].axis("off")
            domain_source = summary.copy()
            if feature_registry is not None and not feature_registry.empty:
                registry_cols = ["feature", "prompt_response_fraction", "coverage_fraction"]
                registry_view = feature_registry[[c for c in registry_cols if c in feature_registry.columns]].drop_duplicates(subset=["feature"]).copy()
                domain_source = domain_source.merge(
                    registry_view,
                    on="feature",
                    how="left",
                    suffixes=("", "_registry"),
                )
                prompt_fraction = to_numeric(domain_source.get("prompt_response_fraction", pd.Series(dtype=float)))
                minute_fraction = to_numeric(domain_source.get("coverage_fraction_registry", pd.Series(dtype=float)))
                fallback_fraction = to_numeric(domain_source.get("coverage_fraction", pd.Series(dtype=float)))
                domain_source["policy_support_fraction"] = prompt_fraction.where(prompt_fraction.notna(), minute_fraction.where(minute_fraction.notna(), fallback_fraction))
            else:
                domain_source["policy_support_fraction"] = to_numeric(domain_source.get("coverage_fraction", pd.Series(dtype=float)))
            domain = domain_source.groupby("domain").agg(
                mean_support=("policy_support_fraction", "mean"),
                n_features=("feature", "count"),
            ).reset_index()
            axes[2].bar(domain["domain"], domain["mean_support"], color="#0f766e")
            axes[2].set_ylim(0, 1)
            axes[2].tick_params(axis="x", rotation=30)
        else:
            axes[0].axis("off")
            axes[1].axis("off")
            axes[2].axis("off")
        if not condition_support.empty:
            cond = condition_support.copy()
            x = np.arange(len(cond))
            width = 0.18
            axes[3].bar(x - 1.5 * width, cond.get("questionnaire_completeness__mean", pd.Series([np.nan] * len(cond))), width=width, label="Questionnaire event support", color="#111827")
            axes[3].bar(x - 0.5 * width, cond.get("empatica_fraction__mean", pd.Series([np.nan] * len(cond))), width=width, label="Empatica signal coverage", color="#2563eb")
            axes[3].bar(x + 0.5 * width, cond.get("biopac_fraction__mean", pd.Series([np.nan] * len(cond))), width=width, label="BIOPAC signal coverage", color="#dc2626")
            axes[3].bar(x + 1.5 * width, cond.get("indoor_fraction__mean", pd.Series([np.nan] * len(cond))), width=width, label="Indoor sensor coverage", color="#059669")
            axes[3].set_xticks(x)
            axes[3].set_xticklabels(cond["condition_code"], rotation=30, ha="right")
            axes[3].set_ylim(0, 1)
            axes[3].legend(frameon=False, fontsize=9.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12), columnspacing=1.4, handletextpad=0.6)
        else:
            axes[3].axis("off")
        fig._cltr_panel_notes = panel_notes
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_cohort_pattern_atlas(self, pattern_summary: pd.DataFrame, inventory: pd.DataFrame):
        if pattern_summary.empty and inventory.empty:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), gridspec_kw={"width_ratios": [1.2, 0.8]})
        fig._cltr_panel_notes = [
            "Left|Dominant recurring patterns across the cohort|Dominant recurring patterns across the cohort.",
            "Right|Strongest session-level motifs|The strongest session-level motifs.",
        ]
        recurrent_patterns = pd.DataFrame()
        if not pattern_summary.empty:
            recurrent_patterns = pattern_summary.loc[pattern_summary["dominant_phase"].astype(str) != "unknown"].copy()
            recurrent_patterns = recurrent_patterns.loc[recurrent_patterns["n_sessions"] >= 2].copy()
            if recurrent_patterns.empty:
                recurrent_patterns = pattern_summary.loc[pattern_summary["dominant_phase"].astype(str) != "unknown"].copy()
            top = recurrent_patterns.head(12).copy()
            top["pattern"] = top["dominant_phase"].map(lambda x: PHASE_ABBR.get(x, str(x)[:3].upper())) + " | " + top["direction"].str.upper()
            pivot = top.pivot_table(index="metric", columns="pattern", values="share_within_metric", aggfunc="max").fillna(0.0)
            pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
            im = axes[0].imshow(pivot.values, aspect="equal", cmap="YlOrRd", vmin=0, vmax=max(0.5, float(np.nanmax(pivot.values)) if pivot.size else 0.5))
            axes[0].grid(False)
            axes[0].text(0.01, 0.98, "Recurring cohort motifs", transform=axes[0].transAxes, ha="left", va="top", fontsize=11, fontweight="bold", color="#172033")
            axes[0].set_yticks(range(len(pivot.index)))
            axes[0].set_yticklabels([FEATURE_LABELS.get(x, x) for x in pivot.index])
            axes[0].set_xticks(range(len(pivot.columns)))
            axes[0].set_xticklabels(pivot.columns, rotation=45, ha="right")
            for yi, metric in enumerate(pivot.index):
                for xi, pattern in enumerate(pivot.columns):
                    val = float(pivot.loc[metric, pattern])
                    if val > 0:
                        axes[0].text(xi, yi, f"{val:.2f}", ha="center", va="center", fontsize=9.5, color="#3f2a0a")
            plt.colorbar(im, ax=axes[0], shrink=0.8, label="Share of sessions within metric")
        else:
            axes[0].axis("off")
        if not inventory.empty:
            display_inventory = inventory.loc[inventory["dominant_phase"].astype(str) != "unknown"].copy()
            if not recurrent_patterns.empty:
                recurrent_keys = recurrent_patterns.loc[:, ["metric", "dominant_phase", "direction"]].drop_duplicates()
                display_inventory = display_inventory.merge(recurrent_keys, on=["metric", "dominant_phase", "direction"], how="inner")
            if display_inventory.empty:
                display_inventory = inventory.loc[inventory["dominant_phase"].astype(str) != "unknown"].copy()
            top_sessions = display_inventory.sort_values(["pattern_strength", "abs_delta"], ascending=[False, False]).head(8).copy()
            labels = [f"{row.session_id}\n{FEATURE_LABELS.get(row.metric, row.metric)}" for row in top_sessions.itertuples()]
            axes[1].barh(labels[::-1], top_sessions["pattern_strength"][::-1], color="#7c3aed")
            axes[1].text(0.01, 0.98, "Strongest session-level motifs", transform=axes[1].transAxes, ha="left", va="top", fontsize=11, fontweight="bold", color="#172033")
            for idx, row in enumerate(top_sessions.iloc[::-1].itertuples()):
                baseline_info = {
                    "phase": "acclimation" if self._uses_acc_assumption(str(row.metric)) and str(row.baseline_phase) != "acclimation" else str(row.baseline_phase),
                    "source_phase": str(row.baseline_phase),
                    "assumed": bool(self._uses_acc_assumption(str(row.metric)) and str(row.baseline_phase) != "acclimation"),
                }
                axes[1].text(
                    float(row.pattern_strength) + 0.01,
                    idx,
                    f"base={self._baseline_phase_abbr(baseline_info)} | {PHASE_ABBR.get(row.dominant_phase, row.dominant_phase[:3].upper())} {row.direction} | c={row.consistency:.2f}",
                    va="center",
                    fontsize=10,
                    color="#475569",
                )
        else:
            axes[1].axis("off")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_cohort_participant_profile_atlas(self, profiles: pd.DataFrame):
        if profiles.empty:
            return None
        metrics = [
            "thermal_comfort",
            "empatica_hr_mean_bpm",
            "biopac_temp_chest_mean_C",
            "master_dpg_C",
        ]
        metrics = [metric for metric in metrics if metric in profiles.columns]
        if not metrics:
            return None
        participants = sorted(profiles["participant_id"].astype(str).unique().tolist())
        conditions = [cond for cond in CONDITION_ORDER if cond in profiles["condition_code"].astype(str).unique()]
        fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.2))
        axes = axes.ravel()
        panel_notes: list[str] = []
        panel_positions = ["Top left", "Top right", "Bottom left", "Bottom right"]
        for ax, metric in zip(axes, metrics):
            pivot = profiles.pivot_table(index="participant_id", columns="condition_code", values=metric, aggfunc="mean")
            pivot = pivot.reindex(index=participants, columns=conditions)
            values = pivot.to_numpy(dtype=float)
            im = ax.imshow(values, aspect="auto", cmap="coolwarm")
            ax.grid(False)
            ax.text(0.01, 0.98, FEATURE_LABELS.get(metric, metric), transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold", color="#172033")
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions, rotation=30, ha="right")
            step = max(1, len(participants) // 10)
            yticks = list(range(0, len(participants), step))
            ax.set_yticks(yticks)
            ax.set_yticklabels([participants[idx] for idx in yticks])
            ax.set_xlabel("Experimental Condition")
            ax.set_ylabel("Participant")
            plt.colorbar(im, ax=ax, shrink=0.82, label=self._axis_label(metric))
            panel_notes.append(f"{panel_positions[len(panel_notes)]} shows participant-by-condition heterogeneity for {FEATURE_LABELS.get(metric, metric)}.")
        for ax in axes[len(metrics):]:
            ax.axis("off")
        fig._cltr_panel_notes = panel_notes
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_cohort_modeling_overview(self, contrasts: pd.DataFrame, mixed: pd.DataFrame, benchmarks: pd.DataFrame):
        if contrasts.empty and mixed.empty:
            return None
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(16.4, 10.0),
            gridspec_kw={"height_ratios": [1.35, 2.05]},
        )
        panel_notes = [
            "Top|Contrast evidence by family|Corrected contrast evidence summarized jointly as burden, breadth, and contributing endpoint identity. Bar length shows the number of Benjamini-Hochberg-significant contrasts, and the annotation reports both the number of contributing endpoints and their names. Table 5.6 gives the strongest-hit rows and Table 5.7 gives the balanced breadth view.",
            "Middle|Mixed-effects evidence profile|Endpoint-level evidence profile shown as a swimlane plot. Each retained marker shows one effect class that survives Benjamini-Hochberg correction for that endpoint, marker size and color reflect evidence strength, and the adjacent callout lists the retained term labels. Table 5.8 provides the coefficient-level interpretation and Table 5.9 gives fit warnings.",
        ]
        contrast_ax, mixed_ax = axes
        def _empty_panel(ax, title: str, message: str) -> None:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor("#f8fafc")
            ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", fontsize=10.5, color="#475569", wrap=True)
        if not contrasts.empty:
            keep = contrasts.copy()
            if "significant_fdr" in keep.columns:
                keep = keep.loc[keep["significant_fdr"] == 1].copy()
            if not keep.empty:
                keep["contrast_family"] = keep["metric"].map(self._contrast_family)
                keep["endpoint_label"] = keep["metric"].map(lambda x: FEATURE_LABELS.get(str(x), str(x).replace("_", " ")))
                summary = (
                    keep.groupby("contrast_family")
                    .agg(
                        n_contrasts=("metric", "size"),
                        n_metrics=("metric", "nunique"),
                        endpoint_labels=("endpoint_label", lambda s: sorted({str(x) for x in s if str(x).strip()})),
                    )
                    .sort_values(["n_contrasts", "n_metrics"], ascending=[True, True])
                    .reset_index()
                )
                labels = summary["contrast_family"].astype(str).tolist()
                values = summary["n_contrasts"].to_numpy(dtype=float)
                contrast_ax.barh(labels, values, color="#2563eb", alpha=0.9, height=0.72)
                contrast_ax.set_xlabel("Number of BH-significant contrasts", labelpad=10)
                contrast_ax.set_xlim(0, max(values.max() * 3.2, 26))
                contrast_ax.grid(True, axis="x", alpha=0.18)
                contrast_ax.set_axisbelow(True)
                contrast_ax.spines["top"].set_visible(False)
                contrast_ax.spines["right"].set_visible(False)
                for idx, row in enumerate(summary.itertuples(index=False)):
                    endpoints_text = fill(", ".join(row.endpoint_labels), width=84)
                    contrast_ax.text(
                        float(row.n_contrasts) + 0.35,
                        idx,
                        f"{int(row.n_contrasts)} contrasts | {int(row.n_metrics)} endpoints\n{endpoints_text}",
                        va="center",
                        fontsize=10.0,
                        color="#475569",
                        linespacing=1.12,
                        bbox={
                            "boxstyle": "round,pad=0.28,rounding_size=0.12",
                            "facecolor": "#ffffff",
                            "edgecolor": "#e2e8f0",
                            "linewidth": 0.8,
                        },
                        clip_on=False,
                    )
            else:
                _empty_panel(contrast_ax, "Contrast evidence by family", "No Benjamini-Hochberg-significant condition contrasts were retained for this cohort run.")
        else:
            _empty_panel(contrast_ax, "Contrast evidence by family", "No eligible cohort contrast results were available for this cohort run.")
        if not mixed.empty:
            keep = self._mixed_effects_fixed_only(mixed)
            if "significant_fdr" in keep.columns:
                keep = keep.loc[keep["significant_fdr"] == 1].copy()
            if not keep.empty:
                matrix_cols = ["Phase", "Light", "Time", "Interaction"]
                rows = []
                class_match = {
                    "Phase": "protocol_phase",
                    "Light": "illuminance_level",
                    "Time": "time_of_day",
                    "Interaction": ":",
                }
                for metric, d in keep.groupby("metric"):
                    label = FEATURE_LABELS.get(metric, metric)
                    for effect_class, token in class_match.items():
                        if effect_class == "Interaction":
                            class_df = d.loc[d["term"].astype(str).str.contains(token, regex=False)].copy()
                        else:
                            class_df = d.loc[d["term"].astype(str).str.contains(token, regex=False) & ~d["term"].astype(str).str.contains(":", regex=False)].copy()
                        if class_df.empty:
                            continue
                        strongest_q = float(to_numeric(class_df["p_value_fdr"]).min())
                        strongest_strength = float(np.clip(-np.log10(max(strongest_q, 1e-300)), 0, 12))
                        term_text = self._mixed_effect_term_tokens(class_df["term"].astype(str).tolist())
                        rows.append(
                            {
                                "metric": metric,
                                "label": label,
                                "effect_class": effect_class,
                                "strength": strongest_strength,
                                "term_text": term_text,
                            }
                        )
                profile = pd.DataFrame(rows)
                endpoint_order = profile.groupby("label")["strength"].max().sort_values(ascending=False).index.tolist()
                y_map = {label: idx for idx, label in enumerate(endpoint_order)}
                x_map = {name: idx for idx, name in enumerate(matrix_cols)}
                profile["x"] = profile["effect_class"].map(x_map)
                profile["y"] = profile["label"].map(y_map)
                cmap = LinearSegmentedColormap.from_list("mixed_profile", ["#cbd5e1", "#a78bfa", "#7c3aed", "#4c1d95"])
                vmax = max(6.0, float(profile["strength"].max()))
                mixed_ax.set_xlim(-0.65, len(matrix_cols) - 0.1)
                mixed_ax.set_ylim(len(endpoint_order) - 0.5, -0.5)
                mixed_ax.set_xticks(range(len(matrix_cols)))
                mixed_ax.set_xticklabels(matrix_cols, rotation=0, ha="center", fontsize=10.5, fontweight="bold")
                mixed_ax.set_yticks(range(len(endpoint_order)))
                mixed_ax.set_yticklabels(endpoint_order, fontsize=10.2)
                mixed_ax.set_xlabel("Retained BH-significant mixed-effects evidence class", labelpad=10)
                for yi in range(len(endpoint_order)):
                    mixed_ax.hlines(yi, -0.45, len(matrix_cols) - 0.55, color="#e2e8f0", lw=1.4, zorder=0)
                for xi in range(len(matrix_cols)):
                    mixed_ax.axvline(xi, color="#f1f5f9", lw=0.8, zorder=0)
                sizes = 280 + 85 * profile["strength"].clip(lower=0.5)
                sc = mixed_ax.scatter(
                    profile["x"],
                    profile["y"],
                    s=sizes,
                    c=profile["strength"],
                    cmap=cmap,
                    vmin=0,
                    vmax=vmax,
                    edgecolors="white",
                    linewidths=1.6,
                    zorder=3,
                )
                mixed_ax.spines["top"].set_visible(False)
                mixed_ax.spines["right"].set_visible(False)
                mixed_ax.spines["left"].set_visible(False)
                mixed_ax.spines["bottom"].set_visible(False)
                for idx, row in enumerate(profile.itertuples(index=False)):
                    offset = 0.24 if idx % 2 == 0 else -0.24
                    ha = "left" if offset > 0 else "right"
                    mixed_ax.text(
                        float(row.x) + offset,
                        float(row.y),
                        str(row.term_text),
                        ha=ha,
                        va="center",
                        fontsize=7.6,
                        color="#1e293b",
                        fontweight="bold",
                        linespacing=1.0,
                        bbox={
                            "boxstyle": "round,pad=0.22,rounding_size=0.12",
                            "facecolor": "#ffffff",
                            "edgecolor": "#cbd5e1",
                            "linewidth": 0.9,
                        },
                        zorder=4,
                    )
                mixed_ax.tick_params(axis="y", length=0)
                mixed_ax.tick_params(axis="x", length=0)
                cbar = plt.colorbar(sc, ax=mixed_ax, fraction=0.03, pad=0.02)
                cbar.set_label("Strongest retained evidence (-log10 BH q)")
            else:
                _empty_panel(mixed_ax, "Mixed-effects evidence class", "No Benjamini-Hochberg-significant mixed-effects fixed terms were retained for this cohort run.")
        else:
            _empty_panel(mixed_ax, "Mixed-effects evidence class", "No eligible mixed-effects primary models were available for this cohort run.")
        fig._cltr_panel_notes = panel_notes
        fig.tight_layout(rect=(0.02, 0.01, 0.995, 0.985))
        return fig

    def _fig_cohort_associations(self, associations: pd.DataFrame):
        if associations.empty:
            return None
        metrics = sorted(set(associations["left_metric"]) | set(associations["right_metric"]))
        mat = pd.DataFrame(np.nan, index=metrics, columns=metrics)
        for metric in metrics:
            mat.loc[metric, metric] = 1.0
        for row in associations.itertuples():
            mat.loc[row.left_metric, row.right_metric] = row.spearman_r
            mat.loc[row.right_metric, row.left_metric] = row.spearman_r
        fig, ax = plt.subplots(figsize=(8.2, 6.8))
        im = ax.imshow(mat.values, aspect="equal", cmap="coolwarm", vmin=-1, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([FEATURE_LABELS.get(x, x) for x in metrics], rotation=45, ha="right")
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels([FEATURE_LABELS.get(x, x) for x in metrics])
        plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman r")
        fig.tight_layout()
        return fig

    def _fig_cohort_agreement_summary(self, summary: pd.DataFrame):
        if summary.empty:
            return None
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        metrics = ["heart_rate", "eda", "temperature"]
        panel_notes = [
            "Left|Median overlap|Median overlap.",
            "Center|Median Spearman correlation|Median Spearman correlation.",
            "Right|Median mean absolute error|Median mean absolute error.",
        ]
        ylabels = {
            "median_overlap_minutes": "Median Overlap Duration (min)",
            "median_spearman_r": "Median Spearman Correlation (r)",
            "median_mae": "Median Mean Absolute Error",
        }
        for ax, col, title in zip(axes, ["median_overlap_minutes", "median_spearman_r", "median_mae"], ["Median overlap", "Median Spearman r", "Median MAE"]):
            vals = []
            colors = []
            for metric in metrics:
                row = summary.loc[summary["metric"] == metric]
                vals.append(float(row[col].iloc[0]) if not row.empty and pd.notna(row[col].iloc[0]) else np.nan)
                colors.append("#2563eb" if not row.empty and row["summary_status"].iloc[0] == "eligible" else "#94a3b8")
            ax.bar(metrics, vals, color=colors)
            ax.set_xlabel("Directly Comparable Modality Pair")
            ax.set_ylabel(ylabels.get(col, title))
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(["Heart Rate", "EDA", "Temperature"], rotation=0)
        fig._cltr_panel_notes = panel_notes
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_cohort_phase_heatmap(self, summary: pd.DataFrame):
        if summary.empty:
            return None
        summary = summary.loc[summary["protocol_phase"].astype(str) != "acclimation"].copy()
        if summary.empty:
            return None
        cols = [f"{m}__mean" for m in ["thermal_comfort", "empatica_hr_mean_bpm", "biopac_temp_chest_mean_C", "indoor_air_velocity_mean_m_s"] if f"{m}__mean" in summary.columns]
        if not cols:
            return None
        mat = summary.copy()
        mat["phase_condition"] = mat["protocol_phase"].astype(str) + " | " + mat["condition_code"].astype(str)
        z = mat.set_index("phase_condition")[cols]
        z = z.apply(lambda col: (col - col.mean()) / col.std(ddof=0) if col.notna().sum() > 1 and col.std(ddof=0) > 0 else col * 0, axis=0).T
        fig = plt.figure(figsize=self._figsize("matrix_tall"))
        gs = fig.add_gridspec(2, 1, height_ratios=[18, 1.4], hspace=0.32)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[1, 0])
        im = ax.imshow(z.values, aspect="equal", cmap="coolwarm", vmin=-2, vmax=2)
        ax.grid(False)
        ax.set_yticks(range(len(z.index)))
        ax.set_yticklabels([FEATURE_LABELS.get(x.replace("__mean", ""), x) for x in z.index])
        ax.set_xticks(range(len(z.columns)))
        ax.set_xticklabels(self._phase_condition_ticklabels(list(z.columns)), rotation=0, ha="center")
        ax.tick_params(axis="x", labelsize=10, pad=8)
        cb = plt.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("z score")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_cohort_participant_heatmap(self, profiles: pd.DataFrame):
        if profiles.empty:
            return None
        if int(profiles["participant_id"].nunique()) <= 1:
            latest = profiles.copy().sort_values("condition_code")
            metrics = [m for m in ["thermal_comfort", "biopac_temp_chest_mean_C", "empatica_hr_mean_bpm"] if m in latest.columns]
            fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.2))
            if len(metrics) == 1:
                axes = [axes]
            for ax, metric in zip(axes, metrics):
                d = latest[["condition_code", metric]].dropna()
                if d.empty:
                    ax.axis("off")
                    continue
                ax.bar(d["condition_code"], d[metric], color=[CONDITION_COLORS.get(str(x), "#475569") for x in d["condition_code"]])
                ax.set_ylabel(self._axis_label(metric))
                ax.tick_params(axis="x", rotation=45)
            fig._cltr_panel_notes = [
                f"{['Left','Center','Right'][idx]}|{FEATURE_LABELS.get(metric, metric)} by condition|{FEATURE_LABELS.get(metric, metric)} by condition."
                for idx, metric in enumerate(metrics)
            ]
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            return fig
        metrics = [m for m in ["thermal_comfort", "biopac_temp_chest_mean_C"] if m in profiles.columns]
        if not metrics:
            return None
        n_participants = int(profiles["participant_id"].nunique())
        fig_height = min(max(6.5, 0.38 * n_participants), 11.5)
        fig, axes = plt.subplots(1, len(metrics), figsize=(7.8 * len(metrics), fig_height))
        if len(metrics) == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            pivot = profiles.pivot(index="participant_id", columns="condition_code", values=metric)
            pivot = pivot.reindex(columns=[x for x in CONDITION_ORDER if x in pivot.columns])
            im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm")
            ax.grid(False)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_ylabel(self._axis_label(metric))
            plt.colorbar(im, ax=ax, shrink=0.8)
        fig._cltr_panel_notes = [
            f"{['Left','Right'][idx]}|Participant-by-condition variation for {FEATURE_LABELS.get(metric, metric)}|Participant-by-condition variation for {FEATURE_LABELS.get(metric, metric)}."
            for idx, metric in enumerate(metrics)
        ]
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _session_observations(self, s: dict) -> list[str]:
        meta = s["processing_metadata"]
        phase = s["phase_df"]
        ev = self._session_evidence(s["aligned_df"], meta)
        story = self._session_story_profile(s)
        confidence = "high" if ev["score"] >= 75 else "moderate" if ev["score"] >= 50 else "limited"
        obs = [story["headline"], f"Overall confidence in the session summary is {confidence}."]
        support_starts = []
        for metric in ["empatica_temp_mean_C", "biopac_temp_chest_mean_C", "empatica_eda_mean_uS", "biopac_eda_mean_uS", "empatica_hr_mean_bpm", "biopac_hr_mean_bpm"]:
            base = self._phase_metric_baseline(phase, metric)
            if base:
                support_starts.append(f"{FEATURE_LABELS.get(metric, metric)} starts at {self._baseline_phase_text(base)}")
        if support_starts:
            obs.append("; ".join(support_starts[:3]) + ".")
        if ev["note"]:
            obs.append(ev["note"] + ".")
        if "thermal_comfort" in phase.columns:
            cur = phase.groupby("protocol_phase")["thermal_comfort"].mean().dropna()
            if len(cur) >= 2:
                obs.append(f"Average thermal comfort was highest during {self._fmt_cell(str(cur.idxmax()))} and lowest during {self._fmt_cell(str(cur.idxmin()))}.")
            consistency = self._phase_repeat_consistency(phase, "thermal_comfort")
            if consistency["dominant_phase"] is not None and consistency["n_blocks"] >= BLOCK_PHASE_NARRATIVE_THRESHOLD:
                obs.append(f"Comfort changes repeat across {consistency['n_blocks']} exposures during {self._fmt_cell(str(consistency['dominant_phase']))}, with consistency {consistency['consistency']:.2f}.")
        overlap = meta["sensor_overlap_minutes"]
        if overlap["heart_rate"] >= self.config.runtime.min_sensor_overlap_minutes:
            pair = s["aligned_df"][["empatica_hr_mean_bpm", "biopac_hr_mean_bpm"]].apply(to_numeric).dropna()
            if len(pair) >= 3:
                obs.append(f"Heart-rate readings from the two devices can be compared across {overlap['heart_rate']} overlapping minutes, with a Spearman correlation of {pair.corr(method='spearman').iloc[0,1]:.2f}.")
        else:
            obs.append(f"Heart-rate readings from the two devices overlap for only {overlap['heart_rate']} minutes, so this comparison should be treated with caution.")
        return obs[:4]

    def _cohort_observations(self, c: dict) -> list[str]:
        sample = c["sample_status"].iloc[0]
        obs = []
        if int(sample["cohort_inference_eligible"]):
            obs.append("The cohort is large enough for full cross-session comparisons.")
        else:
            obs.append(f"This summary is based on `{int(sample['n_sessions'])}` sessions from `{int(sample['n_participants'])}` participants, so the results should be read as directional rather than definitive.")
        signal_audit = c.get("signal_audit_summary", pd.DataFrame())
        if not signal_audit.empty:
            inventory = self._device_stream_inventory_register(c.get("cohort_minute_features", pd.DataFrame()), signal_audit)
            primary = signal_audit.loc[signal_audit["recommended_role"].astype(str).isin(["primary", "primary_with_qc"]), "signal_stream"].astype(str).tolist()
            limited = signal_audit.loc[signal_audit["recommended_role"].astype(str).isin(["secondary_only", "secondary_validation", "subset_only", "not_primary", "not_recommended"]), "signal_stream"].astype(str).tolist()
            if not inventory.empty:
                obs.append(
                    f"The current cohort export catalogs `{len(inventory)}` Empatica/BIOPAC streams, of which `{int(signal_audit['signal_stream'].nunique())}` currently enter the formal signal audit."
                )
                comparable_n = int((inventory["comparison_class"] == "directly_comparable").sum())
                non_comparable_n = int(len(inventory) - comparable_n)
                obs.append(
                    f"`{comparable_n}` streams belong to directly comparable Empatica/BIOPAC pairs, while `{non_comparable_n}` streams are device-specific, same-construct-but-unpaired, or source-only."
                )
            if primary:
                obs.append("Primary device streams in this release are " + ", ".join(self._fmt_cell(x) for x in primary) + ".")
            if limited:
                obs.append("Limited, subset-only, or secondary-use streams include " + ", ".join(self._fmt_cell(x) for x in limited[:6]) + ".")
        else:
            exploratory = c.get("exploratory_feature_summary", pd.DataFrame())
            if not exploratory.empty:
                exploratory = exploratory.loc[~exploratory["feature"].astype(str).str.startswith("support_")].copy()
                top_coverage = exploratory.sort_values(["coverage_fraction", "n_non_null"], ascending=[False, False]).head(3)
                obs.append("The strongest data coverage appears in " + ", ".join(FEATURE_LABELS.get(row.feature, row.feature) for row in top_coverage.itertuples()) + ".")
        return obs[:4]

    def _phase_delta_map(self, phase: pd.DataFrame) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if phase.empty or "protocol_phase" not in phase.columns:
            return out
        for metric, info in SESSION_STORY_METRICS.items():
            if metric not in phase.columns:
                continue
            phase_summary, baseline = self._phase_baseline_delta_summary(phase, metric)
            if phase_summary.empty or not baseline or pd.isna(baseline["value"]):
                continue
            deltas = phase_summary.set_index("protocol_phase")["delta"].dropna()
            deltas = deltas.loc[[p for p in deltas.index if p != baseline["phase"]]]
            if deltas.empty:
                continue
            abs_deltas = deltas.abs()
            best_phase = str(abs_deltas.idxmax())
            best_delta = float(deltas.loc[best_phase])
            repeat = self._phase_repeat_consistency(phase, metric)
            coverage_mean = float(to_numeric(phase_summary["coverage_mean"]).mean()) if "coverage_mean" in phase_summary.columns else 1.0
            support_factor = max(0.2, min(1.0, coverage_mean))
            repeat_factor = max(0.35, 0.45 + 0.55 * float(repeat["consistency"]))
            out[metric] = {
                "metric": metric,
                "label": info["label"],
                "kind": info["kind"],
                "phase": best_phase,
                "delta": best_delta,
                "abs_delta": abs(best_delta),
                "score": (abs(best_delta) / max(float(info["scale"]), 1e-6)) * support_factor * repeat_factor,
                "direction": "rise" if best_delta > 0 else "drop",
                "baseline_phase": baseline["phase"],
                "coverage_mean": coverage_mean,
                "repeat_consistency": float(repeat["consistency"]),
                "repeat_blocks": int(repeat["n_blocks"]),
                "dominant_repeat_phase": repeat["dominant_phase"],
            }
        return out

    def _session_story_profile(self, s: dict) -> dict:
        meta = s["processing_metadata"]
        ev = self._session_evidence(s["aligned_df"], meta)
        overlap = meta.get("sensor_overlap_minutes", {})
        support = meta.get("support", {})
        condition = str(meta.get("condition_code", "")).lower()
        tod = str(meta.get("condition_time_of_day") or meta.get("time_of_day") or "").lower()
        context_parts = [part for part in [condition, tod, ev["label"]] if part]
        context = "/".join(context_parts)
        phase_map = self._phase_delta_map(s["phase_df"])
        base = {
            "archetype": "audit-first",
            "lead_label": f"audit-first | {meta.get('condition_code', '')} | {ev['label']}",
            "headline": "No strong phase-separated signal dominates this session, so interpretation remains audit-first.",
            "tags": ["audit-first", condition, tod, "support", "phase", "qc"],
            "priority_codes": ["S01", "S10", "S07", "S02", "S08"],
        }

        def phase_priority(phase_name: str, kind: str) -> list[str]:
            if phase_name in {"fan_at_constant_speed", "fan_free_control"}:
                return ["S01", "S06", "S07", "S08", "S02"] if kind != "physiology" else ["S01", "S03", "S07", "S09", "S02"]
            if phase_name == "skin_rewarming":
                return ["S01", "S05", "S07", "S08", "S02"]
            if phase_name == "overall_comfort":
                return ["S01", "S08", "S06", "S07", "S02"]
            return ["S01", "S02", "S05", "S08", "S09"] if kind == "thermal" else ["S01", "S03", "S07", "S09", "S02"] if kind == "physiology" else ["S01", "S06", "S02", "S07", "S08"]

        if ev["label"] == "weak":
            lead = f"validation-limited | {meta.get('condition_code', '')} | overlap-limited"
            headline = "Sensor overlap is too weak for a strong physiological read."
            tags = ["validation-limited", condition, tod, "support", "agreement", "qc"]
            return {"archetype": "validation-limited", "lead_label": lead, "headline": headline, "tags": tags, "priority_codes": ["S01", "S10", "S02", "S07", "S06"]}
        if support.get("questionnaire_completeness", 0.0) < 0.8:
            lead = f"questionnaire-sparse | {meta.get('condition_code', '')} | subjective-limited"
            headline = "Subjective interpretation is constrained by incomplete questionnaire capture in Blocks 1 to 3."
            tags = ["questionnaire-sparse", condition, tod, "support", "comfort", "qc"]
            return {"archetype": "questionnaire-sparse", "lead_label": lead, "headline": headline, "tags": tags, "priority_codes": ["S01", "S10", "S06", "S07", "S02"]}

        comfort = phase_map.get("thermal_comfort")
        dpg = phase_map.get("master_dpg_C") or phase_map.get("thermal_gradient_C")
        hr = phase_map.get("empatica_hr_mean_bpm") or phase_map.get("biopac_hr_mean_bpm")
        temp = phase_map.get("biopac_temp_chest_mean_C") or phase_map.get("empatica_temp_mean_C")
        env = phase_map.get("indoor_air_velocity_mean_m_s")
        fan = phase_map.get("fan_control_au")
        ranked = sorted(phase_map.values(), key=lambda x: (x["score"], x["abs_delta"]), reverse=True)
        top = ranked[0] if ranked else None
        second = next((item for item in ranked[1:] if item["label"] != top["label"]), None) if top else None

        if comfort and comfort["phase"] in {"fan_at_constant_speed", "fan_free_control"} and comfort["direction"] == "drop" and comfort["score"] >= 0.75 and comfort.get("repeat_blocks", 0) >= BLOCK_PHASE_NARRATIVE_THRESHOLD:
            phase_name = comfort["phase"].replace("_", " ")
            return {
                "archetype": "comfort-drop",
                "lead_label": f"comfort-drop | {phase_name} | {meta.get('condition_code', '')}",
                "headline": f"Comfort falls most strongly during {phase_name}, and that direction repeats across protocol blocks.",
                "tags": ["comfort-drop", condition, tod, "comfort", "fan", "environment", comfort["phase"], "repeat-supported"],
                "priority_codes": ["S01", "S06", "S07", "S08", "S02"],
            }
        if dpg and dpg["phase"] == "skin_rewarming" and dpg["score"] >= 0.75 and dpg.get("repeat_blocks", 0) >= BLOCK_PHASE_NARRATIVE_THRESHOLD:
            return {
                "archetype": "rewarming-shift",
                "lead_label": f"rewarming-shift | {dpg['label']} | {meta.get('condition_code', '')}",
                "headline": "The strongest thermal departure occurs during skin rewarming, with repetition across blocks.",
                "tags": ["rewarming-shift", condition, tod, dpg["label"], "temperature", "phase", "skin_rewarming", "repeat-supported"],
                "priority_codes": ["S01", "S05", "S07", "S08", "S02"],
            }
        if hr and hr["score"] >= 0.9 and hr.get("repeat_blocks", 0) >= BLOCK_PHASE_NARRATIVE_THRESHOLD:
            phase_name = hr["phase"].replace("_", " ")
            return {
                "archetype": "heart-rate-shift",
                "lead_label": f"heart-rate-{hr['direction']} | {phase_name} | {meta.get('condition_code', '')}",
                "headline": f"Heart-rate change peaks during {phase_name} and is not confined to a single block.",
                "tags": ["heart-rate-shift", condition, tod, "heart_rate", "phase", hr["phase"], hr["direction"], "repeat-supported"],
                "priority_codes": ["S01", "S03", "S07", "S08", "S02"],
            }
        if temp and temp["score"] >= 0.9 and temp.get("repeat_blocks", 0) >= BLOCK_PHASE_NARRATIVE_THRESHOLD:
            phase_name = temp["phase"].replace("_", " ")
            moderate_support = overlap.get("heart_rate", 0) < self.config.runtime.min_sensor_overlap_minutes or ev["label"] == "moderate"
            return {
                "archetype": "thermal-shift-moderate" if moderate_support else "thermal-shift",
                "lead_label": f"{temp['label']}-{temp['direction']} | {phase_name} | {meta.get('condition_code', '')}",
                "headline": f"The clearest thermal signature is a {temp['direction']} in {temp['label']} during {phase_name}, repeated across blocks.",
                "tags": ["thermal-shift", condition, tod, temp["label"], "temperature", "phase", temp["phase"], "repeat-supported"],
                "priority_codes": ["S01", "S10", "S07", "S02", "S08"] if moderate_support else ["S01", "S05", "S07", "S02", "S08"],
            }
        if env and fan and env["phase"] == fan["phase"] and env["score"] >= 0.8:
            phase_name = env["phase"].replace("_", " ")
            return {
                "archetype": "forced-air-response",
                "lead_label": f"forced-air-response | {phase_name} | {meta.get('condition_code', '')}",
                "headline": f"Environmental forcing and fan behavior align most clearly during {phase_name}.",
                "tags": ["forced-air-response", condition, tod, "environment", "fan", "phase", env["phase"]],
                "priority_codes": ["S01", "S06", "S07", "S02", "S05"],
            }
        if top:
            phase_name = top["phase"].replace("_", " ")
            second_piece = ""
            second_tags: list[str] = []
            if second and second["score"] >= 0.55:
                second_piece = f"; secondary signal is {second['label']} in {second['phase'].replace('_', ' ')}"
                second_tags = [second["label"], second["phase"]]
            return {
                "archetype": "support-adjusted-topline",
                "lead_label": f"{top['label']}-{top['direction']} | {phase_name} | {meta.get('condition_code', '')}",
                "headline": f"The strongest support-adjusted phase departure is a {top['direction']} in {top['label']} during {phase_name}{second_piece}.",
                "tags": [f"{top['label']}-shift", condition, tod, top["label"], "phase", top["phase"], top["direction"]] + second_tags + (["repeat-supported"] if top.get("repeat_blocks", 0) >= BLOCK_PHASE_NARRATIVE_THRESHOLD else ["single-block-sensitive"]),
                "priority_codes": phase_priority(top["phase"], top["kind"]),
            }
        return base

    def _session_atlas_tags(self, s: dict, narrative_specs: list[dict]) -> list[str]:
        story = self._session_story_profile(s)
        story_tags = [tag for tag in dict.fromkeys(story["tags"]) if str(tag).strip()]
        tags = set(story_tags)
        for spec in narrative_specs:
            tags.update(t for t in spec["tags"] if t != "appendix" and str(t).strip())
        meta = s["processing_metadata"]
        tags.update(tag for tag in [str(meta.get("condition_code", "")).lower(), str(meta.get("participant_id", "")).lower()] if tag.strip())
        preferred = ["overview", "support", "comfort", "environment", "fan", "phase", "heart_rate", "temperature", "agreement", "statistics"]
        ordered = [tag for tag in story_tags if tag in tags]
        ordered.extend(tag for tag in preferred if tag in tags and tag not in ordered)
        ordered.extend(sorted(tag for tag in tags if tag not in ordered))
        return ordered[:8]

    def _curate_session_specs(self, s: dict, narrative_specs: list[dict], appendix_specs: list[dict]) -> tuple[list[dict], list[dict]]:
        return narrative_specs, appendix_specs

    def _curate_cohort_specs(self, c: dict, narrative_specs: list[dict], appendix_specs: list[dict]) -> tuple[list[dict], list[dict]]:
        return narrative_specs, appendix_specs

    def _session_html(self, session_inputs: dict, narrative_specs: list[dict], appendix_specs: list[dict]) -> str:
        meta = session_inputs["processing_metadata"]
        minute = session_inputs["aligned_df"]
        phase = session_inputs["phase_df"]
        cards = [
            ("Session", session_inputs["session_id"]),
            ("Participant", meta["participant_id"]),
            ("Condition", meta["condition_code"]),
            ("Timeline minutes", len(minute)),
            ("Questionnaire completeness", f"{meta['support']['questionnaire_completeness']:.1%}"),
            ("HR overlap", meta["sensor_overlap_minutes"]["heart_rate"]),
        ]
        return self._html_document(
            title=f"CLTR Session Report: {session_inputs['session_id']}",
            subtitle="",
            cards=cards,
            observations=self._session_observations(session_inputs),
            main_specs=narrative_specs,
            appendix_specs=appendix_specs,
            intro_sections=self._session_stage_sections(session_inputs, phase, meta),
            section_intro_map=self._session_section_intros(),
            doc_kind="session",
            middle_menu_button_id="sessionMenuButton",
            middle_menu_panel_id="sessionMenuPanel",
            middle_menu_label="Sessions",
            middle_menu_title="Sessions",
            middle_menu_items_html="<a href='../../sessions_report.html'>Sessions Report<span>Dedicated session browser and participant-level reports</span></a>",
            secondary_menu_button_id="chapterMenuButton",
            secondary_menu_panel_id="chapterMenuPanel",
            secondary_menu_label="Chapters",
            secondary_menu_title="Cohort Chapters",
            secondary_menu_items_html=self._cohort_chapter_menu_items_html("../../cohort/"),
        )

    def _cohort_html(self, cohort_inputs: dict, narrative_specs: list[dict], appendix_specs: list[dict]) -> str:
        sample = cohort_inputs["sample_status"].iloc[0]
        signal_audit = cohort_inputs.get("signal_audit_summary", pd.DataFrame())
        catalogued_streams = len(DEVICE_STREAM_CATALOG)
        audited_streams = int(signal_audit["signal_stream"].nunique()) if not signal_audit.empty else 0
        comparable_stream_families = int(cohort_inputs["agreement_summary"]["metric"].nunique()) if not cohort_inputs.get("agreement_summary", pd.DataFrame()).empty else 0
        chapter_specs = self._cohort_chapter_specs(cohort_inputs, narrative_specs, appendix_specs)
        chapter_menu_items_html = "".join(
            f"<a href='{html_escape(chapter['filename'])}'>{html_escape(chapter['title'].split(':')[-1].strip())}<span>{html_escape(chapter['subtitle'])}</span></a>"
            for chapter in chapter_specs
        )
        sessions_menu_items_html = (
            "<a href='../sessions_report.html'>Sessions Report<span>Dedicated session browser and participant-level reports</span></a>"
        )
        cards = [
            ("Sessions", int(sample["n_sessions"])),
            ("Participants", int(sample["n_participants"])),
            ("Comparison readiness", "full" if int(sample["cohort_inference_eligible"]) else "limited"),
            ("Catalogued streams", catalogued_streams),
            ("Signal-audited streams", audited_streams),
            ("Comparable stream families", comparable_stream_families),
            ("Comparable agreement records", int((cohort_inputs["sensor_agreement"]["eligible"] == 1).sum()) if not cohort_inputs["sensor_agreement"].empty else 0),
            ("Comparable condition pairs", int((cohort_inputs["condition_contrasts"]["eligible"] == 1).sum()) if not cohort_inputs["condition_contrasts"].empty else 0),
            ("FDR-significant contrasts", int((cohort_inputs["condition_contrasts"].get("significant_fdr", pd.Series(dtype=int)) == 1).sum()) if not cohort_inputs["condition_contrasts"].empty else 0),
            ("Benchmark tasks", int(cohort_inputs["predictive_benchmarks"]["task"].nunique()) if not cohort_inputs.get("predictive_benchmarks", pd.DataFrame()).empty else 0),
            ("Minute-level records", len(cohort_inputs["cohort_minute_features"])),
        ]
        return self._html_document(
            title="CLTR Cohort Report",
            subtitle="",
            cards=cards,
            observations=self._cohort_observations(cohort_inputs),
            main_specs=narrative_specs,
            appendix_specs=appendix_specs,
            intro_sections=self._cohort_stage_sections(cohort_inputs),
            section_intro_map=self._cohort_section_intros(cohort_inputs),
            doc_kind="cohort",
            middle_menu_button_id="sessionMenuButton",
            middle_menu_panel_id="sessionMenuPanel",
            middle_menu_label="Sessions",
            middle_menu_title="Sessions",
            middle_menu_items_html=sessions_menu_items_html,
            secondary_menu_button_id="chapterMenuButton",
            secondary_menu_panel_id="chapterMenuPanel",
            secondary_menu_label="Chapters",
            secondary_menu_title="Cohort Chapters",
            secondary_menu_items_html=chapter_menu_items_html,
        )

    def _cohort_cards(self, cohort_inputs: dict) -> list[tuple[str, object]]:
        sample = cohort_inputs["sample_status"].iloc[0]
        signal_audit = cohort_inputs.get("signal_audit_summary", pd.DataFrame())
        catalogued_streams = len(DEVICE_STREAM_CATALOG)
        audited_streams = int(signal_audit["signal_stream"].nunique()) if not signal_audit.empty else 0
        comparable_stream_families = int(cohort_inputs["agreement_summary"]["metric"].nunique()) if not cohort_inputs.get("agreement_summary", pd.DataFrame()).empty else 0
        return [
            ("Sessions", int(sample["n_sessions"])),
            ("Participants", int(sample["n_participants"])),
            ("Comparison readiness", "full" if int(sample["cohort_inference_eligible"]) else "limited"),
            ("Catalogued streams", catalogued_streams),
            ("Signal-audited streams", audited_streams),
            ("Comparable stream families", comparable_stream_families),
            ("Comparable agreement records", int((cohort_inputs["sensor_agreement"]["eligible"] == 1).sum()) if not cohort_inputs["sensor_agreement"].empty else 0),
            ("Comparable condition pairs", int((cohort_inputs["condition_contrasts"]["eligible"] == 1).sum()) if not cohort_inputs["condition_contrasts"].empty else 0),
            ("FDR-significant contrasts", int((cohort_inputs["condition_contrasts"].get("significant_fdr", pd.Series(dtype=int)) == 1).sum()) if not cohort_inputs["condition_contrasts"].empty else 0),
            ("Benchmark tasks", int(cohort_inputs["predictive_benchmarks"]["task"].nunique()) if not cohort_inputs.get("predictive_benchmarks", pd.DataFrame()).empty else 0),
            ("Minute-level records", len(cohort_inputs["cohort_minute_features"])),
        ]

    def _chapter_observations(self, cohort_inputs: dict, specs: list[dict]) -> list[str]:
        tags = {str(tag) for spec in specs for tag in spec.get("tags", [])}
        observations: list[str] = []
        if {"overview", "qc", "support"} & tags:
            observations.extend(self._cohort_observations(cohort_inputs))
        if {"comfort", "fan"} & tags:
            observations.extend(
                [
                    "This chapter isolates subjective observations and behavioral-control channels before they are reduced into support-gated cohort endpoints.",
                    "Questionnaire figures remain descriptive and event-based, while fan channels provide behavioral context for later environmental and comfort interpretation.",
                ]
            )
        if {"heart_rate", "eda", "temperature", "bloodflow", "optical", "bvp", "motion", "activity"} & tags:
            signal_audit = cohort_inputs.get("signal_audit_summary", pd.DataFrame())
            if not signal_audit.empty:
                primary = signal_audit.loc[signal_audit["recommended_role"].astype(str).isin(["primary", "primary_with_qc"]), "signal_stream"].astype(str).tolist()
                limited = signal_audit.loc[~signal_audit["recommended_role"].astype(str).isin(["primary", "primary_with_qc"]), "signal_stream"].astype(str).tolist()
                observations.extend(
                    [
                        "This chapter groups the audited Empatica and BIOPAC physiological streams before endpoint reduction or relationship screening.",
                        "Primary or QC-qualified physiological streams are " + ", ".join(self._fmt_cell(x) for x in primary[:8]) + ".",
                        "Limited physiological streams include " + ", ".join(self._fmt_cell(x) for x in limited[:6]) + "." if limited else "All displayed physiological streams currently retain at least QC-qualified support.",
                    ]
                )
        if {"environment"} & tags:
            observations.extend(
                [
                    "Environmental figures are separated from physiology so room forcing and ambient context can be inspected without being conflated with body-signal changes.",
                    "These traces are descriptive protocol-context views and should be read alongside support density and condition balance.",
                ]
            )
        if {"statistics", "matrix", "manuscript"} & tags:
            observations.extend(
                [
                    "This chapter contains the support-gated cohort result layer and the audit registers that determine which endpoints remain scientifically defensible.",
                    "Use the scenario and claim registers here before promoting any matrix or model output into manuscript evidence.",
                ]
            )
        if {"relationships", "agreement"} & tags:
            observations.extend(
                [
                    "This chapter separates all-source audit relationships from valid-only claim-supporting relationships.",
                    "Device-agreement figures here are technical validation summaries, not scientific associations between constructs.",
                ]
            )
        deduped: list[str] = []
        for item in observations:
            text = str(item).strip()
            if text and text not in deduped:
                deduped.append(text)
        return deduped[:4]

    def _cohort_chapter_specs(self, cohort_inputs: dict, narrative_specs: list[dict], appendix_specs: list[dict]) -> list[dict]:
        all_specs = narrative_specs + appendix_specs
        intro_map = self._cohort_section_intros(cohort_inputs)
        chapters = [
            {
                "slug": "ch01_overview_audit",
                "chapter_number": 1,
                "filename": "cohort_ch01_overview_audit.html",
                "title": "CLTR Cohort Report: Chapter 1",
                "subtitle": "Study overview and audit registers",
                "description": "The governing chapter for the cohort suite: study coverage, stream inventory, signal-support decisions, comparability classes, scenario definitions, and claim gates.",
                "focus_label": "Audit backbone",
                "sections": ["frontmatter"],
                "intro_sections": self._cohort_stage_sections(cohort_inputs),
                "section_intro_map": {},
            },
            {
                "slug": "ch02_subjective_behavioral",
                "chapter_number": 2,
                "filename": "cohort_ch02_subjective_behavioral.html",
                "title": "CLTR Cohort Report: Chapter 2",
                "subtitle": "Subjective and behavioral data",
                "description": "Comfort, preference, and fan-control views shown as their own evidence family before being blended into derived cohort endpoints or cross-modal interpretations.",
                "focus_label": "Participant response",
                "sections": ["subjective_behavioral"],
                "intro_sections": "",
                "section_intro_map": {"subjective_behavioral": intro_map.get("subjective_behavioral", "")},
            },
            {
                "slug": "ch03_physiological",
                "chapter_number": 3,
                "filename": "cohort_ch03_physiological.html",
                "title": "CLTR Cohort Report: Chapter 3",
                "subtitle": "Physiological data",
                "description": "The full Empatica and BIOPAC body-signal gallery, including audited HR, EDA, temperature, motion, vascular, and optical streams with scenario-aware physiological views.",
                "focus_label": "Wearables + BIOPAC",
                "sections": ["physiological"],
                "intro_sections": "",
                "section_intro_map": {"physiological": intro_map.get("physiological", "")},
            },
            {
                "slug": "ch04_environmental",
                "chapter_number": 4,
                "filename": "cohort_ch04_environmental.html",
                "title": "CLTR Cohort Report: Chapter 4",
                "subtitle": "Environmental data",
                "description": "Ambient and forcing context isolated from physiology so room dynamics can be inspected on their own before causal or relational interpretation is attempted.",
                "focus_label": "Context signals",
                "sections": ["environmental"],
                "intro_sections": "",
                "section_intro_map": {"environmental": intro_map.get("environmental", "")},
            },
            {
                "slug": "ch05_derived_results",
                "chapter_number": 5,
                "filename": "cohort_ch05_derived_results.html",
                "title": "CLTR Cohort Report: Chapter 5",
                "subtitle": "Policy-Gated Scientific Results and Modeling",
                "description": "A scientific-results layer governed by explicit support, quality, modality, and robustness gates: aligned master-table readiness, feature registry, support-screened result matrices, inferential contrasts, mixed-effects estimates, predictive benchmarks, and heterogeneity summaries.",
                "focus_label": "Scientific results",
                "sections": ["analyzed"],
                "intro_sections": "",
                "section_intro_map": {"analyzed": intro_map.get("analyzed", "")},
            },
            {
                "slug": "ch06_relationships_validation",
                "chapter_number": 6,
                "filename": "cohort_ch06_relationships_validation.html",
                "title": "CLTR Cohort Report: Chapter 6",
                "subtitle": "Relationships and validation",
                "description": "All-source and valid-only relationship views, plus device-agreement and validation outputs used to separate exploratory associations from claim-supporting evidence.",
                "focus_label": "Validation layer",
                "sections": ["interpretive"],
                "intro_sections": "",
                "section_intro_map": {"interpretive": intro_map.get("interpretive", "")},
            },
        ]
        out = []
        for chapter in chapters:
            specs = [spec for spec in all_specs if spec.get("display_section", spec.get("section", "analyzed")) in chapter["sections"]]
            enriched = dict(chapter)
            enriched["specs"] = specs
            enriched["observations"] = self._chapter_observations(cohort_inputs, specs)
            out.append(enriched)
        return out

    def _cohort_chapter_html(
        self,
        cohort_inputs: dict,
        title: str,
        subtitle: str,
        specs: list[dict],
        intro_sections: str,
        section_intro_map: dict[str, str],
        chapter_menu_items_html: str,
        chapter_number: int | None = None,
        home_href: str = "../index.html",
        logo_src: str = "../../../../cltr/docs/assets/logos/cltr.png",
        figure_src_prefix: str = "figures/",
        sessions_href: str = "../sessions_report.html",
    ) -> str:
        return self._html_document(
            title=title,
            subtitle=subtitle,
            cards=self._cohort_cards(cohort_inputs),
            observations=self._chapter_observations(cohort_inputs, specs),
            main_specs=specs,
            appendix_specs=[],
            intro_sections=intro_sections,
            section_intro_map=section_intro_map,
            chapter_number=chapter_number,
            doc_kind="cohort",
            home_href=home_href,
            logo_src=logo_src,
            figure_src_prefix=figure_src_prefix,
            middle_menu_button_id="sessionMenuButton",
            middle_menu_panel_id="sessionMenuPanel",
            middle_menu_label="Sessions",
            middle_menu_title="Sessions",
            middle_menu_items_html=f"<a href='{html_escape(sessions_href)}'>Sessions Report<span>Dedicated session browser and participant-level reports</span></a>",
            secondary_menu_button_id="chapterMenuButton",
            secondary_menu_panel_id="chapterMenuPanel",
            secondary_menu_label="Chapters",
            secondary_menu_title="Cohort Chapters",
            secondary_menu_items_html=chapter_menu_items_html,
        )

    def _cohort_index_html(
        self,
        cohort_inputs: dict,
        chapters: list[dict],
        chapter_paths: dict[str, str],
        full_html_path: Path,
    ) -> str:
        cards_html = "".join(
            f"<div class='card'><div class='label'>{html_escape(k)}</div><div class='value'>{html_escape(v)}</div></div>"
            for k, v in self._cohort_cards(cohort_inputs)[:6]
        )
        chapter_cards = "".join(
            f"<a class='chapterLinkCard' href='{html_escape(self._cohort_chapter_route(chapter))}'>"
            "<section class='tablePanel chapterCardPanel'>"
            "<div class='chapterCardHeader'>"
            "<div class='chapterCardTitleGroup'>"
            f"<div class='chapterCardKicker'>{html_escape(chapter.get('focus_label', 'Cohort chapter'))}</div>"
            f"<h3 class='chapterCardHeading'>{html_escape(chapter['title'].split(':')[-1].strip())}</h3>"
            f"<p class='subtitle'>{html_escape(chapter['subtitle'])}</p>"
            "</div>"
            f"<div class='chapterCardBadge'>{len(chapter['specs'])} figures</div>"
            "</div>"
            f"<p class='chapterCardDesc'>{html_escape(chapter.get('description') or chapter['subtitle'])}</p>"
            "<div class='chapterMetaGrid'>"
            f"<div class='chapterMetaItem'><div class='chapterMetaLabel'>Primary focus</div><div class='chapterMetaValue'>{html_escape(chapter['subtitle'])}</div></div>"
            f"<div class='chapterMetaItem'><div class='chapterMetaLabel'>Key note</div><div class='chapterMetaValue'>{html_escape(chapter['observations'][0] if chapter['observations'] else chapter.get('description', chapter['subtitle']))}</div></div>"
            "</div>"
            "<div class='chapterOpenRow'>"
            "<div class='chapterOpenHint'>Open this chapter to inspect the full figure set, supporting tables, and audit framing.</div>"
            "<div class='chapterOpenCta'>Open chapter</div>"
            "</div>"
            "</section>"
            "</a>"
            for chapter in chapters
        )
        chapter_menu_items_html = "".join(
            f"<a href='{html_escape(self._cohort_chapter_route(chapter))}'>{html_escape(chapter['title'].split(':')[-1].strip())}<span>{html_escape(chapter['subtitle'])}</span></a>"
            for chapter in chapters
        )
        masthead = self._shared_chrome(
            home_href="../index.html",
            logo_src="../../../../cltr/docs/assets/logos/cltr.png",
            page_type="Cohort Report",
            page_meta="CLTR Cohort Chapters",
            menu_button_id="sessionMenuButton",
            menu_panel_id="sessionMenuPanel",
            menu_label="Sessions",
            menu_title="Sessions",
            menu_items_html="<a href='../sessions_report.html'>Sessions Report<span>Dedicated session browser and participant-level reports</span></a>",
            secondary_actions_html_after=self._menu_button_html(
                button_id="chapterMenuButton",
                panel_id="chapterMenuPanel",
                label="Chapters",
                title="Cohort Chapters",
                items_html=chapter_menu_items_html,
            ),
        )
        takeaways = self._takeaways_html(
            [
                "The cohort report is now delivered as a chaptered suite so the full audit and result set remains navigable.",
                "Chapter 1 anchors the audit and governance tables; later chapters should be read against those decisions.",
                "A full combined export is still available for continuity and cross-checking.",
            ]
        )
        return f"""<!doctype html><html><head><meta charset='utf-8'><title>CLTR Cohort Report</title>
<style>
{self._shared_report_css()}
</style></head><body class='reportKind--cohort'>{masthead}<div class='page' id='pageRoot'><section class='hero'><div class='panel heroLead'><div class='eyebrow'>Cohort Report</div><div class='title'>CLTR Cohort Report</div><p class='subtitle'>Chapter index for the cohort audit and result suite.</p><div class='cards'>{cards_html}</div><div class='heroActions'><a class='radioCta' href='{html_escape(full_html_path.name)}'>Full Cohort Report</a></div></div><div class='panel heroSide'>{takeaways}</div></section><div class='reportShell'><section class='stack'><section class='sectionBlock'><h3 class='sectionTitle'>Cohort Chapters</h3><section class='chapterGrid'>{chapter_cards}</section></section></section></div></div><div class='copyrightNote'>{COPYRIGHT_NOTE}</div><script>
{self._theme_toggle_script()}
{self._menu_script(button_id='sessionMenuButton', panel_id='sessionMenuPanel', var_prefix='sessionMenu')}
{self._menu_script(button_id='chapterMenuButton', panel_id='chapterMenuPanel', var_prefix='chapterMenu')}
</script></body></html>"""

    def _all_sessions_html(self, manifest: pd.DataFrame, session_reports: list[dict], cohort_report: dict) -> str:
        records = []
        by_session = {r["session_id"]: r for r in session_reports}
        for row in manifest.to_dict("records"):
            report = by_session.get(str(row["session_id"]))
            if not report:
                continue
            evidence = max([spec["evidence_score"] for spec in report["figure_specs"]], default=0)
            records.append(
                {
                    "session_id": row["session_id"],
                    "participant_id": row["participant_id"],
                    "condition_code": row["condition_code"],
                    "html_name": Path(report["html_path"]).name,
                    "evidence_score": evidence,
                    "lead_label": report.get("lead_label", ""),
                    "headline": report.get("headline", ""),
                    "tags": ", ".join(report.get("atlas_tags", [])),
                    "tone_class": self._session_card_tone_class(str(row["condition_code"])),
                }
            )
        cards = "".join(
            f"<article class='sessionCard {html_escape(r['tone_class'])}'>"
            f"<div class='eyebrow'>{html_escape(r['condition_code'])}</div><h3>{html_escape(r['session_id'])}</h3><p><strong>Participant:</strong> {html_escape(r['participant_id'])}<br><strong>Condition:</strong> {html_escape(r['condition_code'])}<br><strong>Evidence score:</strong> {r['evidence_score']}</p><p><strong>Lead story:</strong> {html_escape(r['lead_label'])}</p><p>{html_escape(r['headline'])}</p><p class='tagLine'>{html_escape(r['tags'])}</p><a class='pillLink' href='sessions/{html_escape(r['session_id'])}/{html_escape(r['html_name'])}'>{SESSION_CTA}</a></article>"
            for r in records
        )
        session_nav_items = "".join(
            f"<a href='sessions/{html_escape(r['session_id'])}/{html_escape(r['html_name'])}' title='Open session report for {html_escape(r['session_id'])}'>"
            f"{html_escape(r['session_id'])}<span>{html_escape(r['participant_id'])} | {html_escape(r['condition_code'])}</span></a>"
            for r in records
        )
        intro_panel = (
            f"<section class='panel heroLead heroIntro'>"
            f"<div class='eyebrow'>Sessions Report</div>"
            f"<div class='title'>CLTR Sessions Report</div>"
            f"<p class='subtitle'>Browse all generated participant-session reports from one dedicated index.</p>"
            f"<div class='heroMeta'>"
            f"<p class='heroStatement'>This page is the session-level browsing layer of the CLTR report suite. Use it when you need per-session traces, condition context, and participant-specific evidence.</p>"
            f"<div class='heroFacts'>"
            f"<div class='heroFact'><div class='heroFactLabel'>Coverage</div><div class='heroFactValue'>{len(records)} generated session reports across the full study.</div></div>"
            f"<div class='heroFact'><div class='heroFactLabel'>Study-wide report</div><div class='heroFactValue'><a class='pillLink' href='{html_escape(self._canonical_cohort_href())}'>{COHORT_CTA}</a></div></div>"
            f"</div>"
            f"</div>"
            f"</section>"
        )
        masthead = self._shared_chrome(
            home_href="index.html",
            logo_src="../../../cltr/docs/assets/logos/cltr.png",
            page_type="Sessions Report",
            page_meta=f"{len(records)} generated session reports",
            menu_button_id="sessionMenuButton",
            menu_panel_id="sessionMenuPanel",
            menu_label="Sessions",
            menu_title="Session List",
            menu_items_html=session_nav_items,
            secondary_actions_html_after=self._menu_button_html(
                button_id="chapterMenuButton",
                panel_id="chapterMenuPanel",
                label="Chapters",
                title="Cohort Chapters",
                items_html=self._cohort_chapter_menu_items_html("cohort/"),
            ),
        )
        return f"""<!doctype html><html><head><meta charset='utf-8'><title>CLTR Sessions Report</title>
<style>
{self._shared_index_css()}
</style></head><body class='reportKind--atlas'>{masthead}<div class='page'><section class='hero'>{intro_panel}</section><section id='sessionGrid' class='grid'>{cards}</section></div><div class='copyrightNote'>{COPYRIGHT_NOTE}</div><script>
{self._theme_toggle_script()}
{self._menu_script(button_id='sessionMenuButton', panel_id='sessionMenuPanel', var_prefix='sessionMenu')}
{self._menu_script(button_id='chapterMenuButton', panel_id='chapterMenuPanel', var_prefix='chapterMenu')}
</script></body></html>"""

    def _atlas_html(self, manifest: pd.DataFrame, session_reports: list[dict], cohort_report: dict, sessions_index_name: str) -> str:
        session_count = len(session_reports)
        masthead = self._shared_chrome(
            home_href="index.html",
            logo_src="../../../cltr/docs/assets/logos/cltr.png",
            page_type="Atlas",
            page_meta=f"{session_count} session reports and one cohort summary",
            menu_button_id="atlasMenuButton",
            menu_panel_id="atlasMenuPanel",
            menu_label="Navigate",
            menu_title="Atlas Destinations",
            menu_items_html=(
                f"<a href='{html_escape(self._canonical_cohort_href())}'>Cohort Report<span>Study-wide audit, chapters, and full combined report</span></a>"
                f"<a href='{html_escape(sessions_index_name)}'>Sessions Report<span>Dedicated session browser and participant-level reports</span></a>"
            ),
            show_menu_button=False,
        )
        atlas_intro = (
            f"<section class='panel heroIntro heroSticky'>"
            f"<div class='eyebrow'>CLTR Reporting</div>"
            f"<div class='title'>{WORK_INDEX_TITLE}</div>"
            f"<p class='subtitle'>{WORK_INDEX_SUBTITLE}</p>"
            f"<div class='heroMeta'>"
            f"<p class='heroStatement'>This atlas is the top-level gateway for the CLTR reporting suite. Start with either the study-wide cohort synthesis or the dedicated session-report browser.</p>"
            f"<div class='heroFacts'>"
            f"<div class='heroFact'><div class='heroFactLabel'>Study</div><div class='heroFactValue'>Controlled Laboratory Thermal Response reporting hub.</div></div>"
            f"<div class='heroFact'><div class='heroFactLabel'>Coverage</div><div class='heroFactValue'>{session_count} session reports plus one study-wide cohort suite.</div></div>"
            f"</div>"
            f"</div>"
            f"</section>"
        )
        cohort_card = (
            f"<a class='panel gatewayCard heroSticky' href='{html_escape(self._canonical_cohort_href())}'>"
            f"<div class='eyebrow'>Primary Entry</div>"
            f"<div class='title'>Cohort Report</div>"
            f"<p class='subtitle'>Open the study-wide audit, chapter suite, and full combined cohort synthesis.</p>"
            f"<div class='gatewayMeta'>"
            f"<div class='gatewayFact'><div class='gatewayFactLabel'>Best for</div><div class='gatewayFactValue'>Study-level findings, device audit, scenario logic, and manuscript-facing figures.</div></div>"
            f"<div class='gatewayFact'><div class='gatewayFactLabel'>Includes</div><div class='gatewayFactValue'>Cohort chapters, full report, modality inventory, and validation layers.</div></div>"
            f"</div>"
            f"<div class='gatewayCta'>Open cohort report</div>"
            f"</a>"
        )
        sessions_card = (
            f"<a class='panel gatewayCard heroSticky' href='{html_escape(sessions_index_name)}'>"
            f"<div class='eyebrow'>Primary Entry</div>"
            f"<div class='title'>Sessions Report</div>"
            f"<p class='subtitle'>Open the dedicated session browser for participant-level reports, traces, and session-specific evidence.</p>"
            f"<div class='gatewayMeta'>"
            f"<div class='gatewayFact'><div class='gatewayFactLabel'>Best for</div><div class='gatewayFactValue'>Per-session physiology, environmental context, and within-session narrative inspection.</div></div>"
            f"<div class='gatewayFact'><div class='gatewayFactLabel'>Coverage</div><div class='gatewayFactValue'>{session_count} generated session reports across all participants and conditions.</div></div>"
            f"</div>"
            f"<div class='gatewayCta'>Open sessions report</div>"
            f"</a>"
        )
        return f"""<!doctype html><html><head><meta charset='utf-8'><title>{WORK_INDEX_TITLE}</title>
<style>
{self._shared_index_css()}
</style></head><body class='reportKind--atlas'>{masthead}<div class='page'><section class='hero'>{atlas_intro}</section><section class='gatewayGrid'>{cohort_card}{sessions_card}</section></div><div class='copyrightNote'>{COPYRIGHT_NOTE}</div><script>
{self._theme_toggle_script()}
</script></body></html>"""

    def _render_spec_sections(
        self,
        specs: list[dict],
        intro_sections: str = "",
        section_intro_map: dict[str, str] | None = None,
        chapter_number: int | None = None,
        figure_src_prefix: str = "figures/",
    ) -> str:
        display_map, section_map, kind_map = self._display_numbering(
            specs,
            intro_sections=intro_sections,
            section_intro_map=section_intro_map,
            chapter_number=chapter_number,
        )
        parts = []
        section_intro_map = section_intro_map or {}
        table_index = 0
        for section in SECTION_ORDER:
            section_specs = [spec for spec in specs if spec.get("display_section", "analyzed") == section]
            intro_html = intro_sections if section == "frontmatter" else section_intro_map.get(section, "")
            if not section_specs and not intro_html:
                continue
            if chapter_number is not None and intro_html:
                intro_html, table_index = self._number_table_panels(
                    intro_html,
                    chapter_number=chapter_number,
                    start_index=table_index,
                )
            body = "".join(
                self._spec_subsection(
                    spec,
                    display_map.get(spec["stem"], spec["code"]),
                    kind_map.get(spec["stem"], "figure"),
                    figure_src_prefix=figure_src_prefix,
                )
                for spec in section_specs
            )
            if intro_html:
                body = intro_html + body
            section_label = section_map.get(section, SECTION_TITLES[section])
            parts.append(f"<section class='sectionBlock'><h3 class='sectionTitle'>{html_escape(section_label)}</h3>{body}</section>")
        return "".join(parts)

    def _display_numbering(
        self,
        specs: list[dict],
        intro_sections: str = "",
        section_intro_map: dict[str, str] | None = None,
        chapter_number: int | None = None,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        display_map: dict[str, str] = {}
        section_map: dict[str, str] = {}
        kind_map: dict[str, str] = {}
        section_intro_map = section_intro_map or {}
        section_index = 0
        figure_counter = 0
        table_counter = 0
        for section in SECTION_ORDER:
            section_specs = [spec for spec in specs if spec.get("display_section", "analyzed") == section]
            intro_html = bool(intro_sections) if section == "frontmatter" else bool(section_intro_map.get(section))
            if not section_specs and not intro_html:
                continue
            section_index += 1
            if chapter_number is not None:
                section_map[section] = f"Section {chapter_number}.{section_index}. {SECTION_TITLES[section]}"
            else:
                section_map[section] = f"Section {section_index}. {SECTION_TITLES[section]}"
            local_figure_counter = 0
            local_table_counter = 0
            for local_figure_index, spec in enumerate(section_specs):
                display_kind = self._spec_display_kind(spec)
                kind_map[spec["stem"]] = display_kind
                if chapter_number is not None:
                    if display_kind == "table":
                        table_counter += 1
                        display_map[spec["stem"]] = f"{chapter_number}.{table_counter}"
                    else:
                        figure_counter += 1
                        display_map[spec["stem"]] = f"{chapter_number}.{figure_counter}"
                else:
                    if display_kind == "table":
                        local_table_counter += 1
                        display_map[spec["stem"]] = f"{section_index}.{local_table_counter}"
                    else:
                        local_figure_counter += 1
                        display_map[spec["stem"]] = f"{section_index}.{local_figure_counter}"
        return display_map, section_map, kind_map

    def _html_document(
        self,
        *,
        title: str,
        subtitle: str,
        cards: list[tuple[str, object]],
        observations: list[str],
        main_specs: list[dict],
        appendix_specs: list[dict],
        intro_sections: str = "",
        section_intro_map: dict[str, str] | None = None,
        chapter_number: int | None = None,
        doc_kind: str = "report",
        home_href: str | None = None,
        logo_src: str | None = None,
        figure_src_prefix: str = "figures/",
        hero_actions_html: str = "",
        middle_menu_button_id: str = "",
        middle_menu_panel_id: str = "",
        middle_menu_label: str = "",
        middle_menu_title: str = "",
        middle_menu_items_html: str = "",
        secondary_menu_button_id: str = "",
        secondary_menu_panel_id: str = "",
        secondary_menu_label: str = "",
        secondary_menu_title: str = "",
        secondary_menu_items_html: str = "",
    ) -> str:
        all_specs = main_specs + appendix_specs
        display_map, _, kind_map = self._display_numbering(
            all_specs,
            intro_sections=intro_sections,
            section_intro_map=section_intro_map,
            chapter_number=chapter_number,
        )
        has_tables = any(kind_map.get(spec["stem"]) == "table" for spec in all_specs)
        nav_items = "".join(
            f"<a href='#{html_escape(spec['stem'])}' aria-label='{html_escape(kind_map.get(spec['stem'], 'figure').title())} {html_escape(display_map.get(spec['stem'], spec['code']))}: {html_escape(spec['title'])}' title='{html_escape(kind_map.get(spec['stem'], 'figure').title())} {html_escape(display_map.get(spec['stem'], spec['code']))}: {html_escape(spec['title'])}'>{'Tab' if kind_map.get(spec['stem'], 'figure') == 'table' else 'Fig'} {html_escape(display_map.get(spec['stem'], spec['code']))}<span>{html_escape(spec['title'])}</span></a>"
            for spec in all_specs
        )
        cards_html = "".join(f"<div class='card'><div class='label'>{html_escape(k)}</div><div class='value'>{html_escape(v)}</div></div>" for k, v in cards)
        obs_html = self._takeaways_html(observations)
        sections_html = self._render_spec_sections(
            all_specs,
            intro_sections=intro_sections,
            section_intro_map=section_intro_map,
            chapter_number=chapter_number,
            figure_src_prefix=figure_src_prefix,
        )
        plotly_js = f"<script>{get_plotlyjs()}</script>" if any(spec.get("html_fragment") for spec in all_specs) else ""
        badge = "Cohort Report" if str(doc_kind) == "cohort" else "Session Report"
        home_href = home_href or ("../index.html" if str(doc_kind) == "cohort" else "../../index.html")
        masthead = self._shared_chrome(
            home_href=home_href,
            logo_src=logo_src or ("../../../../cltr/docs/assets/logos/cltr.png" if str(doc_kind) == "cohort" else "../../../../../cltr/docs/assets/logos/cltr.png"),
            page_type=badge,
            page_meta=title,
            menu_button_id="figureMenuButton",
            menu_panel_id="figureMenuPanel",
            menu_label="Figures And Tables" if has_tables else "List of Figures",
            menu_title="Figures And Tables" if has_tables else "List of Figures",
            menu_items_html=nav_items,
            secondary_actions_html_after=(
                (
                    self._menu_button_html(
                        button_id=middle_menu_button_id,
                        panel_id=middle_menu_panel_id,
                        label=middle_menu_label,
                        title=middle_menu_title,
                        items_html=middle_menu_items_html,
                    )
                    if middle_menu_button_id and middle_menu_panel_id and middle_menu_label and middle_menu_title
                    else ""
                )
                +
                (
                    self._menu_button_html(
                        button_id=secondary_menu_button_id,
                        panel_id=secondary_menu_panel_id,
                        label=secondary_menu_label,
                        title=secondary_menu_title,
                        items_html=secondary_menu_items_html,
                    )
                )
                if secondary_menu_button_id and secondary_menu_panel_id and secondary_menu_label and secondary_menu_title
                else ""
            ),
        )
        return f"""<!doctype html><html><head><meta charset='utf-8'><title>{html_escape(title)}</title>
<style>
{self._shared_report_css()}
</style></head><body class='reportKind--{html_escape(doc_kind)}'>{masthead}<div class='page' id='pageRoot'><section class='hero'><div class='panel heroLead'><div class='eyebrow'>{html_escape(badge)}</div><div class='title'>{html_escape(title)}</div><p class='subtitle'>{html_escape(subtitle)}</p><div class='cards'>{cards_html}</div>{hero_actions_html}</div><div class='panel heroSide'>{obs_html}</div></section><div class='reportShell'><section class='stack'>{sections_html}</section></div></div><div id='lightbox' class='lightbox'><img id='lightboxImg' alt='Expanded figure'/></div><div class='copyrightNote'>{COPYRIGHT_NOTE}</div>{plotly_js}<script>
{self._theme_toggle_script()}
const lightbox=document.getElementById('lightbox'); const lightboxImg=document.getElementById('lightboxImg'); document.querySelectorAll('.figureImage').forEach(img=>img.addEventListener('click',()=>{{ lightboxImg.src=img.src; lightbox.classList.add('open'); }})); lightbox.addEventListener('click',()=>lightbox.classList.remove('open'));
const figureMenuButton=document.getElementById('figureMenuButton'); const figureMenuPanel=document.getElementById('figureMenuPanel');
const closeFigureMenu=()=>{{ if(!figureMenuPanel||!figureMenuButton) return; figureMenuPanel.classList.remove('open'); figureMenuButton.setAttribute('aria-expanded','false'); }};
const toggleFigureMenu=()=>{{ if(!figureMenuPanel||!figureMenuButton) return; const open=figureMenuPanel.classList.toggle('open'); figureMenuButton.setAttribute('aria-expanded', open ? 'true' : 'false'); }};
if(figureMenuButton&&figureMenuPanel){{ figureMenuButton.addEventListener('click',(event)=>{{ event.stopPropagation(); toggleFigureMenu(); }}); figureMenuPanel.querySelectorAll('a').forEach(link=>link.addEventListener('click', closeFigureMenu)); document.addEventListener('click',(event)=>{{ if(!figureMenuPanel.contains(event.target) && !figureMenuButton.contains(event.target)) closeFigureMenu(); }}); document.addEventListener('keydown',(event)=>{{ if(event.key==='Escape') closeFigureMenu(); }}); }}
{self._menu_script(button_id=middle_menu_button_id, panel_id=middle_menu_panel_id, var_prefix='sessionMenu') if middle_menu_button_id and middle_menu_panel_id else ''}
const chapterMenuButton=document.getElementById('{html_escape(secondary_menu_button_id)}'); const chapterMenuPanel=document.getElementById('{html_escape(secondary_menu_panel_id)}');
const closeChapterMenu=()=>{{ if(!chapterMenuPanel||!chapterMenuButton) return; chapterMenuPanel.classList.remove('open'); chapterMenuButton.setAttribute('aria-expanded','false'); }};
const toggleChapterMenu=()=>{{ if(!chapterMenuPanel||!chapterMenuButton) return; const open=chapterMenuPanel.classList.toggle('open'); chapterMenuButton.setAttribute('aria-expanded', open ? 'true' : 'false'); }};
if(chapterMenuButton&&chapterMenuPanel){{ chapterMenuButton.addEventListener('click',(event)=>{{ event.stopPropagation(); toggleChapterMenu(); }}); chapterMenuPanel.querySelectorAll('a').forEach(link=>link.addEventListener('click', closeChapterMenu)); document.addEventListener('click',(event)=>{{ if(!chapterMenuPanel.contains(event.target) && !chapterMenuButton.contains(event.target)) closeChapterMenu(); }}); document.addEventListener('keydown',(event)=>{{ if(event.key==='Escape') closeChapterMenu(); }}); }}
const resizePlots=()=>{{ if(!window.Plotly) return; document.querySelectorAll('.js-plotly-plot').forEach(plot=>window.Plotly.Plots.resize(plot)); }};
window.addEventListener('resize', resizePlots); requestAnimationFrame(resizePlots);
</script></body></html>"""

    def _takeaways_html(self, observations: list[str]) -> str:
        cleaned = [str(x).strip() for x in observations if str(x).strip()]
        lead = cleaned[0] if cleaned else "Highlights will appear here once report observations are available."
        supporting = cleaned[1:]
        items = "".join(
            f"<div class='takeawayItem'><div class='takeawayIndex'>{idx}</div><p class='takeawayText'>{html_escape(text)}</p></div>"
            for idx, text in enumerate(supporting, start=1)
        )
        return (
            "<section class='takeawayPanel'>"
            "<div class='takeawayHeader'><h2>Key Takeaways</h2><div class='takeawayBadge'>Summary</div></div>"
            f"<div class='takeawayLead'><div class='takeawayLeadLabel'>Main Finding</div><p class='takeawayLeadText'>{html_escape(lead)}</p></div>"
            f"<div class='takeawayList'>{items}</div>"
            "</section>"
        )

    def _spec_display_kind(self, spec: dict) -> str:
        html_fragment = str(spec.get("html_fragment") or "")
        if "<table" in html_fragment and "tablePanel" in html_fragment:
            return "table"
        return "figure"

    def _spec_subsection(self, spec: dict, display_code: str, display_kind: str, figure_src_prefix: str = "figures/") -> str:
        section_label = f"{display_kind.title()} {display_code}"
        return f"<section class='figureSection'><h3 class='figureSectionTitle'>{html_escape(section_label)}</h3>{self._figure_block(spec, figure_src_prefix=figure_src_prefix)}</section>"

    def _caption_text(self, text: str) -> str:
        cleaned = " ".join(str(text or "").split())
        replacements = [
            ("This panel ", ""),
            ("This exploratory panel ", ""),
            ("This summary ", ""),
            ("This opening panel ", ""),
        ]
        for old, new in replacements:
            if cleaned.startswith(old):
                cleaned = new + cleaned[len(old):]
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned

    def _panel_guide_html(self, panel_notes: list[str]) -> str:
        parts = []
        for note in panel_notes:
            text = str(note).strip().rstrip(".")
            if not text:
                continue
            panel_label = ""
            panel_title = ""
            body = ""
            if "|" in text:
                pieces = [piece.strip() for piece in text.split("|", 2)]
                if len(pieces) == 3:
                    panel_label, panel_title, body = pieces
                elif len(pieces) == 2:
                    panel_label, panel_title = pieces
            else:
                label, sep, tail = text.partition(" shows ")
                if sep:
                    panel_label = label.strip()
                    body = tail.strip()
                else:
                    label, sep, tail = text.partition(":")
                    panel_label = label.strip()
                    body = tail.strip()
            panel_label = " ".join(panel_label.split()).title()
            panel_title = " ".join(panel_title.split())
            body = body.strip()

            if panel_label and panel_title:
                label_html = f"<strong>[{html_escape(panel_label)}] {html_escape(panel_title)}:</strong>"
            elif panel_label:
                label_html = f"<strong>[{html_escape(panel_label)}]</strong>"
            else:
                label_html = ""
            body_html = html_escape(body) if body else ""
            parts.append(f"{label_html} {body_html}".strip())
        if not parts:
            return ""
        return "; ".join(parts)

    def _caption_html(self, summary: str, note: str = "", panel_notes: list[str] | None = None) -> str:
        panel_notes = panel_notes or []
        chunks = []
        panel_html = self._panel_guide_html(panel_notes)
        if panel_html:
            chunks.append(panel_html)
        summary_text = self._caption_text(summary)
        if summary_text:
            chunks.append(html_escape(summary_text))
        note_html = " ".join(str(note or "").split())
        if note_html:
            chunks.append(note_html)
        return " ".join(chunks).strip()

    def _figure_block(self, spec: dict, figure_src_prefix: str = "figures/") -> str:
        path = Path(spec["path"]).name if spec.get("path") else ""
        meta_parts = [f"Evidence: {spec['evidence_label']} ({int(spec['evidence_score'])})"]
        if spec.get("gating_note"):
            meta_parts.append(f"Gate: {spec['gating_note']}")
        meta = f"<p class='figureMeta'>{html_escape(' | '.join(meta_parts))}</p>"
        classes = "figurePanel"
        if spec.get("html_fragment"):
            fragment = str(spec["html_fragment"])
            if "dataTablePanel" in fragment:
                fragment = re.sub(r"<h3>.*?</h3>", "", fragment, count=1, flags=re.DOTALL)
            media = f"<div class='responsiveFigure'>{fragment}</div>"
        else:
            media = f"<img class='figureImage' src='{html_escape(figure_src_prefix)}{html_escape(path)}' alt='{html_escape(spec['title'])}'/>"
        caption = self._caption_html(spec.get("summary", ""), spec.get("caption_note", ""), spec.get("panel_notes", []))
        heading = f"<h2>{html_escape(spec['title'])}</h2>"
        return f"<article id='{html_escape(spec['stem'])}' class='{classes}'>{heading}{media}{meta}<p class='caption'>{caption}</p></article>"

    def _render_table(self, df: pd.DataFrame, title: str, columns: list[str] | None = None, n: int = 8) -> str:
        if df is None or df.empty:
            return ""
        view = df.copy()
        if columns:
            keep = [c for c in columns if c in view.columns]
            view = view[keep]
        view = view.head(n)
        headers = "".join(
            f"<th class='{html_escape(self._table_column_class(c))}'>{html_escape(self._table_column_label(c))}</th>"
            for c in view.columns
        )
        rows = []
        for _, row in view.iterrows():
            rows.append(
                "<tr>" + "".join(
                    f"<td class='{html_escape(self._table_column_class(col))}'>{html_escape(self._fmt_cell(val, column=col))}</td>"
                    for col, val in zip(view.columns, row.tolist())
                ) + "</tr>"
            )
        return f"<section class='tablePanel dataTablePanel'><h3>{html_escape(title)}</h3><div class='tableScroll'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>"

    def _number_table_panels(self, html: str, *, chapter_number: int, start_index: int = 0) -> tuple[str, int]:
        counter = start_index

        def repl(match: re.Match[str]) -> str:
            nonlocal counter
            counter += 1
            return f"{match.group(1)}Table {chapter_number}.{counter}. {match.group(2)}{match.group(3)}"

        numbered = re.sub(
            r"(<section class='tablePanel dataTablePanel'>\s*<h3>)(.*?)(</h3>)",
            repl,
            html,
            flags=re.DOTALL,
        )
        return numbered, counter

    def _table_column_class(self, value: object) -> str:
        text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
        return f"col-{text}" if text else "col-generic"

    def _table_column_label(self, value: object) -> str:
        text = str(value)
        if text in FEATURE_LABELS:
            return FEATURE_LABELS[text]
        return TABLE_COLUMN_LABELS.get(text, text.replace("_", " ").title())

    def _fmt_cell(self, value: object, column: str | None = None) -> str:
        if isinstance(value, float):
            if column and column in {"p_value", "p_value_fdr", "primary_p_value", "t_p_value", "wilcoxon_p_value", "spearman_p_value", "spearman_p_value_fdr"}:
                if not pd.notna(value):
                    return ""
                value = float(value)
                if value == 0:
                    return "<1e-300"
                if value < 1e-3:
                    return f"{value:.2e}"
                return f"{value:.4f}"
            if pd.notna(value) and float(value).is_integer():
                return str(int(value))
            return f"{value:.3f}" if pd.notna(value) else ""
        if isinstance(value, str):
            replacements = {
                "descriptive_only": "Directional summary",
                "inferential": "Full comparison",
                "eligible": "Comparable",
                "insufficient_pairs": "Too few matched pairs",
                "not_eligible": "Limited overlap",
                "strong": "High",
                "moderate": "Moderate",
                "weak": "Limited",
                "partial": "Partial",
                "insufficient": "Insufficient",
                "retained descriptive association": "Retained descriptive association",
                "insufficient paired support": "Insufficient paired support",
                "same-sign across phases": "Same-sign across phases",
                "same-sign across conditions": "Same-sign across conditions",
                "limited phase support": "Limited phase support",
                "limited condition support": "Limited condition support",
                "questionnaire_completeness": "Questionnaire completeness",
                "questionnaire_event_completeness": "Questionnaire event completeness",
                "questionnaire_response_completeness": "Questionnaire response completeness",
                "empatica_fraction": "Empatica coverage",
                "biopac_fraction": "BIOPAC coverage",
                "indoor_fraction": "Indoor sensor coverage",
                "hr_overlap_minutes": "Heart-rate overlap (min)",
                "eda_overlap_minutes": "EDA overlap (min)",
                "temp_overlap_minutes": "Temperature overlap (min)",
                "primary": "Primary",
                "primary_with_qc": "Primary with QC",
                "secondary_only": "Secondary only",
                "secondary_validation": "Secondary validation",
                "subset_only": "Subset only",
                "not_primary": "Not primary",
                "not_recommended": "Not recommended",
                "usable_with_caution": "Usable with caution",
                "yes": "Yes",
                "no": "No",
                "not_audited": "Not audited",
                "all_sources": "All-source",
                "valid_only": "Valid-only",
                "directly_comparable": "Directly comparable pair",
                "device_specific": "Device-specific stream",
                "source_only": "Source-only stream",
                "same_construct_not_paired": "Same construct, not paired",
                "valid-only eligible": "Valid-only eligible",
                "audit-only if included": "Audit-only if included",
                "derived/context endpoint": "Derived/context endpoint",
                "stream-role unclear": "Stream-role unclear",
                "primary_outcome": "Primary outcome",
                "primary_physiological": "Primary physiological",
                "primary_physiological_qc": "QC-qualified primary physiological",
                "protocol_check": "Primary protocol check",
                "secondary_mechanistic": "Secondary mechanistic",
                "audit_only": "Audit-only",
                "not_endpoint_graded": "Not endpoint-graded",
                "direct_analytic_feature": "Direct analytic feature",
                "audit_report_only": "Audit/report-only stream",
            }
            if value in FEATURE_LABELS:
                return FEATURE_LABELS[value]
            if value in replacements:
                return replacements[value]
            signal_labels = {
                "biopac_hr": "BIOPAC HR",
                "empatica_hr": "Empatica HR",
                "biopac_eda": "BIOPAC EDA",
                "empatica_eda": "Empatica EDA",
                "biopac_temp": "BIOPAC chest temperature",
                "empatica_temp": "Empatica temperature",
                "empatica_bvp": "Empatica BVP",
                "empatica_acc": "Empatica acceleration",
                "empatica_enmo": "Empatica ENMO",
                "empatica_steps": "Empatica steps",
                "biopac_temp_thigh": "BIOPAC thigh temperature",
                "biopac_temp_arm": "BIOPAC arm temperature",
                "biopac_temp_tibia": "BIOPAC tibia temperature",
                "biopac_bloodflow": "BIOPAC blood flow",
                "biopac_backscatter": "BIOPAC backscatter",
                "heart_rate": "Heart rate",
                "eda": "EDA",
                "temperature": "Temperature",
                "bvp_source": "BVP source",
                "motion": "Motion",
                "activity": "Activity",
                "temperature_site": "Temperature site",
                "bloodflow": "Blood flow",
                "optical": "Optical",
                "strong": "Strong",
                "limited": "Limited",
                "weak": "Weak",
            }
            if value in signal_labels:
                return signal_labels[value]
            phase_labels = {
                "acclimation": "Acclimation",
                "fan_at_constant_speed": "Fan at constant speed",
                "fan_free_control": "Fan free control",
                "overall_comfort": "Overall comfort",
                "skin_rewarming": "Skin rewarming",
                "steady_state": "Steady state",
            }
            if value in phase_labels:
                return phase_labels[value]
            if "-" in value and value.upper() == value:
                return value
            return value.replace("_", " ")
        return str(value)

    def _report_metric_label(self, metric: str) -> str:
        metric = str(metric)
        if metric in REPORT_METRIC_LABELS:
            return REPORT_METRIC_LABELS[metric]
        if metric in FEATURE_LABELS:
            return FEATURE_LABELS[metric]
        return metric.replace("_", " ").title()

    def _series_title(self, metric: str, *, scope: str, kind: str) -> str:
        label = self._report_metric_label(metric)
        kind = REPORT_METRIC_KINDS.get(str(metric), kind)
        if kind == "observation":
            suffix = "observations"
        elif kind == "distribution":
            suffix = "distributions"
        else:
            suffix = kind
        return f"{scope} {label} {suffix}"

    def _cohort_metric_section(self, metric: str) -> str:
        metric = str(metric)
        if self._is_sparse_observation_channel(metric) or metric in {"fan_control_au", "fan_control_secondary_au", "fan_current_A"}:
            return "subjective_behavioral"
        if metric.startswith("indoor_") or metric.startswith("outdoor_"):
            return "environmental"
        return "physiological"

    def _session_report_tables(self, phase: pd.DataFrame, meta: dict) -> str:
        cols = [
            "protocol_block",
            "protocol_phase",
            "n_minutes",
            "thermal_comfort",
            "master_dpg_C",
            "indoor_air_velocity_mean_m_s",
            "biopac_temp_chest_mean_C",
            "empatica_hr_mean_bpm",
        ]
        support_df = pd.DataFrame(
            [
                {"metric": "questionnaire_completeness", "value": meta["support"]["questionnaire_completeness"]},
                {"metric": "questionnaire_event_completeness", "value": meta["support"]["questionnaire_event_completeness"]},
                {"metric": "questionnaire_response_completeness", "value": meta["support"]["questionnaire_response_completeness"]},
                {"metric": "empatica_fraction", "value": meta["support"]["empatica_fraction"]},
                {"metric": "biopac_fraction", "value": meta["support"]["biopac_fraction"]},
                {"metric": "indoor_fraction", "value": meta["support"]["indoor_fraction"]},
                {"metric": "hr_overlap_minutes", "value": meta["sensor_overlap_minutes"]["heart_rate"]},
                {"metric": "eda_overlap_minutes", "value": meta["sensor_overlap_minutes"]["eda"]},
                {"metric": "temp_overlap_minutes", "value": meta["sensor_overlap_minutes"]["temperature"]},
            ]
        )
        return f"<section class='tableGrid'>{self._render_table(phase, 'Average Results By Phase', cols, n=12)}{self._render_table(support_df, 'Data Coverage And Device Overlap', ['metric','value'], n=12)}</section>"

    def _cohort_report_tables(self, c: dict) -> str:
        sample = c.get("sample_status", pd.DataFrame()).copy()
        support = c.get("condition_support_summary", pd.DataFrame()).copy()
        agreement = c.get("agreement_summary", pd.DataFrame()).copy()
        signal_audit = c.get("signal_audit_summary", pd.DataFrame()).copy()
        session_signal_audit = c.get("session_signal_audit", pd.DataFrame()).copy()
        support_profile = self._cohort_endpoint_support_profile(c.get("cohort_phase_summary", pd.DataFrame()))
        scenario_register = self._scenario_register(signal_audit)
        inventory = self._device_stream_inventory_register(c.get("cohort_minute_features", pd.DataFrame()), signal_audit)
        pathway = self._analysis_pathway_register(c.get("cohort_minute_features", pd.DataFrame()), support_profile, signal_audit)
        comparable_inventory = inventory.loc[inventory["comparison_class"] == "directly_comparable"].copy() if not inventory.empty else pd.DataFrame()
        non_comparable_inventory = inventory.loc[inventory["comparison_class"] != "directly_comparable"].copy() if not inventory.empty else pd.DataFrame()
        b = self._render_table(
            support,
            "Comparison-Window Support By Condition",
            [
                "condition_code",
                "n_sessions",
                "n_participants",
                "questionnaire_completeness__mean",
                "empatica_fraction__mean",
                "biopac_fraction__mean",
                "indoor_fraction__mean",
                "hr_overlap_minutes__mean",
                "eda_overlap_minutes__mean",
                "temp_overlap_minutes__mean",
            ],
            12,
        )
        scenario_table = self._render_table(
            scenario_register,
            "Scenario Definitions Across Device Streams",
            ["scenario", "included_streams", "excluded_streams", "scientific_use"],
            8,
        )
        ctab = self._render_table(
            agreement,
            "Cross-Device Comparable Stream Summary",
            ["metric", "n_sessions", "n_eligible_sessions", "median_overlap_minutes", "median_spearman_r", "median_mae", "summary_status"],
            8,
        )
        comparable_table = self._render_table(
            comparable_inventory,
            "Cross-Device Comparability Register",
            ["stream_label", "device", "construct", "comparison_class", "signal_audited", "recommended_role", "adequacy_status"],
            16,
        )
        signal_table = self._render_table(
            signal_audit,
            "Full Signal Audit Across Device Streams",
            [
                "signal_stream",
                "device",
                "construct",
                "mean_valid_minutes",
                "mean_coverage_fraction",
                "mean_plausible_fraction",
                "median_overlap_minutes",
                "median_spearman_r",
                "adequacy_score",
                "recommended_role",
            ],
            24,
        )
        signal_reading = self._render_table(
            signal_audit,
            "Scientific Reading Of Device Streams",
            ["signal_stream", "adequacy_status", "flagged_sessions", "scientific_reading"],
            24,
        )
        flagged_streams = self._flagged_stream_session_register(session_signal_audit)
        flag_table = self._render_table(
            flagged_streams,
            "High-Concern Device Streams Across Sessions",
            ["signal_stream", "device", "construct", "flagged_session_streams", "affected_sessions", "primary_concern_driver", "concern_profile", "max_concern_score", "top_flagged_sessions"],
            24,
        )
        inventory_table = self._render_table(
            inventory,
            "Empatica And BIOPAC Stream Inventory",
            ["stream_label", "device", "construct", "comparison_class", "present_in_cohort_table", "signal_audited", "cross_device_comparable", "analytic_feature", "stream_usage", "endpoint_policy_role", "recommended_role", "adequacy_status"],
            24,
        )
        pathway_table = self._render_table(
            pathway,
            "Endpoint Analysis Pathway Across Full Modality Set",
            ["endpoint", "metric", "source_streams", "in_cohort_table", "support_grade", "support_basis", "endpoint_policy_role", "pathway_status"],
            36,
        )
        non_comparable_table = self._render_table(
            non_comparable_inventory,
            "Device-Specific, Same-Construct, And Source-Only Stream Register",
            ["stream_label", "device", "construct", "comparison_class", "analytic_feature", "stream_usage", "endpoint_policy_role", "recommended_role", "adequacy_status", "scientific_use"],
            24,
        )
        return (
            f"<section class='tableGrid'>{b}{scenario_table}</section>"
            f"<section class='tableGrid'>{ctab}{comparable_table}</section>"
            f"<section class='tableGrid'>{signal_table}{signal_reading}</section>"
            f"<section class='tableGrid'>{inventory_table}{pathway_table}</section>"
            f"<section class='tableGrid'>{non_comparable_table}{flag_table}</section>"
        )

    def _stage_panel(self, title: str, body: str) -> str:
        return f"<section class='tablePanel'><h3>{html_escape(title)}</h3><p>{html_escape(body)}</p></section>"

    def _section_lead(self, title: str, body: str) -> str:
        return (
            "<section class='tablePanel'>"
            f"<h3>{html_escape(title)}</h3>"
            f"<p>{html_escape(body)}</p>"
            "</section>"
        )

    def _section_lead_list(self, title: str, items: list[str]) -> str:
        bullets = "".join(f"<li>{html_escape(item)}</li>" for item in items if str(item).strip())
        return (
            "<section class='tablePanel'>"
            f"<h3>{html_escape(title)}</h3>"
            f"<ul>{bullets}</ul>"
            "</section>"
        )

    def _session_stage_sections(self, session_inputs: dict, phase: pd.DataFrame, meta: dict) -> str:
        story = self._session_story_profile(session_inputs)
        lead = story["lead_label"].replace("-", " ").replace("/", " / ")
        support = meta["support"]
        overlap = meta["sensor_overlap_minutes"]
        session_minutes = len(session_inputs.get("aligned_df", pd.DataFrame()))
        overview = self._stage_panel(
            "Session Summary",
            (
                f"This session spans {session_minutes} timeline minutes under condition {meta['condition_code']}. "
                f"The main pattern in this report is {lead}. "
                f"Questionnaire completeness is {support['questionnaire_completeness']:.1%} across Blocks 1 to 3, Empatica coverage is {support['empatica_fraction']:.1%}, "
                f"BIOPAC coverage is {support['biopac_fraction']:.1%}, and indoor sensor coverage is {support['indoor_fraction']:.1%}. "
                f"Paired-device overlap is {int(overlap['heart_rate'])} minutes for heart rate, {int(overlap['eda'])} minutes for EDA, "
                f"and {int(overlap['temperature'])} minutes for temperature."
            ),
        )
        return overview + self._session_report_tables(phase, meta)

    def _session_section_intros(self) -> dict[str, str]:
        hr_lo = 35
        hr_hi = 180
        overlap_min = int(self.config.runtime.min_sensor_overlap_minutes)
        outdoor_tol = 10
        phase_min = int(self.config.runtime.min_phase_minutes)
        return {
            "processed_cleaned": self._section_lead_list(
                "Processing And Cleaning Methods",
                [
                    "All sources are placed on the 1-minute session grid and merged by minute timestamp.",
                    "Questionnaire events are kept as discrete minute-level observations rather than converted into continuous traces.",
                    "Indoor temperature and air-velocity channels are averaged across concurrent probes for each minute.",
                    f"Empatica BVP, EDA, and temperature are aggregated to per-minute summaries; accelerometer is reduced to magnitude and ENMO; steps are summed by minute.",
                    f"Empatica heart rate is derived from systolic-peak intervals and values outside {hr_lo} to {hr_hi} bpm are removed before minute averaging.",
                    "BIOPAC channels are converted to numeric values and averaged by minute.",
                    f"Outdoor records are matched to the nearest session minute within {outdoor_tol} minutes.",
                    "Questionnaire completeness is defined from the study-wide expected event templates, and expected question slots are retained when present in at least 95% of those events.",
                    f"Agreement and support checks use a minimum paired-overlap requirement of {overlap_min} minutes.",
                    f"Phase summaries flag segments shorter than {phase_min} minutes as below the reporting minimum.",
                ],
            ),
            "alignment_support": self._section_lead(
                "Why This Comes After Raw Data",
                (
                    "This section follows the cleaned signal views and shows the shared-timeline support layer. "
                    "It shows how the recorded streams line up in Blocks 1 to 3, where support is present, "
                    "and which time windows are comparable before any derived summaries are interpreted."
                ),
            ),
            "derived": self._section_lead_list(
                "How To Read The Derived Results",
                [
                    "This section is support-gated: only endpoints with adequate repeated support are carried into the primary result layer.",
                    "Phase-level values are descriptive medians computed from processed comparison-window summaries, not inferential model estimates.",
                    "Questionnaire endpoints use questionnaire-response support; continuous endpoints use valid minute-summary support.",
                    "Reference-phase deltas are descriptive departures from the earliest supported comparison phase for each endpoint.",
                    "Directional agreement quantifies how often repeated blocks share the same sign of change; it should not be read as a formal reproducibility coefficient.",
                    "Endpoints with incomplete support are moved to the partial-results register instead of being presented as primary findings.",
                ],
            ),
            "agreement_section": self._section_lead_list(
                "How To Read The Relationships",
                [
                    "The first relationship panels are support-gated and include only endpoints retained as primary results in Section 7.",
                    "Pairwise relationships are descriptive Spearman associations on aligned comparison-window data, not causal effects.",
                    "Questionnaire-linked pairs require fewer paired observations than continuous-continuous pairs, but still must meet a predeclared support threshold.",
                    "The device-agreement panels that follow are technical validation figures and should not be interpreted as scientific associations between constructs.",
                ],
            ),
        }

    def _cohort_stage_sections(self, c: dict) -> str:
        sample = c["sample_status"].iloc[0]
        inferential = bool(sample["cohort_inference_eligible"])
        signal_audit = c.get("signal_audit_summary", pd.DataFrame()).copy()
        modality_summary = ""
        if not signal_audit.empty:
            primary = signal_audit.loc[signal_audit["recommended_role"].astype(str).isin(["primary", "primary_with_qc"]), "signal_stream"].astype(str).tolist()
            limited = signal_audit.loc[signal_audit["recommended_role"].astype(str).isin(["secondary_only", "secondary_validation", "subset_only", "not_primary", "not_recommended"]), "signal_stream"].astype(str).tolist()
            pieces = []
            if primary:
                pieces.append("claim-supporting device streams are " + ", ".join(self._fmt_cell(x) for x in primary))
            if limited:
                pieces.append("audit-only or limited streams are " + ", ".join(self._fmt_cell(x) for x in limited))
            if pieces:
                modality_summary = " In the current release, " + "; ".join(pieces) + "."
        synopsis = self._stage_panel(
            "Synopsis",
            (
                "This cohort report is the final synthesis layer of the CLTR pipeline: session timelines are aligned first, modality-specific minute summaries are built next, the full Empatica and BIOPAC stream inventory is audited, comparable and source-only modalities are separated, and only then are cohort-level patterns, contrasts, and device conclusions presented."
                + modality_summary
                + " The overview tables distinguish the full stream inventory from the narrower directly comparable subset so the paired-device agreement layer is not mistaken for the full modality picture."
                + " "
                + (
                    "The study includes enough sessions and participants for cross-session comparison, but the stream inventory, signal audit, and pathway tables below should still be read as part of the result itself, because device adequacy and disagreement determine which endpoints can be defended scientifically."
                    if inferential
                    else "The current sample should still be read as descriptive and support-gated rather than fully inferential."
                )
            ),
        )
        return synopsis + self._cohort_report_tables(c)

    def _cohort_section_intros(self, c: dict) -> dict[str, str]:
        sample = c["sample_status"].iloc[0]
        inferential = bool(sample["cohort_inference_eligible"])
        support_profile = self._cohort_endpoint_support_profile(c.get("cohort_phase_summary", pd.DataFrame()))
        has_strong = bool((support_profile.get("support_grade", pd.Series(dtype=str)) == "strong").any()) if not support_profile.empty else False
        derived_opening = (
            "This section is support-gated: only endpoints with broad condition-by-phase support are carried into the primary cohort result layer."
            if has_strong
            else "This section remains support-gated, but no endpoint reaches strong cohort support in the current sample, so the matrices and heatmaps are descriptive views of partial-support endpoints only."
        )
        relationship_opening = (
            "The first relationship panels are support-gated and include only endpoints retained in the cohort result layer."
            if has_strong
            else "The first relationship panels remain support-gated and are limited to the partial-support endpoints retained in the descriptive cohort result layer."
        )
        sample_reading = (
            "The current sample supports full cross-session comparison, but the cohort result layer should still be read through its endpoint support screen before patterns are generalized."
            if inferential
            else "The current sample is still relatively small, so the cohort result layer should be read as descriptive and support-gated rather than inferential."
        )
        return {
            "subjective_behavioral": self._section_lead_list(
                "Questionnaire Responses And Fan Behavior At A Glance",
                [
                    "This section brings together participant questionnaire responses and fan-setting behavior, showing both the raw observations and the summarized condition patterns.",
                    "Questionnaire panels should be read as ordinal response summaries collected at specific survey moments, not as continuous physiological signals.",
                    "Fan-control and fan-current panels provide behavioral context for how participants interacted with the experimental environment over time.",
                ],
            ),
            "physiological": self._section_lead_list(
                "Physiological Signals At A Glance",
                [
                    "This section brings together the physiological signals captured by Empatica and BIOPAC, including both raw cohort views and condition-level summaries.",
                    "These figures should be read as descriptive summaries of measured support, signal behavior, and cross-condition patterns rather than as stand-alone inferential results.",
                    "Where scenario figures are shown, they distinguish the full audited signal set from the subset retained for claim-supporting interpretation after the modality audit.",
                ],
            ),
            "environmental": self._section_lead_list(
                "Environmental Conditions At A Glance",
                [
                    "This section summarizes the indoor and outdoor environmental conditions measured during the study, including both time-aligned traces and condition-level summaries.",
                    "These figures provide the physical context for the participant responses and physiological signals, showing how air temperature, air movement, humidity, and related ambient conditions varied across the protocol.",
                    "Environmental panels are descriptive context figures and are intended to be read alongside the questionnaire and physiological sections rather than as stand-alone outcome measures.",
                ],
            ),
            "analyzed": self._section_lead_list(
                "Scientific Results And Modeling At A Glance",
                [
                    derived_opening,
                    "This chapter is structured as a policy-gated scientific result layer: aligned readiness and feature registries appear first, then support-screened descriptive matrices, then inferential models, temporal lag evidence, threshold-response fits, validation-aware predictive benchmarks, decision-layer summaries, and robustness checks.",
                    "Condition-phase values and delta matrices remain descriptive views of the endpoint layer, while the contrast and mixed-effects panels provide the inferential model-based layer under explicit eligibility gates.",
                    "All-source versus valid-only scenario matrices are retained here as sensitivity views so modality-inclusion effects remain explicit rather than hidden behind a single result table.",
                    "Pattern and participant-profile figures remain in the chapter to expose recurrent motifs and heterogeneity, so cohort means are not mistaken for uniform participant behavior.",
                    "Predictive benchmarks should be read as multimodal generalization checks under explicit holdout schemes rather than as deployment-ready classifiers.",
                    sample_reading,
                ],
            ),
            "interpretive": self._section_lead_list(
                "How To Read The Relationships",
                [
                    relationship_opening,
                    "Pairwise relationships are descriptive Spearman associations computed on session-level delta summaries, not causal effects.",
                    "Qualified-condition counts and same-sign fractions should be read alongside paired counts before any association is treated as stable.",
                    "The device-agreement panels that follow are technical validation summaries and should not be interpreted as scientific associations between constructs.",
                ],
            ),
        }

    def _filter_specs(self, specs: list[dict], modalities: list[str] | None) -> list[dict]:
        if not modalities:
            return specs
        wanted = {str(x).strip().lower() for x in modalities if str(x).strip()}
        if not wanted or "all" in wanted:
            return specs
        out = []
        for spec in specs:
            tags = {t.lower() for t in spec["tags"]}
            if "overview" in tags or tags & wanted:
                out.append(spec)
        return out

    def _save_specs(self, figures_dir: Path, specs: list[dict]) -> list[Path]:
        saved = []
        for spec in specs:
            fig = spec["fig"]
            if fig is None:
                continue
            if isinstance(fig, go.Figure):
                spec["html_fragment"] = fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False})
            else:
                if len(getattr(fig, "axes", [])) > 1:
                    for ax in fig.axes:
                        try:
                            ax.set_title("")
                        except Exception:
                            continue
                condition_labels = set(CONDITION_ORDER)
                for ax in getattr(fig, "axes", []):
                    self._normalize_legend_layout(ax)
                    for line in ax.get_lines():
                        try:
                            current = float(line.get_linewidth())
                        except Exception:
                            continue
                        if current <= 0:
                            continue
                        line.set_linewidth(max(0.6, current * 0.72))
                    for collection in getattr(ax, "collections", []):
                        try:
                            widths = collection.get_linewidths()
                        except Exception:
                            continue
                        if widths is None or len(widths) == 0:
                            continue
                        collection.set_linewidths([max(0.4, float(width) * 0.72) for width in widths])
                path = figures_dir / f"{spec['stem']}.svg"
                fig.savefig(path, format="svg", bbox_inches="tight")
                plt.close(fig)
                spec["path"] = path
                saved.append(path)
        return saved
