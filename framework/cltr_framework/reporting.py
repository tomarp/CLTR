from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from plotly.offline.offline import get_plotlyjs

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
    "visual_comfort": "Visual Comfort",
    "air_quality_comfort": "Air Quality Comfort",
    "room_comfort": "Room Comfort",
    "empatica_hr_mean_bpm": "Empatica HR",
    "biopac_hr_mean_bpm": "BIOPAC HR",
    "empatica_eda_mean_uS": "Empatica EDA",
    "biopac_eda_mean_uS": "BIOPAC EDA",
    "empatica_temp_mean_C": "Empatica Temperature",
    "biopac_temp_chest_mean_C": "Chest Temperature",
    "indoor_relative_humidity_percent": "Indoor RH",
    "biopac_bloodflow_mean_bpu": "Blood Flow",
    "indoor_air_velocity_mean_m_s": "Air Velocity",
    "indoor_air_temp_mean_C": "Indoor Air Temperature",
    "fan_control_au": "Fan Control",
    "fan_control_secondary_au": "Secondary Fan Control",
    "fan_current_A": "Fan Current",
    "master_dpg_C": "DPG",
    "thermal_gradient_C": "Distal-Proximal Gradient",
}
WORK_INDEX_TITLE = "CLTR Atlas"
WORK_INDEX_SUBTITLE = "Explore the study-wide summary and every individual session in one place."
WORK_HOME_TITLE = "CLTR"
WORK_HOME_SUBTITLE = "Controlled Laboratory Thermal Response"
SESSION_CTA = "Open session report"
COHORT_CTA = "Open cohort report"
COPYRIGHT_NOTE = "&copy; 2026 Puneet Tomar. All rights reserved."
PROJECT_GITHUB_URL = "https://github.com/tomarp/cltr"
PROJECT_ZENODO_URL = "https://doi.org/10.5281/zenodo.17817175"
PROJECT_FRAMEWORK_URL = "https://github.com/tomarp/cltr/tree/main/framework"
MAX_SESSION_MAIN_FIGURES = 5
MAX_COHORT_MAIN_FIGURES = 5
BLOCK_PHASE_NARRATIVE_THRESHOLD = 2
COMPARISON_BLOCKS = {"1", "2", "3"}
SECTION_TITLES = {
    "frontmatter": "Overview",
    "subjective_behavioral": "Subjective And Behavioral Data",
    "physiological": "Physiological Data",
    "environmental": "Environmental Data",
    "processed_cleaned": "Processed And Cleaned Signals",
    "alignment_support": "Alignment And Support Layer",
    "derived": "Derived Results",
    "agreement_section": "Relationships And Validation",
    "raw": "Measured Trends",
    "analyzed": "Derived Results",
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
    "value": "Value",
    "protocol_block": "Block",
    "protocol_phase": "Phase",
    "condition_code": "Condition",
    "n_minutes": "Minutes",
    "n_sessions": "Sessions",
    "n_eligible_sessions": "Comparable sessions",
    "n_participants": "Participants",
    "mean_value": "Average",
    "median": "Median",
    "median_overlap_minutes": "Median overlap (min)",
    "median_spearman_r": "Median correlation",
    "median_mae": "Median average error",
    "coverage_fraction": "Coverage",
    "share_within_metric": "Share within measure",
    "mean_consistency": "Consistency",
    "dominant_phase": "Most pronounced phase",
    "direction": "Direction",
    "domain": "Domain",
    "feature": "Feature",
    "iqr": "IQR",
    "skewness": "Skewness",
    "summary_status": "Summary",
    "evidence_status": "Reading guide",
    "support_grade": "Support Grade",
    "support_basis": "Support Basis",
    "supported_phases": "Supported Phases",
    "supported_conditions": "Supported Conditions",
    "supported_condition_phase_cells": "Supported Condition-Phase Cells",
    "cell_coverage_fraction": "Cell Coverage",
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
    "device": "Device",
    "construct": "Construct",
    "n_sessions_with_any_data": "Sessions With Data",
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
    "visual_comfort",
    "air_quality_comfort",
    "room_comfort",
}
CONTROL_SIGNAL_CHANNELS = {
    "fan_current_A",
    "fan_control_au",
    "fan_control_secondary_au",
}
ACC_ASSUMPTION_CHANNELS = SPARSE_OBSERVATION_CHANNELS | {
    "fan_current_A",
    "fan_control_au",
    "fan_control_secondary_au",
}
REPORT_UI = {
    "page_max_width": "1360px",
    "index_page_max_width": "1100px",
    "page_padding": "24px 28px 48px",
    "index_page_padding": "24px 28px 48px",
    "panel_radius": "22px",
    "card_radius": "16px",
    "image_radius": "14px",
    "panel_shadow": "0 18px 44px rgba(23,32,51,0.08)",
    "panel_border": "1px solid rgba(148,163,184,0.24)",
    "panel_padding": "20px 22px",
    "panel_padding_index": "20px 22px",
    "eyebrow_size": "0.74rem",
    "title_size": "2.4rem",
    "index_title_size": "2.4rem",
    "subtitle_line_height": "1.62",
    "nav_font_size": "0.92rem",
    "section_title_size": "1.28rem",
    "figure_title_size": "1.16rem",
    "hero_gap": "18px",
    "cards_gap": "12px",
    "stack_gap": "28px",
    "table_gap": "18px",
    "report_hero_columns": "1.2fr 0.8fr",
    "index_hero_columns": "1.15fr auto",
    "report_cards_columns": "repeat(3,minmax(0,1fr))",
    "index_grid_columns": "repeat(3,minmax(0,1fr))",
    "mobile_breakpoint": "1000px",
    "index_mobile_breakpoint": "1000px",
}
FIGURE_SIZE_PRESETS = {
    "timeline": (13.2, 3.8),
    "wide_single": (12.8, 4.8),
    "wide_single_short": (13.6, 4.2),
    "wide_single_tall": (14.4, 5.4),
    "three_panel_row": (13.0, 4.5),
    "three_panel_row_wide": (13.8, 4.8),
    "three_panel_stack": (8.8, 12.4),
    "two_panel_row": (13.2, 5.8),
    "two_panel_row_balanced": (12.8, 5.8),
    "two_by_two": (12.0, 8.2),
    "two_by_two_balanced": (11.8, 8.2),
    "two_by_two_wide": (12.5, 8.0),
    "readiness_grid": (12.5, 7.5),
    "matrix": (12.0, 5.8),
    "matrix_tall": (14.2, 9.8),
    "participant_single": (4.6, 4.2),
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
        html_dir = ensure_dir(root / self.o.html_dir)
        narrative_specs, appendix_specs = self._build_session_specs(session_inputs)
        narrative_specs, appendix_specs = self._curate_session_specs(session_inputs, narrative_specs, appendix_specs)
        narrative_specs = self._filter_specs(narrative_specs, modalities)
        appendix_specs = self._filter_specs(appendix_specs, modalities)
        for spec in narrative_specs + appendix_specs:
            spec["display_section"] = spec.get("section", "analyzed")
        saved = self._save_specs(figs, narrative_specs + appendix_specs)
        html_path = html_dir / f"{session_id}_report.html"
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
        html_dir = ensure_dir(root / self.o.html_dir)
        narrative_specs, appendix_specs = self._build_cohort_specs(cohort_inputs)
        narrative_specs, appendix_specs = self._curate_cohort_specs(cohort_inputs, narrative_specs, appendix_specs)
        narrative_specs = self._filter_specs(narrative_specs, modalities)
        appendix_specs = self._filter_specs(appendix_specs, modalities)
        for spec in narrative_specs + appendix_specs:
            spec["display_section"] = spec.get("section", "analyzed")
        saved = self._save_specs(figs, narrative_specs + appendix_specs)
        html_path = html_dir / "cohort_report.html"
        html_path.write_text(self._cohort_html(cohort_inputs, narrative_specs, appendix_specs), encoding="utf-8")
        return {
            "html_path": str(html_path),
            "figure_paths": [str(p) for p in saved],
            "figure_specs": [{"code": s["code"], "title": s["title"], "tags": s["tags"], "evidence_score": s["evidence_score"], "section": s.get("section", "analyzed")} for s in narrative_specs + appendix_specs],
            "narrative_codes": [s["code"] for s in narrative_specs],
            "appendix_codes": [s["code"] for s in appendix_specs],
        }

    def write_all_sessions_index(self, manifest: pd.DataFrame, session_reports: list[dict], cohort_report: dict) -> dict:
        root = ensure_dir(self.outdir / self.o.report_dir / self.o.work_dir)
        index_path = root / "index.html"
        atlas_path = root / "cltr_atlas.html"
        atlas_html = self._all_sessions_html(manifest, session_reports, cohort_report)
        index_path.write_text(atlas_html, encoding="utf-8")
        atlas_path.write_text(atlas_html, encoding="utf-8")
        return {"html_path": str(index_path), "atlas_path": str(atlas_path)}

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
                "grid.linewidth": 0.55,
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
            }
        )

    def _figsize(self, preset: str) -> tuple[float, float]:
        return FIGURE_SIZE_PRESETS[preset]

    def _shared_report_css(self) -> str:
        ui = REPORT_UI
        return f"""
body {{ margin:0; font-family: Georgia, "Times New Roman", serif; color:#172033; background:radial-gradient(circle at top left,#fff6e8 0%,#eef4ff 52%,#f8fafc 100%); }}
.page {{ width:min(100%, {ui['page_max_width']}); margin:0 auto; padding:24px clamp(16px,2.4vw,28px) 48px; box-sizing:border-box; }}
.primaryBar {{ position:sticky; top:0; z-index:24; backdrop-filter:blur(16px); background:rgba(248,250,252,0.92); border-bottom:1px solid rgba(148,163,184,0.18); }}
.primaryBarInner {{ width:min(100%, {ui['page_max_width']}); margin:0 auto; padding:12px clamp(16px,2.4vw,28px); display:flex; align-items:center; justify-content:space-between; gap:16px; box-sizing:border-box; }}
.logoLink {{ display:inline-flex; align-items:center; gap:12px; min-height:58px; text-decoration:none; }}
.logoLink:hover {{ transform:translateY(-1px); }}
.logoMark {{ width:58px; height:58px; display:block; flex-shrink:0; }}
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
.mastheadActions {{ display:flex; align-items:center; gap:12px; flex-shrink:0; }}
.socialLinks {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
.socialLink {{ display:inline-flex; align-items:center; justify-content:center; min-height:44px; padding:0 16px; border-radius:999px; text-decoration:none; color:#172033; background:linear-gradient(180deg,rgba(255,255,255,0.96) 0%,rgba(255,247,237,0.96) 100%); border:1px solid rgba(251,146,60,0.28); box-shadow:0 12px 28px rgba(23,32,51,0.08); font:700 0.92rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0.01em; }}
.socialLink:hover {{ background:#ffffff; border-color:#fb923c; box-shadow:0 16px 34px rgba(23,32,51,0.12); transform:translateY(-1px); }}
.socialLink.isDisabled {{ pointer-events:none; opacity:0.58; }}
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
body.theme-dark .menuPanel {{ background:rgba(15,23,42,0.96); border-color:rgba(71,85,105,0.4); }}
body.theme-dark .nav a {{ color:#f8fafc; background:rgba(30,41,59,0.96); border-color:rgba(71,85,105,0.44); box-shadow:inset 0 0 0 4px rgba(15,23,42,0.75); }}
body.theme-dark table th {{ background:#1e293b; }}
body.theme-dark .figureImage,body.theme-dark .lightbox img {{ background:#e2e8f0; }}
body.theme-dark .themeToggleIconDark {{ display:none; }}
body.theme-dark .themeToggleIconLight {{ display:inline; }}
.menuButton {{ appearance:none; border:1px solid rgba(148,163,184,0.28); background:rgba(255,255,255,0.88); color:#172033; border-radius:999px; padding:10px 14px; display:inline-flex; align-items:center; gap:10px; font:600 0.82rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; cursor:pointer; box-shadow:0 10px 24px rgba(23,32,51,0.08); }}
.menuButton:hover {{ background:#ffffff; border-color:#cbd5e1; }}
.menuButtonBars {{ display:grid; gap:3px; }}
.menuButtonBars span {{ display:block; width:14px; height:2px; border-radius:999px; background:currentColor; }}
.menuPanel {{ position:absolute; right:0; top:calc(100% + 10px); width:min(420px, calc(100vw - 32px)); max-height:min(70vh, 720px); overflow:auto; padding:14px 12px; background:rgba(255,255,255,0.97); border:1px solid rgba(148,163,184,0.22); border-radius:{ui['panel_radius']}; box-shadow:0 22px 54px rgba(23,32,51,0.16); backdrop-filter:blur(18px); display:none; }}
.menuPanel.open {{ display:grid; gap:10px; }}
.menuTitle {{ margin:0 0 2px; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size:0.78rem; letter-spacing:0.14em; text-transform:uppercase; color:#64748b; }}
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
.reportShell {{ display:grid; grid-template-columns:minmax(0,1fr); gap:24px; align-items:start; margin-top:20px; }}
.stack {{ display:grid; gap:{ui['stack_gap']}; min-width:0; }}
.nav {{ display:grid; gap:14px; }}
.navTitle {{ margin:0; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size:0.78rem; letter-spacing:0.14em; text-transform:uppercase; color:#64748b; }}
.navList {{ display:grid; grid-template-columns:1fr; gap:10px; }}
.nav a {{ width:100%; min-height:44px; text-decoration:none; color:#172033; background:rgba(255,247,237,0.95); border:1px solid #fed7aa; border-radius:14px; display:grid; grid-template-columns:auto minmax(0,1fr); align-items:start; gap:10px; padding:10px 12px; font-size:0.72rem; font-weight:700; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height:1.2; box-shadow:inset 0 0 0 4px rgba(255,255,255,0.96); box-sizing:border-box; }}
.nav a:hover {{ background:#ffedd5; border-color:#fdba74; }}
.nav a span {{ display:block; font-size:0.78rem; font-weight:500; color:#475569; overflow-wrap:anywhere; }}
.sectionBlock {{ display:grid; gap:18px; padding-top:8px; border-top:2px solid #e2e8f0; }}
.sectionTitle {{ margin:0; font-size:{ui['section_title_size']}; color:#172033; letter-spacing:0.01em; }}
.figureSection {{ display:grid; gap:10px; margin:8px 0 18px; }}
.figureSectionTitle {{ font-size:1.02rem; letter-spacing:0.02em; color:#52607a; margin:0 0 2px; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.tableGrid {{ display:grid; grid-template-columns:1fr; gap:{ui['table_gap']}; margin:20px 0; }}
.tablePanel {{ max-width:100%; overflow-x:auto; }}
table {{ width:100%; min-width:100%; border-collapse:collapse; font-size:0.92rem; table-layout:fixed; }}
th,td {{ border-bottom:1px solid #e2e8f0; padding:8px 10px; text-align:left; vertical-align:top; overflow-wrap:anywhere; }}
th {{ color:#334155; background:#f8fafc; }}
.figurePanel.hidden {{ display:none; }}
.figurePanel h2 {{ margin:0 0 12px; font-size:{ui['figure_title_size']}; line-height:1.25; white-space:normal; overflow:visible; text-overflow:clip; }}
.figureImage {{ width:100%; height:auto; border-radius:{ui['image_radius']}; border:1px solid #dbeafe; background:white; cursor:zoom-in; }}
.responsiveFigure {{ width:100%; max-width:100%; overflow:hidden; }}
.responsiveFigure > * {{ max-width:100% !important; }}
.responsiveFigure .js-plotly-plot,.responsiveFigure .plot-container,.responsiveFigure .svg-container {{ width:100% !important; max-width:100% !important; }}
.meta {{ color:#64748b; font-size:0.86rem; margin-bottom:10px; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.figureMeta {{ margin:10px 0 0; color:#64748b; font-size:0.82rem; line-height:1.45; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.caption {{ margin:10px 0 0; color:#334155; line-height:1.55; font-size:0.95rem; }}
.lightbox {{ position:fixed; inset:0; background:rgba(15,23,42,0.86); display:none; align-items:center; justify-content:center; padding:30px; z-index:30; }}
.lightbox.open {{ display:flex; }}
.lightbox img {{ max-width:95vw; max-height:90vh; background:white; border-radius:{ui['card_radius']}; }}
.copyrightNote {{ width:min(100%, {ui['page_max_width']}); margin:0 auto; padding:0 clamp(16px,2.4vw,28px) 18px; box-sizing:border-box; text-align:center; color:#64748b; font:500 0.84rem/1.5 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
body.theme-dark .copyrightNote {{ color:#94a3b8; }}
@media (max-width:1280px) {{ .reportShell {{ grid-template-columns:1fr; }} }}
@media (max-width:{ui['mobile_breakpoint']}) {{ .primaryBarInner,.secondaryBarInner,.hero,.tableGrid {{ grid-template-columns:1fr; }} .primaryBarInner,.secondaryBarInner {{ display:grid; padding:12px 20px; }} .mastheadActions,.secondaryBarActions {{ justify-content:space-between; }} .secondaryBarText {{ white-space:normal; }} .page {{ padding:20px 16px 40px; }} .nav a {{ grid-template-columns:auto minmax(0,1fr); }} .menuPanel {{ right:auto; left:0; width:min(100%, 420px); }} .takeawayHeader {{ align-items:start; }} .socialLinks {{ order:2; }} }}
""".strip()

    def _shared_index_css(self) -> str:
        ui = REPORT_UI
        return f"""
body {{ margin:0; font-family: Georgia, "Times New Roman", serif; color:#172033; background:radial-gradient(circle at top left,#fff6e8 0%,#eef4ff 52%,#f8fafc 100%); }}
.page {{ width:min(100%, {ui['page_max_width']}); margin:0 auto; padding:{ui['index_page_padding']}; box-sizing:border-box; }}
.primaryBar {{ position:sticky; top:0; z-index:24; backdrop-filter:blur(16px); background:rgba(248,250,252,0.92); border-bottom:1px solid rgba(148,163,184,0.18); }}
.primaryBarInner {{ width:min(100%, {ui['page_max_width']}); margin:0 auto; padding:12px 28px; display:flex; align-items:center; justify-content:space-between; gap:16px; box-sizing:border-box; }}
.logoLink {{ display:inline-flex; align-items:center; gap:12px; min-height:58px; text-decoration:none; }}
.logoLink:hover {{ transform:translateY(-1px); }}
.logoMark {{ width:58px; height:58px; display:block; flex-shrink:0; }}
.logoWordmark {{ display:inline-flex; align-items:center; height:58px; font:700 2.1rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:-0.04em; color:#172033; }}
.secondaryBar {{ position:sticky; top:71px; z-index:23; backdrop-filter:blur(14px); background:rgba(255,255,255,0.78); border-bottom:1px solid rgba(148,163,184,0.16); }}
.secondaryBarInner {{ width:min(100%, {ui['page_max_width']}); margin:0 auto; padding:10px 28px; display:flex; align-items:center; justify-content:space-between; gap:14px; box-sizing:border-box; }}
.secondaryBarMeta {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size:0.84rem; color:#475569; }}
.secondaryBarActions {{ position:relative; display:flex; align-items:center; gap:10px; flex-shrink:0; }}
.secondaryBarType {{ display:inline-flex; align-items:center; gap:8px; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#172033; }}
.secondaryBarType::before {{ content:""; width:8px; height:8px; border-radius:999px; background:#fb7185; box-shadow:0 0 0 3px rgba(251,113,133,0.14); }}
.reportKind--home .secondaryBarType::before {{ background:#7c3aed; box-shadow:0 0 0 3px rgba(124,58,237,0.14); }}
.secondaryBarText {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.socialLinks {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
.socialLink {{ display:inline-flex; align-items:center; justify-content:center; min-height:44px; padding:0 16px; border-radius:999px; text-decoration:none; color:#172033; background:linear-gradient(180deg,rgba(255,255,255,0.96) 0%,rgba(255,247,237,0.96) 100%); border:1px solid rgba(251,146,60,0.28); box-shadow:0 12px 28px rgba(23,32,51,0.08); font:700 0.92rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0.01em; }}
.socialLink:hover {{ background:#ffffff; border-color:#fb923c; box-shadow:0 16px 34px rgba(23,32,51,0.12); transform:translateY(-1px); }}
.socialLink.isDisabled {{ pointer-events:none; opacity:0.58; }}
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
body.theme-dark .socialLink,body.theme-dark .themeToggle,body.theme-dark .menuButton {{ color:#f8fafc; background:linear-gradient(180deg,rgba(30,41,59,0.96) 0%,rgba(15,23,42,0.96) 100%); border-color:rgba(71,85,105,0.5); }}
body.theme-dark .menuPanel {{ background:rgba(15,23,42,0.96); border-color:rgba(71,85,105,0.4); }}
body.theme-dark .nav a {{ color:#f8fafc; background:rgba(30,41,59,0.96); border-color:rgba(71,85,105,0.44); box-shadow:inset 0 0 0 4px rgba(15,23,42,0.75); }}
body.theme-dark .themeToggleIconDark {{ display:none; }}
body.theme-dark .themeToggleIconLight {{ display:inline; }}
.mastheadActions {{ display:flex; align-items:center; gap:12px; flex-shrink:0; }}
.menuButton {{ appearance:none; border:1px solid rgba(148,163,184,0.28); background:rgba(255,255,255,0.88); color:#172033; border-radius:999px; padding:10px 14px; display:inline-flex; align-items:center; gap:10px; font:600 0.82rem/1 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; cursor:pointer; box-shadow:0 10px 24px rgba(23,32,51,0.08); }}
.menuButton:hover {{ background:#ffffff; border-color:#cbd5e1; }}
.menuButtonBars {{ display:grid; gap:3px; }}
.menuButtonBars span {{ display:block; width:14px; height:2px; border-radius:999px; background:currentColor; }}
.menuPanel {{ position:absolute; right:0; top:calc(100% + 10px); width:min(420px, calc(100vw - 32px)); max-height:min(70vh, 720px); overflow:auto; padding:14px 12px; background:rgba(255,255,255,0.97); border:1px solid rgba(148,163,184,0.22); border-radius:{ui['panel_radius']}; box-shadow:0 22px 54px rgba(23,32,51,0.16); backdrop-filter:blur(18px); display:none; }}
.menuPanel.open {{ display:grid; gap:10px; }}
.menuTitle {{ margin:0 0 2px; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size:0.78rem; letter-spacing:0.14em; text-transform:uppercase; color:#64748b; }}
.nav {{ display:grid; gap:14px; }}
.navList {{ display:grid; grid-template-columns:1fr; gap:10px; }}
.nav a {{ width:100%; min-height:44px; text-decoration:none; color:#172033; background:rgba(255,247,237,0.95); border:1px solid #fed7aa; border-radius:14px; display:grid; grid-template-columns:auto minmax(0,1fr); align-items:start; gap:10px; padding:10px 12px; font-size:0.72rem; font-weight:700; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height:1.2; box-shadow:inset 0 0 0 4px rgba(255,255,255,0.96); box-sizing:border-box; }}
.nav a:hover {{ background:#ffedd5; border-color:#fdba74; }}
.nav a span {{ display:block; font-size:0.78rem; font-weight:500; color:#475569; overflow-wrap:anywhere; }}
.hero,.grid {{ display:grid; gap:16px; }}
.hero {{ grid-template-columns:{ui['index_hero_columns']}; align-items:end; }}
.panel,.sessionCard {{ background:rgba(255,255,255,0.9); border:{ui['panel_border']}; border-radius:{ui['panel_radius']}; box-shadow:{ui['panel_shadow']}; padding:{ui['panel_padding_index']}; backdrop-filter:blur(8px); }}
.grid {{ grid-template-columns:{ui['index_grid_columns']}; margin-top:22px; }}
.eyebrow {{ font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size:{ui['eyebrow_size']}; letter-spacing:0.18em; text-transform:uppercase; color:#9a3412; margin-bottom:8px; }}
.title {{ font-size:{ui['index_title_size']}; font-weight:700; letter-spacing:-0.04em; margin:0 0 8px; }}
.subtitle,.tagLine {{ color:#52607a; line-height:1.6; }}
.tagLine {{ font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size:0.88rem; }}
.heroIntro {{ background:linear-gradient(135deg,#fef3c7 0%,#fed7aa 45%,#fb7185 100%); border-color:rgba(251,113,133,0.28); }}
.heroCta {{ justify-self:end; max-width:320px; background:linear-gradient(135deg,#172033 0%,#1d4ed8 55%,#06b6d4 100%); color:#f8fafc; border:1px solid rgba(191,219,254,0.45); }}
.heroCta .eyebrow {{ color:#dbeafe; }}
.heroCta .subtitle {{ color:rgba(239,246,255,0.92); }}
.heroIntro .subtitle {{ color:#7c2d12; }}
.heroSticky {{ position:sticky; top:18px; align-self:start; }}
.heroMeta {{ display:grid; gap:10px; margin-top:16px; }}
.heroStatement {{ font-size:1rem; line-height:1.65; color:#4a1d0d; max-width:58ch; }}
.heroFacts {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }}
.heroFact {{ background:rgba(255,255,255,0.42); border:1px solid rgba(255,255,255,0.45); border-radius:14px; padding:10px 12px; }}
.heroFactLabel {{ font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:#9a3412; }}
.heroFactValue {{ margin-top:4px; color:#4a1d0d; line-height:1.45; }}
.sessionCard h3 {{ margin:0 0 10px; font-size:1.3rem; }}
.pillLink {{ display:inline-block; margin-top:12px; padding:10px 14px; background:#172033; color:#f8fafc; text-decoration:none; border-radius:999px; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.heroCta .pillLink {{ background:#f8fafc; color:#172033; }}
@media (max-width:{ui['index_mobile_breakpoint']}) {{ .primaryBarInner,.secondaryBarInner,.hero,.grid,.heroFacts {{ grid-template-columns:1fr; }} .primaryBarInner,.secondaryBarInner {{ display:grid; padding:12px 20px; }} .mastheadActions,.secondaryBarActions {{ justify-content:space-between; }} .secondaryBarText {{ white-space:normal; }} .menuPanel {{ right:auto; left:0; width:min(100%, 420px); }} .heroSticky {{ position:static; }} .socialLinks {{ order:2; }} }}
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

    def _logo_image_svg(self) -> str:
        return (
            "<svg class='logoMark' viewBox='0 0 72 72' aria-hidden='true'>"
            "<defs>"
            "<linearGradient id='cltrLogoBg' x1='0%' y1='0%' x2='100%' y2='100%'><stop offset='0%' stop-color='#0f172a'/><stop offset='50%' stop-color='#1d4ed8'/><stop offset='100%' stop-color='#06b6d4'/></linearGradient>"
            "<linearGradient id='cltrLogoLineA' x1='0%' y1='0%' x2='100%' y2='0%'><stop offset='0%' stop-color='#ffffff'/><stop offset='100%' stop-color='#fde68a'/></linearGradient>"
            "<linearGradient id='cltrLogoLineB' x1='0%' y1='0%' x2='100%' y2='0%'><stop offset='0%' stop-color='#bfdbfe'/><stop offset='100%' stop-color='#fca5a5'/></linearGradient>"
            "</defs>"
            "<rect x='3' y='3' width='66' height='66' rx='22' fill='url(#cltrLogoBg)'/>"
            "<g opacity='0.24' stroke='#ffffff' stroke-width='1.5'>"
            "<path d='M16 21h40'/><path d='M16 31h40'/><path d='M16 41h40'/><path d='M16 51h40'/>"
            "<path d='M24 15v42'/><path d='M36 15v42'/><path d='M48 15v42'/>"
            "</g>"
            "<path d='M16 46c5-4 8-17 14-17s9 16 15 16 10-10 13-14' fill='none' stroke='url(#cltrLogoLineA)' stroke-width='5.5' stroke-linecap='round'/>"
            "<path d='M16 33c5 0 8-9 14-9s9 12 15 12 10-6 13-8' fill='none' stroke='url(#cltrLogoLineB)' stroke-width='3.8' stroke-linecap='round'/>"
            "<g fill='#ffffff'><circle cx='30' cy='29' r='3.4'/><circle cx='45' cy='45' r='3.4'/></g>"
            "<circle cx='54' cy='20' r='4.5' fill='#fde68a' fill-opacity='0.94'/>"
            "</svg>"
        )

    def _shared_chrome(
        self,
        *,
        home_href: str,
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
    ) -> str:
        menu_prefix = "<span class='menuButtonBars'><span></span><span></span><span></span></span>" if menu_icon_bars else ""
        menu_html = (
            f"<button id='{html_escape(menu_button_id)}' class='menuButton' type='button' aria-expanded='false' aria-controls='{html_escape(menu_panel_id)}' aria-label='{html_escape(menu_title)}'>{menu_prefix}<span>{html_escape(menu_label)}</span></button>"
            f"<div id='{html_escape(menu_panel_id)}' class='menuPanel'><h2 class='menuTitle'>{html_escape(menu_title)}</h2><nav class='nav'><div class='navList'>{menu_items_html}</div></nav></div>"
            if show_menu_button
            else ""
        )
        secondary_html = (
            f"<div class='secondaryBar'><div class='secondaryBarInner'><div class='secondaryBarMeta'><span class='secondaryBarType'>{html_escape(page_type)}</span><span class='secondaryBarText'>{html_escape(page_meta)}</span></div><div class='secondaryBarActions'>{menu_html}</div></div></div>"
            if show_secondary_bar
            else ""
        )
        return (
            f"<header class='primaryBar'>"
            f"<div class='primaryBarInner'>"
            f"<a class='logoLink' href='{html_escape(home_href)}' title='Open report index' aria-label='Open report index'>{self._logo_image_svg()}<span class='logoWordmark'>CLTR</span></a>"
            f"<div class='mastheadActions'>"
            f"{self._social_links_html()}"
            f"<button class='themeToggle' id='themeToggle' type='button' aria-label='Toggle dark mode'><span class='themeToggleIconDark' aria-hidden='true'>◐</span><span class='themeToggleIconLight' aria-hidden='true'>◑</span></button>"
            f"</div>"
            f"</div>"
            f"</header>"
            f"{secondary_html}"
        )

    def _theme_toggle_script(self) -> str:
        return """const themeToggle=document.getElementById('themeToggle');
const storedTheme=window.localStorage.getItem('cltr-theme');
if(storedTheme==='dark'){document.body.classList.add('theme-dark');}
const syncThemeIcon=()=>{if(themeToggle){themeToggle.setAttribute('aria-pressed', document.body.classList.contains('theme-dark') ? 'true' : 'false');}};
syncThemeIcon();
if(themeToggle){themeToggle.addEventListener('click',()=>{document.body.classList.toggle('theme-dark');window.localStorage.setItem('cltr-theme', document.body.classList.contains('theme-dark') ? 'dark' : 'light');syncThemeIcon();});}"""

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
.copyrightNote {{ padding-bottom:18px; }}
@media (max-width:{ui['mobile_breakpoint']}) {{ .hero {{ gap:16px; }} .heroVisual {{ padding:16px; }} }}
""".strip()

    def _home_html(self, manifest: pd.DataFrame, session_reports: list[dict], cohort_report: dict) -> str:
        cohort_name = Path(cohort_report["html_path"]).name if cohort_report.get("html_path") else "cohort_report.html"
        chrome = self._shared_chrome(
            home_href="index.html",
            page_type="Home",
            page_meta="Report index for the CLTR study",
            menu_button_id="homeMenuButton",
            menu_panel_id="homeMenuPanel",
            menu_label="Navigate",
            menu_title="CLTR Destinations",
            menu_items_html=(
                f"<a href='cltr_atlas.html' title='Open atlas'>Atlas<span>Study-wide hub and session index</span></a>"
                f"<a href='../cohort/html/{html_escape(cohort_name)}' title='Open cohort report'>Cohort<span>Study-wide summary report</span></a>"
                f"<a href='../sessions/{html_escape(session_reports[0]['session_id'])}/html/{html_escape(Path(session_reports[0]['html_path']).name)}' title='Open first session report'>Sessions<span>Session-level analytical reports</span></a>"
                if session_reports else f"<a href='cltr_atlas.html' title='Open atlas'>Atlas<span>Study-wide hub</span></a>"
            ),
            show_secondary_bar=False,
            show_menu_button=False,
        )
        return f"""<!doctype html><html><head><meta charset='utf-8'><title>{WORK_HOME_TITLE}</title>
<style>
{self._home_page_css()}
</style></head><body class='reportKind--home'>{chrome}<main class='page'><section class='landing'><section class='hero'><section class='panel heroLead'><div class='eyebrow'>Report Index</div><h1 class='title'>{WORK_HOME_TITLE}</h1><p class='subtitle'>{WORK_HOME_SUBTITLE}. Framework outputs are generated here first, and the atlas bundle can then be published separately from these report artifacts.</p><div class='heroVisual'>{self._logo_image_svg()}<p class='heroVisualText'>This workspace contains the generated CLTR report bundle only. Public-site pages and publication assets are maintained separately under <code>work/cltr/docs</code>.</p></div></section></section></section></main><div class='copyrightNote'>{COPYRIGHT_NOTE}</div><script>
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
        temp = df[["minute_index", "protocol_phase"]].dropna().sort_values("minute_index")
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
                ax.text((start + end) / 2.0, 1.01, PHASE_ABBR.get(phase, phase[:3].upper()), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color="#475569")

    def _place_condition_legend(self, ax: plt.Axes, handles=None) -> None:
        labels = []
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        else:
            labels = [getattr(handle, "get_label", lambda: "")().strip() for handle in handles]
        labels = [label for label in labels if label]
        if not labels:
            return
        ncols = min(len(labels), len(CONDITION_ORDER))
        ax.legend(
            handles=handles,
            frameon=False,
            ncol=ncols,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.14),
            columnspacing=1.2,
            handletextpad=0.5,
            borderaxespad=0.0,
        )


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
                ax.text((draw_start + draw_end) / 2.0, 1.01, PHASE_ABBR.get(phase, phase[:3].upper()), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color="#475569", clip_on=False)

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
        ax.set_xlabel("Minutes since session start")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
        if trim_to_support and support_x_min is not None and support_x_max is not None:
            pad = max((support_x_max - support_x_min) * 0.02, 0.1)
            visible_start = max(0.0, support_x_min - pad)
            visible_end = support_x_max + pad
            ax.set_xlim(visible_start, visible_end)
            self._add_raw_phase_spans(ax, minute, start_utc, visible_start=visible_start, visible_end=visible_end)
        else:
            self._add_raw_phase_spans(ax, minute, start_utc)
        fig.tight_layout()
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
        ax.set_xlabel("Minutes since session start")
        ax.set_ylabel("Peaks")
        if not x.empty:
            pad = max((float(x.max()) - float(x.min())) * 0.02, 0.1)
            visible_start = max(0.0, float(x.min()) - pad)
            visible_end = float(x.max()) + pad
            ax.set_xlim(visible_start, visible_end)
            self._add_raw_phase_spans(ax, minute, start_utc, visible_start=visible_start, visible_end=visible_end)
        else:
            self._add_raw_phase_spans(ax, minute, start_utc)
        fig.tight_layout()
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
        ax.set_xlabel("Minutes since session start")
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
            ax.text(min(float(row.supported_phases) + 0.08, 4.95), idx, f"{row.support_grade} | min repeats={row.min_block_repeats} | summaries={row.total_block_phase_summaries}", va="center", fontsize=8, color="#172033")
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
            ax.text(float(x.iloc[idx]) + 0.08, idx, f"{row.direction.lower()} | {agreement_txt}", va="center", fontsize=8, color="#172033")
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
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="#172033")
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
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="#172033")
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
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="#172033")
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
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="#172033")
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
            ax.set_xlabel(FEATURE_LABELS.get(metric, metric))
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
        for metric in [m for m in COHORT_DERIVED_ENDPOINTS if m in comparison.columns]:
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
            rows.append(
                {
                    "metric": metric,
                    "endpoint": FEATURE_LABELS.get(metric, metric),
                    "support_grade": grade,
                    "support_basis": "questionnaire responses" if self._is_questionnaire_endpoint(metric) else "valid phase summaries",
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
        strong = support_profile.loc[support_profile["support_grade"] == "strong", "metric"].tolist()
        if strong:
            return strong
        return support_profile.loc[support_profile["support_grade"] == "partial", "metric"].tolist()

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
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="#172033")
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
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="#172033")
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
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="#172033")
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
            ax.text(float(row.cell_coverage_fraction) + 0.01, idx, f"{row.cell_coverage_fraction:.2f}", va="center", fontsize=9, color="#172033")
        ax.set_xlabel("Condition-phase cell coverage")
        ax.set_xlim(0, 1.05)
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
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="#172033")
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
            ax.set_xlabel(f"{FEATURE_LABELS.get(metric, metric)} delta")
            ax.set_ylabel("Thermal Comfort delta")
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
        fig.update_xaxes(title_text="Minute index", showgrid=True, gridcolor="#eef2f7", zeroline=False)
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
            ax.set_xlabel("Minute index")
            ax.grid(True, axis="y")
            ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
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
            axes[-1].set_xlabel("Minute index")
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
        ax.set_ylabel(FEATURE_LABELS.get(column, column))
        ax.set_xlabel("Minute index")
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
        fig.update_yaxes(title_text=FEATURE_LABELS.get(metric, metric), showgrid=True, gridcolor="#eef2f7", zeroline=False)
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
                title="Aligned support map by minute",
                summary="The aligned minute-level support map separates missing support from absent response and shows where questionnaire, physiological, indoor, and overlap support are actually available.",
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
        has_strong = bool((support_profile.get("support_grade", pd.Series(dtype=str)) == "strong").any()) if not support_profile.empty else False
        result_prefix = "Primary-result" if has_strong else "Partial-result descriptive"
        result_summary_suffix = (
            "Only strong-support endpoints are carried into this result layer."
            if has_strong
            else "No endpoint reaches strong cohort support in the current sample, so this layer is shown descriptively from partial-support endpoints only."
        )
        response_matrix = self._cohort_response_matrix(c.get("cohort_phase_summary", pd.DataFrame()), support_profile)
        delta_matrix = self._cohort_delta_matrix(c.get("cohort_phase_summary", pd.DataFrame()), support_profile)
        agreement_matrix = self._cohort_directional_agreement_matrix(c.get("cohort_phase_summary", pd.DataFrame()), support_profile)
        relationship_matrix = self._cohort_relationship_matrix(c.get("cohort_phase_summary", pd.DataFrame()), support_profile)

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
                title="Paired-device validation profile",
                summary="Paired-device validation is separated from session structure so overlap, eligibility, and agreement can be assessed directly for heart rate, electrodermal activity, and temperature.",
                fig=self._fig_cohort_window_validation(c),
                tags=["overview", "qc", "support", "agreement"],
                evidence_score=ev["score"],
                evidence_label=ev["label"],
                gating_note=ev["note"],
                section="frontmatter",
            ),
            self._spec(
                code="C04",
                stem="cohort_support_map",
                title="Phase-annotated cohort support map",
                summary="Support is aggregated by condition and minute across the full protocol timeline so acclimation, intervention, and terminal phases remain visible before cohort trajectories and derived summaries are interpreted.",
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
                    summary="This figure summarizes how completely each endpoint covers the available condition-by-phase grid before the cohort result layer is interpreted.",
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
                self._html_spec(
                    code="C12B",
                    stem="cohort_corrected_contrast_register",
                    title="Multiplicity-corrected contrast register",
                    summary="This register lists the strongest paired condition contrasts after Benjamini-Hochberg correction. Confidence intervals are bootstrap intervals on the matched mean difference.",
                    html_fragment=self._matrix_panel_html(
                        "Corrected Condition Contrast Register",
                        self._cohort_top_contrast_register(c.get("condition_contrasts", pd.DataFrame())),
                        ["metric", "protocol_phase", "comparison", "n_pairs", "mean_difference", "ci_low", "ci_high", "p_value", "p_value_fdr", "inference_label"],
                        n=24,
                    ),
                    tags=["statistics", "contrast", "phase"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._html_spec(
                    code="C12C",
                    stem="cohort_mixed_effects_register",
                    title="Mixed-effects primary-endpoint register",
                    summary="This register reports participant-level mixed-effects estimates for the primary endpoints, screening fixed effects by false-discovery-rate corrected p-values.",
                    html_fragment=self._matrix_panel_html(
                        "Mixed-Effects Primary-Endpoint Register",
                        self._mixed_effects_register(c.get("mixed_effects_primary", pd.DataFrame())),
                        ["metric", "term", "beta", "ci_low", "ci_high", "p_value", "p_value_fdr", "significant_fdr", "n_obs", "n_participants"],
                        n=30,
                    ),
                    tags=["statistics", "mixed_model", "phase"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="analyzed",
                ),
                self._spec(
                    code="C12D",
                    stem="cohort_predictive_benchmarks",
                    title="Participant-grouped predictive benchmarks",
                    summary="This panel reports subject-independent benchmark performance for predicting illuminance level and time of day from the session endpoint feature set. These are grouped-by-participant validation results, not resubstitution scores.",
                    fig=self._fig_predictive_benchmarks(c.get("predictive_benchmarks", pd.DataFrame())),
                    tags=["statistics", "prediction", "benchmark"],
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
                title="How closely the devices agree across sessions",
                summary="This view shows how closely paired devices align across sessions and highlights where comparisons are supported by enough overlapping data.",
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
                title="Agreement summary by modality",
                summary="This summary compares overlap, correlation, and error across the three validation modalities and makes the quality ranking explicit.",
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
            ("C06A", "thermal_comfort", "Condition-stratified comfort distributions", ["appendix", "phase", "comfort"]),
            ("C06B", "empatica_hr_mean_bpm", "Condition-stratified heart-rate trajectories", ["appendix", "phase", "heart_rate"]),
            ("C06C", "biopac_temp_chest_mean_C", "Condition-stratified chest-temperature trajectories", ["appendix", "phase", "temperature"]),
            ("C06D", "indoor_air_velocity_mean_m_s", "Condition-stratified air-velocity trajectories", ["appendix", "phase", "environment"]),
        ]
        specs: list[dict] = []
        for code, metric, title, tags in metric_specs:
            fig = self._fig_cohort_condition_trace(minute, metric)
            specs.append(
                self._spec(
                    code=code,
                    stem=f"cohort_condition_trace_{metric}",
                    title=title,
                    summary=(
                        "Questionnaire endpoints are shown as phase-wise condition distributions with raw observation points, because they are discrete event-based responses rather than continuous trajectories."
                        if self._is_sparse_observation_channel(metric)
                        else "Condition trajectories are shown as standalone plots so each modality can be read without subplot compression or cross-axis crowding."
                    ),
                    fig=fig,
                    tags=tags,
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="raw",
                )
            )
        return specs

    def _build_cohort_burst_specs(self, minute: pd.DataFrame, ev: dict) -> list[dict]:
        channel_specs = [
            ("C04C", "empatica_hr_mean_bpm", "#b91c1c", "Cohort Empatica HR bursts", ["heart_rate", "exploratory"]),
            ("C04D", "empatica_eda_mean_uS", "#1d4ed8", "Cohort Empatica EDA bursts", ["eda", "exploratory"]),
            ("C04E", "empatica_temp_mean_C", "#ea580c", "Cohort Empatica temperature bursts", ["temperature", "exploratory"]),
            ("C04F", "biopac_hr_mean_bpm", "#111827", "Cohort BIOPAC HR bursts", ["heart_rate", "exploratory"]),
            ("C04G", "biopac_eda_mean_uS", "#2563eb", "Cohort BIOPAC EDA bursts", ["eda", "exploratory"]),
            ("C04H", "biopac_temp_chest_mean_C", "#ea580c", "Cohort chest-temperature bursts", ["temperature", "exploratory"]),
            ("C04I", "biopac_bloodflow_mean_bpu", "#7c3aed", "Cohort blood-flow bursts", ["bloodflow", "exploratory"]),
            ("C04J", "indoor_air_temp_mean_C", "#ea580c", "Cohort indoor-air-temperature bursts", ["environment", "exploratory"]),
            ("C04K", "indoor_air_velocity_mean_m_s", "#0f766e", "Cohort air-velocity bursts", ["environment", "exploratory"]),
            ("C04L", "indoor_relative_humidity_percent", "#2563eb", "Cohort indoor-relative-humidity bursts", ["environment", "exploratory"]),
            ("C04M", "fan_control_au", "#111827", "Cohort fan-control bursts", ["fan", "exploratory"]),
            ("C04N", "thermal_sensation", "#b91c1c", "Cohort thermal-sensation observations", ["comfort", "exploratory"]),
            ("C04O", "thermal_comfort", "#0f172a", "Cohort thermal-comfort observations", ["comfort", "exploratory"]),
            ("C04P", "thermal_preference", "#2563eb", "Cohort thermal-preference observations", ["comfort", "exploratory"]),
            ("C04Q", "room_comfort", "#7c3aed", "Cohort room-comfort observations", ["comfort", "exploratory"]),
            ("C04R", "fan_current_A", "#111827", "Cohort fan-current bursts", ["fan", "exploratory"]),
            ("C04S", "fan_control_secondary_au", "#7c3aed", "Cohort secondary-fan-control bursts", ["fan", "exploratory"]),
            ("C04T", "thermal_pleasure", "#ea580c", "Cohort thermal-pleasure observations", ["comfort", "exploratory"]),
            ("C04U", "visual_comfort", "#0f766e", "Cohort visual-comfort observations", ["comfort", "exploratory"]),
            ("C04V", "air_quality_comfort", "#2563eb", "Cohort air-quality-comfort observations", ["comfort", "exploratory"]),
        ]
        specs: list[dict] = []
        for code, column, color, title, tags in channel_specs:
            fig = self._fig_cohort_single_channel_burst(minute, column, color)
            specs.append(
                self._spec(
                    code=code,
                    stem=f"cohort_{column}_bursts",
                    title=title,
                    summary="Each cohort burst figure is dedicated to a single signal so modality-specific support timing and condition-stratified signal shape remain interpretable.",
                    fig=fig,
                    tags=tags + ["phase"],
                    evidence_score=ev["score"],
                    evidence_label=ev["label"],
                    gating_note=ev["note"],
                    section="raw",
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
                    summary="Each modality is shown in its own panel with a renderer matched to its data structure, so sparse questionnaire observations are not misrepresented as continuous signals.",
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
                        phase_ax.text(j, i, int(pivot.values[i, j]), ha="center", va="center", fontsize=8, color="#172033")
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
            overlap_ax.text(bar.get_x() + bar.get_width() / 2.0, value + ymax * 0.03, f"{int(value)}", ha="center", va="bottom", fontsize=9, color="#172033")
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
                axes[2].set_xticklabels([PHASE_ABBR.get(x, x[:3].upper()) for x in pivot.columns], fontsize=8)
                axes[2].set_yticks(range(len(pivot.index)))
                axes[2].set_yticklabels(list(pivot.index))
                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        value = float(pivot.iloc[i, j])
                        if value > 0:
                            axes[2].text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8, color="#172033")
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
                ax.text(row.end_minute + 1.5, idx, f"{row.support_fraction:.0%}", va="center", fontsize=8, color="#475569")
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
                cons_ax.text(min(row[1] + 0.02, 0.98), idx, f"{PHASE_ABBR.get(row[3], row[3][:3].upper())} | {row[4]} | n={row[2]}", va="center", fontsize=8, color="#172033")
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
                ax.text(0.0, 1.02, f"dominant repeat: {PHASE_ABBR.get(consistency['dominant_phase'], consistency['dominant_phase'][:3].upper())} {consistency['dominant_direction']} | consistency={consistency['consistency']:.2f}", transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="#64748b")
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
        ax.set_yticklabels(ylabels, fontsize=9)
        tick_count = min(8, len(comparison_minute))
        xticks = np.linspace(0, len(comparison_minute) - 1, tick_count, dtype=int) if tick_count > 1 else np.array([0])
        ax.set_xticks(xticks)
        minute_tick_values = to_numeric(comparison_minute.iloc[xticks]["minute_index"]).fillna(0).astype(int).tolist()
        ax.set_xticklabels([str(x) for x in minute_tick_values], fontsize=9)
        ax.set_xlabel("Timeline minute")
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
                    fontsize=8,
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
            figsize=(16.8, 4.9),
            gridspec_kw={"width_ratios": [1.0, 1.0, 1.0], "wspace": 0.34},
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
        bars = axes[0].bar(
            cond_counts.index,
            cond_counts.values,
            color=[CONDITION_COLORS.get(x, "#475569") for x in cond_counts.index],
            width=0.68,
        )
        axes[0].set_ylabel("Sessions")
        axes[0].set_ylim(0, max(float(cond_counts.max()) * 1.22, 1.0))
        axes[0].tick_params(axis="x", rotation=0)

        factor_counts = [
            ("Illuminance", session_summary["illuminance_level"].astype(str).value_counts().reindex(["DIM", "BRI"]).fillna(0)),
            ("Diurnal timing", session_summary["time_of_day"].astype(str).value_counts().reindex(["MOR", "MID"]).fillna(0)),
        ]
        factor_palette = {
            "DIM": CONDITION_COLORS.get("DIM-MOR", "#475569"),
            "BRI": CONDITION_COLORS.get("BRI-MID", "#1d4ed8"),
            "MOR": "#0f766e",
            "MID": "#ea580c",
        }
        group_positions = np.array([0.0, 1.6], dtype=float)
        illuminance_counts = factor_counts[0][1]
        timing_counts = factor_counts[1][1]
        illuminance_parts = [("DIM", float(illuminance_counts.get("DIM", 0))), ("BRI", float(illuminance_counts.get("BRI", 0)))]
        timing_parts = [("MOR", float(timing_counts.get("MOR", 0))), ("MID", float(timing_counts.get("MID", 0)))]
        bar_width = 0.7
        bottom = 0.0
        for idx, (level, value) in enumerate(illuminance_parts):
            axes[1].bar(
                group_positions[0],
                value,
                bottom=bottom,
                width=bar_width,
                color=factor_palette[level],
                edgecolor="white",
                linewidth=0.8,
                label=level,
            )
            bottom += value
        illum_total = bottom
        bottom = 0.0
        for idx, (level, value) in enumerate(timing_parts):
            axes[1].bar(
                group_positions[1],
                value,
                bottom=bottom,
                width=bar_width,
                color=factor_palette[level],
                edgecolor="white",
                linewidth=0.8,
                label=level,
            )
            bottom += value
        timing_total = bottom
        axes[1].set_xticks(group_positions)
        axes[1].set_xticklabels(["Illuminance", "Diurnal timing"])
        axes[1].set_ylabel("Sessions")
        axes[1].set_ylim(0, max(float(max(illum_total, timing_total)) * 1.18, 1.0))
        axes[1].set_xlim(-0.75, 2.35)
        axes[1].grid(True, axis="y", alpha=0.2)
        axes[1].grid(False, axis="x")
        axes[1].text(group_positions[0], illum_total + 0.25, f"{int(illum_total)}", ha="center", va="bottom", fontsize=9, color="#334155", fontweight="bold")
        axes[1].text(group_positions[1], timing_total + 0.25, f"{int(timing_total)}", ha="center", va="bottom", fontsize=9, color="#334155", fontweight="bold")
        axes[1].legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))

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
        cmap = LinearSegmentedColormap.from_list("session_support", ["#fff7ed", "#fde68a", "#0f766e"])
        im = axes[2].imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        axes[2].grid(False)
        axes[2].set_xticks(range(len(support_cols)))
        axes[2].set_xticklabels([label for label, _ in support_cols], rotation=0, ha="center")
        axes[2].set_yticks(range(len(cond_order)))
        axes[2].set_yticklabels(cond_order)
        axes[2].tick_params(axis="x", labelsize=9, pad=8)
        for row in range(mat.shape[0]):
            for col in range(mat.shape[1]):
                val = float(mat[row, col])
                axes[2].text(
                    col,
                    row,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if val >= 0.62 else "#172033",
                    fontweight="bold",
                )
        plt.colorbar(im, ax=axes[2], fraction=0.05, pad=0.05, label="Mean support fraction")

        for ax in axes:
            ax.set_axisbelow(True)
            if ax is not axes[2]:
                ax.grid(True, axis="y", alpha=0.18)
        fig._cltr_panel_notes = [
            "Left|Session types|The number of sessions contributing to each condition-defined session type.",
            "Middle|Factor balance|Balance across the two design factors, illuminance and diurnal timing.",
            "Right|Mean analytic support by session type|Mean analytic support by session type for questionnaire, Empatica, BIOPAC, and indoor environmental streams.",
        ]
        return fig

    def _fig_cohort_window_validation(self, c: dict):
        support = c.get("condition_support_summary", pd.DataFrame()).copy()
        agreement = c.get("sensor_agreement", pd.DataFrame()).copy()
        if support.empty:
            return None
        support["condition_label"] = support["condition_code"].astype(str)
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))

        cond_order = [x for x in CONDITION_ORDER if x in support["condition_code"].astype(str).unique()]
        if not cond_order:
            cond_order = support["condition_code"].astype(str).tolist()
        temp = support.set_index("condition_code").reindex(cond_order)

        x = np.arange(len(cond_order), dtype=float)
        overlap_cols = [
            ("hr_overlap_minutes__mean", "Heart rate", "#111827"),
            ("eda_overlap_minutes__mean", "EDA", "#2563eb"),
            ("temp_overlap_minutes__mean", "Temperature", "#ea580c"),
        ]
        for idx, (col, label, color) in enumerate(overlap_cols):
            vals = to_numeric(temp[col]).fillna(0).to_numpy(dtype=float) if col in temp.columns else np.zeros(len(cond_order))
            axes[0, 0].bar(x + (idx - 1) * 0.24, vals, width=0.24, color=color, label=label)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(cond_order)
        axes[0, 0].set_ylabel("Mean overlap minutes")
        axes[0, 0].set_title("Paired overlap by condition")
        axes[0, 0].legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.18))

        metrics = ["heart_rate", "eda", "temperature"]
        metric_labels = {"heart_rate": "Heart rate", "eda": "EDA", "temperature": "Temperature"}
        metric_colors = {"heart_rate": "#111827", "eda": "#2563eb", "temperature": "#ea580c"}

        if agreement.empty:
            axes[0, 1].axis("off")
            axes[1, 0].axis("off")
            axes[1, 1].axis("off")
            fig._cltr_panel_notes = [
                "Top left|Paired overlap by condition|Mean paired-device overlap minutes by condition and modality.",
                "Summary|Agreement summary unavailable|No paired-device agreement summary was available for this cohort export.",
            ]
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            return fig

        agg = (
            agreement.groupby(["metric", "condition_code"], as_index=False)
            .agg(
                eligible_sessions=("eligible", "sum"),
                sessions=("session_id", "nunique"),
            )
        )
        for idx, metric in enumerate(metrics):
            cur = agg.loc[agg["metric"].astype(str) == metric].set_index("condition_code").reindex(cond_order)
            vals = to_numeric(cur["eligible_sessions"]).fillna(0).to_numpy(dtype=float)
            axes[0, 1].bar(
                x + (idx - 1) * 0.24,
                vals,
                width=0.24,
                color=metric_colors[metric],
                label=metric_labels[metric],
            )
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(cond_order)
        axes[0, 1].set_ylabel("Eligible sessions")
        axes[0, 1].set_title("Validation-ready sessions")
        axes[0, 1].legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.18))

        eligible = agreement.loc[to_numeric(agreement["eligible"]).fillna(0) > 0].copy()
        spearman_data = []
        mae_data = []
        for metric in metrics:
            cur = eligible.loc[eligible["metric"].astype(str) == metric]
            spearman_data.append(to_numeric(cur.get("spearman_r", pd.Series(dtype=float))).dropna().to_numpy(dtype=float))
            mae_data.append(to_numeric(cur.get("mae", pd.Series(dtype=float))).dropna().to_numpy(dtype=float))
        box_positions = np.arange(1, len(metrics) + 1, dtype=float)
        rng = np.random.default_rng(42)

        bp = axes[1, 0].boxplot(
            spearman_data,
            positions=box_positions,
            widths=0.56,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 1.4},
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
        for patch, metric in zip(bp["boxes"], metrics):
            patch.set_facecolor(metric_colors[metric])
            patch.set_alpha(0.82)
            patch.set_edgecolor(metric_colors[metric])
        for idx, (metric, values) in enumerate(zip(metrics, spearman_data), start=1):
            if len(values) == 0:
                continue
            jitter = rng.normal(0, 0.045, size=len(values))
            axes[1, 0].scatter(
                np.full(len(values), idx, dtype=float) + jitter,
                values,
                s=14,
                alpha=0.35,
                color=metric_colors[metric],
                edgecolors="none",
                zorder=3,
            )
        axes[1, 0].axhline(0.7, color="#94a3b8", lw=1.0, ls="--")
        axes[1, 0].set_xticks(box_positions)
        axes[1, 0].set_xticklabels([metric_labels[m] for m in metrics])
        axes[1, 0].set_ylabel("Spearman r")
        axes[1, 0].set_title("Agreement strength across eligible sessions")
        axes[1, 0].set_ylim(-1.05, 1.05)

        bp2 = axes[1, 1].boxplot(
            mae_data,
            positions=box_positions,
            widths=0.56,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 1.4},
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
        for patch, metric in zip(bp2["boxes"], metrics):
            patch.set_facecolor(metric_colors[metric])
            patch.set_alpha(0.82)
            patch.set_edgecolor(metric_colors[metric])
        for idx, (metric, values) in enumerate(zip(metrics, mae_data), start=1):
            if len(values) == 0:
                continue
            jitter = rng.normal(0, 0.045, size=len(values))
            axes[1, 1].scatter(
                np.full(len(values), idx, dtype=float) + jitter,
                values,
                s=14,
                alpha=0.35,
                color=metric_colors[metric],
                edgecolors="none",
                zorder=3,
            )
        axes[1, 1].set_xticks(box_positions)
        axes[1, 1].set_xticklabels([metric_labels[m] for m in metrics])
        axes[1, 1].set_ylabel("Mean absolute error")
        axes[1, 1].set_title("Magnitude discrepancy across eligible sessions")

        for ax in axes.ravel():
            ax.set_axisbelow(True)
            ax.grid(True, axis="y", alpha=0.2)

        fig._cltr_panel_notes = [
            "Top left|Paired overlap by condition|Mean paired-device overlap minutes by condition for heart rate, electrodermal activity, and temperature.",
            "Top right|Validation-ready sessions|How many sessions per condition meet the paired-device validation threshold for each modality.",
            "Bottom left|Agreement strength across eligible sessions|Agreement strength across eligible sessions; heart rate validates well in the subset where overlap exists, while electrodermal agreement is weak.",
            "Bottom right|Magnitude discrepancy across eligible sessions|Magnitude discrepancy across eligible sessions so device disagreement remains visible even when overlap is complete.",
        ]
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        return fig

    def _fig_cohort_support_map(self, minute: pd.DataFrame):
        if minute.empty or "minute_index" not in minute.columns or "condition_code" not in minute.columns:
            return None
        mapping = [
            ("Questionnaire", "questionnaire_n"),
            ("Empatica", "support_empatica"),
            ("BIOPAC", "support_biopac"),
            ("Indoor", "support_indoor"),
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
                labels.append(f"{cond} | {label}")
        if not rows:
            return None
        mat = np.vstack(rows)
        fig, ax = plt.subplots(figsize=(13.2, max(5.4, 0.34 * len(labels) + 1.8)))
        cmap = LinearSegmentedColormap.from_list("cohort_support", ["#f8fafc", "#0f766e"])
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
        ax.set_yticklabels(labels, fontsize=8)
        tick_count = min(8, len(minute_index_values))
        tick_indices = np.linspace(0, len(minute_index_values) - 1, tick_count, dtype=int) if tick_count > 1 else np.array([0])
        tick_values = [minute_index_values[idx] for idx in tick_indices]
        ax.set_xticks(tick_values)
        ax.set_xticklabels([str(value) for value in tick_values], fontsize=9)
        ax.set_xlabel("Timeline minute")
        self._add_phase_spans(ax, minute_template)
        plt.colorbar(im, ax=ax, shrink=0.82, label="Fraction of sessions with support")
        fig._cltr_panel_notes = [
            "Rows show condition-by-modality support across the full cohort protocol timeline.",
            "Phase annotations mark the shared protocol structure so validation windows can be read against the study timeline.",
        ]
        fig.tight_layout(rect=(0, 0, 1, 0.96))
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
        ax.set_ylabel(FEATURE_LABELS.get(column, column))
        ax.set_xlabel("Minute index")
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
                    f"{FEATURE_LABELS.get(column, column)} is shown as phase-wise condition distributions with raw observation points; questionnaire responses are discrete event-time observations, not continuous trajectories."
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
        ax.set_ylabel(FEATURE_LABELS.get(metric, metric))
        ax.set_xlabel("Protocol phase")
        ax.grid(True, axis="y")
        self._apply_discrete_y_axis_matplotlib(ax, temp[metric], metric)
        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min if y_max > y_min else 1.0
        y_tail = y_min - 0.12 * y_span
        for xpos, n_obs in lower_annotations:
            ax.text(xpos, y_tail, f"n={n_obs}", ha="center", va="top", fontsize=7, color="#475569", clip_on=False)
        ax.set_ylim(y_min - 0.2 * y_span, y_max + 0.04 * y_span)
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

    def _fig_cohort_condition_trace(self, minute: pd.DataFrame, metric: str):
        if minute.empty or metric not in minute.columns:
            return None
        if self._is_sparse_observation_channel(metric):
            return self._fig_sparse_phase_distribution(minute, metric)
        fig, ax = plt.subplots(figsize=self._figsize("wide_single"))
        self._add_phase_spans(ax, minute.drop_duplicates(subset=["minute_index", "protocol_phase"]))
        self._cohort_band(minute, metric, ax)
        ax.set_xlabel("Minute index")
        self._place_condition_legend(ax)
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
            ax.set_ylabel(FEATURE_LABELS.get(metric, metric))
            panel_notes.append(f"{panel_positions[len(panel_notes)]} shows matched condition differences for {FEATURE_LABELS.get(metric, metric)}.")
            self._apply_discrete_y_axis_matplotlib(ax, d["mean_difference"], metric)
            ax.legend(frameon=False, fontsize=8)
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
        sort_cols = ["significant_fdr", "p_value_fdr", "p_value", "n_pairs"]
        ascending = [False, True, True, False]
        existing_cols = [c for c in sort_cols if c in keep.columns]
        keep = keep.sort_values(existing_cols, ascending=ascending[: len(existing_cols)])
        return keep.head(24).reset_index(drop=True)

    def _mixed_effects_register(self, mixed_effects: pd.DataFrame) -> pd.DataFrame:
        if mixed_effects.empty:
            return pd.DataFrame()
        keep = mixed_effects.copy()
        if "significant_fdr" in keep.columns:
            keep = keep.sort_values(["significant_fdr", "p_value_fdr", "metric"], ascending=[False, True, True])
        return keep.head(30).reset_index(drop=True)

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
        ax.set_xlim(0, 1)
        for idx, row in enumerate(top.itertuples(index=False)):
            ax.text(min(float(row.valid_fraction) + 0.02, 0.98), idx, f"{float(row.valid_fraction):.2f}", va="center", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_predictive_benchmarks(self, benchmarks: pd.DataFrame):
        if benchmarks.empty:
            return None
        tasks = list(benchmarks["task"].astype(str).unique())
        fig, axes = plt.subplots(len(tasks), 1, figsize=(self._figsize("wide_single")[0], 3.2 * len(tasks) + 0.5), sharex=True)
        if len(tasks) == 1:
            axes = [axes]
        for ax, task in zip(axes, tasks):
            d = benchmarks.loc[benchmarks["task"].astype(str) == task].copy()
            if d.empty:
                ax.axis("off")
                continue
            d = d.sort_values("balanced_accuracy_mean", ascending=True)
            y = np.arange(len(d))
            ax.barh(y, d["balanced_accuracy_mean"], color="#1d4ed8", alpha=0.9, label="Balanced accuracy")
            for idx, row in enumerate(d.itertuples(index=False)):
                ax.text(min(float(row.balanced_accuracy_mean) + 0.02, 0.98), idx, f"{float(row.balanced_accuracy_mean):.2f}", va="center", fontsize=9)
            ax.set_yticks(y)
            ax.set_yticklabels(d["model"].astype(str))
            ax.set_xlim(0, 1)
            ax.set_title(task.replace("_", " ").title(), loc="left")
        axes[-1].set_xlabel("Grouped-validation balanced accuracy")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
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
                ax.text(row["spearman_r"] if pd.notna(row["spearman_r"]) else 0, row["mae"] if pd.notna(row["mae"]) else 0, str(row["session_id"]), fontsize=7)
            ax.axvline(0, color="#dbe4ee", lw=0.85, ls="--", zorder=0)
            ax.set_xlabel("Spearman r")
            ax.set_ylabel(f"{metric.replace('_', ' ').title()} MAE")
            panel_notes.append(f"{panel_positions[len(panel_notes)]} shows agreement across sessions for {metric.replace('_', ' ')}.")
            ax.grid(True, axis="both", alpha=0.25)
        fig._cltr_panel_notes = panel_notes
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_cohort_primary_endpoints_raw(self, endpoints: pd.DataFrame):
        if endpoints.empty:
            return None
        metrics = [m for m in ["thermal_comfort", "master_dpg_C", "indoor_air_velocity_mean_m_s", "biopac_temp_chest_mean_C", "empatica_hr_mean_bpm"] if m in endpoints["metric"].unique()]
        if not metrics:
            return None
        endpoints = endpoints.loc[endpoints["protocol_phase"].astype(str) != "acclimation"].copy()
        if endpoints.empty:
            return None
        fig, axes = plt.subplots(len(metrics), 1, figsize=(self._figsize("wide_single")[0], 2.2 * len(metrics) + 0.6), sharex=True)
        panel_notes: list[str] = []
        panel_positions = ["Top", "Upper middle", "Center", "Lower middle", "Bottom"]
        if len(metrics) == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            d = endpoints.loc[endpoints["metric"] == metric].copy()
            if d.empty:
                ax.axis("off")
                continue
            d["phase_condition"] = d["protocol_phase"].astype(str) + "\n" + d["condition_code"].astype(str)
            ax.bar(range(len(d)), d["mean_value"], color=[CONDITION_COLORS.get(str(x), "#475569") for x in d["condition_code"]])
            ax.set_ylabel(FEATURE_LABELS.get(metric, metric))
            panel_notes.append(f"{panel_positions[len(panel_notes)]} shows raw means for {FEATURE_LABELS.get(metric, metric)}.")
            ax.axhline(0, color="#dbe4ee", lw=0.85, ls="--", zorder=0)
            ax.grid(True, axis="y")
            self._apply_discrete_y_axis_matplotlib(ax, d["mean_value"], metric)
        axes[-1].set_xticks(range(len(d)))
        axes[-1].set_xticklabels(list(d["phase_condition"]), rotation=45, ha="right")
        fig._cltr_panel_notes = panel_notes
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_cohort_primary_endpoints(self, endpoints: pd.DataFrame):
        if endpoints.empty:
            return None
        metrics = [m for m in ["thermal_comfort", "master_dpg_C", "indoor_air_velocity_mean_m_s", "biopac_temp_chest_mean_C", "empatica_hr_mean_bpm"] if m in endpoints["metric"].unique()]
        if not metrics:
            return None
        tmp = endpoints.loc[endpoints["metric"].isin(metrics) & (endpoints["protocol_phase"].astype(str) != "acclimation")].copy()
        if tmp.empty:
            return None
        tmp["phase_condition"] = tmp["protocol_phase"].astype(str) + " | " + tmp["condition_code"].astype(str)
        pivot = tmp.pivot(index="metric", columns="phase_condition", values="mean_value")
        pivot = pivot.reindex(metrics)
        z = pivot.apply(lambda col: (col - col.mean()) / col.std(ddof=0) if col.notna().sum() > 1 and col.std(ddof=0) > 0 else col * 0, axis=1)
        fig, ax = plt.subplots(figsize=self._figsize("matrix"))
        im = ax.imshow(z.values, aspect="equal", cmap="coolwarm", vmin=-2, vmax=2)
        ax.grid(False)
        ax.set_yticks(range(len(z.index)))
        ax.set_yticklabels([FEATURE_LABELS.get(x, x) for x in z.index])
        ax.set_xticks(range(len(z.columns)))
        ax.set_xticklabels([x.replace(" | ", "\n") for x in z.columns], rotation=45, ha="right")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Within-metric standardized mean")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def _fig_cohort_exploratory_landscape(self, summary: pd.DataFrame, condition_support: pd.DataFrame):
        if summary.empty and condition_support.empty:
            return None
        fig, axes = plt.subplots(2, 2, figsize=(self._figsize("three_panel_row")[0], 8.8))
        axes = axes.ravel()
        panel_notes = [
            "Top left shows the highest-support scientific features.",
            "Top right shows the most variable supported features when available.",
            "Bottom left shows average scientific coverage by domain.",
            "Bottom right shows condition-level support balance.",
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
            domain = summary.groupby("domain").agg(
                mean_coverage=("coverage_fraction", "mean"),
                n_features=("feature", "count"),
            ).reset_index()
            axes[2].bar(domain["domain"], domain["mean_coverage"], color="#0f766e")
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
            axes[3].bar(x - 1.5 * width, cond.get("questionnaire_completeness__mean", pd.Series([np.nan] * len(cond))), width=width, label="Questionnaire", color="#111827")
            axes[3].bar(x - 0.5 * width, cond.get("empatica_fraction__mean", pd.Series([np.nan] * len(cond))), width=width, label="Empatica", color="#2563eb")
            axes[3].bar(x + 0.5 * width, cond.get("biopac_fraction__mean", pd.Series([np.nan] * len(cond))), width=width, label="BIOPAC", color="#dc2626")
            axes[3].bar(x + 1.5 * width, cond.get("indoor_fraction__mean", pd.Series([np.nan] * len(cond))), width=width, label="Indoor", color="#059669")
            axes[3].set_xticks(x)
            axes[3].set_xticklabels(cond["condition_code"], rotation=30, ha="right")
            axes[3].set_ylim(0, 1)
            axes[3].legend(frameon=False, fontsize=8)
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
            axes[0].set_yticks(range(len(pivot.index)))
            axes[0].set_yticklabels([FEATURE_LABELS.get(x, x) for x in pivot.index])
            axes[0].set_xticks(range(len(pivot.columns)))
            axes[0].set_xticklabels(pivot.columns, rotation=45, ha="right")
            for yi, metric in enumerate(pivot.index):
                for xi, pattern in enumerate(pivot.columns):
                    val = float(pivot.loc[metric, pattern])
                    if val > 0:
                        axes[0].text(xi, yi, f"{val:.2f}", ha="center", va="center", fontsize=7.5, color="#3f2a0a")
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
                    fontsize=8,
                    color="#475569",
                )
        else:
            axes[1].axis("off")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
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
        for ax, col, title in zip(axes, ["median_overlap_minutes", "median_spearman_r", "median_mae"], ["Median overlap", "Median Spearman r", "Median MAE"]):
            vals = []
            colors = []
            for metric in metrics:
                row = summary.loc[summary["metric"] == metric]
                vals.append(float(row[col].iloc[0]) if not row.empty and pd.notna(row[col].iloc[0]) else np.nan)
                colors.append("#2563eb" if not row.empty and row["summary_status"].iloc[0] == "eligible" else "#94a3b8")
            ax.bar(metrics, vals, color=colors)
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
        ax.tick_params(axis="x", labelsize=8, pad=8)
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
                ax.set_ylabel(FEATURE_LABELS.get(metric, metric))
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
            ax.set_ylabel(FEATURE_LABELS.get(metric, metric))
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
        exploratory = c.get("exploratory_feature_summary", pd.DataFrame())
        if not exploratory.empty:
            exploratory = exploratory.loc[~exploratory["feature"].astype(str).str.startswith("support_")].copy()
            top_coverage = exploratory.sort_values(["coverage_fraction", "n_non_null"], ascending=[False, False]).head(3)
            obs.append("The strongest data coverage appears in " + ", ".join(FEATURE_LABELS.get(row.feature, row.feature) for row in top_coverage.itertuples()) + ".")
        patterns = c.get("pattern_summary", pd.DataFrame())
        if not patterns.empty:
            top_pattern = patterns.iloc[0]
            obs.append(
                f"Most recurrent pattern is {FEATURE_LABELS.get(top_pattern['metric'], top_pattern['metric'])} during {top_pattern['dominant_phase']} with a {top_pattern['direction']} direction across {top_pattern['n_sessions']} sessions."
            )
        agreement = c["sensor_agreement"]
        hr = agreement.loc[(agreement["metric"] == "heart_rate") & (agreement["eligible"] == 1), "spearman_r"].dropna()
        if not hr.empty:
            obs.append(f"Median eligible heart-rate agreement is `{hr.median():.2f}`.")
        else:
            obs.append("No eligible heart-rate agreement rows are available yet.")
        signal_audit = c.get("signal_audit_summary", pd.DataFrame())
        if not signal_audit.empty:
            primary = signal_audit.loc[signal_audit["recommended_role"].astype(str).isin(["primary", "primary_with_qc"]), "signal_stream"].astype(str).tolist()
            limited = signal_audit.loc[signal_audit["recommended_role"].astype(str).isin(["secondary_only", "secondary_validation", "subset_only", "not_primary", "not_recommended"]), "signal_stream"].astype(str).tolist()
            if primary:
                obs.append("Primary device streams in this release are " + ", ".join(self._fmt_cell(x) for x in primary) + ".")
            if limited:
                obs.append("Limited or secondary-use streams are " + ", ".join(self._fmt_cell(x) for x in limited[:4]) + ".")
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
        )

    def _cohort_html(self, cohort_inputs: dict, narrative_specs: list[dict], appendix_specs: list[dict]) -> str:
        sample = cohort_inputs["sample_status"].iloc[0]
        cards = [
            ("Sessions", int(sample["n_sessions"])),
            ("Participants", int(sample["n_participants"])),
            ("Comparison readiness", "full" if int(sample["cohort_inference_eligible"]) else "limited"),
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
        )

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
                }
            )
        cards = "".join(
            f"<article class='sessionCard'>"
            f"<div class='eyebrow'>{html_escape(r['condition_code'])}</div><h3>{html_escape(r['session_id'])}</h3><p><strong>Participant:</strong> {html_escape(r['participant_id'])}<br><strong>Condition:</strong> {html_escape(r['condition_code'])}<br><strong>Evidence score:</strong> {r['evidence_score']}</p><p><strong>Lead story:</strong> {html_escape(r['lead_label'])}</p><p>{html_escape(r['headline'])}</p><p class='tagLine'>{html_escape(r['tags'])}</p><a class='pillLink' href='../sessions/{html_escape(r['session_id'])}/html/{html_escape(r['html_name'])}'>{SESSION_CTA}</a></article>"
            for r in records
        )
        session_nav_items = "".join(
            f"<a href='../sessions/{html_escape(r['session_id'])}/html/{html_escape(r['html_name'])}' title='Open session report for {html_escape(r['session_id'])}'>"
            f"{html_escape(r['session_id'])}<span>{html_escape(r['participant_id'])} | {html_escape(r['condition_code'])}</span></a>"
            for r in records
        )
        cohort_name = Path(cohort_report["html_path"]).name if cohort_report.get("html_path") else ""
        atlas_intro = (
            f"<section class='panel heroIntro heroSticky'>"
            f"<div class='eyebrow'>CLTR Reporting</div>"
            f"<div class='title'>{WORK_INDEX_TITLE}</div>"
            f"<p class='subtitle'>{WORK_INDEX_SUBTITLE}</p>"
            f"<div class='heroMeta'>"
            f"<p class='heroStatement'>An interactive report hub for the CLTR study, combining study-wide findings with session-level views of physiology, environmental conditions, and comfort responses.</p>"
            f"<div class='heroFacts'>"
            f"<div class='heroFact'><div class='heroFactLabel'>Study</div><div class='heroFactValue'>Controlled Laboratory Thermal Response study reporting hub.</div></div>"
            f"<div class='heroFact'><div class='heroFactLabel'>Coverage</div><div class='heroFactValue'>{len(records)} session reports plus one study-wide summary.</div></div>"
            f"</div>"
            f"</div>"
            f"</section>"
        )
        cohort_panel = (
            f"<section class='panel heroCta heroSticky'>"
            f"<div class='eyebrow'>Start Here</div>"
            f"<div class='title'>Study Summary</div>"
            f"<p class='subtitle'>Start with the study-wide report for the overall CLTR findings, then explore individual session reports below.</p>"
            f"<a class='pillLink' href='../cohort/html/{html_escape(cohort_name)}'>{COHORT_CTA}</a>"
            f"</section>"
        )
        masthead = self._shared_chrome(
            home_href="index.html",
            page_type="Atlas",
            page_meta=f"{len(records)} session reports and one cohort summary",
            menu_button_id="sessionMenuButton",
            menu_panel_id="sessionMenuPanel",
            menu_label="Sessions",
            menu_title="Session List",
            menu_items_html=session_nav_items,
            menu_icon_bars=True,
        )
        return f"""<!doctype html><html><head><meta charset='utf-8'><title>{WORK_INDEX_TITLE}</title>
<style>
{self._shared_index_css()}
</style></head><body class='reportKind--atlas'>{masthead}<div class='page'><section class='hero'>{atlas_intro}{cohort_panel}</section><section id='sessionGrid' class='grid'>{cards}</section></div><div class='copyrightNote'>{COPYRIGHT_NOTE}</div><script>
{self._theme_toggle_script()}
const sessionMenuButton=document.getElementById('sessionMenuButton'); const sessionMenuPanel=document.getElementById('sessionMenuPanel');
const closeSessionMenu=()=>{{ if(!sessionMenuPanel||!sessionMenuButton) return; sessionMenuPanel.classList.remove('open'); sessionMenuButton.setAttribute('aria-expanded','false'); }};
const toggleSessionMenu=()=>{{ if(!sessionMenuPanel||!sessionMenuButton) return; const open=sessionMenuPanel.classList.toggle('open'); sessionMenuButton.setAttribute('aria-expanded', open ? 'true' : 'false'); }};
if(sessionMenuButton&&sessionMenuPanel){{ sessionMenuButton.addEventListener('click',(event)=>{{ event.stopPropagation(); toggleSessionMenu(); }}); sessionMenuPanel.querySelectorAll('a').forEach(link=>link.addEventListener('click', closeSessionMenu)); document.addEventListener('click',(event)=>{{ if(!sessionMenuPanel.contains(event.target) && !sessionMenuButton.contains(event.target)) closeSessionMenu(); }}); document.addEventListener('keydown',(event)=>{{ if(event.key==='Escape') closeSessionMenu(); }}); }}
</script></body></html>"""

    def _render_spec_sections(
        self,
        specs: list[dict],
        intro_sections: str = "",
        section_intro_map: dict[str, str] | None = None,
    ) -> str:
        display_map, section_map = self._display_numbering(
            specs,
            intro_sections=intro_sections,
            section_intro_map=section_intro_map,
        )
        parts = []
        section_intro_map = section_intro_map or {}
        for section in SECTION_ORDER:
            section_specs = [spec for spec in specs if spec.get("display_section", "analyzed") == section]
            intro_html = intro_sections if section == "frontmatter" else section_intro_map.get(section, "")
            if not section_specs and not intro_html:
                continue
            body = "".join(self._figure_subsection(spec, display_map.get(spec["stem"], spec["code"])) for spec in section_specs)
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
    ) -> tuple[dict[str, str], dict[str, str]]:
        display_map: dict[str, str] = {}
        section_map: dict[str, str] = {}
        section_intro_map = section_intro_map or {}
        section_index = 0
        for section in SECTION_ORDER:
            section_specs = [spec for spec in specs if spec.get("display_section", "analyzed") == section]
            intro_html = bool(intro_sections) if section == "frontmatter" else bool(section_intro_map.get(section))
            if not section_specs and not intro_html:
                continue
            section_index += 1
            section_map[section] = f"Section {section_index}. {SECTION_TITLES[section]}"
            for figure_index, spec in enumerate(section_specs):
                display_map[spec["stem"]] = f"{section_index}.{figure_index + 1}"
        return display_map, section_map

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
        doc_kind: str = "report",
    ) -> str:
        all_specs = main_specs + appendix_specs
        display_map, _ = self._display_numbering(
            all_specs,
            intro_sections=intro_sections,
            section_intro_map=section_intro_map,
        )
        nav_items = "".join(
            f"<a href='#{html_escape(spec['stem'])}' aria-label='Figure {html_escape(display_map.get(spec['stem'], spec['code']))}: {html_escape(spec['title'])}' title='Figure {html_escape(display_map.get(spec['stem'], spec['code']))}: {html_escape(spec['title'])}'>{html_escape(display_map.get(spec['stem'], spec['code']))}<span>{html_escape(spec['title'])}</span></a>"
            for spec in all_specs
        )
        cards_html = "".join(f"<div class='card'><div class='label'>{html_escape(k)}</div><div class='value'>{html_escape(v)}</div></div>" for k, v in cards)
        obs_html = self._takeaways_html(observations)
        sections_html = self._render_spec_sections(
            all_specs,
            intro_sections=intro_sections,
            section_intro_map=section_intro_map,
        )
        plotly_js = f"<script>{get_plotlyjs()}</script>" if any(spec.get("html_fragment") for spec in all_specs) else ""
        badge = "Cohort Report" if str(doc_kind) == "cohort" else "Session Report"
        home_href = "../../work/index.html" if str(doc_kind) == "cohort" else "../../../work/index.html"
        masthead = self._shared_chrome(
            home_href=home_href,
            page_type=badge,
            page_meta=title,
            menu_button_id="figureMenuButton",
            menu_panel_id="figureMenuPanel",
            menu_label="List of Figures",
            menu_title="List of Figures",
            menu_items_html=nav_items,
        )
        return f"""<!doctype html><html><head><meta charset='utf-8'><title>{html_escape(title)}</title>
<style>
{self._shared_report_css()}
</style></head><body class='reportKind--{html_escape(doc_kind)}'>{masthead}<div class='page' id='pageRoot'><section class='hero'><div class='panel heroLead'><div class='eyebrow'>{html_escape(badge)}</div><div class='title'>{html_escape(title)}</div><p class='subtitle'>{html_escape(subtitle)}</p><div class='cards'>{cards_html}</div></div><div class='panel heroSide'>{obs_html}</div></section><div class='reportShell'><section class='stack'>{sections_html}</section></div></div><div id='lightbox' class='lightbox'><img id='lightboxImg' alt='Expanded figure'/></div><div class='copyrightNote'>{COPYRIGHT_NOTE}</div>{plotly_js}<script>
{self._theme_toggle_script()}
const lightbox=document.getElementById('lightbox'); const lightboxImg=document.getElementById('lightboxImg'); document.querySelectorAll('.figureImage').forEach(img=>img.addEventListener('click',()=>{{ lightboxImg.src=img.src; lightbox.classList.add('open'); }})); lightbox.addEventListener('click',()=>lightbox.classList.remove('open'));
const figureMenuButton=document.getElementById('figureMenuButton'); const figureMenuPanel=document.getElementById('figureMenuPanel');
const closeFigureMenu=()=>{{ if(!figureMenuPanel||!figureMenuButton) return; figureMenuPanel.classList.remove('open'); figureMenuButton.setAttribute('aria-expanded','false'); }};
const toggleFigureMenu=()=>{{ if(!figureMenuPanel||!figureMenuButton) return; const open=figureMenuPanel.classList.toggle('open'); figureMenuButton.setAttribute('aria-expanded', open ? 'true' : 'false'); }};
if(figureMenuButton&&figureMenuPanel){{ figureMenuButton.addEventListener('click',(event)=>{{ event.stopPropagation(); toggleFigureMenu(); }}); figureMenuPanel.querySelectorAll('a').forEach(link=>link.addEventListener('click', closeFigureMenu)); document.addEventListener('click',(event)=>{{ if(!figureMenuPanel.contains(event.target) && !figureMenuButton.contains(event.target)) closeFigureMenu(); }}); document.addEventListener('keydown',(event)=>{{ if(event.key==='Escape') closeFigureMenu(); }}); }}
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

    def _figure_subsection(self, spec: dict, display_code: str) -> str:
        section_label = f"Figure {display_code}"
        return f"<section class='figureSection'><h3 class='figureSectionTitle'>{html_escape(section_label)}</h3>{self._figure_block(spec)}</section>"

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

    def _figure_block(self, spec: dict) -> str:
        path = Path(spec["path"]).name if spec.get("path") else ""
        meta_parts = [f"Evidence: {spec['evidence_label']} ({int(spec['evidence_score'])})"]
        if spec.get("gating_note"):
            meta_parts.append(f"Gate: {spec['gating_note']}")
        meta = f"<p class='figureMeta'>{html_escape(' | '.join(meta_parts))}</p>"
        classes = "figurePanel"
        if spec.get("html_fragment"):
            media = f"<div class='responsiveFigure'>{spec['html_fragment']}</div>"
        else:
            media = f"<img class='figureImage' src='../figures/{html_escape(path)}' alt='{html_escape(spec['title'])}'/>"
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
        headers = "".join(f"<th>{html_escape(self._table_column_label(c))}</th>" for c in view.columns)
        rows = []
        for _, row in view.iterrows():
            rows.append("<tr>" + "".join(f"<td>{html_escape(self._fmt_cell(v))}</td>" for v in row.tolist()) + "</tr>")
        return f"<section class='tablePanel'><h3>{html_escape(title)}</h3><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></section>"

    def _table_column_label(self, value: object) -> str:
        text = str(value)
        if text in FEATURE_LABELS:
            return FEATURE_LABELS[text]
        return TABLE_COLUMN_LABELS.get(text, text.replace("_", " ").title())

    def _fmt_cell(self, value: object) -> str:
        if isinstance(value, float):
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
                "heart_rate": "Heart rate",
                "eda": "EDA",
                "temperature": "Temperature",
                "bvp_source": "BVP source",
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
            ],
            12,
        )
        ctab = self._render_table(
            agreement,
            "Device Comparison Summary",
            ["metric", "n_sessions", "n_eligible_sessions", "median_overlap_minutes", "median_spearman_r", "median_mae", "summary_status"],
            8,
        )
        signal_table = self._render_table(
            signal_audit,
            "Signal Adequacy And Recommended Use",
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
            12,
        )
        signal_reading = self._render_table(
            signal_audit,
            "Scientific Reading Of Device Streams",
            ["signal_stream", "adequacy_status", "flagged_sessions", "scientific_reading"],
            12,
        )
        hr_flags = pd.DataFrame()
        if not session_signal_audit.empty:
            hr_flags = (
                session_signal_audit.loc[session_signal_audit["construct"].astype(str) == "heart_rate"]
                .sort_values(["concern_score", "session_id"], ascending=[False, True])
                .head(8)
                .copy()
            )
        flag_table = self._render_table(
            hr_flags,
            "Flagged Heart-Rate Sessions",
            ["session_id", "signal_stream", "n_valid_minutes", "paired_overlap_minutes", "paired_spearman_r", "min_value", "max_value", "concern_score"],
            8,
        )
        return f"<section class='tableGrid'>{b}</section><section class='tableGrid'>{ctab}{signal_table}</section><section class='tableGrid'>{signal_reading}{flag_table}</section>"

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
        hr_rows = signal_audit.loc[signal_audit["construct"].astype(str) == "heart_rate"].copy() if not signal_audit.empty else pd.DataFrame()
        hr_summary = ""
        if not hr_rows.empty:
            primary = hr_rows.loc[hr_rows["recommended_role"].astype(str).isin(["primary", "primary_with_qc"]), "signal_stream"].astype(str).tolist()
            limited = hr_rows.loc[~hr_rows["recommended_role"].astype(str).isin(["primary", "primary_with_qc"]), "signal_stream"].astype(str).tolist()
            pieces = []
            if primary:
                pieces.append("primary HR support is carried by " + ", ".join(self._fmt_cell(x) for x in primary))
            if limited:
                pieces.append("limited HR streams are " + ", ".join(self._fmt_cell(x) for x in limited))
            if pieces:
                hr_summary = " In the current release, " + "; ".join(pieces) + "."
        synopsis = self._stage_panel(
            "Synopsis",
            (
                "This cohort report is the final synthesis layer of the CLTR pipeline: session timelines are aligned first, modality-specific minute summaries are built next, support and agreement are audited before interpretation, and only then are cohort-level patterns, contrasts, and device conclusions presented."
                + hr_summary
                + " "
                + (
                    "The study includes enough sessions and participants for cross-session comparison, but the signal audit below should still be read as part of the result itself, because device adequacy and disagreement determine which endpoints can be defended scientifically."
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
            "raw": self._section_lead_list(
                "How To Read The Measured Trends",
                [
                    "These figures show cohort-level observed trajectories and condition-stratified traces before support-gated endpoint reduction is applied.",
                    "Visual ribbons and condition traces are descriptive summaries of available session support, not inferential estimates.",
                    "Differences in line continuity or spread can reflect support density and condition balance as much as physiology or comfort dynamics.",
                ],
            ),
            "analyzed": self._section_lead_list(
                "How To Read The Derived Results",
                [
                    derived_opening,
                    "Condition-phase values are descriptive medians across available session summaries, not model-based cohort effects.",
                    "Reference-phase deltas are within-condition descriptive departures from the earliest supported phase for each endpoint.",
                    "Session-sign agreement quantifies directional consistency across sessions within a condition; it is not a formal reproducibility coefficient.",
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
                    legend = ax.get_legend()
                    if legend is None:
                        continue
                    labels = [text.get_text().strip() for text in legend.get_texts() if text.get_text().strip()]
                    if len(labels) > 1 and set(labels).issubset(condition_labels):
                        if hasattr(legend, "set_ncols"):
                            legend.set_ncols(len(labels))
                        else:
                            legend._ncols = len(labels)
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
