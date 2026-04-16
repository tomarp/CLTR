from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


@dataclass
class RuntimeConfig:
    timeline_timezone: str = "Europe/Paris"
    output_float_format: str = "%.6f"
    min_phase_minutes: int = 3
    min_contrast_pairs: int = 8
    min_cohort_sessions_for_inference: int = 8
    min_cohort_participants_for_inference: int = 6
    min_sensor_overlap_minutes: int = 20
    min_questionnaire_minutes_for_session_narrative: int = 4
    mpl_config_dirname: str = ".mplconfig"


@dataclass
class DatasetConfig:
    master_dir: str = "master_files"
    env_dir: str = "env"
    empatica_dir: str = "empatica"
    biopac_dir: str = "biopac"
    metadata_dir: str = "metadata"
    timeline_file: str = "timeline_by_minutes.csv"
    sessions_file: str = "sessions.csv"
    questionnaire_file: str = "questionnaire_events.csv"
    skin_temperature_file: str = "skin_temperature_timeseries.csv"
    fan_behavior_file: str = "fan_behavior_timeseries.csv"
    indoor_file: str = "indoor_climate.csv"
    outdoor_file: str = "outdoor_meteorology.csv"


@dataclass
class ReportingConfig:
    session_report_basename: str = "{session_id}_report"
    cohort_report_basename: str = "cohort_report"
    pdf_latex_engine: str = "pdflatex"
    svg_to_pdf_command: str = "rsvg-convert"
    figure_dpi: int = 160
    max_trace_points: int = 3000


@dataclass
class OutputConfig:
    validation_dir: str = "validation"
    manifests_dir: str = "manifests"
    data_dir: str = "data"
    session_dir: str = "sessions"
    cohort_dir: str = "cohort"
    work_dir: str = "work"
    analysis_dir: str = "analysis"
    report_dir: str = "reports"
    figure_dir: str = "figures"
    html_dir: str = "html"
    pdf_dir: str = "pdf"
    tmp_dir: str = "tmp"


@dataclass
class CLTRConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def default_config() -> CLTRConfig:
    return CLTRConfig()
