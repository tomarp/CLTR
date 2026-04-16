from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import CLTRConfig
from .utils import REQUIRED_CHILDREN, ensure_dir, parse_local_datetime, safe_read_csv, session_id_parts, write_json


@dataclass
class DatasetBundle:
    timeline: pd.DataFrame
    sessions: pd.DataFrame
    questionnaire: pd.DataFrame
    skin_temperature: pd.DataFrame
    fan_behavior: pd.DataFrame
    indoor: pd.DataFrame
    outdoor: pd.DataFrame
    manifest: pd.DataFrame


class CLTRDataset:
    def __init__(self, dataset_root: str | Path, config: CLTRConfig):
        self.root = Path(dataset_root)
        self.config = config
        self.ds = config.dataset

    def validate_structure(self) -> dict:
        missing = [name for name in REQUIRED_CHILDREN if not (self.root / name).exists()]
        status = {
            "dataset_root": str(self.root.resolve()),
            "missing_required_children": missing,
            "is_valid": not missing,
        }
        return status

    def build_manifest(self) -> pd.DataFrame:
        emp_root = self.root / self.ds.empatica_dir
        bio_root = self.root / self.ds.biopac_dir
        timeline = safe_read_csv(self.root / self.ds.master_dir / self.ds.timeline_file)
        timeline = timeline.rename(columns={"Session ID": "session_id"})
        timeline_ids = set(timeline["session_id"].astype(str))
        emp = {p.name: p for p in emp_root.glob("*") if p.is_dir()}
        bio = {p.name: p for p in bio_root.glob("*") if p.is_dir()}
        rows = []
        for session_id in sorted(set(emp) | set(bio) | timeline_ids):
            parts = session_id_parts(session_id)
            rows.append({
                **parts,
                "has_empatica": session_id in emp,
                "has_biopac": session_id in bio,
                "in_timeline": session_id in timeline_ids,
                "empatica_path": str(emp[session_id]) if session_id in emp else None,
                "biopac_path": str(bio[session_id]) if session_id in bio else None,
            })
        return pd.DataFrame(rows).sort_values("session_id").reset_index(drop=True)

    def load_bundle(self) -> DatasetBundle:
        master_root = self.root / self.ds.master_dir
        env_root = self.root / self.ds.env_dir

        timeline = safe_read_csv(master_root / self.ds.timeline_file)
        timeline = timeline.rename(columns={
            "datetime": "Datetime",
            "session_id": "Session ID",
            "participant": "Participant ID",
            "minute_index": "Minute index",
            "protocol_block": "Protocol block",
            "protocol_phase": "Protocol phase",
            "expected_fan_mode": "Expected fan mode",
        })
        timeline["minute_local"] = parse_local_datetime(timeline["Datetime"], self.config.runtime.timeline_timezone)
        timeline["minute_utc"] = timeline["minute_local"].dt.tz_convert("UTC")
        timeline = timeline.rename(columns={
            "Session ID": "session_id",
            "Participant ID": "participant_id",
            "Minute index": "minute_index",
            "Protocol block": "protocol_block",
            "Protocol phase": "protocol_phase",
            "Expected fan mode": "expected_fan_mode",
        })
        extra = pd.DataFrame([session_id_parts(x) for x in timeline["session_id"]])
        for col in extra.columns:
            if col not in timeline.columns:
                timeline[col] = extra[col]

        sessions = safe_read_csv(master_root / self.ds.sessions_file).rename(columns={
            "Session ID": "session_id",
            "Participant ID": "participant_id",
        })
        if "session_id" in sessions.columns:
            extra = pd.DataFrame([session_id_parts(x) for x in sessions["session_id"]])
            for col in extra.columns:
                if col not in sessions.columns:
                    sessions[col] = extra[col]

        questionnaire = self._load_event_minute_table(master_root / self.ds.questionnaire_file)
        skin_temperature = self._load_event_minute_table(master_root / self.ds.skin_temperature_file)
        fan_behavior = self._load_event_minute_table(master_root / self.ds.fan_behavior_file)
        indoor = self._load_event_minute_table(env_root / self.ds.indoor_file)
        outdoor = safe_read_csv(env_root / self.ds.outdoor_file)
        outdoor["minute_local"] = parse_local_datetime(outdoor["datetime"], self.config.runtime.timeline_timezone).dt.floor("min")
        outdoor["minute_utc"] = outdoor["minute_local"].dt.tz_convert("UTC")
        manifest = self.build_manifest()
        return DatasetBundle(
            timeline=timeline,
            sessions=sessions,
            questionnaire=questionnaire,
            skin_temperature=skin_temperature,
            fan_behavior=fan_behavior,
            indoor=indoor,
            outdoor=outdoor,
            manifest=manifest,
        )

    def _load_event_minute_table(self, path: Path) -> pd.DataFrame:
        df = safe_read_csv(path)
        rename = {"Session ID": "session_id", "Participant ID": "participant_id"}
        df = df.rename(columns=rename)
        if "datetime" in df.columns:
            df["minute_local"] = parse_local_datetime(df["datetime"], self.config.runtime.timeline_timezone).dt.floor("min")
            df["minute_utc"] = df["minute_local"].dt.tz_convert("UTC")
        return df

    def write_validation(self, outdir: str | Path) -> dict:
        outdir = ensure_dir(outdir)
        structure = self.validate_structure()
        write_json(structure, outdir / "dataset_structure.json")
        manifest = self.build_manifest()
        manifest.to_csv(outdir / "session_manifest.csv", index=False)
        return structure
