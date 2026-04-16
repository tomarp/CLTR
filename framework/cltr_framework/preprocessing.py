from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLTRConfig
from .dataset import DatasetBundle
from .utils import minute_floor, parse_any_datetime, safe_read_csv, session_id_parts, to_numeric


CANONICAL_MINUTE_COLUMNS = [
    "session_id",
    "participant_id",
    "study_day",
    "condition_code",
    "illuminance_level",
    "time_of_day",
    "minute_index",
    "protocol_block",
    "protocol_phase",
    "expected_fan_mode",
    "minute_local",
    "minute_utc",
    "questionnaire_n",
    "thermal_sensation",
    "thermal_comfort",
    "thermal_pleasure",
    "thermal_preference",
    "visual_sensation",
    "color_sensation",
    "visual_comfort",
    "sound_comfort_dbA",
    "air_quality_comfort",
    "room_comfort",
    "discomfort_source",
    "master_skin_chest_C",
    "master_skin_thigh_C",
    "master_skin_arm_C",
    "master_skin_tibia_C",
    "master_hand_C",
    "master_neck_C",
    "master_shoulder_C",
    "master_mst_C",
    "master_distal_C",
    "master_proximal_C",
    "master_dpg_C",
    "fan_current_A",
    "fan_control_au",
    "fan_control_secondary_au",
    "fan_state_primary",
    "fan_state_secondary",
    "fan_mode_observed_primary",
    "fan_mode_observed_secondary",
    "indoor_air_temp_mean_C",
    "indoor_air_velocity_mean_m_s",
    "indoor_relative_humidity_percent",
    "indoor_illuminance_lux",
    "indoor_sound_dbA",
    "indoor_co2_ppm",
    "indoor_pmv_fanger",
    "outdoor_air_temp_C",
    "outdoor_relative_humidity_percent",
    "outdoor_wind_speed_m_s",
    "outdoor_solar_radiation_W_m2",
    "empatica_bvp_mean",
    "empatica_bvp_sd",
    "empatica_eda_mean_uS",
    "empatica_eda_p95_uS",
    "empatica_temp_mean_C",
    "empatica_temp_sd_C",
    "empatica_acc_mean_g",
    "empatica_enmo_mean_g",
    "empatica_steps",
    "empatica_hr_mean_bpm",
    "empatica_hr_sd_bpm",
    "biopac_bloodflow_mean_bpu",
    "biopac_temp_chest_mean_C",
    "biopac_temp_thigh_mean_C",
    "biopac_temp_arm_mean_C",
    "biopac_temp_tibia_mean_C",
    "biopac_eda_mean_uS",
    "biopac_hr_mean_bpm",
    "biopac_backscatter_mean_percent",
    "thermal_gradient_C",
    "thermal_state_index_C",
    "hr_delta_bpm",
    "eda_delta_uS",
    "temp_delta_C",
    "support_questionnaire",
    "support_fan",
    "support_empatica",
    "support_biopac",
    "support_indoor",
    "support_outdoor",
    "support_core_overlap_hr",
    "support_core_overlap_eda",
    "support_core_overlap_temp",
    "eligible_session_hr_agreement",
    "eligible_session_eda_agreement",
    "eligible_session_temp_agreement",
    "quality_empatica_hr",
    "quality_empatica_eda",
    "quality_empatica_temp",
    "quality_biopac_hr",
    "quality_biopac_eda",
    "quality_biopac_temp",
    "quality_empatica_bvp",
]

PHASE_METRICS = [
    "thermal_comfort",
    "thermal_sensation",
    "thermal_pleasure",
    "thermal_preference",
    "visual_comfort",
    "sound_comfort_dbA",
    "air_quality_comfort",
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

COMPARISON_BLOCKS = {"1", "2", "3"}
QUESTIONNAIRE_SET_COLUMNS = [
    "thermal_sensation",
    "thermal_comfort",
    "thermal_pleasure",
    "thermal_preference",
    "visual_sensation",
    "color_sensation",
    "visual_comfort",
    "sound_comfort_dbA",
    "air_quality_comfort",
    "room_comfort",
    "discomfort_source",
]


@dataclass
class SessionArtifacts:
    aligned_minute: pd.DataFrame
    phase_summary: pd.DataFrame
    processing_metadata: dict


class SessionPreprocessor:
    def __init__(self, dataset_root: str | Path, config: CLTRConfig, bundle: DatasetBundle):
        self.root = Path(dataset_root)
        self.config = config
        self.bundle = bundle

    def process_session(self, session_id: str) -> SessionArtifacts:
        grid = self._session_grid(session_id)
        minute = grid.merge(self._questionnaire(session_id), on=["session_id", "minute_utc"], how="left")
        minute = minute.merge(self._skin_temperature(session_id), on=["session_id", "minute_utc"], how="left")
        minute = minute.merge(self._fan_behavior(session_id), on=["session_id", "minute_utc"], how="left")
        minute = minute.merge(self._indoor(session_id), on=["session_id", "minute_utc"], how="left")
        minute = self._merge_outdoor(minute)

        emp_dir = self.root / self.config.dataset.empatica_dir / session_id
        bio_file = self.root / self.config.dataset.biopac_dir / session_id / "biopac.csv"
        if emp_dir.exists():
            minute = minute.merge(self._empatica(emp_dir), on="minute_utc", how="left")
        if bio_file.exists():
            minute = minute.merge(self._biopac(bio_file), on="minute_utc", how="left")

        minute = self._canonicalize(minute)
        minute = minute.sort_values("minute_index").reset_index(drop=True)
        phase_summary = self._phase_summary(minute)
        metadata = self._processing_metadata(minute, phase_summary)
        return SessionArtifacts(aligned_minute=minute, phase_summary=phase_summary, processing_metadata=metadata)

    def _session_grid(self, session_id: str) -> pd.DataFrame:
        grid = self.bundle.timeline.loc[self.bundle.timeline["session_id"] == session_id].copy()
        session_meta = self.bundle.sessions.loc[self.bundle.sessions["session_id"] == session_id].copy()
        if not session_meta.empty:
            meta = session_meta.iloc[0]
            for key in ["study_day", "condition_code", "illuminance_level", "time_of_day"]:
                if key not in grid.columns and key in meta:
                    grid[key] = meta[key]
        return grid[
            [
                "session_id",
                "participant_id",
                "study_day",
                "condition_code",
                "illuminance_level",
                "time_of_day",
                "Datetime",
                "minute_local",
                "minute_utc",
                "minute_index",
                "protocol_block",
                "protocol_phase",
                "expected_fan_mode",
            ]
        ].copy()

    def _questionnaire(self, session_id: str) -> pd.DataFrame:
        d = self.bundle.questionnaire.loc[self.bundle.questionnaire["session_id"] == session_id].copy()
        if d.empty:
            return pd.DataFrame({"session_id": pd.Series(dtype=str), "minute_utc": pd.Series(dtype="datetime64[ns, UTC]")})
        renames = {
            "q n": "questionnaire_n",
            "thermal sensation": "thermal_sensation",
            "thermal comfort": "thermal_comfort",
            "thermal pleasure": "thermal_pleasure",
            "thermal preference": "thermal_preference",
            "visual sensation": "visual_sensation",
            "color sensation": "color_sensation",
            "visual comfort": "visual_comfort",
            "sound comfort (dB(A))": "sound_comfort_dbA",
            "airQuality comfort": "air_quality_comfort",
            "room comfort": "room_comfort",
            "source of discomfort": "discomfort_source",
        }
        d = d.rename(columns=renames)
        keep = ["session_id", "minute_utc"] + [v for v in renames.values() if v in d.columns]
        d = d[keep]
        num_cols = [c for c in keep if c not in {"session_id", "minute_utc", "discomfort_source"}]
        for col in num_cols:
            d[col] = to_numeric(d[col])
        out = d[["session_id", "minute_utc"]].drop_duplicates()
        if num_cols:
            out = out.merge(d.groupby(["session_id", "minute_utc"])[num_cols].mean(numeric_only=True).reset_index(), on=["session_id", "minute_utc"], how="left")
        if "discomfort_source" in d.columns:
            collapsed = d.groupby(["session_id", "minute_utc"])["discomfort_source"].agg(lambda s: "; ".join(sorted({str(x).strip() for x in s.dropna() if str(x).strip()}))).reset_index()
            out = out.merge(collapsed, on=["session_id", "minute_utc"], how="left")
        return out

    def _skin_temperature(self, session_id: str) -> pd.DataFrame:
        d = self.bundle.skin_temperature.loc[self.bundle.skin_temperature["session_id"] == session_id].copy()
        if d.empty:
            return pd.DataFrame({"session_id": pd.Series(dtype=str), "minute_utc": pd.Series(dtype="datetime64[ns, UTC]")})
        renames = {
            "biopac skin temp chest": "master_skin_chest_C",
            "biopac skin temp thigh": "master_skin_thigh_C",
            "biopac skin temp arm": "master_skin_arm_C",
            "biopac skin temp tibia": "master_skin_tibia_C",
            "thermocouple skin temp hand": "master_hand_C",
            "thermocouple skin temp neck": "master_neck_C",
            "thermocouple skin temp shoulder": "master_shoulder_C",
            "mst (°C)": "master_mst_C",
            "distal": "master_distal_C",
            "proximal": "master_proximal_C",
            "dpg": "master_dpg_C",
        }
        d = d.rename(columns=renames)
        keep = ["session_id", "minute_utc"] + [v for v in renames.values() if v in d.columns]
        d = d[keep]
        for col in keep:
            if col not in {"session_id", "minute_utc"}:
                d[col] = to_numeric(d[col])
        return d.groupby(["session_id", "minute_utc"]).mean(numeric_only=True).reset_index()

    def _fan_behavior(self, session_id: str) -> pd.DataFrame:
        d = self.bundle.fan_behavior.loc[self.bundle.fan_behavior["session_id"] == session_id].copy()
        if d.empty:
            return pd.DataFrame({"session_id": pd.Series(dtype=str), "minute_utc": pd.Series(dtype="datetime64[ns, UTC]")})
        renames = {
            "fan current (A)": "fan_current_A",
            "fan control (a.u.)": "fan_control_au",
            "fan control (a.u.).1": "fan_control_secondary_au",
            "fan state": "fan_state_primary",
            "fan state.1": "fan_state_secondary",
            "fan mode observed": "fan_mode_observed_primary",
            "fan mode observed.1": "fan_mode_observed_secondary",
        }
        d = d.rename(columns=renames)
        keep = ["session_id", "minute_utc"] + [v for v in renames.values() if v in d.columns]
        out = d[["session_id", "minute_utc"]].drop_duplicates()
        numeric = [c for c in ["fan_current_A", "fan_control_au", "fan_control_secondary_au"] if c in d.columns]
        text_cols = [c for c in ["fan_state_primary", "fan_state_secondary", "fan_mode_observed_primary", "fan_mode_observed_secondary"] if c in d.columns]
        if numeric:
            out = out.merge(d.groupby(["session_id", "minute_utc"])[numeric].mean(numeric_only=True).reset_index(), on=["session_id", "minute_utc"], how="left")
        for col in text_cols:
            agg = d.groupby(["session_id", "minute_utc"])[col].agg(lambda s: "; ".join(sorted({str(x).strip() for x in s.dropna() if str(x).strip()}))).reset_index()
            out = out.merge(agg, on=["session_id", "minute_utc"], how="left")
        return out

    def _indoor(self, session_id: str) -> pd.DataFrame:
        d = self.bundle.indoor.loc[self.bundle.indoor["session_id"] == session_id].copy()
        if d.empty:
            return pd.DataFrame({"session_id": pd.Series(dtype=str), "minute_utc": pd.Series(dtype="datetime64[ns, UTC]")})
        temp_cols = [c for c in d.columns if "air temperature" in c.lower()]
        vel_cols = [c for c in d.columns if "air velocity" in c.lower()]
        out = d[["session_id", "minute_utc"]].drop_duplicates().copy()
        out["indoor_air_temp_mean_C"] = d[temp_cols].mean(axis=1) if temp_cols else np.nan
        out["indoor_air_velocity_mean_m_s"] = d[vel_cols].mean(axis=1) if vel_cols else np.nan
        renames = {
            "relative humidity  1034 (%RH)": "indoor_relative_humidity_percent",
            "illuminance  1035 (lux)": "indoor_illuminance_lux",
            "sound level  1037 (dB(A))": "indoor_sound_dbA",
            "CO2 (ppm)": "indoor_co2_ppm",
            "PMV fanger": "indoor_pmv_fanger",
        }
        for src, dst in renames.items():
            if src in d.columns:
                out[dst] = to_numeric(d[src])
        return out.groupby(["session_id", "minute_utc"]).mean(numeric_only=True).reset_index()

    def _merge_outdoor(self, minute: pd.DataFrame) -> pd.DataFrame:
        outdoor = self.bundle.outdoor.sort_values("minute_utc").copy()
        if outdoor.empty:
            return minute
        keep = [
            "minute_utc",
            "air_temp_C_mean",
            "rel_humidity_percent_mean",
            "wind_speed_mean_m_s_5min",
            "solar_radiation_W_m2_mean",
        ]
        keep = [c for c in keep if c in outdoor.columns]
        outdoor = outdoor[keep].rename(
            columns={
                "air_temp_C_mean": "outdoor_air_temp_C",
                "rel_humidity_percent_mean": "outdoor_relative_humidity_percent",
                "wind_speed_mean_m_s_5min": "outdoor_wind_speed_m_s",
                "solar_radiation_W_m2_mean": "outdoor_solar_radiation_W_m2",
            }
        )
        return pd.merge_asof(
            minute.sort_values("minute_utc"),
            outdoor.sort_values("minute_utc"),
            on="minute_utc",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=10),
        )

    def _empatica(self, emp_dir: Path) -> pd.DataFrame:
        pieces = []
        for filename, fn in {
            "bvp.csv": self._empatica_bvp,
            "eda.csv": self._empatica_eda,
            "temperature.csv": self._empatica_temperature,
            "accelerometer.csv": self._empatica_acc,
            "steps.csv": self._empatica_steps,
            "systolic_peaks.csv": self._empatica_peaks,
        }.items():
            path = emp_dir / filename
            if path.exists():
                pieces.append(fn(path))
        if not pieces:
            return pd.DataFrame({"minute_utc": pd.Series(dtype="datetime64[ns, UTC]")})
        out = pieces[0]
        for piece in pieces[1:]:
            out = out.merge(piece, on="minute_utc", how="outer")
        return out

    def _empatica_bvp(self, path: Path) -> pd.DataFrame:
        df = safe_read_csv(path)
        df["minute_utc"] = minute_floor(df["datetime"])
        df["bvp_nW"] = to_numeric(df["bvp_nW"])
        g = df.groupby("minute_utc")["bvp_nW"]
        return pd.DataFrame({"minute_utc": g.mean().index, "empatica_bvp_mean": g.mean().to_numpy(), "empatica_bvp_sd": g.std().to_numpy()})

    def _empatica_eda(self, path: Path) -> pd.DataFrame:
        df = safe_read_csv(path)
        df["minute_utc"] = minute_floor(df["datetime"])
        df["eda_uS"] = to_numeric(df["eda_uS"])
        g = df.groupby("minute_utc")["eda_uS"]
        return pd.DataFrame({"minute_utc": g.mean().index, "empatica_eda_mean_uS": g.mean().to_numpy(), "empatica_eda_p95_uS": g.quantile(0.95).to_numpy()})

    def _empatica_temperature(self, path: Path) -> pd.DataFrame:
        df = safe_read_csv(path)
        df["minute_utc"] = minute_floor(df["datetime"])
        df["temperature_C"] = to_numeric(df["temperature_C"])
        g = df.groupby("minute_utc")["temperature_C"]
        return pd.DataFrame({"minute_utc": g.mean().index, "empatica_temp_mean_C": g.mean().to_numpy(), "empatica_temp_sd_C": g.std().to_numpy()})

    def _empatica_acc(self, path: Path) -> pd.DataFrame:
        df = safe_read_csv(path)
        df["minute_utc"] = minute_floor(df["datetime"])
        mag = np.sqrt(to_numeric(df["x_g"]) ** 2 + to_numeric(df["y_g"]) ** 2 + to_numeric(df["z_g"]) ** 2)
        enmo = np.maximum(0.0, mag - 1.0)
        d = pd.DataFrame({"minute_utc": df["minute_utc"], "mag": mag, "enmo": enmo})
        g = d.groupby("minute_utc")
        return pd.DataFrame({"minute_utc": g["mag"].mean().index, "empatica_acc_mean_g": g["mag"].mean().to_numpy(), "empatica_enmo_mean_g": g["enmo"].mean().to_numpy()})

    def _empatica_steps(self, path: Path) -> pd.DataFrame:
        df = safe_read_csv(path)
        df["minute_utc"] = minute_floor(df["datetime"])
        df["steps"] = to_numeric(df["steps"])
        g = df.groupby("minute_utc")["steps"].sum(min_count=1)
        return pd.DataFrame({"minute_utc": g.index, "empatica_steps": g.to_numpy()})

    def _empatica_peaks(self, path: Path) -> pd.DataFrame:
        df = safe_read_csv(path)
        ts = pd.to_datetime(df["datetime"], utc=True, errors="coerce").sort_values()
        ibi = ts.diff().dt.total_seconds()
        hr = 60.0 / ibi
        out = pd.DataFrame({"minute_utc": ts.dt.floor("min"), "empatica_hr_mean_bpm": hr})
        out.loc[~out["empatica_hr_mean_bpm"].between(35, 180), "empatica_hr_mean_bpm"] = np.nan
        g = out.groupby("minute_utc")["empatica_hr_mean_bpm"]
        return pd.DataFrame({"minute_utc": g.mean().index, "empatica_hr_mean_bpm": g.mean().to_numpy(), "empatica_hr_sd_bpm": g.std().to_numpy()})

    def _biopac(self, path: Path) -> pd.DataFrame:
        df = safe_read_csv(path)
        df["minute_utc"] = parse_any_datetime(df["Datetime"], self.config.runtime.timeline_timezone).dt.floor("min")
        renames = {
            "BloodFlow [BPU]": "biopac_bloodflow_mean_bpu",
            "Chest [deg. C]": "biopac_temp_chest_mean_C",
            "Thigh [deg. C]": "biopac_temp_thigh_mean_C",
            "UpperArm [deg. C]": "biopac_temp_arm_mean_C",
            "Tibia [deg. C]": "biopac_temp_tibia_mean_C",
            "EDA [microsiemens]": "biopac_eda_mean_uS",
            "HR [BPM]": "biopac_hr_mean_bpm",
            "Backscatter [%]": "biopac_backscatter_mean_percent",
        }
        df = df.rename(columns=renames)
        keep = ["minute_utc"] + [c for c in renames.values() if c in df.columns]
        for col in keep:
            if col != "minute_utc":
                df[col] = to_numeric(df[col])
        return df[keep].groupby("minute_utc").mean(numeric_only=True).reset_index()

    def _canonicalize(self, minute: pd.DataFrame) -> pd.DataFrame:
        minute = minute.rename(
            columns={
                "Datetime": "source_datetime",
            }
        )
        minute["thermal_gradient_C"] = to_numeric(minute.get("master_distal_C")) - to_numeric(minute.get("master_proximal_C"))
        minute["thermal_state_index_C"] = to_numeric(minute.get("empatica_temp_mean_C")) - to_numeric(minute.get("indoor_air_temp_mean_C"))
        minute["hr_delta_bpm"] = to_numeric(minute.get("biopac_hr_mean_bpm")) - to_numeric(minute.get("empatica_hr_mean_bpm"))
        minute["eda_delta_uS"] = to_numeric(minute.get("biopac_eda_mean_uS")) - to_numeric(minute.get("empatica_eda_mean_uS"))
        minute["temp_delta_C"] = to_numeric(minute.get("biopac_temp_chest_mean_C")) - to_numeric(minute.get("empatica_temp_mean_C"))

        questionnaire_cols = [c for c in QUESTIONNAIRE_SET_COLUMNS if c in minute.columns]
        minute["support_questionnaire"] = minute[questionnaire_cols].notna().any(axis=1).astype(int) if questionnaire_cols else 0
        fan_support_cols = [c for c in ["fan_current_A", "fan_control_au"] if c in minute.columns]
        minute["support_fan"] = minute[fan_support_cols].notna().any(axis=1).astype(int) if fan_support_cols else 0
        minute["support_empatica"] = minute.filter(regex=r"^empatica_").notna().any(axis=1).astype(int)
        minute["support_biopac"] = minute.filter(regex=r"^biopac_").notna().any(axis=1).astype(int)
        minute["support_indoor"] = minute.filter(regex=r"^indoor_").notna().any(axis=1).astype(int)
        minute["support_outdoor"] = minute.filter(regex=r"^outdoor_").notna().any(axis=1).astype(int)
        minute["support_core_overlap_hr"] = (minute["empatica_hr_mean_bpm"].notna() & minute["biopac_hr_mean_bpm"].notna()).astype(int) if {"empatica_hr_mean_bpm", "biopac_hr_mean_bpm"} <= set(minute.columns) else 0
        minute["support_core_overlap_eda"] = (minute["empatica_eda_mean_uS"].notna() & minute["biopac_eda_mean_uS"].notna()).astype(int) if {"empatica_eda_mean_uS", "biopac_eda_mean_uS"} <= set(minute.columns) else 0
        minute["support_core_overlap_temp"] = (minute["empatica_temp_mean_C"].notna() & minute["biopac_temp_chest_mean_C"].notna()).astype(int) if {"empatica_temp_mean_C", "biopac_temp_chest_mean_C"} <= set(minute.columns) else 0
        for metric, flag in [
            ("support_core_overlap_hr", "eligible_session_hr_agreement"),
            ("support_core_overlap_eda", "eligible_session_eda_agreement"),
            ("support_core_overlap_temp", "eligible_session_temp_agreement"),
        ]:
            minute[flag] = int(minute[metric].sum() >= self.config.runtime.min_sensor_overlap_minutes)

        minute["quality_empatica_hr"] = (
            minute["empatica_hr_mean_bpm"].notna()
            & to_numeric(minute["empatica_hr_mean_bpm"]).between(40, 180)
            & (to_numeric(minute["empatica_hr_sd_bpm"]).fillna(0) <= 20)
        ).astype(int) if {"empatica_hr_mean_bpm", "empatica_hr_sd_bpm"} <= set(minute.columns) else 0
        minute["quality_empatica_eda"] = (
            minute["empatica_eda_mean_uS"].notna()
            & to_numeric(minute["empatica_eda_mean_uS"]).between(0, 40)
            & (to_numeric(minute["empatica_eda_p95_uS"]).fillna(to_numeric(minute["empatica_eda_mean_uS"])) >= to_numeric(minute["empatica_eda_mean_uS"]).fillna(-np.inf))
        ).astype(int) if {"empatica_eda_mean_uS", "empatica_eda_p95_uS"} <= set(minute.columns) else 0
        minute["quality_empatica_temp"] = (
            minute["empatica_temp_mean_C"].notna()
            & to_numeric(minute["empatica_temp_mean_C"]).between(20, 40)
            & (to_numeric(minute["empatica_temp_sd_C"]).fillna(0) <= 2.5)
        ).astype(int) if {"empatica_temp_mean_C", "empatica_temp_sd_C"} <= set(minute.columns) else 0
        minute["quality_biopac_hr"] = (
            minute["biopac_hr_mean_bpm"].notna()
            & to_numeric(minute["biopac_hr_mean_bpm"]).between(40, 180)
        ).astype(int) if "biopac_hr_mean_bpm" in minute.columns else 0
        minute["quality_biopac_eda"] = (
            minute["biopac_eda_mean_uS"].notna()
            & to_numeric(minute["biopac_eda_mean_uS"]).between(0, 60)
        ).astype(int) if "biopac_eda_mean_uS" in minute.columns else 0
        minute["quality_biopac_temp"] = (
            minute["biopac_temp_chest_mean_C"].notna()
            & to_numeric(minute["biopac_temp_chest_mean_C"]).between(20, 42)
        ).astype(int) if "biopac_temp_chest_mean_C" in minute.columns else 0
        minute["quality_empatica_bvp"] = (
            minute["empatica_bvp_mean"].notna()
            & to_numeric(minute["empatica_bvp_sd"]).fillna(0) > 0
        ).astype(int) if {"empatica_bvp_mean", "empatica_bvp_sd"} <= set(minute.columns) else 0

        ordered = []
        minute = minute.rename(columns={"source_datetime": "Datetime"})
        for col in CANONICAL_MINUTE_COLUMNS:
            if col in minute.columns:
                ordered.append(col)
            else:
                minute[col] = np.nan
                ordered.append(col)
        return minute[ordered].copy()

    def _phase_summary(self, minute: pd.DataFrame) -> pd.DataFrame:
        group_cols = ["session_id", "participant_id", "study_day", "condition_code", "illuminance_level", "time_of_day", "protocol_block", "protocol_phase"]
        rows = []
        for keys, d in minute.groupby(group_cols):
            row = {col: keys[idx] for idx, col in enumerate(group_cols)}
            row["n_minutes"] = int(len(d))
            row["phase_meets_min_minutes"] = int(len(d) >= self.config.runtime.min_phase_minutes)
            for metric in PHASE_METRICS:
                if metric not in d.columns:
                    continue
                vals = to_numeric(d[metric]).dropna()
                row[metric] = float(vals.mean()) if not vals.empty else np.nan
                row[f"{metric}__n_valid"] = int(vals.notna().sum())
                row[f"{metric}__coverage"] = float(vals.notna().mean()) if len(d) else np.nan
            row["support_questionnaire_fraction"] = float(d["support_questionnaire"].mean())
            row["support_empatica_fraction"] = float(d["support_empatica"].mean())
            row["support_biopac_fraction"] = float(d["support_biopac"].mean())
            row["support_indoor_fraction"] = float(d["support_indoor"].mean())
            for col in [
                "quality_empatica_hr",
                "quality_empatica_eda",
                "quality_empatica_temp",
                "quality_biopac_hr",
                "quality_biopac_eda",
                "quality_biopac_temp",
                "quality_empatica_bvp",
            ]:
                if col in d.columns:
                    row[f"{col}__fraction"] = float(to_numeric(d[col]).mean())
            rows.append(row)
        return pd.DataFrame(rows)

    def _processing_metadata(self, minute: pd.DataFrame, phase_summary: pd.DataFrame) -> dict:
        comparison_minute = minute.loc[minute["protocol_block"].astype(str).isin(COMPARISON_BLOCKS)].copy()
        questionnaire_stats = self._questionnaire_completeness(comparison_minute)
        overlap = {
            "heart_rate": int(comparison_minute["support_core_overlap_hr"].sum()),
            "eda": int(comparison_minute["support_core_overlap_eda"].sum()),
            "temperature": int(comparison_minute["support_core_overlap_temp"].sum()),
        }
        support = {
            "questionnaire_completeness": float(questionnaire_stats["overall_completeness"]),
            "questionnaire_event_completeness": float(questionnaire_stats["event_completeness"]),
            "questionnaire_response_completeness": float(questionnaire_stats["response_completeness"]),
            "fan_fraction": float(comparison_minute["support_fan"].mean()) if not comparison_minute.empty else 0.0,
            "empatica_fraction": float(comparison_minute["support_empatica"].mean()) if not comparison_minute.empty else 0.0,
            "biopac_fraction": float(comparison_minute["support_biopac"].mean()) if not comparison_minute.empty else 0.0,
            "indoor_fraction": float(comparison_minute["support_indoor"].mean()) if not comparison_minute.empty else 0.0,
            "outdoor_fraction": float(comparison_minute["support_outdoor"].mean()) if not comparison_minute.empty else 0.0,
        }
        quality = {}
        for col in [
            "quality_empatica_hr",
            "quality_empatica_eda",
            "quality_empatica_temp",
            "quality_biopac_hr",
            "quality_biopac_eda",
            "quality_biopac_temp",
            "quality_empatica_bvp",
        ]:
            if col in comparison_minute.columns:
                quality[col] = float(to_numeric(comparison_minute[col]).mean()) if not comparison_minute.empty else 0.0
        return {
            "session_id": str(minute["session_id"].iloc[0]),
            "participant_id": str(minute["participant_id"].iloc[0]),
            "condition_code": str(minute["condition_code"].iloc[0]),
            "n_minutes_timeline": int(len(minute)),
            "n_minutes_comparison_window": int(len(comparison_minute)),
            "n_phase_rows": int(len(phase_summary)),
            "support": support,
            "quality": quality,
            "questionnaire_design": questionnaire_stats,
            "sensor_overlap_minutes": overlap,
            "eligible_agreement_metrics": {
                "heart_rate": bool(overlap["heart_rate"] >= self.config.runtime.min_sensor_overlap_minutes),
                "eda": bool(overlap["eda"] >= self.config.runtime.min_sensor_overlap_minutes),
                "temperature": bool(overlap["temperature"] >= self.config.runtime.min_sensor_overlap_minutes),
            },
        }

    def _questionnaire_completeness(self, minute: pd.DataFrame) -> dict:
        base = self.bundle.questionnaire.copy()
        if base.empty:
            return {
                "expected_event_count": 0,
                "observed_event_count": 0,
                "expected_questions_per_event": 0,
                "answered_question_cells": 0,
                "expected_question_cells": 0,
                "event_completeness": 0.0,
                "response_completeness": 0.0,
                "overall_completeness": 0.0,
            }
        base = base.rename(columns={
            "q n": "questionnaire_n",
            "thermal sensation": "thermal_sensation",
            "thermal comfort": "thermal_comfort",
            "thermal pleasure": "thermal_pleasure",
            "thermal preference": "thermal_preference",
            "visual sensation": "visual_sensation",
            "color sensation": "color_sensation",
            "visual comfort": "visual_comfort",
            "sound comfort (dB(A))": "sound_comfort_dbA",
            "airQuality comfort": "air_quality_comfort",
            "room comfort": "room_comfort",
            "source of discomfort": "discomfort_source",
            "protocol block": "protocol_block",
        })
        base["protocol_block"] = base["protocol_block"].astype(str)
        question_cols = [c for c in QUESTIONNAIRE_SET_COLUMNS if c in minute.columns]
        expected_events = sorted(to_numeric(base.loc[base["protocol_block"].isin(COMPARISON_BLOCKS), "questionnaire_n"]).dropna().astype(int).unique().tolist())
        expected_event_count = len(expected_events)
        expected_question_map: dict[int, list[str]] = {}
        for qn, d in base.loc[base["protocol_block"].isin(COMPARISON_BLOCKS)].groupby("questionnaire_n"):
            qn_int = int(qn)
            expected_question_map[qn_int] = [col for col in question_cols if col in d.columns and float(d[col].notna().mean()) >= 0.95]
        expected_question_cells = int(sum(len(expected_question_map.get(qn, [])) for qn in expected_events))
        expected_questions_per_event = int(round(expected_question_cells / expected_event_count)) if expected_event_count else 0
        comparison = minute.loc[minute["protocol_block"].astype(str).isin(COMPARISON_BLOCKS)].copy()
        if comparison.empty or "questionnaire_n" not in comparison.columns or expected_event_count == 0:
            return {
                "expected_event_count": expected_event_count,
                "observed_event_count": 0,
                "expected_questions_per_event": expected_questions_per_event,
                "answered_question_cells": 0,
                "expected_question_cells": expected_question_cells,
                "event_completeness": 0.0,
                "response_completeness": 0.0,
                "overall_completeness": 0.0,
            }
        q_rows = comparison.loc[to_numeric(comparison["questionnaire_n"]).notna()].copy()
        if q_rows.empty:
            observed_event_count = 0
            answered_question_cells = 0
        else:
            q_rows["questionnaire_n"] = to_numeric(q_rows["questionnaire_n"]).astype(int)
            q_rows = q_rows.loc[q_rows["questionnaire_n"].isin(expected_events)].copy()
            observed_event_count = int(q_rows["questionnaire_n"].nunique())
            answered_question_cells = 0
            if question_cols:
                event_table = q_rows.groupby("questionnaire_n", as_index=False)[question_cols].first()
                for row in event_table.itertuples(index=False):
                    qn = int(row.questionnaire_n)
                    expected_cols = expected_question_map.get(qn, [])
                    answered_question_cells += sum(pd.notna(getattr(row, col)) for col in expected_cols)
        event_completeness = float(observed_event_count / expected_event_count) if expected_event_count else 0.0
        response_denominator = int(sum(len(expected_question_map.get(int(qn), [])) for qn in q_rows["questionnaire_n"].drop_duplicates().tolist())) if observed_event_count else 0
        response_completeness = float(answered_question_cells / response_denominator) if response_denominator else 0.0
        overall_completeness = float(answered_question_cells / expected_question_cells) if expected_question_cells else 0.0
        return {
            "expected_event_count": expected_event_count,
            "observed_event_count": observed_event_count,
            "expected_questions_per_event": expected_questions_per_event,
            "answered_question_cells": answered_question_cells,
            "expected_question_cells": expected_question_cells,
            "event_completeness": event_completeness,
            "response_completeness": response_completeness,
            "overall_completeness": overall_completeness,
        }
