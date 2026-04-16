from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


DEFAULT_SESSION_IDS = [
    "P01_D01_DIM-MOR",
    "P01_D02_BRI-MID",
    "P01_D03_BRI-MOR",
    "P01_D04_DIM-MID",
    "P02_D01_BRI-MID",
    "P02_D02_DIM-MOR",
    "P02_D03_DIM-MID",
    "P02_D04_BRI-MOR",
]

SESSION_ID_COLUMNS = ("Session ID", "session_id")
PARTICIPANT_ID_COLUMNS = ("Participant ID", "participant_id")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _first_present(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _filter_by_session_ids(df: pd.DataFrame, session_ids: set[str]) -> pd.DataFrame:
    session_col = _first_present(df, SESSION_ID_COLUMNS)
    if session_col is None:
        return df.copy()
    return df.loc[df[session_col].astype(str).isin(session_ids)].copy()


def _filter_outdoor(df: pd.DataFrame, subset_timeline: pd.DataFrame) -> pd.DataFrame:
    if df.empty or subset_timeline.empty or "Datetime" not in subset_timeline.columns or "datetime" not in df.columns:
        return df.copy()
    session_times = pd.to_datetime(subset_timeline["Datetime"], errors="coerce").dropna()
    if session_times.empty:
        return df.copy()
    start = session_times.min().floor("min")
    end = session_times.max().ceil("min")
    outdoor_times = pd.to_datetime(df["datetime"], errors="coerce")
    return df.loc[(outdoor_times >= start) & (outdoor_times <= end)].copy()


def _copy_session_dirs(src_root: Path, dst_root: Path, session_ids: list[str]) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    for session_id in session_ids:
        src = src_root / session_id
        if not src.exists():
            continue
        shutil.copytree(src, dst_root / session_id, dirs_exist_ok=True)


def build_subset(dataset_root: Path, outdir: Path, session_ids: list[str]) -> None:
    session_ids = [str(x) for x in session_ids]
    session_set = set(session_ids)
    if outdir.exists():
        shutil.rmtree(outdir)
    (outdir / "master_files").mkdir(parents=True, exist_ok=True)
    (outdir / "env").mkdir(parents=True, exist_ok=True)
    (outdir / "empatica").mkdir(parents=True, exist_ok=True)
    (outdir / "biopac").mkdir(parents=True, exist_ok=True)

    timeline = _read_csv(dataset_root / "master_files" / "timeline_by_minutes.csv")
    subset_timeline = _filter_by_session_ids(timeline, session_set)
    subset_timeline.to_csv(outdir / "master_files" / "timeline_by_minutes.csv", index=False)

    master_files = [
        "sessions.csv",
        "questionnaire_events.csv",
        "skin_temperature_timeseries.csv",
        "fan_behavior_timeseries.csv",
    ]
    for filename in master_files:
        df = _read_csv(dataset_root / "master_files" / filename)
        _filter_by_session_ids(df, session_set).to_csv(outdir / "master_files" / filename, index=False)

    indoor = _read_csv(dataset_root / "env" / "indoor_climate.csv")
    _filter_by_session_ids(indoor, session_set).to_csv(outdir / "env" / "indoor_climate.csv", index=False)

    outdoor = _read_csv(dataset_root / "env" / "outdoor_meteorology.csv")
    _filter_outdoor(outdoor, subset_timeline).to_csv(outdir / "env" / "outdoor_meteorology.csv", index=False)

    _copy_session_dirs(dataset_root / "empatica", outdir / "empatica", session_ids)
    _copy_session_dirs(dataset_root / "biopac", outdir / "biopac", session_ids)
    shutil.copytree(dataset_root / "metadata", outdir / "metadata", dirs_exist_ok=True)

    manifest = pd.DataFrame({"session_id": session_ids})
    participant_df = _read_csv(dataset_root / "master_files" / "sessions.csv")
    session_col = _first_present(participant_df, SESSION_ID_COLUMNS)
    participant_col = _first_present(participant_df, PARTICIPANT_ID_COLUMNS)
    if session_col and participant_col:
        keep = participant_df.loc[participant_df[session_col].astype(str).isin(session_set), [session_col, participant_col]].copy()
        keep.columns = ["session_id", "participant_id"]
        manifest = manifest.merge(keep, on="session_id", how="left")
    manifest.to_csv(outdir / "subset_manifest.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a small root-level CLTR test subset dataset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("CLTR_dataset"))
    parser.add_argument("--outdir", type=Path, default=Path("CLTR_dataset_subset"))
    parser.add_argument("--session-ids", nargs="*", default=DEFAULT_SESSION_IDS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_subset(args.dataset_root, args.outdir, args.session_ids)


if __name__ == "__main__":
    main()
