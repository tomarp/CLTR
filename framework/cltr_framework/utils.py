from __future__ import annotations

from pathlib import Path
import html
import json
import os
import re
import shutil
import subprocess
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_CHILDREN = ("master_files", "env", "empatica", "biopac", "metadata")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def safe_read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def session_id_parts(session_id: str) -> dict:
    parts = str(session_id).split("_")
    day = parts[1] if len(parts) > 1 else None
    condition = parts[2] if len(parts) > 2 else None
    return {
        "session_id": str(session_id),
        "participant_id": parts[0] if parts else None,
        "study_day": day,
        "condition_code": condition,
        "illuminance_level": condition.split("-")[0] if condition and "-" in condition else None,
        "time_of_day": condition.split("-")[1] if condition and "-" in condition else None,
    }


def parse_local_datetime(series: pd.Series, timezone: str) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        return dt.dt.tz_localize(timezone)
    return dt.dt.tz_convert(timezone)


def parse_any_datetime(series: pd.Series, timezone: str) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if dt.isna().all():
        dt = pd.to_datetime(series, errors="coerce")
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(timezone).dt.tz_convert("UTC")
        else:
            dt = dt.dt.tz_convert("UTC")
    return dt


def minute_floor(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.floor("min")


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def numeric_cols(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    banned = set(exclude)
    return [c for c in df.columns if c not in banned and pd.api.types.is_numeric_dtype(df[c])]


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def html_escape(value: object) -> str:
    return html.escape("" if value is None else str(value))


def set_mplconfigdir(root: str | Path, dirname: str) -> Path:
    target = ensure_dir(Path(root) / dirname)
    os.environ.setdefault("MPLCONFIGDIR", str(target))
    return target


def which(name: str) -> str | None:
    return shutil.which(name)


def run_command(args: list[str], cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, check=False, capture_output=True, text=True)


def relative_delta(a: pd.Series, b: pd.Series) -> pd.Series:
    return to_numeric(a) - to_numeric(b)


def paired_ttest(x: pd.Series, y: pd.Series) -> dict:
    from scipy import stats

    xy = pd.DataFrame({"x": to_numeric(x), "y": to_numeric(y)}).dropna()
    if len(xy) < 2:
        mean_diff = float((xy["x"] - xy["y"]).mean()) if len(xy) else np.nan
        return {
            "n_pairs": int(len(xy)),
            "statistic": np.nan,
            "p_value": np.nan,
            "mean_difference": mean_diff,
            "cohens_dz": np.nan,
        }
    stat, pval = stats.ttest_rel(xy["x"], xy["y"], nan_policy="omit")
    diff = xy["x"] - xy["y"]
    effect = float(diff.mean() / diff.std(ddof=1)) if len(diff) > 1 and diff.std(ddof=1) not in (0, np.nan) else np.nan
    return {
        "n_pairs": int(len(xy)),
        "statistic": float(stat),
        "p_value": float(pval),
        "mean_difference": float(diff.mean()),
        "cohens_dz": effect,
    }


def benjamini_hochberg(p_values: pd.Series | list[float] | np.ndarray) -> pd.Series:
    vals = pd.to_numeric(pd.Series(p_values), errors="coerce")
    out = pd.Series(np.nan, index=vals.index, dtype=float)
    mask = vals.notna()
    if not mask.any():
        return out
    valid = vals.loc[mask].astype(float)
    order = np.argsort(valid.to_numpy())
    ranked = valid.to_numpy()[order]
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        cur = min(prev, ranked[i] * n / rank)
        adjusted[i] = cur
        prev = cur
    restored = np.empty(n, dtype=float)
    restored[order] = adjusted
    out.loc[mask] = restored
    return out


def bootstrap_mean_ci(values: pd.Series | list[float] | np.ndarray, n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return (np.nan, np.nan)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = float(sample.mean())
    alpha = (1.0 - ci) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))
