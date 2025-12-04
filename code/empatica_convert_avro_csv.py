#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from typing import List

from avro.datafile import DataFileReader
from avro.io import DatumReader


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert local Empatica Avro files into CSVs (all participants/dates)."
    )
    p.add_argument(
        "--indir",
        required=True,
        help="Root directory containing .avro files (mirrored from S3).",
    )
    p.add_argument(
        "--csv-root",
        required=False,
        default=None,
        help=(
            "Optional root directory for CSV output. "
            "If omitted, CSVs are created next to each Avro file."
        ),
    )
    return p.parse_args()


# ---------- helpers ----------

def reconstruct_timestamps(start_us: int, sampling_hz: float, n: int) -> List[int]:
    """
    Reconstruct timestamps in microseconds:
        t_i = start_us + i * (1e6 / sampling_hz)

    Assumes sampling_hz > 0 and n >= 1 (caller must check).
    """
    step = 1_000_000.0 / float(sampling_hz)
    return [int(round(start_us + i * step)) for i in range(n)]


def convert_imu_counts_to_physical(values, phys_min, phys_max, dig_min, dig_max):
    """Map ADC counts to physical units (g for acc, deg/s for gyro)."""
    delta_physical = phys_max - phys_min
    delta_digital = dig_max - dig_min
    if delta_digital == 0:
        return [0.0 for _ in values]
    scale = delta_physical / float(delta_digital)
    return [val * scale for val in values]


def write_csv(path: Path, header: List[str], rows: List[List]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[csv] {path}")


def convert_single_avro(avro_path: Path, indir_root: Path, csv_root: Path | None):
    """
    Convert a single Empatica Avro file into CSVs for each sensor.

    Path rules:

      rel = avro_path.relative_to(indir_root)

      If csv_root is not None:
          sample_out_dir = csv_root / rel.parent / avro_path.stem
      else:
          sample_out_dir = avro_path.parent / avro_path.stem

    Sensors with invalid Fs (<=0) or empty arrays are skipped with a warning.
    """
    record_name = avro_path.stem

    if csv_root is not None:
        rel = avro_path.relative_to(indir_root)
        base_out_dir = csv_root / rel.parent
    else:
        base_out_dir = avro_path.parent

    sample_out_dir = base_out_dir / record_name
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    with DataFileReader(avro_path.open("rb"), DatumReader()) as reader:
        try:
            data = next(reader)
        except StopIteration:
            print(f"[warn] Avro file {avro_path} is empty.")
            return

    raw = data.get("rawData", {})

    # -------- Accelerometer --------
    acc = raw.get("accelerometer")
    if acc:
        sf = float(acc.get("samplingFrequency", 0) or 0)
        x_vals = acc.get("x") or []
        if sf <= 0 or len(x_vals) == 0:
            print(
                f"[warn] accelerometer in {avro_path} has invalid Fs={sf} or n={len(x_vals)}; skipping."
            )
        else:
            ts = reconstruct_timestamps(
                int(acc["timestampStart"]),
                sf,
                len(x_vals),
            )
            imu = acc["imuParams"]
            x = convert_imu_counts_to_physical(
                x_vals, imu["physicalMax"], imu["physicalMin"], imu["digitalMax"], imu["digitalMin"]
            )
            y = convert_imu_counts_to_physical(
                acc["y"], imu["physicalMax"], imu["physicalMin"], imu["digitalMax"], imu["digitalMin"]
            )
            z = convert_imu_counts_to_physical(
                acc["z"], imu["physicalMax"], imu["physicalMin"], imu["digitalMax"], imu["digitalMin"]
            )
            rows = [[t, xv, yv, zv] for t, xv, yv, zv in zip(ts, x, y, z)]
            write_csv(sample_out_dir / "accelerometer.csv",
                      ["unix_timestamp_us", "x_g", "y_g", "z_g"], rows)

    # -------- Gyroscope --------
    gyro = raw.get("gyroscope")
    if gyro:
        sf = float(gyro.get("samplingFrequency", 0) or 0)
        x_vals = gyro.get("x") or []
        if sf <= 0 or len(x_vals) == 0:
            print(
                f"[warn] gyroscope in {avro_path} has invalid Fs={sf} or n={len(x_vals)}; skipping."
            )
        else:
            ts = reconstruct_timestamps(
                int(gyro["timestampStart"]),
                sf,
                len(x_vals),
            )
            imu = gyro["imuParams"]
            x = convert_imu_counts_to_physical(
                x_vals, imu["physicalMax"], imu["physicalMin"], imu["digitalMax"], imu["digitalMin"]
            )
            y = convert_imu_counts_to_physical(
                gyro["y"], imu["physicalMax"], imu["physicalMin"], imu["digitalMax"], imu["digitalMin"]
            )
            z = convert_imu_counts_to_physical(
                gyro["z"], imu["physicalMax"], imu["physicalMin"], imu["digitalMax"], imu["digitalMin"]
            )
            rows = [[t, xv, yv, zv] for t, xv, yv, zv in zip(ts, x, y, z)]
            write_csv(sample_out_dir / "gyroscope.csv",
                      ["unix_timestamp_us", "x_dps", "y_dps", "z_dps"], rows)

    # -------- EDA --------
    eda = raw.get("eda")
    if eda:
        sf = float(eda.get("samplingFrequency", 0) or 0)
        values = eda.get("values") or []
        if sf <= 0 or len(values) == 0:
            print(
                f"[warn] EDA in {avro_path} has invalid Fs={sf} or n={len(values)}; skipping."
            )
        else:
            ts = reconstruct_timestamps(
                int(eda["timestampStart"]),
                sf,
                len(values),
            )
            rows = [[t, v] for t, v in zip(ts, values)]
            write_csv(sample_out_dir / "eda.csv",
                      ["unix_timestamp_us", "eda_uS"], rows)

    # -------- Temperature --------
    tmp = raw.get("temperature")
    if tmp:
        sf = float(tmp.get("samplingFrequency", 0) or 0)
        values = tmp.get("values") or []
        if sf <= 0 or len(values) == 0:
            print(
                f"[warn] temperature in {avro_path} has invalid Fs={sf} or n={len(values)}; skipping."
            )
        else:
            ts = reconstruct_timestamps(
                int(tmp["timestampStart"]),
                sf,
                len(values),
            )
            rows = [[t, v] for t, v in zip(ts, values)]
            write_csv(sample_out_dir / "temperature.csv",
                      ["unix_timestamp_us", "temperature_C"], rows)

    # -------- BVP --------
    bvp = raw.get("bvp")
    if bvp:
        sf = float(bvp.get("samplingFrequency", 0) or 0)
        values = bvp.get("values") or []
        if sf <= 0 or len(values) == 0:
            print(
                f"[warn] BVP in {avro_path} has invalid Fs={sf} or n={len(values)}; skipping."
            )
        else:
            ts = reconstruct_timestamps(
                int(bvp["timestampStart"]),
                sf,
                len(values),
            )
            rows = [[t, v] for t, v in zip(ts, values)]
            write_csv(sample_out_dir / "bvp.csv",
                      ["unix_timestamp_us", "bvp_nW"], rows)

    # -------- Steps --------
    steps = raw.get("steps")
    if steps:
        sf = float(steps.get("samplingFrequency", 0) or 0)
        values = steps.get("values") or []
        if sf <= 0 or len(values) == 0:
            print(
                f"[warn] steps in {avro_path} has invalid Fs={sf} or n={len(values)}; skipping."
            )
        else:
            ts = reconstruct_timestamps(
                int(steps["timestampStart"]),
                sf,
                len(values),
            )
            rows = [[t, v] for t, v in zip(ts, values)]
            write_csv(sample_out_dir / "steps.csv",
                      ["unix_timestamp_us", "steps"], rows)

    # -------- Tags --------
    tags = raw.get("tags")
    if tags:
        times = tags.get("tagsTimeMicros", []) or []
        if len(times) == 0:
            print(f"[warn] tags in {avro_path} are empty; skipping.")
        else:
            rows = [[t] for t in times]
            write_csv(sample_out_dir / "tags.csv",
                      ["tags_timestamp_us"], rows)

    # -------- Systolic peaks --------
    sps = raw.get("systolicPeaks")
    if sps:
        times = sps.get("peaksTimeNanos", []) or []
        if len(times) == 0:
            print(f"[warn] systolicPeaks in {avro_path} are empty; skipping.")
        else:
            rows = [[t] for t in times]
            write_csv(sample_out_dir / "systolic_peaks.csv",
                      ["peak_timestamp_ns"], rows)


def main():
    args = parse_args()
    indir = Path(args.indir).expanduser().resolve()
    if not indir.exists():
        raise SystemExit(f"[error] indir does not exist: {indir}")

    if args.csv_root:
        csv_root = Path(args.csv_root).expanduser().resolve()
        csv_root.mkdir(parents=True, exist_ok=True)
    else:
        csv_root = None

    avro_files = list(indir.rglob("*.avro"))
    if not avro_files:
        print(f"[error] No .avro files found under {indir}")
        return

    print(f"[info] Found {len(avro_files)} .avro files under {indir}. Converting...")

    for av in avro_files:
        print(f"[info] Converting {av}")
        convert_single_avro(av, indir, csv_root)

    print("[done] All Avro files processed.")


if __name__ == "__main__":
    main() 
    