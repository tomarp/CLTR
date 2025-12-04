#!/usr/bin/env python3
"""
Download all Empatica raw .avro files under a participant_data base prefix,
mirroring the S3 directory structure locally and writing a manifest CSV.

Example:

  python empatica_raw_avro_all.py \
      --access-key YOUR_ACCESS_KEY \
      --secret-key YOUR_SECRET_KEY \
      --endpoint-url https://s3.amazonaws.com \
      --bucket empatica-us-east-1-prod-data \
      --base-prefix v2/829/1/1/participant_data/ \
      --out-dir ../datasets/raw/empatica_avro \
      --manifest ../empatica_avro/manifest_raw_avro.csv
"""

import argparse
import csv
from pathlib import Path

import boto3
from botocore.config import Config


def parse_args():
    p = argparse.ArgumentParser(
        description="Mirror Empatica raw .avro files from S3 under a participant_data/ base prefix."
    )
    p.add_argument("--access-key", required=True, help="Empatica S3 access key ID")
    p.add_argument("--secret-key", required=True, help="Empatica S3 secret access key")
    p.add_argument("--endpoint-url", required=True, help="Empatica S3 endpoint URL, e.g. https://s3.amazonaws.com")
    p.add_argument("--bucket", required=True, help="Empatica S3 bucket name")
    p.add_argument(
        "--base-prefix",
        required=True,
        help=(
            "Base S3 prefix under which to search for raw_data .avro files, "
            "e.g. v2/829/1/1/participant_data/"
        ),
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Local directory where the Avro files will be mirrored",
    )
    p.add_argument(
        "--manifest",
        required=False,
        default=None,
        help="Optional path to a manifest CSV summarizing downloaded files",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="If set, do not re-download files that already exist locally.",
    )
    return p.parse_args()


def make_s3_client(access_key, secret_key, endpoint_url):
    session = boto3.session.Session()
    return session.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
        region_name="us-east-1",
        config=Config(signature_version="s3v4"),
    )


def main():
    args = parse_args()
    s3 = make_s3_client(args.access_key, args.secret_key, args.endpoint_url)

    base_prefix = args.base_prefix
    if not base_prefix.endswith("/"):
        base_prefix += "/"

    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] Scanning for .avro under s3://{args.bucket}/{base_prefix}")

    paginator = s3.get_paginator("list_objects_v2")
    total_objects = 0
    total_avro = 0
    downloaded = 0
    skipped = 0

    manifest_rows = []

    for page in paginator.paginate(Bucket=args.bucket, Prefix=base_prefix):
        for obj in page.get("Contents", []):
            total_objects += 1
            key = obj["Key"]
            if not key.lower().endswith(".avro"):
                continue
            total_avro += 1

            # Derive relative path under base_prefix to preserve date/participant schema
            rel_key = key[len(base_prefix):]  # may contain date/participant/raw_data/vX/filename
            local_path = out_root / rel_key
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and local_path.exists():
                print(f"[skip] {key} -> {local_path} (already exists)")
                skipped += 1
            else:
                print(f"[download] {key} -> {local_path}")
                s3.download_file(args.bucket, key, str(local_path))
                downloaded += 1

            manifest_rows.append(
                {
                    "s3_key": key,
                    "relative_path": rel_key,
                    "local_path": str(local_path),
                    "size_bytes": obj.get("Size", ""),
                    "etag": obj.get("ETag", "").strip('"'),
                }
            )

    print(f"\n[summary]")
    print(f"  Total S3 objects scanned: {total_objects}")
    print(f"  Total .avro objects found: {total_avro}")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (existing): {skipped}")

    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["s3_key", "relative_path", "local_path", "size_bytes", "etag"],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"[manifest] Wrote manifest with {len(manifest_rows)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
