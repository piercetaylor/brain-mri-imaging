#!/usr/bin/env python
"""Fetch the cohort listed in ``data/manifest.csv`` and verify what arrived.

Every file comes from the public Imaging Data Commons object store, which needs
no credentials and no download client. The license field returned by the index
is checked against the license recorded in the manifest before anything is
written, and each series is digested after it is written.

    python data/download_data.py              # download, then verify
    python data/download_data.py --verify     # verify what is already on disk

The digests are written to ``data/checksums.txt`` on the first run and compared
against it on every run after that, so a series that changes upstream is
reported and not silently absorbed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.s99_utils import (
    Metrics, banner, idc_sql, parallel, s3_download, s3_list, sha256_series
)


def read_manifest() -> list[dict]:
    if not config.MANIFEST.exists():
        raise SystemExit(
            "data/manifest.csv is absent. Run: python -m src.s01_manifest"
        )
    with open(config.MANIFEST, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def series_directory(row: dict) -> Path:
    return config.RAW / row["patient_id"] / row["role"] / row["series_uid"]


def check_license(rows: list[dict]) -> str:
    """Ask the index what license it now reports for the manifested series."""
    uids = ", ".join("'" + r["series_uid"] + "'" for r in rows)
    reported = idc_sql(
        "SELECT DISTINCT license_short_name AS license FROM index "
        f"WHERE SeriesInstanceUID IN ({uids})"
    )
    names = sorted(r["license"] for r in reported)
    if names != [config.EXPECTED_LICENSE]:
        raise SystemExit(
            f"the index reports license {names}, expected "
            f"['{config.EXPECTED_LICENSE}']. Nothing was downloaded."
        )
    print(f"license reported for all {len(rows)} series: {names[0]}")
    return names[0]


def download(rows: list[dict]) -> int:
    total = 0
    for index, row in enumerate(rows, start=1):
        destination = series_directory(row)
        expected = int(row["instance_count"])
        present = len(list(destination.glob("*.dcm"))) if destination.exists() else 0
        if present == expected:
            total += present
            continue
        keys = s3_list(row["series_uuid"])
        parallel(
            lambda kv: s3_download(kv[0], destination / Path(kv[0]).name), keys
        )
        total += len(keys)
        print(
            f"  [{index:>3}/{len(rows)}] {row['patient_id']} {row['role']:<12} "
            f"{len(keys):>4} files  {row['series_desc'][:44]}",
            flush=True,
        )
    return total


def verify(rows: list[dict]) -> list[dict]:
    def digest(row: dict) -> dict:
        outer, count, size = sha256_series(series_directory(row))
        return {
            "series_uid": row["series_uid"],
            "patient_id": row["patient_id"],
            "role": row["role"],
            "sha256": outer,
            "files": count,
            "bytes": size,
        }

    return sorted(parallel(digest, rows, workers=8), key=lambda r: r["series_uid"])


def compare_to_record(digests: list[dict]) -> None:
    if not config.CHECKSUMS.exists():
        with open(config.CHECKSUMS, "w", encoding="utf-8") as handle:
            handle.write(
                "# SHA-256 of each downloaded series. The series digest is the\n"
                "# SHA-256 of its per-file digests, sorted and joined by newlines,\n"
                "# so it does not depend on the order files arrive in.\n"
                "# sha256  files  bytes  series_uid\n"
            )
            for row in digests:
                handle.write(
                    f"{row['sha256']}  {row['files']}  {row['bytes']}  "
                    f"{row['series_uid']}\n"
                )
        print(f"wrote {len(digests)} digests to {config.CHECKSUMS}")
        return

    recorded = {}
    for line in config.CHECKSUMS.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        sha, files, size, uid = line.split()
        recorded[uid] = (sha, int(files), int(size))

    problems = []
    for row in digests:
        want = recorded.get(row["series_uid"])
        if want is None:
            problems.append(f"{row['series_uid']} is not in data/checksums.txt")
        elif want != (row["sha256"], row["files"], row["bytes"]):
            problems.append(
                f"{row['series_uid']} digest {row['sha256'][:12]} "
                f"does not match the recorded {want[0][:12]}"
            )
    missing = set(recorded) - {r["series_uid"] for r in digests}
    problems += [f"{uid} is recorded but absent on disk" for uid in sorted(missing)]
    if problems:
        for problem in problems:
            print(f"  FAIL {problem}")
        raise SystemExit(f"{len(problems)} series failed verification")
    print(f"all {len(digests)} series match data/checksums.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="digest what is on disk without downloading")
    args = parser.parse_args()

    banner("acquisition")
    rows = read_manifest()
    license = config.EXPECTED_LICENSE
    if not args.verify:
        license = check_license(rows)
        files = download(rows)
        print(f"{files} files present across {len(rows)} series")

    digests = verify(rows)
    compare_to_record(digests)

    metrics = Metrics()
    metrics.update({
        "download_series": len(digests),
        "download_files": sum(r["files"] for r in digests),
        "download_bytes": sum(r["bytes"] for r in digests),
        "download_gb": sum(r["bytes"] for r in digests) / 1024 ** 3,
        "download_license_verified": license,
    })
    metrics.save()


if __name__ == "__main__":
    main()
