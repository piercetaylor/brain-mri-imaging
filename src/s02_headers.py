"""Stage 2: read every DICOM header in the cohort into one typed table.

Tags are addressed by group and element number, which is what the DICOM
standard fixes; keyword spellings belong to the reading library and can change.
The result is one row per image instance, written to
``data/interim/dicom_headers.csv``, together with a committed series-level
inventory in ``results/series_inventory.csv``.
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pydicom

from . import config
from .s99_utils import Metrics, banner, write_table

INTERIM = config.DATA / "interim"
HEADER_TABLE = INTERIM / "dicom_headers.csv"
SERIES_TABLE = config.RESULTS / "series_inventory.csv"

HEADER_COLUMNS = ["series_uid", "sop_instance_uid", "file"] + list(config.HEADER_TAGS)


def _coerce(value, kind: str):
    if value is None or value == "":
        return ""
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "decimal_string":
        # A multi-valued decimal string, written back in the DICOM form with a
        # backslash between the values. The reading library's own text form
        # brackets the list and separates it with commas, which is not what the
        # standard stores and is not what stage 3 parses.
        if isinstance(value, str):
            return value.strip()
        return "\\".join(str(item) for item in value)
    if kind == "date":
        text = str(value)
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}" if len(text) == 8 else text
    return str(value).strip()


def read_header(path_text: str) -> dict:
    """Read one instance and return the configured tags, typed."""
    path = Path(path_text)
    dataset = pydicom.dcmread(path, stop_before_pixels=True, force=False)
    row = {
        "series_uid": str(dataset.SeriesInstanceUID),
        "sop_instance_uid": str(dataset.SOPInstanceUID),
        "file": path.name,
    }
    for name, (tag, kind) in config.HEADER_TAGS.items():
        element = dataset.get(tag)
        row[name] = _coerce(None if element is None else element.value, kind)
    return row


def extract() -> list[dict]:
    banner("stage 02 headers")
    files = sorted(str(p) for p in (config.RAW).rglob("image/*/*.dcm"))
    if not files:
        raise SystemExit(
            "no image files under data/raw. Run: python data/download_data.py"
        )
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(read_header, files, chunksize=200))
    rows.sort(key=lambda r: (r["patient_id"], r["series_uid"], r["sop_instance_uid"]))

    INTERIM.mkdir(parents=True, exist_ok=True)
    write_table(rows, HEADER_TABLE, HEADER_COLUMNS)

    series: dict[str, dict] = {}
    for row in rows:
        entry = series.setdefault(row["series_uid"], {
            "series_uid": row["series_uid"],
            "patient_id": row["patient_id"],
            "series_desc": row["series_desc"],
            "study_date": row["study_date"],
            "series_date": row["series_date"],
            "modality": row["modality"],
            "manufacturer": row["manufacturer"],
            "model_name": row["model_name"],
            "body_part": row["body_part"],
            "rows_px": row["rows"],
            "cols_px": row["cols"],
            "photometric_interpretation": row["photometric_interpretation"],
            "bits_allocated": row["bits_allocated"],
            "instances": 0,
            "rows_varies_within_series": 0,
            "cols_varies_within_series": 0,
            "photometric_varies_within_series": 0,
        })
        entry["instances"] += 1
        entry["rows_varies_within_series"] |= int(row["rows"] != entry["rows_px"])
        entry["cols_varies_within_series"] |= int(row["cols"] != entry["cols_px"])
        entry["photometric_varies_within_series"] |= int(
            row["photometric_interpretation"] != entry["photometric_interpretation"]
        )
    inventory = sorted(series.values(), key=lambda r: (r["patient_id"], r["series_uid"]))
    write_table(inventory, SERIES_TABLE)

    metrics = Metrics()
    metrics.update({
        "headers_instances": len(rows),
        "headers_series": len(inventory),
        "headers_patients": len({r["patient_id"] for r in rows}),
        "headers_tags_extracted": len(config.HEADER_TAGS),
        "headers_series_with_varying_rows": sum(
            r["rows_varies_within_series"] for r in inventory),
        "headers_series_with_varying_cols": sum(
            r["cols_varies_within_series"] for r in inventory),
        "headers_series_with_varying_photometric": sum(
            r["photometric_varies_within_series"] for r in inventory),
    })
    metrics.save()
    print(
        f"{len(rows)} instances, {len(inventory)} series, "
        f"{metrics.get('headers_patients')} patients"
    )
    return rows


def load_headers() -> list[dict]:
    if not HEADER_TABLE.exists():
        raise SystemExit("data/interim/dicom_headers.csv is absent. Run stage 02.")
    with open(HEADER_TABLE, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for name, (_, kind) in config.HEADER_TAGS.items():
            if kind == "int":
                row[name] = int(row[name])
            elif kind == "float":
                row[name] = float(row[name]) if row[name] else float("nan")
    return rows


if __name__ == "__main__":
    sys.exit(0 if extract() else 1)
