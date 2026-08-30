"""Stage 1: select the cohort and write the download manifest.

The full UPENN-GBM collection is 630 patients and 139.4 GB, which is more than
a portfolio pipeline should move. The subset is chosen by a rule and not by
hand: keep the patients whose axial T1 post-contrast series carries a
radiologist-corrected tumor segmentation, sort them by patient identifier, and
take at most ``config.COHORT_MAX``. Running the rule again on the same index
release returns the same patients, and ``data/manifest.csv`` records which ones
they were.
"""

from __future__ import annotations

import sys

from . import config
from .s99_utils import Metrics, banner, idc_sql, idc_version, write_table

COHORT_QUERY = f"""
SELECT DISTINCT i.PatientID AS patient_id
FROM index i
JOIN seg_index s ON s.SeriesInstanceUID = i.SeriesInstanceUID
JOIN index m ON m.SeriesInstanceUID = s.segmented_SeriesInstanceUID
WHERE i.collection_id = '{config.IDC_COLLECTION}'
  AND s.AlgorithmType LIKE '%{config.CORRECTED_ALGORITHM_TYPE}%'
  AND m.SeriesDescription LIKE '{config.LABEL_SERIES_PATTERN}'
ORDER BY patient_id
"""

MR_QUERY = """
SELECT PatientID AS patient_id, SeriesInstanceUID AS series_uid,
       StudyInstanceUID AS study_uid, Modality AS modality,
       SeriesDescription AS series_desc, SeriesDate AS series_date,
       Manufacturer AS manufacturer, ManufacturerModelName AS model_name,
       instanceCount AS instance_count, series_size_MB AS size_mb,
       crdc_series_uuid AS series_uuid, license_short_name AS license,
       source_DOI AS source_doi
FROM index
WHERE collection_id = '{collection}' AND Modality = 'MR'
  AND PatientID IN ({patients})
ORDER BY patient_id, series_uid
"""

SEG_QUERY = """
SELECT i.PatientID AS patient_id, i.SeriesInstanceUID AS series_uid,
       i.StudyInstanceUID AS study_uid, i.Modality AS modality,
       i.SeriesDescription AS series_desc, i.SeriesDate AS series_date,
       i.Manufacturer AS manufacturer, i.ManufacturerModelName AS model_name,
       i.instanceCount AS instance_count, i.series_size_MB AS size_mb,
       i.crdc_series_uuid AS series_uuid, i.license_short_name AS license,
       i.source_DOI AS source_doi,
       s.segmented_SeriesInstanceUID AS labels_series_uid,
       s.AlgorithmType AS algorithm_type, s.total_segments AS total_segments
FROM index i
JOIN seg_index s ON s.SeriesInstanceUID = i.SeriesInstanceUID
JOIN index m ON m.SeriesInstanceUID = s.segmented_SeriesInstanceUID
WHERE i.collection_id = '{collection}'
  AND s.AlgorithmType LIKE '%{corrected}%'
  AND m.SeriesDescription LIKE '{pattern}'
  AND i.PatientID IN ({patients})
ORDER BY patient_id, series_uid
"""

COLLECTION_EQUIPMENT_QUERY = """
SELECT Manufacturer AS manufacturer, ManufacturerModelName AS model_name,
       count(*) AS series
FROM index
WHERE collection_id = '{collection}' AND Modality = 'MR'
GROUP BY 1, 2
ORDER BY series DESC
"""

COLUMNS = [
    "patient_id", "role", "series_uid", "study_uid", "modality", "series_desc",
    "series_date", "manufacturer", "model_name", "instance_count", "size_mb",
    "series_uuid", "license", "source_doi", "labels_series_uid",
    "algorithm_type", "total_segments",
]


def quoted(values) -> str:
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def build() -> list[dict]:
    banner("stage 01 manifest")
    version = idc_version()
    eligible = [row["patient_id"] for row in idc_sql(COHORT_QUERY)]
    cohort = sorted(eligible)[: config.COHORT_MAX]
    if len(cohort) < config.COHORT_MIN:
        raise RuntimeError(
            f"only {len(cohort)} eligible patients were found, "
            f"at least {config.COHORT_MIN} are needed"
        )
    print(f"eligible patients {len(eligible)}, cohort {len(cohort)}")

    patients = quoted(cohort)
    mr = idc_sql(MR_QUERY.format(collection=config.IDC_COLLECTION, patients=patients))
    seg = idc_sql(
        SEG_QUERY.format(
            collection=config.IDC_COLLECTION,
            corrected=config.CORRECTED_ALGORITHM_TYPE,
            pattern=config.LABEL_SERIES_PATTERN,
            patients=patients,
        )
    )

    # One patient can hold more than one corrected segmentation of the same
    # kind. Keep one label series per patient, the first by series identifier,
    # so that every patient contributes the same number of labeled volumes.
    seen: set[str] = set()
    seg_rows = []
    for row in sorted(seg, key=lambda r: (r["patient_id"], r["series_uid"])):
        if row["patient_id"] in seen:
            continue
        seen.add(row["patient_id"])
        seg_rows.append(row)

    rows = []
    for row in mr:
        rows.append({**row, "role": "image", "labels_series_uid": "",
                     "algorithm_type": "", "total_segments": ""})
    for row in seg_rows:
        rows.append({**row, "role": "segmentation"})
    rows.sort(key=lambda r: (r["patient_id"], r["role"], r["series_uid"]))

    licenses = sorted({str(r["license"]) for r in rows})
    if licenses != [config.EXPECTED_LICENSE]:
        raise RuntimeError(f"unexpected license in the manifest: {licenses}")

    write_table(rows, config.MANIFEST, COLUMNS)

    # What the whole collection was acquired on, so that the equipment mix of
    # the cohort can be compared against the mix it was drawn from.
    equipment = idc_sql(
        COLLECTION_EQUIPMENT_QUERY.format(collection=config.IDC_COLLECTION))

    image_rows = [r for r in rows if r["role"] == "image"]
    metrics = Metrics()
    metrics.update({
        "cohort_eligible_patients": len(eligible),
        "cohort_patients": len(cohort),
        "cohort_first_patient": cohort[0],
        "cohort_last_patient": cohort[-1],
        "manifest_series": len(rows),
        "manifest_image_series": len(image_rows),
        "manifest_label_series": len(seg_rows),
        "manifest_instances": sum(int(r["instance_count"]) for r in rows),
        "manifest_size_gb": sum(float(r["size_mb"]) for r in rows) / 1024,
        "manifest_license": licenses[0],
        "idc_index_version": version["idc_version"],
        "idc_index_data_version": version["idc_index_data_version"],
        "collection_total_patients": 630,
        "collection_scanner_combinations": len(equipment),
        "collection_manufacturers": len({r["manufacturer"] for r in equipment}),
        "collection_mr_series": sum(int(r["series"]) for r in equipment),
    })
    metrics.save()
    print(
        f"manifest rows {len(rows)} "
        f"({len(image_rows)} image series, {len(seg_rows)} label series), "
        f"{metrics.number('manifest_size_gb'):.2f} GB"
    )
    return rows


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
