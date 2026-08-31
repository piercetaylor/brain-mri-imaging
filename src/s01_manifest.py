"""Stage 1: select the cohort and write the download manifest.

The full UPENN-GBM collection is 630 patients and 139.4 GB, which is more than
a portfolio pipeline should move. The subset is chosen by a rule and not by
hand: keep the patients whose post-contrast T1 series carries a tumor
segmentation with at least one segment marked SEMIAUTOMATIC, and take at most
``config.COHORT_MAX`` of them. The algorithm-type filter is a property of the
segmentation series and not of every segment inside it, so a selected series
can still contain segments no person reviewed; stage 4 measures how much of
each label that accounts for.

The post-contrast T1 series is identified by the marker rule in
``config.label_series_predicate`` and not by one vendor's protocol name, so the
cohort keeps the scanner heterogeneity of the collection it is drawn from.
Running the rule again on the same index release returns the same patients, and
``data/manifest.csv`` records which ones they were.

Two description tables are written beside the manifest.
``results/label_series_descriptions.csv`` holds what the rule matched inside the
selected cohort. ``results/segmented_series_descriptions.csv`` holds every
description in the collection that carries a reviewed segmentation, admitted or
not, so that what the rule excluded can be read off a table as well.
"""

from __future__ import annotations

import sys
from collections import defaultdict

from . import config
from .s99_utils import Metrics, banner, idc_sql, idc_version, write_table

LABEL_RULE = config.label_series_predicate("m.SeriesDescription")

COHORT_QUERY = f"""
SELECT m.PatientID AS patient_id,
       min(m.ManufacturerModelName) AS label_model_name,
       min(m.Manufacturer) AS label_manufacturer
FROM index i
JOIN seg_index s ON s.SeriesInstanceUID = i.SeriesInstanceUID
JOIN index m ON m.SeriesInstanceUID = s.segmented_SeriesInstanceUID
WHERE i.collection_id = '{config.IDC_COLLECTION}'
  AND s.AlgorithmType LIKE '%{config.CORRECTED_ALGORITHM_TYPE}%'
  AND {LABEL_RULE}
GROUP BY patient_id
ORDER BY patient_id
"""

# What the rule actually matched in the selected cohort, so that the selection
# can be read off a table instead of re-derived. One row per description and
# scanner model. A pre-contrast or T2 description appearing here would mean the
# marker rule had let one through.
LABEL_DESCRIPTION_QUERY = """
SELECT m.SeriesDescription AS series_desc,
       m.Manufacturer AS manufacturer,
       m.ManufacturerModelName AS model_name,
       count(DISTINCT m.PatientID) AS patients,
       count(DISTINCT m.SeriesInstanceUID) AS series
FROM index i
JOIN seg_index s ON s.SeriesInstanceUID = i.SeriesInstanceUID
JOIN index m ON m.SeriesInstanceUID = s.segmented_SeriesInstanceUID
WHERE i.collection_id = '{collection}'
  AND s.AlgorithmType LIKE '%{corrected}%'
  AND {rule}
  AND m.PatientID IN ({patients})
GROUP BY 1, 2, 3
ORDER BY 3, 1
"""

# Every series description in the collection that carries a segmentation with a
# reviewed segment, whether or not the marker rule admits it. The cohort table
# above is restricted to the rule and to the selected patients, so on its own it
# cannot show what the rule left out. This query drops both restrictions, and
# the admitted flag written beside each row is the rule applied in Python, so
# the boundary of the selection is on the record and not asserted in prose.
SEGMENTED_DESCRIPTION_QUERY = """
SELECT m.SeriesDescription AS series_desc,
       m.Manufacturer AS manufacturer,
       m.ManufacturerModelName AS model_name,
       count(DISTINCT m.PatientID) AS patients,
       count(DISTINCT m.SeriesInstanceUID) AS series
FROM index i
JOIN seg_index s ON s.SeriesInstanceUID = i.SeriesInstanceUID
JOIN index m ON m.SeriesInstanceUID = s.segmented_SeriesInstanceUID
WHERE i.collection_id = '{collection}'
  AND s.AlgorithmType LIKE '%{corrected}%'
GROUP BY 1, 2, 3
ORDER BY 3, 1
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
  AND {rule}
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

LABEL_DESCRIPTION_TABLE = config.RESULTS / "label_series_descriptions.csv"
SEGMENTED_DESCRIPTION_TABLE = config.RESULTS / "segmented_series_descriptions.csv"


def names_no_contrast_state(description: str) -> bool:
    """True when a description carries neither contrast-state token."""
    text = str(description).lower()
    return not any(marker in text for marker in config.CONTRAST_STATE_MARKERS)

COLUMNS = [
    "patient_id", "role", "series_uid", "study_uid", "modality", "series_desc",
    "series_date", "manufacturer", "model_name", "instance_count", "size_mb",
    "series_uuid", "license", "source_doi", "labels_series_uid",
    "algorithm_type", "total_segments",
]


def quoted(values) -> str:
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def cap_cohort(eligible: list[dict], limit: int) -> list[str]:
    """Reduce the eligible patients to at most ``limit``, keeping every scanner.

    Taking the first ``limit`` identifiers in sorted order is deterministic but
    is not neutral with respect to equipment: the models that contributed
    fewest patients are the ones a sorted cap is most likely to remove, and a
    cohort recut to hold six scanner models would lose them again. Patients are
    therefore taken from each scanner model in turn, and in sorted order within
    a model, until the cap is filled. The result depends only on the eligible
    set and is returned sorted, so the manifest is byte-identical between runs.
    Below the cap the function is the plain sort.
    """
    if len(eligible) <= limit:
        return sorted(row["patient_id"] for row in eligible)
    by_model: dict[str, list[str]] = defaultdict(list)
    for row in sorted(eligible, key=lambda r: r["patient_id"]):
        by_model[str(row["label_model_name"])].append(row["patient_id"])
    kept: list[str] = []
    for position in range(max(len(v) for v in by_model.values())):
        for model in sorted(by_model):
            if position < len(by_model[model]) and len(kept) < limit:
                kept.append(by_model[model][position])
        if len(kept) >= limit:
            break
    return sorted(kept)


def build() -> list[dict]:
    banner("stage 01 manifest")
    version = idc_version()
    eligible = idc_sql(COHORT_QUERY)
    cohort = cap_cohort(eligible, config.COHORT_MAX)
    if len(cohort) < config.COHORT_MIN:
        raise RuntimeError(
            f"only {len(cohort)} eligible patients were found, "
            f"at least {config.COHORT_MIN} are needed"
        )
    eligible_models = {str(r["label_model_name"]) for r in eligible}
    kept_models = {str(r["label_model_name"]) for r in eligible
                   if r["patient_id"] in set(cohort)}
    if eligible_models != kept_models:
        raise RuntimeError(
            "capping the cohort at {} dropped every patient of scanner "
            "model {}".format(config.COHORT_MAX,
                              ", ".join(sorted(eligible_models - kept_models)))
        )
    print(f"eligible patients {len(eligible)}, cohort {len(cohort)}, "
          f"label scanner models {len(kept_models)}")

    patients = quoted(cohort)
    matched = idc_sql(LABEL_DESCRIPTION_QUERY.format(
        collection=config.IDC_COLLECTION,
        corrected=config.CORRECTED_ALGORITHM_TYPE,
        rule=LABEL_RULE,
        patients=patients,
    ))
    for row in matched:
        print("  matched {:<20} {:<44} {} patients".format(
            str(row["model_name"]), str(row["series_desc"]), row["patients"]))

    mr = idc_sql(MR_QUERY.format(collection=config.IDC_COLLECTION, patients=patients))
    seg = idc_sql(
        SEG_QUERY.format(
            collection=config.IDC_COLLECTION,
            corrected=config.CORRECTED_ALGORITHM_TYPE,
            rule=LABEL_RULE,
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
    write_table(
        [{"series_desc": r["series_desc"], "manufacturer": r["manufacturer"],
          "model_name": r["model_name"], "patients": r["patients"],
          "series": r["series"]} for r in matched],
        LABEL_DESCRIPTION_TABLE,
    )

    # The vocabulary the rule was applied to, so that what it rejected can be
    # read beside what it admitted. The flag is the Python form of the rule and
    # the query carries no rule of its own, so the two cannot drift apart.
    vocabulary = idc_sql(SEGMENTED_DESCRIPTION_QUERY.format(
        collection=config.IDC_COLLECTION,
        corrected=config.CORRECTED_ALGORITHM_TYPE,
    ))
    vocabulary_rows = [
        {"series_desc": str(r["series_desc"]),
         "manufacturer": str(r["manufacturer"]),
         "model_name": str(r["model_name"]),
         "patients": int(r["patients"]),
         "series": int(r["series"]),
         "admitted": int(config.is_label_series_description(r["series_desc"]))}
        for r in vocabulary
    ]
    write_table(vocabulary_rows, SEGMENTED_DESCRIPTION_TABLE)
    admitted_rows = [r for r in vocabulary_rows if r["admitted"]]
    rejected_rows = [r for r in vocabulary_rows if not r["admitted"]]
    # The rule's blind spot, counted. A T1 acquisition whose description names
    # no contrast state cannot be admitted, whichever side of the injection it
    # was acquired on.
    unnamed_t1 = [r for r in rejected_rows
                  if "t1" in r["series_desc"].lower()
                  and not any(m in r["series_desc"].lower()
                              for m in config.LABEL_EXCLUDED_MARKERS)
                  and names_no_contrast_state(r["series_desc"])]
    print("segmented description vocabulary {} rows over {} descriptions, "
          "{} rows admitted over {} descriptions".format(
              len(vocabulary_rows),
              len({r["series_desc"] for r in vocabulary_rows}),
              len(admitted_rows),
              len({r["series_desc"] for r in admitted_rows})))

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
        "cohort_capped_below_eligible": int(len(cohort) < len(eligible)),
        "cohort_label_series_descriptions": len(
            {str(r["series_desc"]) for r in matched}),
        "cohort_label_scanner_models": len(kept_models),
        "cohort_label_manufacturers": len(
            {str(r["manufacturer"]) for r in matched}),
        "cohort_scanner_models": len(
            {str(r["model_name"]) for r in image_rows}),
        "cohort_manufacturers": len(
            {str(r["manufacturer"]) for r in image_rows}),
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
        "label_rule_vocabulary_rows": len(vocabulary_rows),
        "label_rule_vocabulary_descriptions": len(
            {r["series_desc"] for r in vocabulary_rows}),
        "label_rule_admitted_rows": len(admitted_rows),
        "label_rule_admitted_descriptions": len(
            {r["series_desc"] for r in admitted_rows}),
        "label_rule_rejected_rows": len(rejected_rows),
        "label_rule_rejected_descriptions": len(
            {r["series_desc"] for r in rejected_rows}),
        "label_rule_rejected_t1_naming_no_contrast_state_rows": len(unnamed_t1),
        "label_rule_rejected_t1_naming_no_contrast_state_descriptions": len(
            {r["series_desc"] for r in unnamed_t1}),
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
