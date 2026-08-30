"""Stage 3: cohort quality control and de-identification audit.

The checks answer the questions that decide whether a cohort can be pooled at
all: what equipment produced it, whether the image grid is stable within a
patient, whether the pixel encoding is consistent, and whether the header still
carries anything that identifies a person. Findings are written to
``results/qc_findings.csv`` and ``results/phi_audit.csv``.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict

import pydicom

from . import config
from .s02_headers import load_headers
from .s99_utils import Metrics, banner, write_table

QC_TABLE = config.RESULTS / "qc_findings.csv"
SCANNER_TABLE = config.RESULTS / "scanner_inventory.csv"
PHI_TABLE = config.RESULTS / "phi_audit.csv"

# The structural acquisitions the labeled task depends on. Everything else in
# the cohort is a diffusion or perfusion series and is reported as such.
STRUCTURAL_KEYWORDS = ("t1 axial", "t2", "flair")


def _series_view(rows):
    series = {}
    for row in rows:
        entry = series.setdefault(row["series_uid"], {
            "series_uid": row["series_uid"],
            "patient_id": row["patient_id"],
            "series_desc": row["series_desc"],
            "series_date": row["series_date"],
            "study_date": row["study_date"],
            "modality": row["modality"],
            "manufacturer": row["manufacturer"],
            "model_name": row["model_name"],
            "rows_px": row["rows"],
            "cols_px": row["cols"],
            "photometric_interpretation": row["photometric_interpretation"],
            "bits_allocated": row["bits_allocated"],
            "instances": 0,
        })
        entry["instances"] += 1
    return sorted(
        series.values(),
        key=lambda r: (r["patient_id"], r["series_date"], r["series_uid"]),
    )


def _is_structural(description):
    text = description.lower()
    return any(keyword in text for keyword in STRUCTURAL_KEYWORDS)


def run():
    banner("stage 03 quality control")
    rows = load_headers()
    series = _series_view(rows)
    findings = []

    def record(check, subject, detail, flag):
        findings.append({"check": check, "subject": subject, "detail": detail,
                         "flagged": int(flag)})

    # 1. Equipment. Pooling across scanners is a modeling decision, so the
    # combinations present are counted and not assumed.
    combos = Counter((s["manufacturer"], s["model_name"]) for s in series)
    scanner_rows = []
    for (manufacturer, model), count in sorted(combos.items(), key=lambda kv: -kv[1]):
        patients = {s["patient_id"] for s in series
                    if (s["manufacturer"], s["model_name"]) == (manufacturer, model)}
        scanner_rows.append({"manufacturer": manufacturer, "model_name": model,
                             "series": count, "patients": len(patients)})
    write_table(scanner_rows, SCANNER_TABLE)
    for entry in scanner_rows:
        record("scanner_combination",
               entry["manufacturer"] + " " + entry["model_name"],
               "{} series across {} patients".format(entry["series"], entry["patients"]),
               False)

    # 2. Modality. Every series must be MR.
    modalities = Counter(s["modality"] for s in series)
    for modality, count in sorted(modalities.items()):
        record("modality", modality, "{} series".format(count), modality != "MR")

    # 3. Image grid stability within a patient, comparing the first and the
    # last series a patient contributed, as the original exercise did.
    by_patient = defaultdict(list)
    for entry in series:
        by_patient[entry["patient_id"]].append(entry)
    dimension_changes = 0
    for patient, entries in sorted(by_patient.items()):
        first, last = entries[0], entries[-1]
        changed = ((first["rows_px"], first["cols_px"])
                   != (last["rows_px"], last["cols_px"]))
        dimension_changes += int(changed)
        record("first_last_dimensions", patient,
               "{}x{} then {}x{}".format(first["rows_px"], first["cols_px"],
                                         last["rows_px"], last["cols_px"]),
               changed)

    # 4. How many distinct grids a patient carries across all of their series.
    grids_per_patient = {
        patient: len({(e["rows_px"], e["cols_px"]) for e in entries})
        for patient, entries in by_patient.items()
    }
    for patient, count in sorted(grids_per_patient.items()):
        record("distinct_grids_per_patient", patient,
               "{} distinct row and column combinations".format(count), count > 1)

    # 5. Photometric interpretation and bit depth.
    photometric = Counter(s["photometric_interpretation"] for s in series)
    for value, count in sorted(photometric.items()):
        record("photometric_interpretation", value, "{} series".format(count),
               value != "MONOCHROME2")
    bits = Counter(s["bits_allocated"] for s in series)
    for value, count in sorted(bits.items()):
        record("bits_allocated", str(value), "{} series".format(count), value != 16)

    # 6. Series types. Diffusion and perfusion acquisitions sit in the same
    # studies as the structural series and are not interchangeable with them,
    # so they are counted and excluded from the labeled task.
    structural = [s for s in series if _is_structural(s["series_desc"])]
    other = [s for s in series if not _is_structural(s["series_desc"])]
    for description, count in sorted(Counter(s["series_desc"] for s in other).items()):
        record("non_structural_series", description, "{} series".format(count), True)

    # 7. Tag constancy inside a series.
    fields = ("rows", "cols", "photometric_interpretation", "bits_allocated",
              "manufacturer", "model_name")
    per_series_values = defaultdict(lambda: defaultdict(set))
    for row in rows:
        for field in fields:
            per_series_values[row["series_uid"]][field].add(row[field])
    inconsistent = {
        field: sum(1 for uid in per_series_values
                   if len(per_series_values[uid][field]) > 1)
        for field in fields
    }
    for field, count in sorted(inconsistent.items()):
        record("within_series_constant", field,
               "{} series carry more than one value".format(count), count > 0)

    write_table(findings, QC_TABLE)

    metrics = Metrics()
    metrics.update({
        "qc_series": len(series),
        "qc_patients": len(by_patient),
        "qc_scanner_combinations": len(combos),
        "qc_manufacturers": len({m for m, _ in combos}),
        "qc_modalities": len(modalities),
        "qc_patients_with_dimension_change": dimension_changes,
        "qc_patients_with_multiple_grids": sum(
            1 for c in grids_per_patient.values() if c > 1),
        "qc_max_grids_one_patient": max(grids_per_patient.values()),
        "qc_distinct_grids": len({(s["rows_px"], s["cols_px"]) for s in series}),
        "qc_photometric_values": len(photometric),
        "qc_bits_allocated_values": len(bits),
        "qc_structural_series": len(structural),
        "qc_non_structural_series": len(other),
        "qc_non_structural_descriptions": len({s["series_desc"] for s in other}),
        "qc_within_series_inconsistent_fields": sum(
            1 for count in inconsistent.values() if count > 0),
        "qc_findings_flagged": sum(f["flagged"] for f in findings),
        "qc_findings_total": len(findings),
    })
    metrics.save()
    print("{} series, {} scanner combinations, {} patients whose grid changes "
          "between first and last series, {} non-structural series".format(
              len(series), len(combos), dimension_changes, len(other)))
    return findings


def audit_phi():
    """Check the de-identification tags on one instance from every series."""
    banner("stage 03 de-identification audit")
    directories = [d for d in sorted(config.RAW.glob("*/*/*")) if any(d.glob("*.dcm"))]
    audit = []
    for directory in directories:
        path = sorted(directory.glob("*.dcm"))[0]
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
        patient_id = str(dataset.get(config.PHI_TAG_PATIENT_ID).value)
        name = dataset.get(config.PHI_TAG_PATIENT_NAME)
        name_value = "" if name is None else str(name.value)
        removed = dataset.get(config.PHI_TAG_IDENTITY_REMOVED)
        burned = dataset.get(config.PHI_TAG_BURNED_IN)
        populated = [label for label, tag in config.PHI_TAGS_MUST_BE_EMPTY.items()
                     if dataset.get(tag) is not None
                     and str(dataset.get(tag).value).strip()]
        audit.append({
            "series_uid": str(dataset.SeriesInstanceUID),
            "patient_id": patient_id,
            "modality": str(dataset.Modality),
            "patient_id_is_collection_form": int(
                patient_id.startswith(config.NBIA_COLLECTION + "-")),
            "patient_name_matches_patient_id": int(name_value == patient_id),
            "identity_removed": "" if removed is None else str(removed.value),
            "burned_in_annotation": "" if burned is None else str(burned.value),
            "private_tags": sum(1 for element in dataset if element.tag.is_private),
            "populated_identifier_tags": ";".join(populated),
        })
    write_table(sorted(audit, key=lambda r: r["series_uid"]), PHI_TABLE)

    metrics = Metrics()
    metrics.update({
        "phi_series_audited": len(audit),
        "phi_patient_id_non_conforming": sum(
            1 for r in audit if not r["patient_id_is_collection_form"]),
        "phi_patient_name_differs_from_id": sum(
            1 for r in audit if not r["patient_name_matches_patient_id"]),
        "phi_identity_removed_yes": sum(
            1 for r in audit if r["identity_removed"].upper() == "YES"),
        "phi_burned_in_annotation_yes": sum(
            1 for r in audit if r["burned_in_annotation"].upper() == "YES"),
        "phi_series_with_populated_identifier_tags": sum(
            1 for r in audit if r["populated_identifier_tags"]),
        "phi_series_with_private_tags": sum(1 for r in audit if r["private_tags"] > 0),
        "phi_max_private_tags": max(r["private_tags"] for r in audit),
    })
    metrics.save()
    print("{} series audited, {} carry a populated identifier tag, "
          "{} declare identity removed".format(
              len(audit),
              metrics.get("phi_series_with_populated_identifier_tags"),
              metrics.get("phi_identity_removed_yes")))
    return audit


if __name__ == "__main__":
    run()
    audit_phi()
    sys.exit(0)
