"""Stage 3: cohort quality control and de-identification audit.

The checks answer the questions that decide whether a cohort can be pooled at
all: what equipment produced it, whether the image grid is stable within a
patient, whether the slices of a series agree on their orientation, whether the
pixel encoding is consistent, and whether the header still carries anything
that identifies a person. Findings are written to
``results/qc_findings.csv`` and ``results/phi_audit.csv``.
"""

from __future__ import annotations

import math
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
# the cohort is a diffusion or perfusion series and is reported as such. The
# markers come from config so that they cover the naming of every scanner model
# in the cohort. A marker tied to one vendor's protocol name, such as the
# string "t1 axial", matches that vendor alone and files the T1 series of the
# others among the diffusion and perfusion acquisitions.
STRUCTURAL_KEYWORDS = config.STRUCTURAL_MARKERS


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


def _cosine_deviation(first, second):
    """1 minus the cosine similarity of two direction vectors."""
    dot = sum(a * b for a, b in zip(first, second))
    norm = (math.sqrt(sum(a * a for a in first))
            * math.sqrt(sum(b * b for b in second)))
    return 1.0 - dot / norm if norm else 1.0


def _orientation_deviation(counts):
    """Largest disagreement with the modal orientation of one series.

    ``counts`` maps each distinct ImageOrientationPatient text in a series to
    the number of instances carrying it. The modal value is the reference,
    which is the value highdicom takes as the reference in the consistency
    check it applies before assembling a volume, and each of the two direction
    unit vectors of every other value is compared with the corresponding vector
    of the modal value.

    The modal value is not compared with itself. A series carrying one
    orientation has no disagreement to report and returns 0.0; comparing that
    value with itself would return the rounding of a unit vector's own dot
    product and report roughly 1e-15 of disagreement that is not present.
    """
    if len(counts) < 2:
        return 0.0
    modal = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    reference = [float(v) for v in modal.split("\\")]
    deviation = 0.0
    for value in counts:
        if value == modal:
            continue
        vector = [float(v) for v in value.split("\\")]
        deviation = max(
            deviation,
            _cosine_deviation(vector[:3], reference[:3]),
            _cosine_deviation(vector[3:], reference[3:]),
        )
    return deviation


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

    # Whether the cohort is heterogeneous is the question the rest of the
    # pipeline is entitled to assume an answer to, so it is measured here and
    # not taken on trust. A cohort drawn from one scanner cannot show whether a
    # score survives a change of equipment; one drawn from several can, and the
    # finding is flagged so that the pooling decision is visible in the table.
    largest_share = (max(combos.values()) / len(series)) if series else 0.0
    record("scanner_heterogeneity", "cohort",
           "{} manufacturer and model combinations, largest holds {:.1f} percent "
           "of series".format(len(combos), 100 * largest_share),
           len(combos) > 1)
    by_patient = defaultdict(list)
    for entry in series:
        by_patient[entry["patient_id"]].append(entry)
    scanners_per_patient = {
        patient: len({(e["manufacturer"], e["model_name"]) for e in entries})
        for patient, entries in by_patient.items()
    }
    for patient, count in sorted(scanners_per_patient.items()):
        if count > 1:
            record("scanners_per_patient", patient,
                   "{} manufacturer and model combinations".format(count), True)

    # 2. Modality. Every series must be MR.
    modalities = Counter(s["modality"] for s in series)
    for modality, count in sorted(modalities.items()):
        record("modality", modality, "{} series".format(count), modality != "MR")

    # 3. Image grid stability within a patient, comparing the first and the
    # last series a patient contributed, as the original exercise did.
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

    # 8. In-plane orientation inside a series. ImageOrientationPatient is left
    # out of the constancy check above and given its own, because a series can
    # carry several values that describe the same plane to the precision the
    # DICOM decimal string holds. highdicom compares those values for equality
    # before it will assemble a volume, so a difference in the sixth
    # significant figure stops stage 4 outright. Nothing in the cohort showed
    # this while every patient came from one scanner model.
    if rows and "image_orientation" not in rows[0]:
        raise SystemExit(
            "data/interim/dicom_headers.csv carries no image_orientation "
            "column. Re-run stage 02."
        )
    orientations = defaultdict(Counter)
    missing_orientation = 0
    for row in rows:
        value = row["image_orientation"]
        if value:
            orientations[row["series_uid"]][value] += 1
        else:
            missing_orientation += 1
    record("orientation_present", "cohort",
           "{} of {} instances carry no ImageOrientationPatient".format(
               missing_orientation, len(rows)),
           missing_orientation > 0)
    variants = {}
    deviations = {}
    for entry in series:
        uid = entry["series_uid"]
        counts = orientations[uid]
        variants[uid] = len(counts)
        deviations[uid] = _orientation_deviation(counts)
    for count, number in sorted(Counter(variants.values()).items()):
        record("orientation_variants_per_series", str(count),
               "{} series".format(number), count > 1)
    for entry in series:
        uid = entry["series_uid"]
        if variants[uid] < 2:
            continue
        record("series_orientation_variants", uid,
               "{} on {} {} carries {} distinct ImageOrientationPatient values "
               "across {} instances, largest deviation {:.3e} from the modal "
               "value against a tolerance of {:.3e}".format(
                   entry["patient_id"], entry["manufacturer"],
                   entry["model_name"], variants[uid],
                   sum(orientations[uid].values()), deviations[uid],
                   config.ORIENTATION_TOLERANCE),
               True)
    multi_orientation = sum(1 for count in variants.values() if count > 1)
    max_deviation = max(deviations.values()) if deviations else 0.0
    record("orientation_consistency", "cohort",
           "{} of {} series carry more than one ImageOrientationPatient value, "
           "largest deviation {:.3e} against a tolerance of {:.3e}".format(
               multi_orientation, len(series), max_deviation,
               config.ORIENTATION_TOLERANCE),
           multi_orientation > 0)

    write_table(findings, QC_TABLE)

    metrics = Metrics()
    metrics.update({
        "qc_series": len(series),
        "qc_patients": len(by_patient),
        "qc_scanner_combinations": len(combos),
        "qc_manufacturers": len({m for m, _ in combos}),
        "qc_scanner_models": len({model for _, model in combos}),
        "qc_largest_scanner_series_share": float(largest_share),
        "qc_patients_with_multiple_scanners": sum(
            1 for count in scanners_per_patient.values() if count > 1),
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
        "qc_instances_without_orientation": missing_orientation,
        "qc_series_with_multiple_orientations": multi_orientation,
        "qc_max_orientation_variants_one_series": max(variants.values()),
        "qc_max_orientation_deviation": max_deviation,
        "qc_orientation_tolerance": config.ORIENTATION_TOLERANCE,
        "qc_findings_flagged": sum(f["flagged"] for f in findings),
        "qc_findings_total": len(findings),
    })
    metrics.save()
    print("{} series, {} scanner combinations across {} manufacturers, largest "
          "holds {:.1f} percent of series, {} patients whose grid changes "
          "between first and last series, {} non-structural series".format(
              len(series), len(combos), len({m for m, _ in combos}),
              100 * largest_share, dimension_changes, len(other)))
    print("{} series carry more than one ImageOrientationPatient value, "
          "largest deviation {:.3e} against a tolerance of {:.3e}".format(
              multi_orientation, max_deviation, config.ORIENTATION_TOLERANCE))
    return findings


# BurnedInAnnotation is absent from every series this collection holds, so the
# count of series declaring YES is zero because nothing was declared at all. A
# zero read on its own would say the tag was populated and read negative. The
# count of series where the tag is absent is recorded beside it, and this role
# string states which of the two the zero is. ImageType 0008,0008 and the pixel
# data would have to be inspected to make a positive statement, and neither is
# read here.
def burned_in_annotation_role(absent: int, audited: int) -> str:
    """What ``phi_burned_in_annotation_yes`` is, stated beside the number."""
    return ("the tag is absent from {} of the {} audited series, so this zero "
            "counts a tag that carries no value and is not a negative "
            "reading".format(absent, audited))


# PatientIdentityRemoved is populated on the image series and absent on the
# segmentation series, which carry no such attribute. The count of series
# declaring YES is therefore a count over part of the audit, and this role
# string names the part.
def identity_removed_role(absent: int, audited: int) -> str:
    """What ``phi_identity_removed_yes`` covers, stated beside the number."""
    return ("the tag is absent from {} of the {} audited series, all of them "
            "segmentation series, so this count covers the remainder and not "
            "the whole audit".format(absent, audited))


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

    burned_in_absent = sum(
        1 for r in audit if not r["burned_in_annotation"].strip())
    identity_removed_absent = sum(
        1 for r in audit if not r["identity_removed"].strip())

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
        "phi_burned_in_annotation_no": sum(
            1 for r in audit if r["burned_in_annotation"].upper() == "NO"),
        "phi_burned_in_annotation_absent": burned_in_absent,
        "phi_burned_in_annotation_yes_role": burned_in_annotation_role(
            burned_in_absent, len(audit)),
        "phi_identity_removed_absent": identity_removed_absent,
        "phi_identity_removed_yes_role": identity_removed_role(
            identity_removed_absent, len(audit)),
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
    print("burned-in annotation: {} series declare YES, {} declare NO, "
          "{} do not carry the tag".format(
              metrics.get("phi_burned_in_annotation_yes"),
              metrics.get("phi_burned_in_annotation_no"),
              metrics.get("phi_burned_in_annotation_absent")))
    return audit


if __name__ == "__main__":
    run()
    audit_phi()
    sys.exit(0)
