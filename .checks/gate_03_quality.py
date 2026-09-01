#!/usr/bin/env python
"""Phase 3 gate: the quality control findings and the audit are internally consistent.

The gate does not decide whether the cohort is heterogeneous. It confirms that
whatever the checks found was recorded, that the recorded counts match the
tables they were derived from, and that the de-identification audit found no
identifier still populated.
"""

from __future__ import annotations

from gate_lib import check, config, finish, gate, metrics, table

gate("gate 03 quality control")

findings = table("qc_findings")
scanners = table("scanner_inventory")
inventory = table("series_inventory")
audit = table("phi_audit")
recorded = metrics()

check("findings were recorded", len(findings) > 0, "{} findings".format(len(findings)))
check("the recorded finding count matches the table",
      len(findings) == int(recorded.number("qc_findings_total")))
check("the flagged count matches the table",
      sum(int(f["flagged"]) for f in findings)
      == int(recorded.number("qc_findings_flagged")),
      "{} flagged".format(sum(int(f["flagged"]) for f in findings)))

check("every check name appears in the findings",
      {"scanner_combination", "modality", "first_last_dimensions",
       "distinct_grids_per_patient", "photometric_interpretation",
       "bits_allocated", "within_series_constant"}
      <= {f["check"] for f in findings},
      ", ".join(sorted({f["check"] for f in findings})))

check("the scanner inventory matches the recorded combination count",
      len(scanners) == int(recorded.number("qc_scanner_combinations")),
      "{} combinations".format(len(scanners)))
check("the scanner inventory accounts for every series",
      sum(int(s["series"]) for s in scanners) == len(inventory),
      "{} series".format(len(inventory)))

check("the cohort is magnetic resonance throughout",
      int(recorded.number("qc_modalities")) == 1)
check("pixel data is single channel grayscale",
      {s["photometric_interpretation"] for s in inventory} == {"MONOCHROME2"},
      ", ".join(sorted({s["photometric_interpretation"] for s in inventory})))
check("no header tag varies inside a series",
      int(recorded.number("qc_within_series_inconsistent_fields")) == 0)

flagged_dimension = sum(1 for f in findings
                        if f["check"] == "first_last_dimensions" and int(f["flagged"]))
check("the recorded dimension changes match the findings",
      flagged_dimension == int(recorded.number("qc_patients_with_dimension_change")),
      "{} patients".format(flagged_dimension))

# Whether the slices of a series agree on where they point. Stage 4 assembles
# each series into a volume with a tolerance on that comparison, so the
# tolerance has to be shown to stand above every disagreement the cohort holds
# and by the margin config states. A tolerance a cohort has grown to fill is no
# longer a bound on rounding, and the margin is what would catch that while
# stage 4 still ran.
check("the orientation finding was recorded",
      {"orientation_consistency", "orientation_variants_per_series"}
      <= {f["check"] for f in findings},
      ", ".join(sorted({f["check"] for f in findings
                        if "orientation" in f["check"]})))
check("every instance carries an orientation",
      int(recorded.number("qc_instances_without_orientation")) == 0,
      "{} instances carry none".format(
          recorded.get("qc_instances_without_orientation")))
check("the recorded tolerance is the configured tolerance",
      recorded.number("qc_orientation_tolerance") == config.ORIENTATION_TOLERANCE,
      "{:.3e}".format(config.ORIENTATION_TOLERANCE))
check("no series exceeds the orientation tolerance",
      recorded.number("qc_max_orientation_deviation")
      < config.ORIENTATION_TOLERANCE,
      "largest deviation {:.3e} against a tolerance of {:.3e}".format(
          recorded.number("qc_max_orientation_deviation"),
          config.ORIENTATION_TOLERANCE))
check("the largest orientation deviation clears the tolerance by the stated margin",
      recorded.number("qc_max_orientation_deviation")
      * config.ORIENTATION_TOLERANCE_MARGIN < config.ORIENTATION_TOLERANCE,
      "largest deviation {:.3e}, tolerance {:.3e}, margin of at least {:.0f}".format(
          recorded.number("qc_max_orientation_deviation"),
          config.ORIENTATION_TOLERANCE, config.ORIENTATION_TOLERANCE_MARGIN))
check("every series carrying more than one orientation was reported on its own",
      sum(1 for f in findings if f["check"] == "series_orientation_variants")
      == int(recorded.number("qc_series_with_multiple_orientations")),
      "{} of {} series".format(
          recorded.get("qc_series_with_multiple_orientations"), len(inventory)))

# De-identification. Anything populated here would be a reason to stop.
check("every series was audited",
      len(audit) == int(recorded.number("phi_series_audited")),
      "{} series".format(len(audit)))
check("no identifier tag is populated",
      int(recorded.number("phi_series_with_populated_identifier_tags")) == 0,
      "{} series carry one".format(
          recorded.get("phi_series_with_populated_identifier_tags")))
# The burned-in annotation tag is unpopulated across the cohort, so a count of
# zero declarations is an absence and not a clearance. The three counts are
# required to add up to the audit, which is what keeps the zero readable: a
# series that dropped out of the accounting would fail here.
check("the burned-in annotation tag is accounted for on every audited series",
      int(recorded.number("phi_burned_in_annotation_yes")) == 0
      and (int(recorded.number("phi_burned_in_annotation_yes"))
           + int(recorded.number("phi_burned_in_annotation_no"))
           + int(recorded.number("phi_burned_in_annotation_absent")))
      == int(recorded.number("phi_series_audited")),
      "{} declare it, {} declare none, {} carry no such tag, of {} audited".format(
          recorded.get("phi_burned_in_annotation_yes"),
          recorded.get("phi_burned_in_annotation_no"),
          recorded.get("phi_burned_in_annotation_absent"),
          recorded.get("phi_series_audited")))
check("every patient identifier uses the collection form",
      int(recorded.number("phi_patient_id_non_conforming")) == 0)
check("patient name carries no name",
      int(recorded.number("phi_patient_name_differs_from_id")) == 0,
      "{} series differ".format(recorded.get("phi_patient_name_differs_from_id")))
check("the audit covers image and segmentation series",
      {r["modality"] for r in audit} == {"MR", "SEG"},
      ", ".join(sorted({r["modality"] for r in audit})))

finish()
