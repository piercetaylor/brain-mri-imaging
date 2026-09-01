#!/usr/bin/env python
"""Phase 4 gate: the patch set is what the configuration says it should be.

Two properties are asserted and not merely recorded. The negative class must be
tissue and not air, because a network separating head from background would
report a score that says nothing about tumor. The geometry residuals between
each segmentation and the image it labels must sit below the tolerances stage 4
places one on the other with, by the margin config states, and the largest
displacement must stay inside the quantum of the coordinate strings that
produced it, because a residual larger than that is a position the label and
the image genuinely disagree on.
"""

from __future__ import annotations

import sys

import numpy as np

from gate_lib import ROOT, check, config, finish, gate, metrics, table

sys.path.insert(0, str(ROOT))
from src.s04_patches import load_patches  # noqa: E402

gate("gate 04 preparation")

X, y, patients, index = load_patches()
summary = table("patch_summary")
recorded = metrics()

check("patches are square and the configured size",
      X.shape[1:] == (config.PATCH_SIZE, config.PATCH_SIZE), str(X.shape))
check("labels and patches agree in length", len(X) == len(y) == len(patients))
check("labels are binary", set(np.unique(y).tolist()) == {0, 1})
check("the recorded patch count matches the array",
      len(y) == int(recorded.number("patches_total")), "{} patches".format(len(y)))

check("pixel values are normalized to the unit interval",
      float(X.min()) >= 0.0 and float(X.max()) <= 1.0,
      "{:.4f} to {:.4f}".format(float(X.min()), float(X.max())))
check("patches are not constant",
      float(X.std()) > 0.01, "standard deviation {:.4f}".format(float(X.std())))

check("every cohort patient contributed patches",
      len(set(patients.tolist())) == int(recorded.number("patches_patients")),
      "{} patients".format(len(set(patients.tolist()))))
check("the per-patient summary covers every patient",
      {r["patient_id"] for r in summary} == set(patients.tolist()))

for row in summary:
    expected = int(row["patches_positive"]) * config.NEGATIVES_PER_POSITIVE
    if int(row["candidate_negative_patches"]) < expected:
        check("negative sampling for {}".format(row["patient_id"]), False,
              "fewer candidates than requested")
        break
else:
    check("every patient had enough negative candidates for the sampling ratio", True,
          "ratio {}:1".format(config.NEGATIVES_PER_POSITIVE))

ratio = [int(r["patches_negative"]) / max(int(r["patches_positive"]), 1)
         for r in summary]
check("the sampled ratio is the configured ratio for every patient",
      all(abs(v - config.NEGATIVES_PER_POSITIVE) < 1e-9 for v in ratio),
      "{:.3f} to {:.3f}".format(min(ratio), max(ratio)))

check("no patient exceeds the positive cap",
      max(int(r["patches_positive"]) for r in summary)
      <= config.MAX_POSITIVE_PATCHES_PER_PATIENT,
      "largest {}".format(max(int(r["patches_positive"]) for r in summary)))

check("the class balance matches the sampling design",
      abs(float(y.mean()) - 1 / (1 + config.NEGATIVES_PER_POSITIVE)) < 1e-6,
      "{:.4f} positive".format(float(y.mean())))

natural = [float(r["natural_positive_rate"]) for r in summary]
check("the natural positive rate is far below the sampled rate",
      max(natural) < float(y.mean()),
      "largest natural rate {:.4f}".format(max(natural)))

# The check that would catch a negative class made of empty space. A patch of
# air would let the network separate tissue from background and report a score
# that says nothing about tumor.
check("negative patches are tissue and not background",
      recorded.number("patch_negative_share_near_empty") < 0.01,
      "{:.4f} of negatives have mean intensity below 0.05".format(
          recorded.number("patch_negative_share_near_empty")))
check("the two classes overlap in mean intensity",
      recorded.number("patch_negative_mean_intensity")
      > 0.5 * recorded.number("patch_positive_mean_intensity"),
      "negative {:.3f} against positive {:.3f}".format(
          recorded.number("patch_negative_mean_intensity"),
          recorded.number("patch_positive_mean_intensity")))
check("the tissue threshold leaves a plausible head fraction",
      0.05 < recorded.number("patch_brain_fraction_median") < 0.60,
      "median {:.3f} of the volume".format(
          recorded.number("patch_brain_fraction_median")))

check("ambiguous patches were discarded and counted",
      int(recorded.number("patches_ambiguous_discarded")) > 0,
      recorded.get("patches_ambiguous_discarded"))
check("every patient has tumor on more than one slice",
      min(int(r["slices_with_tumor"]) for r in summary) > 1,
      "fewest {}".format(min(int(r["slices_with_tumor"]) for r in summary)))
check("every segment type contributed voxels",
      all(int(recorded.number("segment_voxels_" + key)) > 0
          for key in ("necrosis", "edema", "enhancing_lesion")))

# Where the segmentation sits against the image grid. Stage 4 places one on the
# other with a tolerance on the translation between the two origins, so the
# tolerance has to be shown to stand above everything the cohort holds and by
# the margin config states. The margin is what separates a tolerance set from a
# measurement from one set to make an error go away: a residual that grew
# toward the tolerance would fail here while stage 4 still ran. The direction
# vectors and the spacings are held to a tighter bound of their own, because
# the single argument highdicom accepts would otherwise let a segmentation
# 11.5 degrees out of plane pass as aligned.
check("the recorded translation tolerance is the configured tolerance",
      recorded.number("geometry_translation_tolerance")
      == config.GEOMETRY_TRANSLATION_TOLERANCE,
      "{:.3e}".format(config.GEOMETRY_TRANSLATION_TOLERANCE))
check("the recorded direction tolerance is the configured tolerance",
      recorded.number("geometry_direction_tolerance")
      == config.GEOMETRY_DIRECTION_TOLERANCE,
      "{:.3e}".format(config.GEOMETRY_DIRECTION_TOLERANCE))
check("every pair was measured",
      int(recorded.number("geometry_pairs_measured")) == len(summary)
      == int(recorded.number("patches_patients")),
      "{} pairs".format(len(summary)))

pair_translation = [float(r["geometry_translation_residual"]) for r in summary]
pair_displacement = [float(r["geometry_translation_displacement_mm"])
                     for r in summary]
check("no pair exceeds the translation tolerance",
      max(pair_translation) < config.GEOMETRY_TRANSLATION_TOLERANCE,
      "largest residual {:.3e} voxels against a tolerance of {:.3e}".format(
          max(pair_translation), config.GEOMETRY_TRANSLATION_TOLERANCE))
check("the largest translation residual clears the tolerance by the stated margin",
      max(pair_translation) * config.GEOMETRY_TOLERANCE_MARGIN
      < config.GEOMETRY_TRANSLATION_TOLERANCE,
      "largest residual {:.3e}, tolerance {:.3e}, margin of at least {:.0f}".format(
          max(pair_translation), config.GEOMETRY_TRANSLATION_TOLERANCE,
          config.GEOMETRY_TOLERANCE_MARGIN))
check("the recorded maximum matches the per-pair table",
      abs(recorded.number("geometry_max_translation_residual")
          - max(pair_translation)) <= 1e-5 * max(pair_translation),
      "{} against {:.6e}".format(
          recorded.get("geometry_max_translation_residual"),
          max(pair_translation)))

# What the residual costs in millimetres, and the line between rounding and
# disagreement. A residual is rounded away when the mask is placed, and the
# displacement is the furthest that moves the label along any one axis. The
# coarsest ImagePositionPatient in the cohort is written to six significant
# figures, which quantizes a coordinate near 200 mm at 1e-03 mm, and a
# displacement above that quantum is one the two series report and not one
# their decimal strings invented.
check("the largest displacement stays inside the coordinate string quantum",
      max(pair_displacement) < 1e-3,
      "{:.3e} mm against a quantum of 1.000e-03 mm".format(
          max(pair_displacement)))
check("the recorded displacement matches the per-pair table",
      abs(recorded.number("geometry_max_translation_displacement_mm")
          - max(pair_displacement)) <= 1e-5 * max(pair_displacement),
      "{} mm".format(recorded.get("geometry_max_translation_displacement_mm")))
check("the tolerance stays far inside the half voxel that would move the label",
      config.GEOMETRY_TRANSLATION_TOLERANCE < 0.5 / 20,
      "{:.3e} voxels against half a voxel".format(
          config.GEOMETRY_TRANSLATION_TOLERANCE))

# The direction vectors and the spacings, which the loosened tolerance must not
# also loosen.
for column, label in (("geometry_direction_defect", "direction vector defect"),
                      ("geometry_scale_residual", "spacing ratio residual")):
    values = [float(r[column]) for r in summary]
    check("no pair exceeds the direction tolerance in {}".format(label),
          max(values) < config.GEOMETRY_DIRECTION_TOLERANCE,
          "largest {:.3e} against a tolerance of {:.3e}".format(
              max(values), config.GEOMETRY_DIRECTION_TOLERANCE))
    check("the largest {} clears its tolerance by the stated margin".format(label),
          max(values) * config.GEOMETRY_TOLERANCE_MARGIN
          < config.GEOMETRY_DIRECTION_TOLERANCE,
          "largest {:.3e}, margin of at least {:.0f}".format(
              max(values), config.GEOMETRY_TOLERANCE_MARGIN))
check("the direction tolerance is tighter than the translation tolerance",
      config.GEOMETRY_DIRECTION_TOLERANCE
      < config.GEOMETRY_TRANSLATION_TOLERANCE,
      "{:.3e} against {:.3e}".format(config.GEOMETRY_DIRECTION_TOLERANCE,
                                     config.GEOMETRY_TRANSLATION_TOLERANCE))

# Why the tolerance was needed at all. The count is recorded so that the reason
# the library default does not serve is a number and not an assertion.
above = sum(1 for v in pair_translation if v > config.GEOMETRY_DEFAULT_TOLERANCE)
check("the pairs above the library default were counted",
      above == int(recorded.number("geometry_pairs_above_default_tolerance")),
      "{} of {} pairs exceed {:.3e}".format(
          above, len(summary), config.GEOMETRY_DEFAULT_TOLERANCE))
check("the worst pair is the one recorded",
      max(summary, key=lambda r: float(r["geometry_translation_residual"]))
      ["patient_id"] == recorded.get("geometry_worst_pair"),
      recorded.get("geometry_worst_pair"))

# The residuals also reach results/qc_findings.csv, which is the table a reader
# opens to see what quality control found. Stage 3 writes it from the image
# headers and stage 4 adds these rows, so what stage 4 added is checked against
# the per-pair measurements it added them from.
findings = table("qc_findings")
geometry_rows = [f for f in findings
                 if f["check"].startswith("segmentation_geometry")]
check("the geometry finding was recorded",
      {"segmentation_geometry_residual", "segmentation_geometry_by_scanner",
       "segmentation_geometry_consistency"}
      <= {f["check"] for f in geometry_rows},
      ", ".join(sorted({f["check"] for f in geometry_rows})))
reported = {f["subject"] for f in geometry_rows
            if f["check"] == "segmentation_geometry_residual"}
exceeding = {r["patient_id"] for r in summary
             if float(r["geometry_translation_residual"])
             > config.GEOMETRY_DEFAULT_TOLERANCE}
check("every pair above the library default was reported on its own",
      reported == exceeding,
      "{} of {} pairs: {}".format(len(exceeding), len(summary),
                                  ", ".join(sorted(exceeding))))
check("every pair above the library default is flagged",
      all(int(f["flagged"]) for f in geometry_rows
          if f["check"] == "segmentation_geometry_residual"),
      "{} findings".format(len(reported)))
scanner_rows = [f for f in geometry_rows
                if f["check"] == "segmentation_geometry_by_scanner"]
check("the scanner rows account for every pair",
      sum(int(f["detail"].split(" of ")[1].split(" ")[0])
          for f in scanner_rows) == len(summary),
      "{} scanner models across {} pairs".format(len(scanner_rows), len(summary)))
check("a scanner row is flagged when and only when one of its pairs exceeds "
      "the library default",
      all(int(f["flagged"])
          == (int(f["detail"].split(" of ")[0]) > 0) for f in scanner_rows),
      ", ".join(sorted(f["subject"] for f in scanner_rows if int(f["flagged"]))))
check("the cohort row is flagged on what it measured",
      all(int(f["flagged"]) == (len(exceeding) > 0) for f in geometry_rows
          if f["check"] == "segmentation_geometry_consistency"),
      "{} pairs exceed the library default".format(len(exceeding)))

# Patch coordinates must lie inside the volume they came from.
check("patch coordinates are non-negative", int(index.min()) >= 0)
check("patch coordinates leave room for the patch",
      int(index[:, 1].max()) + config.PATCH_SIZE <= 512
      and int(index[:, 2].max()) + config.PATCH_SIZE <= 512)

finish()
