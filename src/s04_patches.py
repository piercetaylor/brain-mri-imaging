"""Stage 4: turn each labeled volume into 32 by 32 image patches.

For every patient the post-contrast T1 volume is paired with the segmentation
selected in stage 1, which carries at least one segment a person reviewed but
need not have been reviewed throughout. The share of the positive label that
rests on a reviewed segment is measured here and recorded per patient in
``results/patch_summary.csv``. The segmentation is stored on its own grid,
whose in-plane orientation is the negative of the image orientation, so the
mask is resampled onto the image geometry through the patient coordinate system
before any pixel is read. The overlap is checked afterwards and recorded.

That placement only succeeds when the two grids agree, and the two series write
their geometry as decimal strings that round differently. The residual the
translation leaves against a whole number of voxels is measured for every pair
and recorded in ``results/patch_summary.csv``. The largest is 1.736e-03 voxels,
and that pair's mask would move 8.138e-04 mm along one axis; ``config`` carries
the tolerance and the measurement it was set from. The five pairs whose
residual exceeds the 1e-05 the library defaults to are added to
``results/qc_findings.csv``, beside the orientation findings stage 3 records,
together with a count for each of the six scanner models in the cohort.

A patch is positive when at least ``POSITIVE_TUMOR_FRACTION`` of its pixels lie
inside the union of the necrosis, edema and enhancing-lesion segments. A patch
is negative when none of its pixels do and at least
``NEGATIVE_MIN_BRAIN_FRACTION`` of them are brain, which stops the classifier
from separating tissue from surrounding air. Patches between the two are
ambiguous and are discarded.
"""

from __future__ import annotations

import csv
import sys

import highdicom as hd
import numpy as np
import pydicom

from . import config
from .s99_utils import Metrics, banner, otsu_threshold, write_table

INTERIM = config.DATA / "interim"
PATCH_FILE = INTERIM / "patches.npz"
PATCH_TABLE = config.RESULTS / "patch_summary.csv"
QC_TABLE = config.RESULTS / "qc_findings.csv"

# The check names stage 4 contributes to results/qc_findings.csv. Stage 3
# writes that table and cannot carry these findings: the residual is a property
# of an image and segmentation pair, and stage 3 reads neither the segmentation
# series nor ImagePositionPatient, which is not among the tags stage 2
# extracts. The rows are keyed by these names so that a repeated stage 4
# replaces its own findings and leaves the stage 3 rows untouched.
GEOMETRY_CHECKS = (
    "segmentation_geometry_residual",
    "segmentation_geometry_by_scanner",
    "segmentation_geometry_consistency",
)


def read_manifest():
    with open(config.MANIFEST, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def series_directory(patient_id, role, series_uid):
    return config.RAW / patient_id / role / series_uid


def labeled_pairs():
    """One (patient, image series, segmentation series) triple per patient."""
    rows = read_manifest()
    images = {r["series_uid"]: r for r in rows if r["role"] == "image"}
    pairs = []
    for row in rows:
        if row["role"] != "segmentation":
            continue
        image = images.get(row["labels_series_uid"])
        if image is None:
            raise SystemExit(
                "the manifest lists a segmentation of series "
                + row["labels_series_uid"] + " but not the series itself"
            )
        pairs.append((row["patient_id"], image, row))
    return sorted(pairs, key=lambda p: p[0])


def load_volume(patient_id, image_row):
    """The image series of one patient, assembled into a volume.

    ``orientation_tol`` is supplied because five of the 35 slices of
    UPENN-GBM-00452, the GE MEDICAL SYSTEMS DISCOVERY MR750w patient, carry
    ImageOrientationPatient values that describe the same plane to the six
    significant figures the DICOM decimal string holds; stage 3 measures the
    disagreement at 7.77e-15 and ``config.ORIENTATION_TOLERANCE`` carries the
    rest of that measurement. Left unset, the argument makes the comparison an
    equality test on the stored text, which that series fails. The check itself
    is kept: a series whose slices point in genuinely different directions
    exceeds the tolerance and raises, as it should.
    """
    directory = series_directory(patient_id, "image", image_row["series_uid"])
    datasets = [pydicom.dcmread(path) for path in sorted(directory.glob("*.dcm"))]
    return hd.get_volume_from_series(
        datasets, orientation_tol=config.ORIENTATION_TOLERANCE)


def geometry_residuals(volume, mask_volume):
    """The three quantities ``Volume.match_geometry`` holds to its tolerance.

    highdicom pairs each direction vector of the image with the direction
    vector of the segmentation it is closest to, compares the two by their dot
    product, compares the two spacings by the ratio between them, and compares
    the two origins by the remainder the translation leaves in voxels. One
    tolerance covers all three. The three are measured separately here so that
    the translation can be given the tolerance the cohort needs while the other
    two keep a bound 0.81 degrees wide; ``config`` carries both numbers and
    what was measured to arrive at them.

    The residual and the displacement are each maximized over the three axes
    independently, so the two need not fall on the same axis. The displacement
    is the furthest the segmentation would move along any one axis, and that is
    what the quantum of the coordinate strings bounds.
    """
    permutation = []
    direction_defect = 0.0
    scale_residual = 0.0
    for target, target_spacing in zip(volume.unit_vectors(), volume.spacing):
        best = None
        for axis, (vector, spacing) in enumerate(
                zip(mask_volume.unit_vectors(), mask_volume.spacing)):
            dot = float(target @ vector)
            defect = min(abs(dot - 1.0), abs(dot + 1.0))
            if best is None or defect < best[0]:
                best = (defect, axis, target_spacing / spacing)
        defect, axis, scale = best
        permutation.append(axis)
        direction_defect = max(direction_defect, defect)
        scale_residual = max(scale_residual, abs(scale - round(scale)))

    # Two image directions can be closest to one mask axis, which leaves
    # permutation without one of the three. Orthonormality bounds the dot
    # product of two such directions onto one axis at 1/sqrt(2), so
    # direction_defect is then at least 0.293 against a tolerance of 1e-04 and
    # the caller raises on the defect. The residual is returned as infinite so
    # that the caller reports that cause, and not the ValueError
    # permute_spatial_axes raises on a permutation missing an axis.
    if sorted(permutation) != [0, 1, 2]:
        return {
            "geometry_direction_defect": direction_defect,
            "geometry_scale_residual": scale_residual,
            "geometry_translation_residual": float("inf"),
            "geometry_translation_displacement_mm": float("inf"),
        }

    placed = (mask_volume if permutation == [0, 1, 2]
              else mask_volume.permute_spatial_axes(permutation))
    origin_offset = np.array(volume.position) - np.array(placed.position)
    translation = 0.0
    displacement = 0.0
    for vector, spacing in zip(placed.unit_vectors(), placed.spacing):
        start = float(vector @ origin_offset) / spacing
        residual = abs(start - round(start))
        translation = max(translation, residual)
        displacement = max(displacement, residual * spacing)
    return {
        "geometry_direction_defect": direction_defect,
        "geometry_scale_residual": scale_residual,
        "geometry_translation_residual": translation,
        "geometry_translation_displacement_mm": displacement,
    }


def load_mask(patient_id, seg_row, volume):
    """Whole-tumor mask resampled onto the image grid, with its provenance.

    The cohort filter requires that a segmentation series carry at least one
    SEMIAUTOMATIC segment, which is the algorithm type DICOM records for a
    contour a person reviewed. It does not require that of every segment, and
    the necrosis, edema and enhancing-lesion segments are unioned here whatever
    their individual type. The algorithm type of each tumor segment is
    therefore read separately and its voxels attributed to it, so that the
    share of the positive label nobody reviewed is a measured number.

    A voxel can lie inside more than one segment, so the shares are computed on
    the union and not by adding per-segment counts: a voxel counts as reviewed
    when at least one SEMIAUTOMATIC segment covers it, and as unreviewed when
    only segments of another type do. The two shares partition the mask and sum
    to one.
    """
    directory = series_directory(patient_id, "segmentation", seg_row["series_uid"])
    segmentation = hd.seg.segread(sorted(directory.glob("*.dcm"))[0])
    numbers, labels, algorithms = [], [], []
    for number in range(1, segmentation.number_of_segments + 1):
        description = segmentation.get_segment_description(number)
        label = str(description.SegmentedPropertyTypeCodeSequence[0].CodeMeaning)
        if label in config.TUMOR_SEGMENT_LABELS:
            numbers.append(number)
            labels.append(label)
            algorithms.append(
                str(getattr(description, "SegmentAlgorithmType", "") or "UNKNOWN"))
    if not numbers:
        raise SystemExit("no tumor segments in " + seg_row["series_uid"])
    mask_volume = segmentation.get_volume(
        combine_segments=False, segment_numbers=numbers, allow_missing_positions=True)
    geometry = geometry_residuals(volume, mask_volume)
    for key, tolerance in (
            ("geometry_direction_defect", config.GEOMETRY_DIRECTION_TOLERANCE),
            ("geometry_scale_residual", config.GEOMETRY_DIRECTION_TOLERANCE),
            ("geometry_translation_residual",
             config.GEOMETRY_TRANSLATION_TOLERANCE)):
        if geometry[key] > tolerance:
            raise SystemExit(
                "{} of {} is {:.3e} against a tolerance of {:.3e}. The label "
                "and the image describe different geometry and the mask would "
                "be placed on tissue it does not describe.".format(
                    key, patient_id, geometry[key], tolerance))
    mask_volume = mask_volume.match_geometry(
        volume, tol=config.GEOMETRY_TRANSLATION_TOLERANCE)
    array = mask_volume.array
    per_segment = {
        label: int(array[..., index].sum()) for index, label in enumerate(labels)
    }

    reviewed = np.zeros(array.shape[:-1], dtype=bool)
    for index, algorithm in enumerate(algorithms):
        if algorithm == config.CORRECTED_ALGORITHM_TYPE:
            reviewed |= array[..., index] > 0
    mask = array.sum(axis=-1) > 0
    tumor_voxels = int(mask.sum())
    reviewed_voxels = int((mask & reviewed).sum())
    provenance = dict(geometry)
    provenance.update({
        "segment_algorithm_types": ";".join(
            "{}={}".format(label, algorithm)
            for label, algorithm in zip(labels, algorithms)
        ),
        # Counted from the segment list and not from per_segment, which is
        # keyed by label and would collapse two segments sharing a label.
        "segments_total": len(numbers),
        "segments_reviewed": sum(
            1 for a in algorithms if a == config.CORRECTED_ALGORITHM_TYPE),
        "segments_unreviewed": sum(
            1 for a in algorithms if a != config.CORRECTED_ALGORITHM_TYPE),
        "tumor_voxels_reviewed": reviewed_voxels,
        "tumor_voxels_unreviewed": tumor_voxels - reviewed_voxels,
        "reviewed_voxel_share": (
            reviewed_voxels / tumor_voxels if tumor_voxels else 0.0),
        "unreviewed_voxel_share": (
            1.0 - reviewed_voxels / tumor_voxels if tumor_voxels else 0.0),
    })
    return mask, per_segment, provenance


def patch_grid(shape, size, stride):
    rows = range(0, shape[0] - size + 1, stride)
    cols = range(0, shape[1] - size + 1, stride)
    return [(r, c) for r in rows for c in cols]


def extract_patient(patient_id, volume, mask, rng):
    image = volume.array.astype(np.float32)
    # Head tissue is separated from air by Otsu's threshold on the volume. A
    # test for a non-zero value does not work here: reconstruction noise fills
    # the air compartment, so nearly all voxels are non-zero and a patch of
    # empty space would qualify as a negative example. The classifier would
    # then be separating tissue from air, and the score would not mean what it
    # appears to mean.
    threshold = otsu_threshold(image)
    brain = image > threshold
    # Magnetic resonance intensities carry no absolute unit, so a value that
    # means gray matter in one patient can mean white matter in another. Each
    # volume is divided by its own 99.5th percentile of tissue intensity and
    # clipped to the unit interval. The scale uses no label, so it does not
    # leak across the split boundary.
    scale = float(np.percentile(image[brain], 99.5)) if brain.any() else 1.0
    image = np.clip(image / max(scale, 1e-6), 0.0, 1.0)
    size = config.PATCH_SIZE
    positions = patch_grid(image.shape[1:], size, config.PATCH_STRIDE)

    positives, negatives = [], []
    slices_used = 0
    ambiguous = 0
    for z in range(image.shape[0]):
        if brain[z].mean() < config.SLICE_MIN_BRAIN_FRACTION:
            continue
        slices_used += 1
        tumor_slice = mask[z]
        brain_slice = brain[z]
        if tumor_slice.any():
            for r, c in positions:
                fraction = tumor_slice[r:r + size, c:c + size].mean()
                if fraction >= config.POSITIVE_TUMOR_FRACTION:
                    positives.append((z, r, c))
                elif fraction > 0:
                    ambiguous += 1
                elif brain_slice[r:r + size, c:c + size].mean() >= \
                        config.NEGATIVE_MIN_BRAIN_FRACTION:
                    negatives.append((z, r, c))
        else:
            for r, c in positions:
                if brain_slice[r:r + size, c:c + size].mean() >= \
                        config.NEGATIVE_MIN_BRAIN_FRACTION:
                    negatives.append((z, r, c))

    candidate_positives = len(positives)
    candidate_negatives = len(negatives)
    natural_positive_rate = (
        candidate_positives / (candidate_positives + candidate_negatives)
        if candidate_positives or candidate_negatives else 0.0
    )

    if len(positives) > config.MAX_POSITIVE_PATCHES_PER_PATIENT:
        keep = rng.choice(len(positives), config.MAX_POSITIVE_PATCHES_PER_PATIENT,
                          replace=False)
        positives = [positives[i] for i in sorted(keep)]
    wanted = len(positives) * config.NEGATIVES_PER_POSITIVE
    if len(negatives) > wanted:
        keep = rng.choice(len(negatives), wanted, replace=False)
        negatives = [negatives[i] for i in sorted(keep)]

    coordinates = [(p, 1) for p in positives] + [(n, 0) for n in negatives]
    patches = np.empty((len(coordinates), size, size), dtype=np.float32)
    labels = np.empty(len(coordinates), dtype=np.int64)
    index = np.empty((len(coordinates), 3), dtype=np.int32)
    for i, ((z, r, c), label) in enumerate(coordinates):
        patches[i] = image[z, r:r + size, c:c + size]
        labels[i] = label
        index[i] = (z, r, c)

    summary = {
        "patient_id": patient_id,
        "slices_total": int(image.shape[0]),
        "slices_used": slices_used,
        "slices_with_tumor": int((mask.reshape(image.shape[0], -1).sum(1) > 0).sum()),
        "tumor_voxels": int(mask.sum()),
        "brain_voxels": int(brain.sum()),
        "tumor_fraction_of_brain": float(mask.sum() / max(brain.sum(), 1)),
        "candidate_positive_patches": candidate_positives,
        "candidate_negative_patches": candidate_negatives,
        "ambiguous_patches_discarded": ambiguous,
        "natural_positive_rate": natural_positive_rate,
        "patches_positive": int(labels.sum()),
        "patches_negative": int((labels == 0).sum()),
        "intensity_scale": scale,
        "otsu_threshold": threshold,
        "brain_fraction_of_volume": float(brain.mean()),
        "negative_patch_mean_intensity": float(
            patches[labels == 0].mean()) if (labels == 0).any() else 0.0,
        "positive_patch_mean_intensity": float(
            patches[labels == 1].mean()) if (labels == 1).any() else 0.0,
    }
    return patches, labels, index, summary


def geometry_findings(summaries, models):
    """The geometry residuals, in the finding form stage 3 writes.

    Three kinds of row are produced. One row per pair whose translation
    residual exceeds ``config.GEOMETRY_DEFAULT_TOLERANCE`` names the pair that
    the library default would have stopped, because that count is the reason a
    tolerance is passed at all. One row per scanner model reports how many of
    its pairs exceed that default, so that the residual is attributed to the
    equipment it came from and not to the cohort as a whole. One cohort row
    carries the largest residual and the furthest any mask would move along one
    axis. A per-pair row is written only for a pair above the default, and every
    such row is flagged.
    The scanner and cohort rows are flagged when at least one pair they cover
    exceeds it, which is how stage 3 flags its own cohort row.

    ``models`` maps a patient identifier to the manufacturer and model of its
    image series, which the manifest records.
    """
    findings = []
    ordered = sorted(summaries,
                     key=lambda s: -s["geometry_translation_residual"])
    for summary in ordered:
        if (summary["geometry_translation_residual"]
                <= config.GEOMETRY_DEFAULT_TOLERANCE):
            continue
        findings.append({
            "check": "segmentation_geometry_residual",
            "subject": summary["patient_id"],
            "detail": "on {} the translation between the image origin and the "
                      "segmentation origin leaves {:.3e} voxels against a whole "
                      "number and would move the mask {:.3e} mm, each the "
                      "largest of the three axes and not necessarily the same "
                      "axis, above the {:.3e} library default and against a "
                      "tolerance of {:.3e}".format(
                          models[summary["patient_id"]],
                          summary["geometry_translation_residual"],
                          summary["geometry_translation_displacement_mm"],
                          config.GEOMETRY_DEFAULT_TOLERANCE,
                          config.GEOMETRY_TRANSLATION_TOLERANCE),
            "flagged": 1,
        })

    by_model = {}
    for summary in summaries:
        by_model.setdefault(models[summary["patient_id"]], []).append(summary)
    for model, group in sorted(by_model.items()):
        above = [s for s in group
                 if s["geometry_translation_residual"]
                 > config.GEOMETRY_DEFAULT_TOLERANCE]
        findings.append({
            "check": "segmentation_geometry_by_scanner",
            "subject": model,
            "detail": "{} of {} pairs exceed the {:.3e} library default, "
                      "largest {:.3e} voxels".format(
                          len(above), len(group),
                          config.GEOMETRY_DEFAULT_TOLERANCE,
                          max(s["geometry_translation_residual"] for s in group)),
            "flagged": int(bool(above)),
        })

    worst = ordered[0]
    exceeded = sum(1 for s in summaries
                   if s["geometry_translation_residual"]
                   > config.GEOMETRY_DEFAULT_TOLERANCE)
    findings.append({
        "check": "segmentation_geometry_consistency",
        "subject": "cohort",
        "detail": "{} of {} image and segmentation pairs leave a translation "
                  "residual above the {:.3e} library default, largest {:.3e} "
                  "voxels on {}, whose mask would move {:.3e} mm, each figure "
                  "the largest of the three axes, against a tolerance of "
                  "{:.3e}".format(
                      exceeded, len(summaries),
                      config.GEOMETRY_DEFAULT_TOLERANCE,
                      worst["geometry_translation_residual"],
                      worst["patient_id"],
                      worst["geometry_translation_displacement_mm"],
                      config.GEOMETRY_TRANSLATION_TOLERANCE),
        "flagged": int(exceeded > 0),
    })
    return findings


def qc_findings_columns():
    """The columns of the table stage 3 wrote, read before stage 4 does its work.

    Stage 4 adds its rows at the end of an 80.8 second run that has already
    written ``patches.npz`` and ``results/patch_summary.csv``, so all three
    conditions that would stop it there are tested first. An interrupted stage
    3 leaves a table that exists and carries no header, so the header is tested
    and not the file alone.
    """
    if not QC_TABLE.exists():
        raise SystemExit(
            "results/qc_findings.csv is absent. Stage 03 writes it and stage 04 "
            "adds to it, so stage 03 has to run first.")
    with open(QC_TABLE, newline="", encoding="utf-8") as handle:
        columns = csv.DictReader(handle).fieldnames
    if not columns:
        raise SystemExit(
            "results/qc_findings.csv carries no header. Stage 03 was "
            "interrupted before it wrote one. Re-run stage 03.")
    if set(columns) != {"check", "subject", "detail", "flagged"}:
        raise SystemExit(
            "results/qc_findings.csv carries the columns {} where stage 04 "
            "writes check, subject, detail and flagged. Re-run stage 03.".format(
                ", ".join(columns)))
    return list(columns)


def append_geometry_findings(findings, columns):
    """Add the geometry findings to the table stage 3 wrote, and return it.

    Stage 3 writes ``results/qc_findings.csv`` from the image headers and stage
    4 runs after it, so the table is read back and rewritten with these rows
    appended. Rows carrying one of :data:`GEOMETRY_CHECKS` are dropped first,
    which makes a second stage 4 over the same cohort leave the table it found.
    ``columns`` comes from :func:`qc_findings_columns`, which stage 4 calls
    before it reads the first image.
    """
    with open(QC_TABLE, newline="", encoding="utf-8") as handle:
        kept = [row for row in csv.DictReader(handle)
                if row["check"] not in GEOMETRY_CHECKS]
    rows = kept + [{column: finding[column] for column in columns}
                   for finding in findings]
    write_table(rows, QC_TABLE, columns)
    return rows


def build():
    banner("stage 04 patches")
    qc_columns = qc_findings_columns()
    pairs = labeled_pairs()
    rng = np.random.default_rng(config.SEED)

    all_patches, all_labels, all_index, all_patients, summaries = [], [], [], [], []
    segment_totals = {label: 0 for label in config.TUMOR_SEGMENT_LABELS}
    for position, (patient_id, image_row, seg_row) in enumerate(pairs, start=1):
        volume = load_volume(patient_id, image_row)
        mask, per_segment, provenance = load_mask(patient_id, seg_row, volume)
        for label, count in per_segment.items():
            segment_totals[label] += count
        patches, labels, index, summary = extract_patient(
            patient_id, volume, mask, np.random.default_rng(config.SEED + position)
        )
        summary["segments_present"] = len(per_segment)
        summary["algorithm_type"] = seg_row["algorithm_type"]
        summary.update(provenance)
        all_patches.append(patches)
        all_labels.append(labels)
        all_index.append(index)
        all_patients.append(np.full(len(labels), patient_id))
        summaries.append(summary)
        print("  [{:>2}/{}] {} {:>5} patches, {:>4} positive".format(
            position, len(pairs), patient_id, len(labels), int(labels.sum())),
            flush=True)

    X = np.concatenate(all_patches)
    y = np.concatenate(all_labels)
    index = np.concatenate(all_index)
    patients = np.concatenate(all_patients)

    INTERIM.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PATCH_FILE, X=X, y=y, index=index, patients=patients)
    write_table(summaries, PATCH_TABLE)

    metrics = Metrics()
    metrics.update({
        "patch_size": config.PATCH_SIZE,
        "patch_stride": config.PATCH_STRIDE,
        "patch_positive_tumor_fraction": config.POSITIVE_TUMOR_FRACTION,
        "patches_total": int(len(y)),
        "patches_positive": int(y.sum()),
        "patches_negative": int((y == 0).sum()),
        # The rate the sampling produces and not a property of the tumors.
        # config.SAMPLED_POSITIVE_RATE fixes it; the measured rate before
        # sampling is patches_natural_positive_rate_median below.
        "patches_positive_rate_fixed_by_sampling": float(y.mean()),
        "patches_patients": len(pairs),
        "patches_min_per_patient": int(min(len(a) for a in all_labels)),
        "patches_max_per_patient": int(max(len(a) for a in all_labels)),
        "patches_ambiguous_discarded": sum(
            s["ambiguous_patches_discarded"] for s in summaries),
        "patches_natural_positive_rate_median": float(np.median(
            [s["natural_positive_rate"] for s in summaries])),
        "patches_tumor_fraction_of_brain_median": float(np.median(
            [s["tumor_fraction_of_brain"] for s in summaries])),
        "patch_pixel_min": float(X.min()),
        "patch_pixel_max": float(X.max()),
        "patch_intensity_scale_min": float(min(s["intensity_scale"] for s in summaries)),
        "patch_intensity_scale_max": float(max(s["intensity_scale"] for s in summaries)),
        "patch_brain_fraction_median": float(np.median(
            [s["brain_fraction_of_volume"] for s in summaries])),
        "patch_negative_mean_intensity": float(X[y == 0].mean()),
        "patch_positive_mean_intensity": float(X[y == 1].mean()),
        "patch_negative_share_near_empty": float(
            (X[y == 0].reshape((y == 0).sum(), -1).mean(1) < 0.05).mean()),
        "segment_voxels_necrosis": segment_totals["Necrosis"],
        "segment_voxels_edema": segment_totals["Edema"],
        "segment_voxels_enhancing_lesion": segment_totals["Enhancing Lesion"],
    })

    # Where the segmentation sits against the image grid. highdicom will not
    # place one on the other until the direction vectors, the spacings and the
    # translation between the two origins all agree to within its tolerance,
    # and five of the 49 pairs exceed the 1e-5 that tolerance defaults to. The
    # residuals are recorded per pair in results/patch_summary.csv and
    # summarized here, so the tolerance config carries can be read against what
    # the cohort holds. The residual and the displacement are each the largest
    # over the three axes and need not fall on the same one, so the
    # displacement is the furthest the mask would move along any one axis were
    # the residual a position the two series disagree on.
    translation = [s["geometry_translation_residual"] for s in summaries]
    displacement = [s["geometry_translation_displacement_mm"] for s in summaries]
    metrics.update({
        "geometry_pairs_measured": len(summaries),
        "geometry_translation_tolerance": config.GEOMETRY_TRANSLATION_TOLERANCE,
        "geometry_direction_tolerance": config.GEOMETRY_DIRECTION_TOLERANCE,
        "geometry_default_tolerance": config.GEOMETRY_DEFAULT_TOLERANCE,
        "geometry_max_translation_residual": float(max(translation)),
        "geometry_median_translation_residual": float(np.median(translation)),
        "geometry_max_translation_displacement_mm": float(max(displacement)),
        "geometry_max_direction_defect": float(max(
            s["geometry_direction_defect"] for s in summaries)),
        "geometry_max_scale_residual": float(max(
            s["geometry_scale_residual"] for s in summaries)),
        "geometry_pairs_above_default_tolerance": sum(
            1 for v in translation if v > config.GEOMETRY_DEFAULT_TOLERANCE),
        "geometry_worst_pair": max(
            summaries, key=lambda s: s["geometry_translation_residual"])["patient_id"],
    })

    # The same residuals as findings, so that a reader who opens the quality
    # control table sees them beside the orientation findings stage 3 recorded
    # from the image headers. Stage 3 counts its own rows into
    # qc_findings_total and qc_findings_flagged; those two are restated here
    # over the table stage 4 leaves behind, so that the counts continue to
    # describe the whole file and not the part of it one stage wrote.
    models = {patient_id: image_row["manufacturer"] + " " + image_row["model_name"]
              for patient_id, image_row, _ in pairs}
    findings = append_geometry_findings(
        geometry_findings(summaries, models), qc_columns)
    metrics.update({
        "qc_findings_total": len(findings),
        "qc_findings_flagged": sum(int(row["flagged"]) for row in findings),
    })

    # How much of the positive label a person actually reviewed. The cohort
    # filter accepts a segmentation series when any one of its segments is
    # SEMIAUTOMATIC, so a positive patch can rest on a segment nobody looked
    # at. These are the numbers that say how often that happens and by how
    # much; a claim that the labels are radiologist-corrected is only as strong
    # as label_reviewed_voxel_share.
    tumor_total = sum(s["tumor_voxels"] for s in summaries)
    reviewed_total = sum(s["tumor_voxels_reviewed"] for s in summaries)
    shares = [s["reviewed_voxel_share"] for s in summaries]
    metrics.update({
        "label_tumor_voxels": tumor_total,
        "label_tumor_voxels_reviewed": reviewed_total,
        "label_tumor_voxels_unreviewed": tumor_total - reviewed_total,
        "label_reviewed_voxel_share": float(reviewed_total / max(tumor_total, 1)),
        "label_unreviewed_voxel_share": float(
            1.0 - reviewed_total / max(tumor_total, 1)),
        "label_reviewed_voxel_share_min": float(min(shares)),
        "label_reviewed_voxel_share_median": float(np.median(shares)),
        "label_reviewed_voxel_share_max": float(max(shares)),
        "label_patients_fully_reviewed": sum(1 for v in shares if v >= 1.0),
        "label_patients_with_unreviewed_voxels": sum(1 for v in shares if v < 1.0),
        "label_segments_reviewed": sum(s["segments_reviewed"] for s in summaries),
        "label_segments_unreviewed": sum(s["segments_unreviewed"] for s in summaries),
    })
    metrics.save()
    print("{:.1f} percent of positive tumor voxels come from a reviewed segment; "
          "{} of {} patients carry unreviewed voxels".format(
              100 * reviewed_total / max(tumor_total, 1),
              sum(1 for v in shares if v < 1.0), len(shares)))
    print("largest translation residual {:.3e} voxels on {}, and the furthest "
          "any mask would move along one axis {:.3e} mm, against a tolerance "
          "of {:.3e}; {} of {} pairs exceed {:.3e}".format(
              max(translation),
              metrics.get("geometry_worst_pair"), max(displacement),
              config.GEOMETRY_TRANSLATION_TOLERANCE,
              metrics.get("geometry_pairs_above_default_tolerance"),
              len(summaries), config.GEOMETRY_DEFAULT_TOLERANCE))
    print("{} patches from {} patients, {:.1f} percent positive".format(
        len(y), len(pairs), 100 * y.mean()))
    return X, y, patients


def load_patches():
    if not PATCH_FILE.exists():
        raise SystemExit("data/interim/patches.npz is absent. Run stage 04.")
    data = np.load(PATCH_FILE, allow_pickle=False)
    return data["X"], data["y"], data["patients"], data["index"]


if __name__ == "__main__":
    build()
    sys.exit(0)
