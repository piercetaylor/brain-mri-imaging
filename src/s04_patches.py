"""Stage 4: turn each labeled volume into 32 by 32 image patches.

For every patient the axial T1 post-contrast volume is paired with its
radiologist-corrected segmentation. The segmentation is stored on its own grid,
whose in-plane orientation is the negative of the image orientation, so the
mask is resampled onto the image geometry through the patient coordinate system
before any pixel is read. The overlap is checked afterwards and recorded.

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
    directory = series_directory(patient_id, "image", image_row["series_uid"])
    datasets = [pydicom.dcmread(path) for path in sorted(directory.glob("*.dcm"))]
    return hd.get_volume_from_series(datasets)


def load_mask(patient_id, seg_row, volume):
    """Whole-tumor mask resampled onto the image grid."""
    directory = series_directory(patient_id, "segmentation", seg_row["series_uid"])
    segmentation = hd.seg.segread(sorted(directory.glob("*.dcm"))[0])
    numbers, labels = [], []
    for number in range(1, segmentation.number_of_segments + 1):
        description = segmentation.get_segment_description(number)
        label = str(description.SegmentedPropertyTypeCodeSequence[0].CodeMeaning)
        if label in config.TUMOR_SEGMENT_LABELS:
            numbers.append(number)
            labels.append(label)
    if not numbers:
        raise SystemExit("no tumor segments in " + seg_row["series_uid"])
    mask_volume = segmentation.get_volume(
        combine_segments=False, segment_numbers=numbers, allow_missing_positions=True
    ).match_geometry(volume)
    per_segment = {
        label: int(mask_volume.array[..., index].sum())
        for index, label in enumerate(labels)
    }
    return mask_volume.array.sum(axis=-1) > 0, per_segment


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


def build():
    banner("stage 04 patches")
    pairs = labeled_pairs()
    rng = np.random.default_rng(config.SEED)

    all_patches, all_labels, all_index, all_patients, summaries = [], [], [], [], []
    segment_totals = {label: 0 for label in config.TUMOR_SEGMENT_LABELS}
    for position, (patient_id, image_row, seg_row) in enumerate(pairs, start=1):
        volume = load_volume(patient_id, image_row)
        mask, per_segment = load_mask(patient_id, seg_row, volume)
        for label, count in per_segment.items():
            segment_totals[label] += count
        patches, labels, index, summary = extract_patient(
            patient_id, volume, mask, np.random.default_rng(config.SEED + position)
        )
        summary["segments_present"] = len(per_segment)
        summary["algorithm_type"] = seg_row["algorithm_type"]
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
        "patches_positive_rate": float(y.mean()),
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
    metrics.save()
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
