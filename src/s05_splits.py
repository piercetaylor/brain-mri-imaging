"""Stage 5: partition the patches into training, validation and test sets.

The partition is over patients. Patches drawn from one volume on an eight-pixel
stride overlap one another, so patches from the same patient are near
duplicates. A partition over patches would put copies of the same tissue on
both sides of the boundary and the test score would measure memorisation.

Both partitions are built. The patient-level partition is the one the reported
model uses. The patch-level partition is kept so that the difference between
the two can be measured and reported.
"""

from __future__ import annotations

import csv
import sys

import numpy as np

from . import config
from .s04_patches import PATCH_TABLE, load_patches
from .s99_utils import Metrics, banner, write_table

SPLIT_TABLE = config.RESULTS / "splits.csv"


def natural_positive_rates() -> dict[str, str]:
    """The positive rate each patient carried before the negatives were cut.

    Read from ``results/patch_summary.csv``, which stage 4 wrote, so the split
    table reports the measured rate beside the constructed one without
    recomputing either.
    """
    with open(PATCH_TABLE, newline="", encoding="utf-8") as handle:
        return {row["patient_id"]: row["natural_positive_rate"]
                for row in csv.DictReader(handle)}


def patient_level(patients, y, seed=config.SEED):
    """Assign whole patients to the three parts, then report the balance."""
    unique = np.array(sorted(set(patients.tolist())))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    shuffled = unique[order]

    n = len(shuffled)
    n_train = int(round(config.SPLIT_FRACTIONS["train"] * n))
    n_validation = int(round(config.SPLIT_FRACTIONS["validation"] * n))
    assignment = {}
    for index, patient in enumerate(shuffled):
        if index < n_train:
            assignment[patient] = "train"
        elif index < n_train + n_validation:
            assignment[patient] = "validation"
        else:
            assignment[patient] = "test"
    return np.array([assignment[p] for p in patients]), assignment


def patch_level(y, seed=config.SEED):
    """The partition the patient-level design is meant to avoid."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    n_train = int(round(config.SPLIT_FRACTIONS["train"] * len(y)))
    n_validation = int(round(config.SPLIT_FRACTIONS["validation"] * len(y)))
    labels = np.empty(len(y), dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train:n_train + n_validation]] = "validation"
    labels[order[n_train + n_validation:]] = "test"
    return labels.astype(str)


def build():
    banner("stage 05 splits")
    X, y, patients, _ = load_patches()
    by_patient, assignment = patient_level(patients, y)
    by_patch = patch_level(y)

    natural = natural_positive_rates()
    rows = []
    for patient in sorted(set(patients.tolist())):
        selector = patients == patient
        rows.append({
            "patient_id": patient,
            "split": assignment[patient],
            # Which patients the epoch selection reads. Stage 6 keeps the epoch
            # that maximizes validation ROC-AUC, so the validation patients
            # listed here are the patients that choice was made on.
            config.SELECTION_PARTICIPATION_COLUMN:
                config.selection_participation(assignment[patient]),
            "patches": int(selector.sum()),
            "positive": int(y[selector].sum()),
            "negative": int((y[selector] == 0).sum()),
            # Two positive rates, named apart. The first is what the sampling
            # was built to produce, one positive for every
            # config.NEGATIVES_PER_POSITIVE negatives, and it is 0.333333 on
            # every patient because the ratio fixes it. The second is what the
            # patient's volume actually held, and it runs from 0.000743 to
            # 0.098428 across the cohort. A column named positive_rate alone
            # would read as the second while holding the first.
            "positive_rate_fixed_by_sampling": float(y[selector].mean()),
            "natural_positive_rate": natural[patient],
        })
    write_table(rows, SPLIT_TABLE)

    np.savez_compressed(
        config.DATA / "interim" / "splits.npz",
        patient_split=by_patient, patch_split=by_patch,
    )

    # The property the whole design rests on: no patient appears in more than
    # one part of the patient-level partition.
    overlap = 0
    for part_a in ("train", "validation", "test"):
        for part_b in ("train", "validation", "test"):
            if part_a >= part_b:
                continue
            shared = set(patients[by_patient == part_a]) & set(patients[by_patient == part_b])
            overlap += len(shared)

    patch_level_overlap = len(
        set(patients[by_patch == "train"]) & set(patients[by_patch == "test"]))

    metrics = Metrics()
    updates = {
        "split_seed": config.SEED,
        "split_patient_overlap": overlap,
        "split_patch_level_patient_overlap": patch_level_overlap,
        # The patch-level partition exists to be the comparison, so every one
        # of the 49 patients lying on both sides of it is the design and not a
        # defect. split_patient_overlap is the same quantity on the partition
        # the reported model uses, and it is 0.
        "split_patch_level_patient_overlap_role": (
            "the count the patch-level comparison is built to produce: all {} "
            "patients contribute to both sides of it, while the reported "
            "partition leaves split_patient_overlap at {}".format(
                patch_level_overlap, overlap)),
    }
    for part in ("train", "validation", "test"):
        selector = by_patient == part
        updates["split_" + part + "_patients"] = len(set(patients[selector].tolist()))
        updates["split_" + part + "_patches"] = int(selector.sum())
        updates["split_" + part + "_positive_rate_fixed_by_sampling"] = float(
            y[selector].mean())
    metrics.update(updates)
    metrics.save()
    print("train {} patients / {} patches, validation {} / {}, test {} / {}".format(
        updates["split_train_patients"], updates["split_train_patches"],
        updates["split_validation_patients"], updates["split_validation_patches"],
        updates["split_test_patients"], updates["split_test_patches"]))
    print("patients appearing in more than one part: {}".format(overlap))
    return by_patient, by_patch


def load_splits():
    path = config.DATA / "interim" / "splits.npz"
    if not path.exists():
        raise SystemExit("data/interim/splits.npz is absent. Run stage 05.")
    data = np.load(path, allow_pickle=False)
    return data["patient_split"].astype(str), data["patch_split"].astype(str)


if __name__ == "__main__":
    build()
    sys.exit(0)
