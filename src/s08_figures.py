"""Stage 8: the figures.

Each figure is drawn from a table in ``results/`` or from the patch array, so
that a figure and the prose beside it cannot disagree.
"""

from __future__ import annotations

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config
from .s04_patches import load_patches, labeled_pairs, load_mask, load_volume
from .s05_splits import load_splits
from .s06_model import GRID_SELECTION_CRITERION, load_model, predict
from .s99_utils import Metrics, banner

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE = "#2f6f9f"
ORANGE = "#c1610a"
GRAY = "#7a7a7a"


def table(name):
    with open(config.RESULTS / (name + ".csv"), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(figure, name):
    path = config.FIGURES / name
    figure.savefig(path)
    plt.close(figure)
    print("  " + name)
    return path


def fig_scanners():
    rows = table("scanner_inventory")
    labels = [r["manufacturer"] + "\n" + r["model_name"] for r in rows]
    counts = [int(r["series"]) for r in rows]
    patients = [int(r["patients"]) for r in rows]
    figure, axis = plt.subplots(figsize=(7, 3.4))
    positions = np.arange(len(labels))
    axis.bar(positions - 0.2, counts, 0.4, label="series", color=BLUE)
    axis.bar(positions + 0.2, patients, 0.4, label="patients", color=ORANGE)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=7)
    axis.set_ylabel("count")
    axis.set_title("Acquisition equipment in the cohort")
    axis.legend(frameon=False)
    return save(figure, "fig01_scanner_inventory.png")


def fig_series_types():
    rows = table("series_inventory")
    counts = {}
    for row in rows:
        counts[row["series_desc"]] = counts.get(row["series_desc"], 0) + 1
    order = sorted(counts.items(), key=lambda kv: kv[1])
    figure, axis = plt.subplots(figsize=(7, 0.22 * len(order) + 1.2))
    axis.barh([k for k, _ in order], [v for _, v in order], color=BLUE)
    axis.set_xlabel("series")
    axis.set_title("Series descriptions present in the cohort")
    axis.tick_params(axis="y", labelsize=6)
    return save(figure, "fig02_series_descriptions.png")


def fig_grids():
    rows = table("series_inventory")
    grids = {}
    for row in rows:
        key = (int(row["rows_px"]), int(row["cols_px"]))
        grids[key] = grids.get(key, 0) + 1
    per_patient = {}
    for row in rows:
        per_patient.setdefault(row["patient_id"], set()).add(
            (int(row["rows_px"]), int(row["cols_px"])))
    distinct = sorted(len(v) for v in per_patient.values())

    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    labels = ["{}x{}".format(*k) for k in sorted(grids)]
    axes[0].bar(labels, [grids[k] for k in sorted(grids)], color=BLUE)
    axes[0].set_ylabel("series")
    axes[0].set_title("Image grids across the cohort")
    axes[0].tick_params(axis="x", rotation=45, labelsize=7)
    values, counts = np.unique(distinct, return_counts=True)
    axes[1].bar(values, counts, color=ORANGE)
    axes[1].set_xlabel("distinct grids held by one patient")
    axes[1].set_ylabel("patients")
    axes[1].set_xticks(values)
    axes[1].set_title("Grid stability within a patient")
    return save(figure, "fig03_image_grids.png")


def fig_alignment():
    """One slice with its resampled mask, the check that the label lines up."""
    patient_id, image_row, seg_row = labeled_pairs()[0]
    volume = load_volume(patient_id, image_row)
    # load_mask returns the mask, the per-segment voxel counts and the
    # provenance of the label; only the mask is drawn here.
    mask, _, _ = load_mask(patient_id, seg_row, volume)
    image = volume.array.astype(np.float32)
    image = image / max(float(np.percentile(image[image > 0], 99.5)), 1e-6)
    per_slice = mask.reshape(len(image), -1).sum(1)
    z = int(per_slice.argmax())

    figure, axes = plt.subplots(1, 3, figsize=(8.4, 3.2))
    axes[0].imshow(np.clip(image[z], 0, 1), cmap="gray")
    axes[0].set_title("T1 post-contrast, slice {}".format(z))
    axes[1].imshow(mask[z], cmap="gray")
    axes[1].set_title("whole-tumor mask")
    axes[2].imshow(np.clip(image[z], 0, 1), cmap="gray")
    axes[2].imshow(np.ma.masked_where(~mask[z], mask[z]), cmap="autumn", alpha=0.45)
    axes[2].set_title("mask on the image")
    for axis in axes:
        axis.axis("off")
        axis.grid(False)
    figure.suptitle("Segmentation resampled onto the image grid, " + patient_id,
                    fontsize=9)
    return save(figure, "fig04_mask_alignment.png")


def fig_patches():
    X, y, patients, _ = load_patches()
    rng = np.random.default_rng(config.SEED)
    figure, axes = plt.subplots(4, 10, figsize=(8.4, 3.8))
    for row_index, (label, name) in enumerate([(1, "tumor"), (0, "non-tumor")]):
        pool = np.flatnonzero(y == label)
        picked = rng.choice(pool, 20, replace=False)
        for column, chosen in enumerate(picked):
            axis = axes[row_index * 2 + column // 10, column % 10]
            axis.imshow(X[chosen], cmap="gray", vmin=0, vmax=1)
            axis.axis("off")
            axis.grid(False)
            if column == 0:
                axis.set_title(name, loc="left", fontsize=8)
    figure.suptitle("Patches drawn from the labeled volumes", fontsize=9)
    return save(figure, "fig05_patch_examples.png")


def fig_class_balance():
    rows = table("patch_summary")
    natural = [float(r["natural_positive_rate"]) for r in rows]
    total = [int(r["patches_positive"]) + int(r["patches_negative"]) for r in rows]
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].hist(np.array(natural) * 100, bins=15, color=BLUE)
    axes[0].set_xlabel("positive patches before sampling (percent)")
    axes[0].set_ylabel("patients")
    axes[0].set_title("Tumor is a small part of the brain")
    axes[1].bar(range(len(total)), sorted(total), color=ORANGE)
    axes[1].set_xlabel("patient, ordered by patch count")
    axes[1].set_ylabel("patches retained")
    axes[1].set_title("Patches contributed per patient")
    return save(figure, "fig06_class_balance.png")


def fig_splits():
    rows = table("splits")
    order = {"train": 0, "validation": 1, "test": 2}
    rows.sort(key=lambda r: (order[r["split"]], r["patient_id"]))
    colors = {"train": BLUE, "validation": GRAY, "test": ORANGE}
    figure, axis = plt.subplots(figsize=(8, 3.2))
    axis.bar(range(len(rows)), [int(r["patches"]) for r in rows],
             color=[colors[r["split"]] for r in rows])
    axis.set_xlabel("patient")
    axis.set_ylabel("patches")
    axis.set_title("Patients are assigned whole to one part of the split")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in order]
    axis.legend(handles, list(order), frameon=False)
    return save(figure, "fig07_splits.png")


def fig_grid_search():
    rows = table("model_grid")
    dropouts = sorted({float(r["dropout"]) for r in rows})
    rates = sorted({float(r["learning_rate"]) for r in rows})
    figure, axis = plt.subplots(figsize=(5.4, 3.2))
    width = 0.35
    positions = np.arange(len(rates))
    for offset, dropout in enumerate(dropouts):
        values = [
            float(next(r for r in rows
                       if float(r["learning_rate"]) == rate
                       and float(r["dropout"]) == dropout)[GRID_SELECTION_CRITERION])
            for rate in rates
        ]
        axis.bar(positions + (offset - 0.5) * width, values, width,
                 label="dropout {:g}".format(dropout),
                 color=[BLUE, ORANGE][offset % 2])
    axis.set_xticks(positions)
    axis.set_xticklabels(["{:g}".format(r) for r in rates])
    axis.set_xlabel("Adadelta learning rate")
    axis.set_ylabel("validation ROC-AUC")
    axis.set_ylim(0.5, 1.0)
    # The bars are the criterion the configuration was chosen on, which is the
    # score at the epoch each grid run would keep.
    axis.set_title("Grid search, scored on the validation patients")
    axis.legend(frameon=False)
    return save(figure, "fig08_grid_search.png")


def fig_history():
    """Training loss and validation ROC-AUC, with the retained epoch marked.

    The vertical line is the epoch whose weights the reported model uses. It is
    drawn because the curve alone does not say which point on it was kept, and
    the last epoch is not that point.
    """
    rows = table("training_history")
    epochs = [int(r["epoch"]) for r in rows]
    figure, axis = plt.subplots(figsize=(5.6, 3.2))
    axis.plot(epochs, [float(r["train_loss"]) for r in rows], color=BLUE,
              label="training loss")
    axis.set_xlabel("epoch")
    axis.set_ylabel("binary cross-entropy", color=BLUE)
    twin = axis.twinx()
    twin.plot(epochs, [float(r["validation_roc_auc"]) for r in rows], color=ORANGE,
              label="validation ROC-AUC")
    twin.set_ylabel("validation ROC-AUC", color=ORANGE)
    twin.grid(False)
    selected = next(int(r["epoch"]) for r in rows if int(r["is_selected_epoch"]))
    axis.axvline(selected, color=GRAY, linestyle="--", linewidth=1)
    axis.annotate("epoch {} kept".format(selected), (selected, 0.98),
                  xycoords=("data", "axes fraction"), ha="right", va="top",
                  fontsize=8, color=GRAY, rotation=90)
    axis.set_title("Training")
    return save(figure, "fig09_training_history.png")


def fig_roc():
    metrics = Metrics()
    rows = table("roc_test")
    fpr = [float(r["false_positive_rate"]) for r in rows]
    tpr = [float(r["true_positive_rate"]) for r in rows]
    figure, axis = plt.subplots(figsize=(4.4, 4.0))
    # The interval comes from resampling the thirteen test patients, so the label
    # carries the room the point estimate actually has.
    axis.plot(fpr, tpr, color=BLUE,
              label="test, AUC {:.3f}\n{:.0f} percent interval {:.3f} to "
                    "{:.3f}".format(
                        metrics.number("eval_test_roc_auc"),
                        100 * metrics.number("eval_test_roc_auc_bootstrap_interval"),
                        metrics.number("eval_test_roc_auc_bootstrap_low"),
                        metrics.number("eval_test_roc_auc_bootstrap_high")))
    axis.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1)
    axis.set_xlabel("false positive rate")
    axis.set_ylabel("true positive rate")
    axis.set_title("Held-out patients")
    axis.legend(frameon=False, loc="lower right")
    return save(figure, "fig10_roc_test.png")


def fig_confusion():
    rows = [r for r in table("confusion_matrix") if r["split"] == "test"]
    names = ["non-tumor", "tumor"]
    matrix = np.zeros((2, 2), dtype=int)
    for row in rows:
        matrix[names.index(row["actual"]), names.index(row["predicted"])] = int(row["count"])
    figure, axis = plt.subplots(figsize=(4.0, 3.6))
    axis.imshow(matrix, cmap="Blues")
    axis.grid(False)
    for i in range(2):
        for j in range(2):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    axis.set_xticks([0, 1], names)
    axis.set_yticks([0, 1], names)
    axis.set_xlabel("predicted")
    axis.set_ylabel("actual")
    axis.set_title("Test confusion matrix at threshold 0.5")
    return save(figure, "fig11_confusion_matrix.png")


def fig_per_patient():
    rows = [r for r in table("per_patient_test") if r["roc_auc"] != ""]
    rows.sort(key=lambda r: float(r["roc_auc"]))
    figure, axis = plt.subplots(figsize=(7, 3.2))
    axis.bar([r["patient_id"][-5:] for r in rows],
             [float(r["roc_auc"]) for r in rows], color=BLUE)
    axis.axhline(0.5, color=GRAY, linestyle="--", linewidth=1)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("ROC-AUC")
    axis.set_xlabel("test patient")
    axis.tick_params(axis="x", rotation=90, labelsize=6)
    axis.set_title("Performance is not uniform across held-out patients")
    return save(figure, "fig12_per_patient.png")


def fig_leakage():
    """The two partitions at the primary seed, drawn over the range across seeds.

    The bars are the primary-seed scores the text quotes. The points are the
    scores at every seed in ``config.SEED_LIST``, so that the difference
    between the bars is read against the spread the seed alone produces and not
    as a fixed quantity.
    """
    metrics = Metrics()
    honest = metrics.number("leakage_test_roc_auc_patient_split")
    leaky = metrics.number("leakage_test_roc_auc_patch_split")
    variance = table("seed_variance")
    labels = ["split by patient", "split by patch"]
    per_seed = [
        [float(r["test_roc_auc"]) for r in variance if r["split_unit"] == part]
        for part in ("patient", "patch")
    ]
    figure, axis = plt.subplots(figsize=(4.8, 3.4))
    axis.bar(labels, [honest, leaky], color=[BLUE, ORANGE])
    for position, values in enumerate(per_seed):
        axis.plot([position] * len(values), values, "o", color=GRAY,
                  markersize=4, zorder=3)
        axis.text(position, max(values) + 0.015,
                  "{:.3f}".format([honest, leaky][position]), ha="center")
    axis.set_ylim(0.5, 1.05)
    axis.set_ylabel("test ROC-AUC")
    axis.set_title("The same model under two partitions")
    axis.set_xlabel("bars are the primary seed, points are {} seeds".format(
        len(config.SEED_LIST)), fontsize=8)
    return save(figure, "fig13_split_leakage.png")


def fig_errors():
    X, y, patients, _ = load_patches()
    patient_split, _ = load_splits()
    scores = predict(load_model(), X)
    test = patient_split == "test"
    predicted = (scores >= 0.5).astype(int)
    false_positive = np.flatnonzero(test & (predicted == 1) & (y == 0))
    false_negative = np.flatnonzero(test & (predicted == 0) & (y == 1))
    rng = np.random.default_rng(config.SEED)

    figure, axes = plt.subplots(2, 8, figsize=(8.4, 2.6))
    for row_index, (pool, name) in enumerate(
            [(false_positive, "false positive"), (false_negative, "false negative")]):
        picked = rng.choice(pool, min(8, len(pool)), replace=False) if len(pool) else []
        for column in range(8):
            axis = axes[row_index, column]
            axis.axis("off")
            axis.grid(False)
            if column < len(picked):
                axis.imshow(X[picked[column]], cmap="gray", vmin=0, vmax=1)
            if column == 0:
                axis.set_title(name, loc="left", fontsize=8)
    figure.suptitle("Errors on the held-out patients", fontsize=9)
    return save(figure, "fig14_errors.png")


def build():
    banner("stage 08 figures")
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    paths = [
        fig_scanners(), fig_series_types(), fig_grids(), fig_alignment(),
        fig_patches(), fig_class_balance(), fig_splits(), fig_grid_search(),
        fig_history(), fig_roc(), fig_confusion(), fig_per_patient(),
        fig_leakage(), fig_errors(),
    ]
    metrics = Metrics()
    metrics.set("figures_written", len(paths))
    metrics.save()
    return paths


if __name__ == "__main__":
    build()
    sys.exit(0)
