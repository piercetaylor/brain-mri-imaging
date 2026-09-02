"""Stage 7: evaluate the trained network on the held-out patients.

The reported numbers are ROC-AUC, accuracy, per-class precision, recall and F1,
and the confusion matrix, on training and test alike. Errors are also broken
down by patient, because a score pooled over patches hides a model that works
on most patients and fails on a few. The pooled test ROC-AUC is given a
patient-level bootstrap interval, since thirteen test patients leave the point
estimate with more room than one number admits.

The validation rows of both tables are read from the weights of the epoch that
was retained for maximizing validation ROC-AUC, which reached 0.9629 against
the 0.9071 of the last epoch. Both tables therefore carry a column stating, per
row, whether the split took part in that selection, so a validation row cannot
be read as an unbiased estimate by someone who reads the file alone.
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, roc_curve,
)

from . import config
from .s04_patches import load_patches
from .s05_splits import load_splits, natural_positive_rates
from .s06_model import REPORTED_MODEL_VALIDATION_ROLE, load_model, predict
from .s99_utils import Metrics, banner, write_table

CLASS_NAMES = ("non-tumor", "tumor")


def report_rows(part, y_true, y_score, threshold=0.5):
    predicted = (y_score >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=[0, 1], zero_division=0
    )
    rows = []
    for index, name in enumerate(CLASS_NAMES):
        rows.append({
            "split": part,
            "class": name,
            # Every figure on this row is read from the weights of the retained
            # epoch, and on the validation split that epoch was retained for
            # maximizing a score on those same patients. The column says so on
            # the row, so a reader of this file alone cannot take a validation
            # figure for an unbiased estimate.
            config.SELECTION_PARTICIPATION_COLUMN:
                config.selection_participation(part),
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
            "f1": round(float(f1[index]), 4),
            "support": int(support[index]),
        })
    return rows


def bootstrap_test_roc_auc(y_true, y_score, patient_ids):
    """Percentile interval for the pooled test ROC-AUC, resampling patients.

    Thirteen patients carry the test score, so the quantity that varies
    between plausible test sets is which patients are in it. The resampling unit is
    therefore the patient and not the patch: patients are drawn with
    replacement, every patch of a drawn patient enters the resample, and the
    pooled ROC-AUC is recomputed. Resampling patches instead would treat the
    thousands of overlapping patches of one patient as independent and return
    an interval far narrower than the evidence supports.

    The point estimate reported elsewhere is the score on the test set as it
    stands and is not replaced by the median of this distribution. A resample
    that happens to contain one class alone admits no score and is counted and
    discarded.
    """
    rng = np.random.default_rng(config.SEED)
    unique = np.array(sorted(set(patient_ids.tolist())))
    members = {patient: np.flatnonzero(patient_ids == patient)
               for patient in unique}
    scores = []
    for _ in range(config.BOOTSTRAP_RESAMPLES):
        drawn = rng.integers(0, len(unique), len(unique))
        rows = np.concatenate([members[unique[i]] for i in drawn])
        truth = y_true[rows]
        if len(set(truth.tolist())) < 2:
            continue
        scores.append(float(roc_auc_score(truth, y_score[rows])))
    if not scores:
        raise RuntimeError("no bootstrap resample carried both classes")
    tail = (1.0 - config.BOOTSTRAP_INTERVAL) / 2.0
    return {
        "resamples": config.BOOTSTRAP_RESAMPLES,
        "usable": len(scores),
        "median": float(np.median(scores)),
        "low": float(np.percentile(scores, 100 * tail)),
        "high": float(np.percentile(scores, 100 * (1.0 - tail))),
    }


def evaluate():
    banner("stage 07 evaluation")
    X, y, patients, index = load_patches()
    patient_split, _ = load_splits()
    model = load_model()

    scores = predict(model, X)
    metrics = Metrics()

    report, confusion_rows = [], []
    for part in ("train", "validation", "test"):
        selector = patient_split == part
        truth, score = y[selector], scores[selector]
        predicted = (score >= 0.5).astype(int)
        matrix = confusion_matrix(truth, predicted, labels=[0, 1])
        report += report_rows(part, truth, score)
        for i, actual in enumerate(CLASS_NAMES):
            for j, guess in enumerate(CLASS_NAMES):
                confusion_rows.append({
                    "split": part, "actual": actual, "predicted": guess,
                    config.SELECTION_PARTICIPATION_COLUMN:
                        config.selection_participation(part),
                    "count": int(matrix[i, j])})
        metrics.update({
            "eval_" + part + "_roc_auc": round(float(roc_auc_score(truth, score)), 4),
            "eval_" + part + "_accuracy": round(float(accuracy_score(truth, predicted)), 4),
            "eval_" + part + "_true_negative": int(matrix[0, 0]),
            "eval_" + part + "_false_positive": int(matrix[0, 1]),
            "eval_" + part + "_false_negative": int(matrix[1, 0]),
            "eval_" + part + "_true_positive": int(matrix[1, 1]),
            "eval_" + part + "_patches": int(selector.sum()),
        })
        print("  {:<11} ROC-AUC {:.4f}  accuracy {:.4f}  n {}".format(
            part, roc_auc_score(truth, score), accuracy_score(truth, predicted),
            int(selector.sum())))

    # Which of the three recorded validation ROC-AUC figures belongs to the
    # model that was kept. This one does: it is measured here from the reloaded
    # retained weights, and it is 0.9629 against the 0.9071 the final epoch
    # reached. The role is recorded beside it so the two cannot be confused.
    metrics.set("eval_validation_roc_auc_role", REPORTED_MODEL_VALIDATION_ROLE)

    write_table(report, config.RESULTS / "classification_report.csv")
    write_table(confusion_rows, config.RESULTS / "confusion_matrix.csv")

    for row in report:
        key = "eval_{}_{}_".format(row["split"], row["class"].replace("-", "_"))
        metrics.update({key + "precision": row["precision"],
                        key + "recall": row["recall"],
                        key + "f1": row["f1"]})

    # Per-patient performance on the held-out patients.
    test = patient_split == "test"
    natural = natural_positive_rates()
    per_patient = []
    for patient in sorted(set(patients[test].tolist())):
        selector = test & (patients == patient)
        truth, score = y[selector], scores[selector]
        predicted = (score >= 0.5).astype(int)
        per_patient.append({
            "patient_id": patient,
            "patches": int(selector.sum()),
            # The sampling fixes this at 0.333333 on every patient, so it is a
            # property of how the patches were drawn. The rate the patient's
            # volume held before the negatives were cut is beside it.
            "positive_rate_fixed_by_sampling": round(float(truth.mean()), 4),
            "natural_positive_rate": round(float(natural[patient]), 6),
            "roc_auc": round(float(roc_auc_score(truth, score)), 4)
            if len(set(truth.tolist())) > 1 else "",
            "accuracy": round(float(accuracy_score(truth, predicted)), 4),
            "false_positive": int(((predicted == 1) & (truth == 0)).sum()),
            "false_negative": int(((predicted == 0) & (truth == 1)).sum()),
        })
    write_table(per_patient, config.RESULTS / "per_patient_test.csv")
    patient_aucs = [float(r["roc_auc"]) for r in per_patient if r["roc_auc"] != ""]

    # The ROC curve on the test patients, saved so the figure and the gate read
    # the same numbers.
    fpr, tpr, thresholds = roc_curve(y[test], scores[test])
    write_table(
        [{"false_positive_rate": round(float(a), 6),
          "true_positive_rate": round(float(b), 6),
          "threshold": round(float(c), 6)}
         for a, b, c in zip(fpr, tpr, thresholds)],
        config.RESULTS / "roc_test.csv",
    )

    # A patch-level baseline: always predict the majority class.
    majority = 1 if y[patient_split == "train"].mean() >= 0.5 else 0
    baseline = float((y[test] == majority).mean())

    interval = bootstrap_test_roc_auc(y[test], scores[test], patients[test])

    metrics.update({
        "eval_test_roc_auc_bootstrap_resamples": interval["resamples"],
        "eval_test_roc_auc_bootstrap_usable_resamples": interval["usable"],
        "eval_test_roc_auc_bootstrap_median": round(interval["median"], 4),
        "eval_test_roc_auc_bootstrap_low": round(interval["low"], 4),
        "eval_test_roc_auc_bootstrap_high": round(interval["high"], 4),
        "eval_test_roc_auc_bootstrap_interval": config.BOOTSTRAP_INTERVAL,
        "eval_test_roc_auc_bootstrap_unit": "patient",
        "eval_test_patients": len(per_patient),
        # A patient whose test patches carry one class admits no ROC-AUC, and
        # per_patient_test.csv leaves the column empty for it. The three figures
        # below are then taken over fewer patients than eval_test_patients
        # counts, so the count they are taken over is recorded beside them.
        "eval_test_patients_with_roc_auc": len(patient_aucs),
        "eval_test_patient_roc_auc_role":
            "taken over the {} of {} test patients whose patches carry both "
            "classes, because a patient carrying one class admits no "
            "ROC-AUC".format(len(patient_aucs), len(per_patient)),
        "eval_test_patient_roc_auc_min": round(min(patient_aucs), 4),
        "eval_test_patient_roc_auc_median": round(float(np.median(patient_aucs)), 4),
        "eval_test_patient_roc_auc_max": round(max(patient_aucs), 4),
        "eval_test_patients_below_roc_auc_0.8": sum(1 for a in patient_aucs if a < 0.8),
        "eval_test_majority_class_accuracy": round(baseline, 4),
        "eval_threshold": 0.5,
    })
    metrics.save()
    print("  test patients {}, per-patient ROC-AUC median {:.3f}, "
          "range {:.3f} to {:.3f}".format(
              len(per_patient), np.median(patient_aucs),
              min(patient_aucs), max(patient_aucs)))
    print("  pooled test ROC-AUC {}, {:.0f} percent patient bootstrap interval "
          "{:.4f} to {:.4f} over {} usable resamples".format(
              metrics.get("eval_test_roc_auc"), 100 * config.BOOTSTRAP_INTERVAL,
              interval["low"], interval["high"], interval["usable"]))
    return scores


if __name__ == "__main__":
    evaluate()
    sys.exit(0)
