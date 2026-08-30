"""Stage 7: evaluate the trained network on the held-out patients.

The reported numbers are ROC-AUC, accuracy, per-class precision, recall and F1,
and the confusion matrix, on training and test alike. Errors are also broken
down by patient, because a score pooled over patches hides a model that works
on most patients and fails on a few.
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
from .s05_splits import load_splits
from .s06_model import load_model, predict
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
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
            "f1": round(float(f1[index]), 4),
            "support": int(support[index]),
        })
    return rows


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
                confusion_rows.append({"split": part, "actual": actual,
                                       "predicted": guess, "count": int(matrix[i, j])})
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

    write_table(report, config.RESULTS / "classification_report.csv")
    write_table(confusion_rows, config.RESULTS / "confusion_matrix.csv")

    for row in report:
        key = "eval_{}_{}_".format(row["split"], row["class"].replace("-", "_"))
        metrics.update({key + "precision": row["precision"],
                        key + "recall": row["recall"],
                        key + "f1": row["f1"]})

    # Per-patient performance on the held-out patients.
    test = patient_split == "test"
    per_patient = []
    for patient in sorted(set(patients[test].tolist())):
        selector = test & (patients == patient)
        truth, score = y[selector], scores[selector]
        predicted = (score >= 0.5).astype(int)
        per_patient.append({
            "patient_id": patient,
            "patches": int(selector.sum()),
            "positive_rate": round(float(truth.mean()), 4),
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

    metrics.update({
        "eval_test_patients": len(per_patient),
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
    return scores


if __name__ == "__main__":
    evaluate()
    sys.exit(0)
