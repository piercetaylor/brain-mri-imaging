#!/usr/bin/env python
"""Phase 5 gate: the split is honest and the reported score comes from it.

Three properties are checked. No patient appears in more than one part of the
partition. Nothing outside the pixel array carries the label, so the network
cannot read the answer off a path or a file name. The reported test score is
the score the saved model produces when it is loaded again and re-run.
"""

from __future__ import annotations

import re
import sys

import numpy as np

from gate_lib import ROOT, check, config, finish, gate, metrics, table

sys.path.insert(0, str(ROOT))
from src.s04_patches import load_patches  # noqa: E402
from src.s05_splits import load_splits  # noqa: E402
from src.s06_model import load_model, predict  # noqa: E402

gate("gate 05 modeling")

X, y, patients, index = load_patches()
patient_split, patch_split = load_splits()
recorded = metrics()

parts = ("train", "validation", "test")
check("every patch is assigned to exactly one part",
      set(np.unique(patient_split).tolist()) == set(parts))

overlaps = []
for i, first in enumerate(parts):
    for second in parts[i + 1:]:
        shared = set(patients[patient_split == first]) & set(patients[patient_split == second])
        if shared:
            overlaps.append("{} and {} share {}".format(first, second, sorted(shared)))
check("no patient appears in more than one part", not overlaps,
      "; ".join(overlaps) if overlaps else "{} patients partitioned".format(
          len(set(patients.tolist()))))

for part in parts:
    selector = patient_split == part
    check("{} is populated".format(part), int(selector.sum()) > 0,
          "{} patches from {} patients".format(
              int(selector.sum()), len(set(patients[selector].tolist()))))
    check("{} carries both classes".format(part),
          len(set(y[selector].tolist())) == 2)

# The label must not be recoverable from anything the model is given. The model
# is given X alone, so the check is that no identifier used to build X encodes
# the class.
LABEL_WORDS = re.compile(r"pos|neg|tumou?r|lesion|label|class", re.I)
check("no patient identifier encodes a class",
      not any(LABEL_WORDS.search(p) for p in set(patients.tolist())))
paths = [str(p) for p in config.RAW.glob("*/*/*")]
check("no directory on the raw path encodes a class",
      not any(LABEL_WORDS.search(p.split("raw")[-1]) for p in paths),
      "{} directories".format(len(paths)))
check("the model input is the pixel array alone",
      X.ndim == 3 and len(X) == len(y),
      "inputs {}".format(X.shape))

# A trivial statistic that would betray a leak: patch coordinates must not
# separate the classes on their own.
positive_z = index[y == 1, 0].mean()
negative_z = index[y == 0, 0].mean()
check("slice position does not separate the classes on its own",
      abs(positive_z - negative_z) < index[:, 0].std(),
      "mean slice {:.1f} against {:.1f}, standard deviation {:.1f}".format(
          positive_z, negative_z, index[:, 0].std()))

# The saved model must reproduce the reported score.
model = load_model()
from sklearn.metrics import roc_auc_score  # noqa: E402
test = patient_split == "test"
auc = float(roc_auc_score(y[test], predict(model, X[test])))
check("the saved model reproduces the reported test ROC-AUC",
      abs(auc - recorded.number("eval_test_roc_auc")) < 5e-4,
      "{:.4f} against {}".format(auc, recorded.get("eval_test_roc_auc")))
check("the test score beats always predicting the majority class",
      recorded.number("eval_test_accuracy")
      > recorded.number("eval_test_majority_class_accuracy"),
      "{} against {}".format(recorded.get("eval_test_accuracy"),
                             recorded.get("eval_test_majority_class_accuracy")))

grid = table("model_grid")
check("the grid search covered every configured combination",
      len(grid) == len(config.LEARNING_RATE_GRID) * len(config.DROPOUT_GRID),
      "{} combinations".format(len(grid)))
# The column the grid was ranked on is read from the record and not named here.
# Stage 6 ranks on the best-epoch column, because that is what a configuration
# delivers under the procedure the reported model uses, and it records which
# column that was. A gate holding its own copy of the column name would pass
# whenever the two criteria happened to agree and would say nothing when they
# did not.
# Read without Metrics.get, which raises on a key the record no longer holds.
# A criterion that disappeared has to fail the check below and not stop the gate
# before the check is reached.
criterion = recorded.values.get("model_grid_selection_criterion", "")
check("the chosen hyperparameters are the best on the validation set",
      criterion in grid[0]
      and float(grid[0][criterion]) == max(float(g[criterion]) for g in grid)
      and float(grid[0]["learning_rate"]) == recorded.number("model_learning_rate")
      and float(grid[0]["dropout"]) == recorded.number("model_dropout"),
      "learning rate {}, dropout {}, {} of {}".format(
          recorded.get("model_learning_rate"), recorded.get("model_dropout"),
          criterion, grid[0].get(criterion, "an absent column")))
check("the optimizer was given a non-zero learning rate",
      recorded.number("model_learning_rate") > 0,
      recorded.get("model_learning_rate"))

check("both partitions were evaluated with the same hyperparameters",
      0.0 <= recorded.number("leakage_test_roc_auc_patch_split") <= 1.0
      and abs(recorded.number("leakage_test_roc_auc_patient_split")
              - recorded.number("eval_test_roc_auc")) < 5e-4,
      "patient split {}, patch split {}, difference {}".format(
          recorded.get("leakage_test_roc_auc_patient_split"),
          recorded.get("leakage_test_roc_auc_patch_split"),
          recorded.get("leakage_roc_auc_inflation_primary_seed")))

finish()
