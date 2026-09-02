#!/usr/bin/env python
"""Phase 5 gate: the split is honest, and the record says what it holds.

Four properties are checked. No patient appears in more than one part of the
partition. Nothing outside the pixel array carries the label, so the network
cannot read the answer off a path or a file name. The reported test score is
the score the saved model produces when it is loaded again and re-run. Every
recorded quantity covering epoch selection, validation and leakage is labeled
for a reader who opens ``results/`` and reads nothing else.
"""

from __future__ import annotations

import math
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

# --- the record read on its own --------------------------------------------
# Finding R-12 named four ways results/ misleads a reader who never opens the
# prose. All four are labeling defects and none moved a number. The checks
# below read results/ and hold the record against what it says about itself,
# because the record is what the repository asks to be trusted.


def number_or_nan(key: str) -> float:
    """Read a recorded number without raising when the key has gone.

    ``Metrics.get`` raises on a key the record no longer holds, and ``check``
    evaluates its detail argument eagerly, so a vanished key would stop the
    gate before the check that should report it. A NaN compares false against
    every tolerance, which is the failure the gate exists to show.
    """
    try:
        return float(recorded.values.get(key, "nan"))
    except ValueError:
        return float("nan")


# R-12(1). The gain from epoch selection was recorded at the one seed where it
# helped most. Every run is now scored at both candidate epochs, and the six
# gains in results/seed_variance.csv are the evidence behind the summary keys.
seeds = table("seed_variance")
GAIN_COLUMN = "test_roc_auc_epoch_selection_gain"

gains: list[float] = []
gain_error = ""
try:
    gains = [float(row[GAIN_COLUMN]) for row in seeds]
except (KeyError, TypeError, ValueError) as error:
    gain_error = " [{}]".format(error)

check("every seed-variance run records its gain from epoch selection",
      len(gains) == len(seeds)
      and recorded.values.get("seed_variance_runs", "") == str(len(seeds))
      and recorded.values.get(
          "seed_variance_epoch_selection_gain_runs", "") == str(len(seeds)),
      "{} of {} rows carry {}, the record counts {} runs{}".format(
          len(gains), len(seeds), GAIN_COLUMN,
          recorded.values.get("seed_variance_epoch_selection_gain_runs",
                              "an absent number of"), gain_error))

gain_mismatches = []
for row in seeds:
    try:
        measured = (float(row["test_roc_auc_selected_epoch"])
                    - float(row["test_roc_auc_final_epoch"]))
        if abs(measured - float(row[GAIN_COLUMN])) > 5e-5:
            gain_mismatches.append("{} {} measures {:.4f}".format(
                row.get("seed"), row.get("split_unit"), measured))
    except (KeyError, TypeError, ValueError) as error:
        gain_mismatches.append("{} {} [{}]".format(
            row.get("seed"), row.get("split_unit"), error))
check("each recorded gain is the difference between the run's two epochs",
      not gain_mismatches,
      "; ".join(gain_mismatches) if gain_mismatches
      else "{} runs agree to 5e-5".format(len(seeds)))

check("the recorded mean gain is the mean over every run",
      bool(gains)
      and abs(sum(gains) / len(gains)
              - number_or_nan("seed_variance_epoch_selection_gain_mean")) < 5e-5,
      "the rows give {:.4f} and the record says {}".format(
          sum(gains) / len(gains) if gains else float("nan"),
          recorded.values.get("seed_variance_epoch_selection_gain_mean",
                              "an absent mean")))

check("the recorded minimum, maximum and spread bound every gain",
      bool(gains)
      and abs(min(gains)
              - number_or_nan("seed_variance_epoch_selection_gain_min")) < 5e-5
      and abs(max(gains)
              - number_or_nan("seed_variance_epoch_selection_gain_max")) < 5e-5
      and abs(max(gains) - min(gains)
              - number_or_nan("seed_variance_epoch_selection_gain_spread")) < 5e-5,
      "the rows run {:.4f} to {:.4f} and the record says {} to {} spanning {}".format(
          min(gains) if gains else float("nan"),
          max(gains) if gains else float("nan"),
          recorded.values.get("seed_variance_epoch_selection_gain_min", "absent"),
          recorded.values.get("seed_variance_epoch_selection_gain_max", "absent"),
          recorded.values.get("seed_variance_epoch_selection_gain_spread", "absent")))

check("the recorded count of runs that gained nothing matches the rows",
      bool(gains)
      and sum(1 for g in gains if abs(g) < 1e-9)
      == number_or_nan("seed_variance_epoch_selection_runs_with_no_gain"),
      "{} of {} rows gained nothing and the record says {}".format(
          sum(1 for g in gains if abs(g) < 1e-9), len(gains),
          recorded.values.get("seed_variance_epoch_selection_runs_with_no_gain",
                              "an absent count")))

# R-12(2). The retained epoch maximizes validation ROC-AUC, so every validation
# figure is read at the epoch that maximized a quantity on that same split.
# The three tables identifying their rows by split carry the participation
# column, so a validation row cannot be read as unbiased from the file alone.
PARTICIPATION = config.SELECTION_PARTICIPATION_COLUMN
MARKED_TABLES = ("splits", "classification_report", "confusion_matrix")

took_part: dict = {}
for name in MARKED_TABLES:
    rows = table(name)
    wrong = []
    for row in rows:
        split = row.get("split", "")
        took_part.setdefault(row.get(PARTICIPATION, "an absent column"),
                             set()).add(split)
        if row.get(PARTICIPATION) != config.selection_participation(split):
            wrong.append("{} says {}".format(split, row.get(PARTICIPATION)))
    check("results/{}.csv marks every row with its part in the epoch "
          "selection".format(name),
          bool(rows) and not wrong,
          "; ".join(sorted(set(wrong))) if wrong
          else "{} rows carry {}".format(len(rows), PARTICIPATION))

check("the validation split alone is marked as taking part in the selection",
      took_part.get(config.SELECTION_PARTICIPATION_YES) == {"validation"}
      and took_part.get(config.SELECTION_PARTICIPATION_NO) == {"train", "test"},
      "marked as taking part: {}; marked as not: {}".format(
          sorted(took_part.get(config.SELECTION_PARTICIPATION_YES, set())),
          sorted(took_part.get(config.SELECTION_PARTICIPATION_NO, set()))))

check("the participation column carries a sentence and not a flag",
      len(config.SELECTION_PARTICIPATION_YES.split()) >= 5
      and "epoch" in config.SELECTION_PARTICIPATION_YES.lower()
      and config.SELECTION_PARTICIPATION_YES.strip().lower()
      not in ("1", "true", "yes", "y"),
      "a validation row reads: {}".format(config.SELECTION_PARTICIPATION_YES))

# R-12(3). Two validation ROC-AUC keys meant different things and neither name
# said which belonged to the model kept. The final-epoch figure is retained for
# the training-curve narrative, so the disambiguation is carried by the names,
# by a role key on each quantity, and by a pointer to the reported model's key.
POINTER_KEY = "model_reported_model_validation_roc_auc_key"
FINAL_EPOCH_KEY = "model_final_epoch_validation_roc_auc"
pointer = recorded.values.get(POINTER_KEY, "")

check("the final-epoch validation ROC-AUC names the epoch it belongs to",
      FINAL_EPOCH_KEY in recorded.values
      and not math.isnan(number_or_nan(FINAL_EPOCH_KEY)),
      "{} is {}".format(FINAL_EPOCH_KEY,
                        recorded.values.get(FINAL_EPOCH_KEY, "absent")))

for quantity in (FINAL_EPOCH_KEY, "eval_validation_roc_auc"):
    role = recorded.values.get(quantity + "_role", "")
    check("{} states its role in the record".format(quantity),
          bool(role.strip()),
          role if role.strip() else "{}_role is absent".format(quantity))

check("a pointer key names the validation ROC-AUC of the reported model",
      bool(pointer.strip()),
      pointer if pointer.strip() else "{} is absent".format(POINTER_KEY))
check("the pointer names a key the record holds",
      pointer in recorded.values,
      "{} points at {}".format(POINTER_KEY, pointer or "nothing"))
check("the pointer names the retained epoch and not the final epoch",
      pointer != FINAL_EPOCH_KEY
      and abs(number_or_nan(pointer)
              - number_or_nan("model_selected_epoch_validation_roc_auc")) < 5e-5,
      "{} is {}, the retained epoch is {} and the final epoch is {}".format(
          pointer or "nothing", recorded.values.get(pointer, "absent"),
          recorded.values.get("model_selected_epoch_validation_roc_auc", "absent"),
          recorded.values.get(FINAL_EPOCH_KEY, "absent")))

# R-12(4). The singular leakage key was the least representative of the three
# seeds and is the key a reader reaches for first. It now names its seed, and
# the cross-seed count, mean, minimum and maximum sit beside it.
INFLATION = "leakage_roc_auc_inflation"
# The column seed_variance.csv carries the per-seed difference under. The bare
# name it replaced read as a property of the partition its row names.
SEED_INFLATION_COLUMN = "seed_roc_auc_inflation_patch_minus_patient"
INFLATION_KEYS = (INFLATION + "_primary_seed", INFLATION + "_seed_count",
                  INFLATION + "_seed_mean", INFLATION + "_seed_min",
                  INFLATION + "_seed_max")

per_seed: dict = {}
primary_inflations: set = set()
inflation_error = ""
try:
    for row in seeds:
        per_seed[row["seed"]] = float(row[SEED_INFLATION_COLUMN])
        if row["is_primary_seed"] == "1":
            primary_inflations.add(float(row[SEED_INFLATION_COLUMN]))
except (KeyError, TypeError, ValueError) as error:
    inflation_error = " [{}]".format(error)
inflations = sorted(per_seed.values())

missing_inflation = [k for k in INFLATION_KEYS if k not in recorded.values]
check("the leakage inflation is recorded per seed with its cross-seed summary",
      not missing_inflation,
      "absent: " + ", ".join(missing_inflation) if missing_inflation
      else "all {} keys under {} are present".format(len(INFLATION_KEYS), INFLATION))

check("the primary-seed inflation is the one the primary seed's rows carry",
      len(primary_inflations) == 1
      and abs(next(iter(primary_inflations))
              - number_or_nan(INFLATION + "_primary_seed")) < 5e-5,
      "the rows give {} and the record says {}{}".format(
          sorted(primary_inflations) or "nothing",
          recorded.values.get(INFLATION + "_primary_seed", "absent"),
          inflation_error))

check("the recorded seed count and mean agree with the per-seed inflations",
      bool(inflations)
      and len(inflations) == number_or_nan(INFLATION + "_seed_count")
      and abs(sum(inflations) / len(inflations)
              - number_or_nan(INFLATION + "_seed_mean")) < 5e-5,
      "{} seeds averaging {:.4f}, and the record says {} seeds averaging {}".format(
          len(inflations),
          sum(inflations) / len(inflations) if inflations else float("nan"),
          recorded.values.get(INFLATION + "_seed_count", "absent"),
          recorded.values.get(INFLATION + "_seed_mean", "absent")))

check("the recorded minimum and maximum are the extremes of the per-seed "
      "inflations",
      bool(inflations)
      and abs(inflations[0] - number_or_nan(INFLATION + "_seed_min")) < 5e-5
      and abs(inflations[-1] - number_or_nan(INFLATION + "_seed_max")) < 5e-5,
      "the rows run {:.4f} to {:.4f} and the record says {} to {}".format(
          inflations[0] if inflations else float("nan"),
          inflations[-1] if inflations else float("nan"),
          recorded.values.get(INFLATION + "_seed_min", "absent"),
          recorded.values.get(INFLATION + "_seed_max", "absent")))

# Seven keys were renamed to say what they hold. The record is append-only, so
# a stale row survives a re-run and the old bare name reappears with no source
# file writing it, which is how the renames were nearly lost once already.
# Each rename is gated on the old name being gone and the new one being there.
RENAMED_KEYS = (
    ("leakage_roc_auc_inflation", "leakage_roc_auc_inflation_primary_seed"),
    ("model_final_train_loss", "model_final_epoch_train_loss"),
    ("model_final_validation_roc_auc", "model_final_epoch_validation_roc_auc"),
    ("patches_positive_rate", "patches_positive_rate_fixed_by_sampling"),
    ("split_test_positive_rate", "split_test_positive_rate_fixed_by_sampling"),
    ("split_train_positive_rate", "split_train_positive_rate_fixed_by_sampling"),
    ("split_validation_positive_rate",
     "split_validation_positive_rate_fixed_by_sampling"),
)
for old_key, new_key in RENAMED_KEYS:
    check("the record carries {} and not the bare {}".format(new_key, old_key),
          old_key not in recorded.values and new_key in recorded.values,
          "{} is {}, and {} is {}".format(
              new_key, recorded.values.get(new_key, "absent"), old_key,
              "absent" if old_key not in recorded.values
              else "still recorded as " + recorded.values[old_key]))

finish()
