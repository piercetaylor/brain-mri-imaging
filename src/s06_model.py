"""Stage 6: the convolutional network, a grid search, and training.

The architecture is the one the original coursework specified: three
convolutional blocks with 16, 16 and 64 filters, batch normalization on the
second and third, max pooling after each, then a 64 unit dense layer, dropout,
and a single sigmoid output.

Two things the original left undone are done here. The optimizer is given a
learning rate at construction, and the grid search the exercise was named after
is actually run. The grid is scored on a validation set that shares no patient
with either the training or the test set, at ``config.GRID_EPOCHS`` epochs,
while the chosen configuration is then trained for ``config.MAX_EPOCHS``. Both
counts are recorded, as is the column the choice was read from.

Every training run here keeps the weights of one epoch. The validation score is
measured after each of the ``config.MAX_EPOCHS`` epochs, the weights of the
epoch that scored best under ``config.EPOCH_SELECTION_MONITOR`` are retained,
and they are restored before the model is returned. The previous run reported
the last epoch and paid 0.0558 of validation ROC-AUC for it. Training still
runs to the full epoch count, so the history covers every epoch and the
final-epoch weights are kept beside the retained ones and scored on the test
set, which is what makes the size of the selection effect readable.

That pair of test scores is taken on every one of the six runs the stage
performs, and not on the primary seed alone. The retained epochs on the
patient-level partition are 14, 13 and 20, so one of the three runs retains its
final epoch and gains nothing, and the primary seed is the run that scores
highest of the three. The mean and the range of the gain are recorded beside
the primary-seed figure, which carries the seed in its name.

The stage closes by re-running both partitions over ``config.SEED_LIST``, so
that the difference between a patient-level and a patch-level test score is
reported with the spread it carries across seeds and not as one number.
"""

from __future__ import annotations

import copy
import sys
import time

import numpy as np
import torch
from torch import nn

from . import config
from .s04_patches import load_patches
from .s05_splits import load_splits, patch_level, patient_level
from .s99_utils import Metrics, banner, write_table

MODEL_FILE = config.DATA / "interim" / "model.pt"
GRID_TABLE = config.RESULTS / "model_grid.csv"
SEED_TABLE = config.RESULTS / "seed_variance.csv"

# The column of results/model_grid.csv the chosen configuration is taken from.
# Both criteria are built from the monitored quantity, so the configuration and
# the epoch are chosen on the same measurement. A grid run and the reported
# model both keep their best epoch, so the best-epoch column is what a
# configuration delivers under the procedure that is actually used, and the
# final-epoch column is the alternative. Selecting the epoch on one quantity
# and the configuration on another would rank configurations by a score no
# retained model ever attains. What the other column would have chosen is
# recorded beside the choice instead of being left for a reader to derive.
GRID_SELECTION_CRITERION = config.EPOCH_SELECTION_MONITOR + "_best"
GRID_ALTERNATIVE_CRITERION = config.EPOCH_SELECTION_MONITOR + "_final"

# The grid is scored on a column that is already a maximum, and the kept
# configuration is the maximum of that column, so the figure the table reports
# for the kept row is a maximum of maxima over 4 configurations of 8 epochs
# each. The chosen row scored 0.9543 and the worst 0.9208, a spread of 0.0335
# over the four. Each row states which of the two it is, so the double
# selection is readable from the file and not only from this comment.
GRID_POINTS = len(config.LEARNING_RATE_GRID) * len(config.DROPOUT_GRID)
GRID_SELECTED_NOTE = (
    "selected: the largest of {} configurations of a column that is itself "
    "the largest of {} epochs".format(GRID_POINTS, config.GRID_EPOCHS))
GRID_NOT_SELECTED_NOTE = (
    "not selected: the largest of {} epochs of this configuration".format(
        config.GRID_EPOCHS))

# What ``model_selected_epoch_validation_roc_auc`` is, recorded beside it so
# that the figure cannot be read as an estimate of validation performance. It
# is the largest of MAX_EPOCHS measurements and the epoch it belongs to was
# chosen because it produced it.
SELECTED_VALIDATION_ROLE = (
    "selection statistic: the maximum over epochs, optimistically biased as an "
    "estimate of validation performance")
# The recorded quantity the epoch selection does not bias, because the test
# patients enter no part of it.
UNBIASED_SCORE_KEY = "eval_test_roc_auc"

# Two validation ROC-AUC figures are recorded and they differ by 0.0558 at the
# primary seed, 0.9071 against 0.9629. One belongs to the weights of the last
# epoch, which the reported model does not carry, and one belongs to the
# retained epoch, which it does. The role strings below are written beside each
# of them, and REPORTED_MODEL_VALIDATION_KEY names the key that belongs to the
# model stage 7 scores and stage 5 of the gates reloads. A reader of
# results/metrics.csv can therefore tell the two apart from the file alone.
FINAL_EPOCH_VALIDATION_ROLE = (
    "training-curve figure: the last epoch of the run, whose weights the "
    "reported model does not use")
REPORTED_MODEL_VALIDATION_ROLE = (
    "the reported model at its retained epoch, and a selection statistic, "
    "because that epoch was retained for maximizing this quantity")
REPORTED_MODEL_VALIDATION_KEY = "eval_validation_roc_auc"

# Where the gain from epoch selection is summarized over every run. The
# singular gain keys end in _primary_seed, because at the primary seed the
# patient-level partition gains 0.0141 while the third seed retains its final
# epoch and gains nothing. This key points from one to the other.
EPOCH_SELECTION_GAIN_SUMMARY_KEY = "seed_variance_epoch_selection_gain_mean"

# Which extreme of the monitored column each mode wants. A strict comparison
# keeps the earliest epoch when two epochs tie, so the retained epoch does not
# depend on the order the comparison happens to run in.
MONITOR_IMPROVES = {
    "max": lambda candidate, incumbent: candidate > incumbent,
    "min": lambda candidate, incumbent: candidate < incumbent,
}


class PatchCNN(nn.Module):
    """Conv(16) - Conv(16)+BN - Conv(64)+BN - Dense(64) - Dropout - sigmoid."""

    def __init__(self, dropout=0.5, size=config.PATCH_SIZE):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flattened = 64 * (size // 8) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(-1)


def as_tensors(X, y):
    return (torch.from_numpy(X).unsqueeze(1).float(),
            torch.from_numpy(y).float())


def select_epoch(history, monitor, mode):
    """Return the one-based epoch whose ``monitor`` column is most extreme."""
    improves = MONITOR_IMPROVES[mode]
    chosen = history[0]
    for row in history[1:]:
        if improves(row[monitor], chosen[monitor]):
            chosen = row
    return chosen["epoch"]


def train_model(X_train, y_train, X_validation, y_validation, learning_rate,
                dropout, epochs, seed=config.SEED, verbose=False):
    """Train for ``epochs`` epochs and return the weights of the best of them.

    The validation set is scored after every epoch on two quantities, ROC-AUC
    and cross-entropy loss, and the weights of the epoch that is best under
    ``config.EPOCH_SELECTION_MONITOR`` are copied and restored before the model
    is returned. Training is not cut short: every epoch runs, so the history is
    complete and the final-epoch weights survive in the returned selection
    record, where the caller that needs the size of the selection effect can
    score them.

    Neither the ROC-AUC nor the loss measurement draws from a random generator,
    because both run the network in evaluation mode under ``no_grad``, so the
    training trajectory is the one the previous run produced.
    """
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = PatchCNN(dropout=dropout)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()

    inputs, targets = as_tensors(X_train, y_train)
    generator = torch.Generator().manual_seed(seed)
    history = []
    improves = MONITOR_IMPROVES[config.EPOCH_SELECTION_MODE]
    selected_epoch, selected_state = 0, None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(inputs), generator=generator)
        running = 0.0
        for start in range(0, len(order), config.BATCH_SIZE):
            batch = order[start:start + config.BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_function(model(inputs[batch]), targets[batch])
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(batch)
        validation_auc = roc_auc(model, X_validation, y_validation)
        history.append({"epoch": epoch + 1,
                        "train_loss": running / len(order),
                        "validation_roc_auc": validation_auc,
                        "validation_loss": cross_entropy(
                            model, X_validation, y_validation)})
        monitored = history[-1][config.EPOCH_SELECTION_MONITOR]
        if selected_state is None or improves(monitored, selected_value):
            selected_epoch, selected_value = epoch + 1, monitored
            selected_state = copy.deepcopy(model.state_dict())
        if verbose:
            print("    epoch {:>2}  loss {:.4f}  validation ROC-AUC {:.4f}  "
                  "validation loss {:.4f}{}".format(
                      epoch + 1, history[-1]["train_loss"], validation_auc,
                      history[-1]["validation_loss"],
                      "  kept" if selected_epoch == epoch + 1 else ""),
                  flush=True)

    final_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(selected_state)
    for row in history:
        row["is_selected_epoch"] = int(row["epoch"] == selected_epoch)
        # Which column the retained epoch was read from and which extreme of it
        # was wanted, on every row. is_selected_epoch alone marks one epoch of
        # the 20 without saying that the mark is the maximum of a column the
        # same table carries, so a reader of the file has to be told elsewhere.
        row["epoch_selection_monitor"] = config.EPOCH_SELECTION_MONITOR
        row["epoch_selection_mode"] = config.EPOCH_SELECTION_MODE
    alternative_epoch = select_epoch(history,
                                     config.EPOCH_SELECTION_ALTERNATIVE_MONITOR,
                                     config.EPOCH_SELECTION_ALTERNATIVE_MODE)
    selection = {
        "monitor": config.EPOCH_SELECTION_MONITOR,
        "mode": config.EPOCH_SELECTION_MODE,
        "epochs": epochs,
        "selected_epoch": selected_epoch,
        "selected_validation_roc_auc": history[selected_epoch - 1][
            "validation_roc_auc"],
        "selected_validation_loss": history[selected_epoch - 1][
            "validation_loss"],
        "final_validation_roc_auc": history[-1]["validation_roc_auc"],
        "final_validation_loss": history[-1]["validation_loss"],
        "alternative_monitor": config.EPOCH_SELECTION_ALTERNATIVE_MONITOR,
        "alternative_epoch": alternative_epoch,
        "alternative_validation_roc_auc": history[alternative_epoch - 1][
            "validation_roc_auc"],
        "monitors_agree": int(alternative_epoch == selected_epoch),
        # The weights themselves, so that a caller can score the epoch it did
        # not keep without training the network a second time.
        "selected_state": selected_state,
        "final_state": final_state,
    }
    return model, history, selection


def logits(model, X, batch_size=1024):
    model.eval()
    inputs, _ = as_tensors(X, np.zeros(len(X), dtype=np.int64))
    output = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            output.append(model(inputs[start:start + batch_size]))
    return torch.cat(output)


def predict(model, X, batch_size=1024):
    return torch.sigmoid(logits(model, X, batch_size)).numpy()


def cross_entropy(model, X, y):
    """Mean binary cross-entropy of ``model`` on ``X``, computed from logits.

    The training loss is a mean over batches of a network that changes between
    them, so it is not comparable with a loss measured on a fixed network. This
    is the second quantity the epoch selection could have been read from, and
    it is measured the same way the training loss is, on logits and not on the
    probabilities :func:`predict` returns.
    """
    target = torch.from_numpy(np.asarray(y)).float()
    return float(nn.functional.binary_cross_entropy_with_logits(
        logits(model, X), target))


def roc_auc(model, X, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, predict(model, X)))


def grid_search(X, y, split):
    """Score every combination on the validation patients."""
    banner("stage 06 grid search")
    train = split == "train"
    validation = split == "validation"
    rows = []
    for learning_rate in config.LEARNING_RATE_GRID:
        for dropout in config.DROPOUT_GRID:
            started = time.time()
            model, history, selection = train_model(
                X[train], y[train], X[validation], y[validation],
                learning_rate, dropout, config.GRID_EPOCHS,
            )
            best = selection["selected_validation_roc_auc"]
            final = selection["final_validation_roc_auc"]
            rows.append({
                "learning_rate": learning_rate,
                "dropout": dropout,
                "epochs": config.GRID_EPOCHS,
                "validation_roc_auc_final": round(final, 4),
                "validation_roc_auc_best": round(best, 4),
                "selected_epoch": selection["selected_epoch"],
                "train_loss_final": round(history[-1]["train_loss"], 4),
                "seconds": round(time.time() - started, 1),
            })
            print("  learning rate {:<5} dropout {:<4} validation ROC-AUC {:.4f} "
                  "at epoch {} of {} ({:.0f}s)".format(
                      learning_rate, dropout, best, selection["selected_epoch"],
                      config.GRID_EPOCHS, rows[-1]["seconds"]), flush=True)
    # Python's sort is stable, so equal scores keep the order the grid was
    # enumerated in and the choice is reproducible without a tie-break.
    rows.sort(key=lambda r: -r[GRID_SELECTION_CRITERION])
    for position, row in enumerate(rows):
        row["is_selected_configuration"] = int(position == 0)
        row["selection_note"] = (GRID_SELECTED_NOTE if position == 0
                                 else GRID_NOT_SELECTED_NOTE)
    write_table(rows, GRID_TABLE)
    return rows


def seed_variance(X, y, patients, learning_rate, dropout, primary):
    """Re-run both partitions over ``config.SEED_LIST`` and record the spread.

    The difference between the test score of a patient-level partition and that
    of a patch-level one is the quantity the leakage comparison reports. At one
    seed that difference cannot be distinguished from the variation a different
    draw of thirteen test patients would produce. Each seed here draws its own
    partition and initializes its own network, so the spread across seeds
    measures both sources at once.

    ``primary`` holds the two scores already computed at ``config.SEED``,
    together with the epoch each of them was retained at and the score the
    final-epoch weights reached. Those are the numbers the reported model and
    its comparison were built from. The table reuses them and does not
    recompute them, so the table and the headline cannot disagree.

    Every run here keeps its best epoch, on both partitions and at every seed.
    The retained epoch is written to the table, so that the selection can be
    seen to have been applied uniformly and not read on trust.

    Each run is scored on its test patches twice, once at the final epoch and
    once at the retained epoch, and the difference is written beside the pair.
    One seed cannot carry that difference: the retained epochs on the
    patient-level partition are 14, 13 and 20, and the run that retains its
    twentieth epoch retains the final epoch and gains nothing. The mean and the
    range over the six runs are recorded from this table.
    """
    banner("stage 06 seed variance")
    rows = []
    for seed in config.SEED_LIST:
        partitions = {
            "patient": patient_level(patients, y, seed=seed)[0],
            "patch": patch_level(y, seed=seed),
        }
        for name, split in partitions.items():
            train = split == "train"
            validation = split == "validation"
            test = split == "test"
            if seed == config.SEED and name in primary:
                auc = primary[name]["test_roc_auc"]
                final_auc = primary[name]["final_epoch_test_roc_auc"]
                selected_epoch = primary[name]["selected_epoch"]
            else:
                model, _, selection = train_model(
                    X[train], y[train], X[validation], y[validation],
                    learning_rate, dropout, config.MAX_EPOCHS, seed=seed,
                )
                # The final-epoch weights are scored first and the retained
                # weights are then put back, so the two scores come from one
                # run and differ only in which epoch was reloaded.
                model.load_state_dict(selection["final_state"])
                final_auc = roc_auc(model, X[test], y[test])
                model.load_state_dict(selection["selected_state"])
                auc = roc_auc(model, X[test], y[test])
                selected_epoch = selection["selected_epoch"]
            rows.append({
                "seed": seed,
                # The unit the partition was drawn over, and the number of
                # patients the resulting test patches came from. On the
                # patient-level rows that number is the test set, 13 of the 49
                # patients. On the patch-level rows it is the whole cohort,
                # because a partition drawn over patches leaves every patient
                # contributing to both sides; a column named as a test-set size
                # would report 49 there and read as one.
                "split_unit": name,
                "is_primary_seed": int(seed == config.SEED),
                "train_patches": int(train.sum()),
                "test_patches": int(test.sum()),
                "test_patients_contributing_patches": len(
                    set(patients[test].tolist())),
                "epochs": config.MAX_EPOCHS,
                "epoch_selection_monitor": config.EPOCH_SELECTION_MONITOR,
                "selected_epoch": selected_epoch,
                "test_roc_auc": round(auc, 4),
                # The same score under the name that says which epoch it comes
                # from, the score the discarded final epoch reaches, and what
                # the selection was worth on this run.
                "test_roc_auc_selected_epoch": round(auc, 4),
                "test_roc_auc_final_epoch": round(final_auc, 4),
                "test_roc_auc_epoch_selection_gain": round(auc - final_auc, 4),
            })
            print("  seed {} {:<8} test ROC-AUC {:.4f} from epoch {} of {}, "
                  "{:.4f} at epoch {}, a gain of {:+.4f}".format(
                      seed, name + " split", auc, selected_epoch,
                      config.MAX_EPOCHS, final_auc, config.MAX_EPOCHS,
                      auc - final_auc), flush=True)

    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["split_unit"]] = row["test_roc_auc"]
    inflation = [by_seed[s]["patch"] - by_seed[s]["patient"] for s in config.SEED_LIST]
    for row in rows:
        row["roc_auc_inflation"] = round(
            by_seed[row["seed"]]["patch"] - by_seed[row["seed"]]["patient"], 4)
    write_table(rows, SEED_TABLE)
    gains = [row["test_roc_auc_epoch_selection_gain"] for row in rows]
    print("  epoch selection over {} runs: mean gain {:+.4f}, range {:+.4f} to "
          "{:+.4f}".format(len(gains), float(np.mean(gains)), min(gains),
                           max(gains)))
    return rows, inflation


def build():
    X, y, patients, _ = load_patches()
    patient_split, patch_split = load_splits()

    grid = grid_search(X, y, patient_split)
    chosen = grid[0]

    banner("stage 06 training")
    train = patient_split == "train"
    validation = patient_split == "validation"
    started = time.time()
    model, history, selection = train_model(
        X[train], y[train], X[validation], y[validation],
        chosen["learning_rate"], chosen["dropout"], config.MAX_EPOCHS, verbose=True,
    )
    seconds = time.time() - started
    torch.save({"state_dict": model.state_dict(),
                "learning_rate": chosen["learning_rate"],
                "dropout": chosen["dropout"],
                "selected_epoch": selection["selected_epoch"]}, MODEL_FILE)
    write_table(history, config.RESULTS / "training_history.csv")

    # What the epoch selection was worth, measured and not asserted. The test
    # set takes no part in the selection, so it is the set on which the two
    # candidate epochs can be compared. The final-epoch weights are scored
    # first, then the retained weights are put back.
    test = patient_split == "test"
    model.load_state_dict(selection["final_state"])
    final_epoch_test_auc = roc_auc(model, X[test], y[test])
    model.load_state_dict(selection["selected_state"])
    honest_auc = roc_auc(model, X[test], y[test])
    print("epoch {} of {} kept; test ROC-AUC {:.4f} against {:.4f} at the last "
          "epoch, a difference of {:+.4f}".format(
              selection["selected_epoch"], config.MAX_EPOCHS, honest_auc,
              final_epoch_test_auc, honest_auc - final_epoch_test_auc))

    # The same architecture and the same hyperparameters, trained on the
    # partition that splits patches and not patients, and with its own epoch
    # kept the same way. The gap between the two test scores is the cost of
    # that mistake, and it stays a comparison of like with like only because
    # the selection is applied to both arms.
    banner("stage 06 patch-level comparison")
    leak_train = patch_split == "train"
    leak_validation = patch_split == "validation"
    leaky_model, _, leaky_selection = train_model(
        X[leak_train], y[leak_train], X[leak_validation], y[leak_validation],
        chosen["learning_rate"], chosen["dropout"], config.MAX_EPOCHS,
    )
    leak_test = patch_split == "test"
    # The patch-level arm is scored at both candidate epochs as well, so that
    # the seed-variance table holds a final-epoch score for every one of its
    # six runs and reuses the two computed here at the primary seed.
    leaky_model.load_state_dict(leaky_selection["final_state"])
    leaky_final_epoch_test_auc = roc_auc(leaky_model, X[leak_test], y[leak_test])
    leaky_model.load_state_dict(leaky_selection["selected_state"])
    leaky_auc = roc_auc(leaky_model, X[leak_test], y[leak_test])
    print("test ROC-AUC, patients split {:.4f} from epoch {}; patches split "
          "{:.4f} from epoch {}".format(
              honest_auc, selection["selected_epoch"], leaky_auc,
              leaky_selection["selected_epoch"]))

    # Both saved partitions must be the ones the primary seed produces, or the
    # seed-variance table would reuse the two scores above against partitions
    # other than the ones that produced them.
    if not (np.array_equal(patient_level(patients, y, seed=config.SEED)[0],
                           patient_split)
            and np.array_equal(patch_level(y, seed=config.SEED), patch_split)):
        raise RuntimeError(
            "data/interim/splits.npz was not built at the primary seed")
    seed_rows, inflation = seed_variance(
        X, y, patients, chosen["learning_rate"], chosen["dropout"],
        {"patient": {"test_roc_auc": honest_auc,
                     "final_epoch_test_roc_auc": final_epoch_test_auc,
                     "selected_epoch": selection["selected_epoch"]},
         "patch": {"test_roc_auc": leaky_auc,
                   "final_epoch_test_roc_auc": leaky_final_epoch_test_auc,
                   "selected_epoch": leaky_selection["selected_epoch"]}},
    )
    patient_aucs = [r["test_roc_auc"] for r in seed_rows
                    if r["split_unit"] == "patient"]
    patch_aucs = [r["test_roc_auc"] for r in seed_rows if r["split_unit"] == "patch"]
    # What epoch selection was worth, over every run of the comparison. The
    # gain at one seed is the gain on one draw of test patients, so the mean
    # and the range are recorded beside it.
    gains = {
        "patient": [r["test_roc_auc_epoch_selection_gain"] for r in seed_rows
                    if r["split_unit"] == "patient"],
        "patch": [r["test_roc_auc_epoch_selection_gain"] for r in seed_rows
                  if r["split_unit"] == "patch"],
    }
    gains["all"] = gains["patient"] + gains["patch"]

    # What the other grid column would have chosen, recorded so that the
    # sensitivity of the choice to the criterion is visible.
    alternative = max(grid, key=lambda r: r[GRID_ALTERNATIVE_CRITERION])
    validation_patients = len(set(patients[validation].tolist()))

    metrics = Metrics()
    metrics.update({
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "model_grid_points": len(grid),
        "model_grid_learning_rates": ";".join(
            str(v) for v in config.LEARNING_RATE_GRID),
        "model_grid_dropouts": ";".join(str(v) for v in config.DROPOUT_GRID),
        "model_grid_epochs": config.GRID_EPOCHS,
        "model_grid_best_validation_roc_auc": chosen[GRID_SELECTION_CRITERION],
        "model_grid_best_validation_roc_auc_role": GRID_SELECTED_NOTE,
        "model_grid_worst_validation_roc_auc": grid[-1][GRID_SELECTION_CRITERION],
        "model_learning_rate": chosen["learning_rate"],
        "model_dropout": chosen["dropout"],
        "model_epochs": config.MAX_EPOCHS,
        "model_batch_size": config.BATCH_SIZE,
        "model_train_seconds": round(seconds, 1),
        # The last epoch of the run. Its weights were kept and scored, and the
        # reported model does not use them, so every key in this group names
        # the epoch it belongs to and the first carries its role beside it.
        "model_final_epoch_train_loss": round(history[-1]["train_loss"], 4),
        "model_final_epoch_validation_roc_auc": round(
            history[-1]["validation_roc_auc"], 4),
        "model_final_epoch_validation_roc_auc_role": FINAL_EPOCH_VALIDATION_ROLE,
        "model_final_epoch_validation_loss": round(
            history[-1]["validation_loss"], 4),
        # Which recorded validation score belongs to the model that was kept.
        # Stage 7 writes it under this key from the reloaded weights.
        "model_reported_model_validation_roc_auc_key":
            REPORTED_MODEL_VALIDATION_KEY,
        # Which epoch the reported model comes from, and on what.
        "model_epoch_selection_monitor": config.EPOCH_SELECTION_MONITOR,
        "model_epoch_selection_mode": config.EPOCH_SELECTION_MODE,
        "model_epoch_selection_max_epochs": config.MAX_EPOCHS,
        "model_selected_epoch": selection["selected_epoch"],
        # The next two are the largest of the 20 validation measurements the
        # run produced and the epoch that produced it was chosen because it was
        # the largest. Neither is an unbiased estimate of validation
        # performance and neither is reported as one; SELECTED_VALIDATION_ROLE
        # is recorded beside them to say so, and UNBIASED_SCORE_KEY names the
        # recorded score this selection does not bias.
        "model_selected_epoch_validation_roc_auc": round(
            selection["selected_validation_roc_auc"], 4),
        "model_selected_epoch_validation_loss": round(
            selection["selected_validation_loss"], 4),
        "model_selected_epoch_validation_roc_auc_role": SELECTED_VALIDATION_ROLE,
        "model_epoch_selection_unbiased_score_key": UNBIASED_SCORE_KEY,
        # What monitoring validation loss instead would have kept.
        "model_epoch_selection_alternative_monitor": selection[
            "alternative_monitor"],
        "model_epoch_selection_alternative_mode":
            config.EPOCH_SELECTION_ALTERNATIVE_MODE,
        "model_epoch_selection_alternative_epoch": selection["alternative_epoch"],
        "model_epoch_selection_alternative_epoch_validation_roc_auc": round(
            selection["alternative_validation_roc_auc"], 4),
        "model_epoch_selection_monitors_agree": selection["monitors_agree"],
        # What the selection was worth on the test patients, who take no part
        # in it. Both figures come from the same run and differ only in which
        # epoch's weights were scored. The three keys carry the primary seed in
        # their names because they are one of the six runs the stage performs,
        # and it is the run that scores highest of the three patient-level
        # seeds. The summary over all six is recorded below and is named here.
        "model_epoch_selection_test_roc_auc_final_epoch_primary_seed": round(
            final_epoch_test_auc, 4),
        "model_epoch_selection_test_roc_auc_selected_epoch_primary_seed": round(
            honest_auc, 4),
        "model_epoch_selection_test_roc_auc_gain_primary_seed": round(
            honest_auc - final_epoch_test_auc, 4),
        "model_epoch_selection_test_roc_auc_gain_summary_key":
            EPOCH_SELECTION_GAIN_SUMMARY_KEY,
        # The selection is applied to every training run the stage performs:
        # the grid, the reported model, the patch-level comparison and the two
        # partitions at each of the seeds beyond the primary one.
        "model_epoch_selection_runs": len(grid) + 2 + 2 * (len(config.SEED_LIST) - 1),
        "model_patch_split_selected_epoch": leaky_selection["selected_epoch"],
        "leakage_test_roc_auc_patient_split": round(honest_auc, 4),
        "leakage_test_roc_auc_patch_split": round(leaky_auc, 4),
        # The inflation at the primary seed and the inflation across the three.
        # The primary seed gives 0.0043, the smallest of the three, and the
        # mean is 0.0407, so the singular figure names the seed it comes from
        # and the summary sits beside it under the same prefix.
        "leakage_roc_auc_inflation_primary_seed": round(leaky_auc - honest_auc, 4),
        "leakage_roc_auc_inflation_seed_count": len(inflation),
        "leakage_roc_auc_inflation_seed_mean": float(np.mean(inflation)),
        "leakage_roc_auc_inflation_seed_min": float(min(inflation)),
        "leakage_roc_auc_inflation_seed_max": float(max(inflation)),
        # How the grid was read, and on how little.
        "model_grid_selection_criterion": GRID_SELECTION_CRITERION,
        "model_grid_alternative_criterion": GRID_ALTERNATIVE_CRITERION,
        # The criterion is the monitored quantity at the epoch the run would
        # keep, so the configuration and the epoch are chosen on one
        # measurement and the grid ranks configurations by a score the retained
        # model attains.
        "model_grid_criterion_monitor": config.EPOCH_SELECTION_MONITOR,
        "model_grid_selected_epoch": chosen["selected_epoch"],
        "model_grid_alternative_learning_rate": alternative["learning_rate"],
        "model_grid_alternative_dropout": alternative["dropout"],
        "model_grid_criteria_agree": int(
            alternative["learning_rate"] == chosen["learning_rate"]
            and alternative["dropout"] == chosen["dropout"]),
        "model_grid_validation_patients": validation_patients,
        "model_grid_validation_patches": int(validation.sum()),
        "model_final_epochs": config.MAX_EPOCHS,
        # The seed the headline numbers come from, and how far they move when
        # the seed changes.
        "model_primary_seed": config.SEED,
        "seed_variance_seeds": ";".join(str(s) for s in config.SEED_LIST),
        "seed_variance_seed_count": len(config.SEED_LIST),
        "seed_variance_runs": len(seed_rows),
        "seed_variance_patient_split_roc_auc_mean": float(np.mean(patient_aucs)),
        "seed_variance_patient_split_roc_auc_min": float(min(patient_aucs)),
        "seed_variance_patient_split_roc_auc_max": float(max(patient_aucs)),
        "seed_variance_patch_split_roc_auc_mean": float(np.mean(patch_aucs)),
        "seed_variance_patch_split_roc_auc_min": float(min(patch_aucs)),
        "seed_variance_patch_split_roc_auc_max": float(max(patch_aucs)),
        "seed_variance_inflation_mean": float(np.mean(inflation)),
        "seed_variance_inflation_min": float(min(inflation)),
        "seed_variance_inflation_max": float(max(inflation)),
        "seed_variance_inflation_spread": float(max(inflation) - min(inflation)),
        "seed_variance_inflation_sd": float(np.std(inflation, ddof=1)),
        "seed_variance_inflation_seeds_positive": sum(1 for v in inflation if v > 0),
        # What epoch selection was worth over every run, on the test patches
        # of each partition and over the two partitions together. The mean says
        # what it is worth on average and the minimum says what it is worth at
        # worst, which the primary-seed figure alone does not.
        "seed_variance_patient_split_epoch_selection_gain_mean": float(
            np.mean(gains["patient"])),
        "seed_variance_patient_split_epoch_selection_gain_min": min(gains["patient"]),
        "seed_variance_patient_split_epoch_selection_gain_max": max(gains["patient"]),
        "seed_variance_patch_split_epoch_selection_gain_mean": float(
            np.mean(gains["patch"])),
        "seed_variance_patch_split_epoch_selection_gain_min": min(gains["patch"]),
        "seed_variance_patch_split_epoch_selection_gain_max": max(gains["patch"]),
        "seed_variance_epoch_selection_gain_runs": len(gains["all"]),
        "seed_variance_epoch_selection_gain_mean": float(np.mean(gains["all"])),
        "seed_variance_epoch_selection_gain_min": min(gains["all"]),
        "seed_variance_epoch_selection_gain_max": max(gains["all"]),
        "seed_variance_epoch_selection_gain_spread": float(
            max(gains["all"]) - min(gains["all"])),
        "seed_variance_epoch_selection_runs_with_no_gain": sum(
            1 for v in gains["all"] if v == 0.0),
    })
    metrics.save()
    print("inflation across {} seeds: mean {:.4f}, range {:.4f} to {:.4f}, "
          "positive on {} of {}".format(
              len(inflation), float(np.mean(inflation)), min(inflation),
              max(inflation), sum(1 for v in inflation if v > 0), len(inflation)))
    print("epoch selection on {}: {} runs, reported model from epoch {} of {}; "
          "monitoring {} would have kept epoch {}".format(
              config.EPOCH_SELECTION_MONITOR,
              metrics.get("model_epoch_selection_runs"),
              selection["selected_epoch"], config.MAX_EPOCHS,
              selection["alternative_monitor"], selection["alternative_epoch"]))
    print("epoch selection gain over {} runs: mean {:+.4f}, range {:+.4f} to "
          "{:+.4f}; {:+.4f} at the primary seed on the patient split".format(
              len(gains["all"]), float(np.mean(gains["all"])),
              min(gains["all"]), max(gains["all"]),
              honest_auc - final_epoch_test_auc))
    return model


def load_model():
    if not MODEL_FILE.exists():
        raise SystemExit("data/interim/model.pt is absent. Run stage 06.")
    saved = torch.load(MODEL_FILE, weights_only=True)
    model = PatchCNN(dropout=saved["dropout"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    build()
    sys.exit(0)
