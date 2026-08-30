"""Stage 6: the convolutional network, a grid search, and training.

The architecture is the one the original coursework specified: three
convolutional blocks with 16, 16 and 64 filters, batch normalization on the
second and third, max pooling after each, then a 64 unit dense layer, dropout,
and a single sigmoid output.

Two things the original left undone are done here. The optimizer is given a
learning rate at construction, and the grid search the exercise was named after
is actually run. The grid is scored on a validation set that shares no patient
with either the training or the test set.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch
from torch import nn

from . import config
from .s04_patches import load_patches
from .s05_splits import load_splits
from .s99_utils import Metrics, banner, write_table

MODEL_FILE = config.DATA / "interim" / "model.pt"
GRID_TABLE = config.RESULTS / "model_grid.csv"


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


def train_model(X_train, y_train, X_validation, y_validation, learning_rate,
                dropout, epochs, seed=config.SEED, verbose=False):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = PatchCNN(dropout=dropout)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()

    inputs, targets = as_tensors(X_train, y_train)
    generator = torch.Generator().manual_seed(seed)
    history = []
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
                        "validation_roc_auc": validation_auc})
        if verbose:
            print("    epoch {:>2}  loss {:.4f}  validation ROC-AUC {:.4f}".format(
                epoch + 1, history[-1]["train_loss"], validation_auc), flush=True)
    return model, history


def predict(model, X, batch_size=1024):
    model.eval()
    inputs, _ = as_tensors(X, np.zeros(len(X), dtype=np.int64))
    output = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            output.append(torch.sigmoid(model(inputs[start:start + batch_size])))
    return torch.cat(output).numpy()


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
            model, history = train_model(
                X[train], y[train], X[validation], y[validation],
                learning_rate, dropout, config.GRID_EPOCHS,
            )
            best = max(h["validation_roc_auc"] for h in history)
            final = history[-1]["validation_roc_auc"]
            rows.append({
                "learning_rate": learning_rate,
                "dropout": dropout,
                "epochs": config.GRID_EPOCHS,
                "validation_roc_auc_final": round(final, 4),
                "validation_roc_auc_best": round(best, 4),
                "train_loss_final": round(history[-1]["train_loss"], 4),
                "seconds": round(time.time() - started, 1),
            })
            print("  learning rate {:<5} dropout {:<4} validation ROC-AUC {:.4f} "
                  "({:.0f}s)".format(learning_rate, dropout, final,
                                     rows[-1]["seconds"]), flush=True)
    rows.sort(key=lambda r: -r["validation_roc_auc_final"])
    write_table(rows, GRID_TABLE)
    return rows


def build():
    X, y, patients, _ = load_patches()
    patient_split, patch_split = load_splits()

    grid = grid_search(X, y, patient_split)
    chosen = grid[0]

    banner("stage 06 training")
    train = patient_split == "train"
    validation = patient_split == "validation"
    started = time.time()
    model, history = train_model(
        X[train], y[train], X[validation], y[validation],
        chosen["learning_rate"], chosen["dropout"], config.MAX_EPOCHS, verbose=True,
    )
    seconds = time.time() - started
    torch.save({"state_dict": model.state_dict(),
                "learning_rate": chosen["learning_rate"],
                "dropout": chosen["dropout"]}, MODEL_FILE)
    write_table(history, config.RESULTS / "training_history.csv")

    # The same architecture and the same hyperparameters, trained on the
    # partition that splits patches and not patients. The gap between the
    # two test scores is the cost of that mistake.
    banner("stage 06 patch-level comparison")
    leak_train = patch_split == "train"
    leak_validation = patch_split == "validation"
    leaky_model, _ = train_model(
        X[leak_train], y[leak_train], X[leak_validation], y[leak_validation],
        chosen["learning_rate"], chosen["dropout"], config.MAX_EPOCHS,
    )
    leaky_auc = roc_auc(leaky_model, X[patch_split == "test"], y[patch_split == "test"])
    honest_auc = roc_auc(model, X[patient_split == "test"], y[patient_split == "test"])
    print("test ROC-AUC, patients split {:.4f}; patches split {:.4f}".format(
        honest_auc, leaky_auc))

    metrics = Metrics()
    metrics.update({
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "model_grid_points": len(grid),
        "model_grid_learning_rates": ";".join(
            str(v) for v in config.LEARNING_RATE_GRID),
        "model_grid_dropouts": ";".join(str(v) for v in config.DROPOUT_GRID),
        "model_grid_epochs": config.GRID_EPOCHS,
        "model_grid_best_validation_roc_auc": chosen["validation_roc_auc_final"],
        "model_grid_worst_validation_roc_auc": grid[-1]["validation_roc_auc_final"],
        "model_learning_rate": chosen["learning_rate"],
        "model_dropout": chosen["dropout"],
        "model_epochs": config.MAX_EPOCHS,
        "model_batch_size": config.BATCH_SIZE,
        "model_train_seconds": round(seconds, 1),
        "model_final_train_loss": round(history[-1]["train_loss"], 4),
        "model_final_validation_roc_auc": round(
            history[-1]["validation_roc_auc"], 4),
        "leakage_test_roc_auc_patient_split": round(honest_auc, 4),
        "leakage_test_roc_auc_patch_split": round(leaky_auc, 4),
        "leakage_roc_auc_inflation": round(leaky_auc - honest_auc, 4),
    })
    metrics.save()
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
