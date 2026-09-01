"""Shared harness for the phase gates.

A gate verifies one phase of the pipeline and exits 0 or non-zero. Every check
prints what it looked at and what it found, so that a passing gate is evidence
and not a silent success.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.s99_utils import Metrics  # noqa: E402

_STARTED = time.time()
_STATE = {"pass": 0, "fail": 0, "name": "gate"}


def gate(name: str) -> None:
    _STATE["name"] = name
    print("\n=== {} ===".format(name))


def check(label: str, condition, detail: str = "") -> bool:
    try:
        ok = bool(condition() if callable(condition) else condition)
    except Exception as error:  # a check that cannot run has not passed
        ok = False
        detail = (detail + " [error: {}]".format(error)).strip()
    _STATE["pass" if ok else "fail"] += 1
    print("  [{}] {}{}".format("PASS" if ok else "FAIL", label,
                               " -- " + detail if detail else ""))
    return ok


def skip(label: str, reason: str) -> None:
    print("  [SKIP] {} -- {}".format(label, reason))


def finish() -> None:
    total = _STATE["pass"] + _STATE["fail"]
    print("\n{}: {} ({} of {} checks passed, {:.1f}s)".format(
        _STATE["name"], "PASS" if _STATE["fail"] == 0 else "FAIL",
        _STATE["pass"], total, time.time() - _STARTED))
    sys.exit(0 if _STATE["fail"] == 0 else 1)


def metrics() -> Metrics:
    if not config.METRICS.exists():
        print("results/metrics.csv is absent. Run: python analysis/run_all.py")
        sys.exit(1)
    return Metrics()


def table(name: str) -> list[dict]:
    import csv
    path = config.RESULTS / (name + ".csv")
    if not path.exists():
        raise FileNotFoundError("results/{}.csv is absent".format(name))
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
