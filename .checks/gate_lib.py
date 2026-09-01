"""Shared harness for the phase gates.

A gate verifies one phase of the pipeline and exits 0 or non-zero. Every check
prints what it looked at and what it found, so that a passing gate is evidence
and not a silent success.

Two properties here are load-bearing. A check that cannot be run has not passed,
so an absent quantity fails one check and does not stop the gate. A service that
did not answer is skipped only when a skip is asked for, so a check does not
turn itself off when the thing it tests moves.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.s99_utils import Metrics  # noqa: E402

_STARTED = time.time()
_STATE = {"pass": 0, "fail": 0, "skip": 0, "name": "gate"}
_READERS: list = []

# What a service that cannot be reached raises. urllib.error.URLError covers a
# refused connection and an HTTP status alike, and the other two cover a socket
# that times out or is reset. A service that answers with something unexpected
# raises KeyError, TypeError or ValueError, and those are deliberately not
# caught, so a renamed field fails a check and is not skipped past.
NETWORK_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError)

# The variable that permits an unreachable service to be skipped in place of
# failed, so that running without a network is a stated choice.
OFFLINE = "ALLOW_OFFLINE"


def gate(name: str) -> None:
    _STATE["name"] = name
    print("\n=== {} ===".format(name))


class _Record:
    """The metrics record as a gate reads it.

    ``Metrics.get`` raises on a key it does not hold. A gate builds the label,
    the condition and the detail of a check before ``check`` is entered, so one
    absent key stopped the whole gate where one failed check was wanted. Here an
    absent key reads as a value that fails every comparison, and the key is kept
    so that :func:`finish` reports what was read and never found.
    """

    def __init__(self, record: Metrics) -> None:
        self._record = record
        self.values = record.values
        self.absent: list[str] = []

    def get(self, key: str) -> str:
        if key in self.values:
            return self._record.get(key)
        self.absent.append(key)
        return "absent"

    def number(self, key: str) -> float:
        if key in self.values:
            return self._record.number(key)
        self.absent.append(key)
        return float("nan")


def check(label: str, condition, detail="") -> bool:
    """Record one check. ``condition`` and ``detail`` may each be a callable.

    A callable is evaluated inside this function, so an expression that raises
    fails the check it belongs to. An argument built at the call site is
    evaluated before this function is entered and cannot be caught here, which
    is why the record a gate reads returns a failing value in place of raising.

    A check whose detail cannot be read has not passed either. The detail is
    what the check prints as its evidence, and a pass with no evidence behind it
    is the silent success this harness exists to prevent.
    """
    try:
        text = str(detail() if callable(detail) else detail)
        evidenced = True
    except Exception as error:
        text = "the detail of this check could not be read: {}".format(error)
        evidenced = False
    try:
        ok = bool(condition() if callable(condition) else condition)
    except Exception as error:  # a check that cannot run has not passed
        ok = False
        text = (text + " [error: {}]".format(error)).strip()
    ok = ok and evidenced
    _STATE["pass" if ok else "fail"] += 1
    print("  [{}] {}{}".format("PASS" if ok else "FAIL", label,
                               " -- " + text if text else ""))
    return ok


def skip(label: str, reason: str) -> None:
    """Record a check that was not run. A skip asserts nothing."""
    _STATE["skip"] += 1
    print("  [SKIP] {} -- {}".format(label, reason))


def unreachable(label: str, error) -> None:
    """Record a check whose service did not answer.

    A service that cannot be reached is a fact about the network and not about
    this repository, so it may be skipped, and it is skipped only when OFFLINE
    is set. Finding A-04 recorded the shape this avoids: a check that turned
    itself off whenever the service moved, while the gate reported PASS.
    """
    if os.environ.get(OFFLINE):
        skip(label, "the service did not answer and {} is set: {}".format(
            OFFLINE, error))
    else:
        check(label, False,
              "the service did not answer: {}. Set {} to permit a skip".format(
                  error, OFFLINE))


def metrics() -> _Record:
    if not config.METRICS.exists():
        print("results/metrics.csv is absent. Run: python analysis/run_all.py")
        sys.exit(1)
    reader = _Record(Metrics())
    _READERS.append(reader)
    return reader


def finish() -> None:
    for reader in _READERS:
        absent = sorted(set(reader.absent))
        check("every quantity a check read is present in the record",
              not absent,
              ", ".join(absent) if absent
              else "{} quantities available".format(len(reader.values)))
    total = _STATE["pass"] + _STATE["fail"]
    skipped = _STATE["skip"]
    print("\n{}: {} ({} of {} checks passed{}, {:.1f}s)".format(
        _STATE["name"], "PASS" if _STATE["fail"] == 0 else "FAIL",
        _STATE["pass"], total,
        ", {} skipped".format(skipped) if skipped else "",
        time.time() - _STARTED))
    sys.exit(0 if _STATE["fail"] == 0 else 1)


def table(name: str) -> list[dict]:
    import csv
    path = config.RESULTS / (name + ".csv")
    if not path.exists():
        raise FileNotFoundError("results/{}.csv is absent".format(name))
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
