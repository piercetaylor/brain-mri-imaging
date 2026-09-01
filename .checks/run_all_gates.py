#!/usr/bin/env python
"""Run every gate in order and stop at the first failure.

    python .checks/run_all_gates.py
    SKIP_RERUN=1 python .checks/run_all_gates.py   (skip the reproducibility re-run)
    ALLOW_OFFLINE=1 python .checks/run_all_gates.py   (permit a skip where a
                                                       service does not answer)

A failing gate is not a warning. The phase it guards is not complete, and the
work that depends on it is not to be trusted until the cause is fixed. A skipped
check asserts nothing, so the summary reports skips beside the gate that took
them.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

CHECKS = Path(__file__).resolve().parent
ROOT = CHECKS.parent

# The count each gate prints for itself, read back so that the summary can
# report it. A gate that skipped a check still exits zero, and a summary
# carrying PASS alone would present the phase as verified.
SKIPPED = re.compile(r"(\d+) skipped")

gates = sorted(CHECKS.glob("gate_[0-9][0-9]_*.py"))
results = []


def run(path):
    """Run one gate, echo its output as it arrives, and keep its summary line."""
    process = subprocess.Popen(
        [sys.executable, str(path)], cwd=ROOT, text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    summary = ""
    for line in process.stdout:
        sys.stdout.write(line)
        if " checks passed" in line:
            summary = line
    return process.wait(), summary


def report():
    print("\n--- summary ---")
    for name, state, elapsed, skipped in results:
        print("  {:<32} {:<5} {:>7.1f}s{}".format(
            name, state, elapsed,
            "   {} skipped".format(skipped) if skipped else ""))
    total = sum(row[3] for row in results)
    if total:
        print("\n{} checks were skipped and assert nothing.".format(total))


for path in gates:
    started = time.time()
    code, summary = run(path)
    found = SKIPPED.search(summary)
    results.append((path.name, "PASS" if code == 0 else "FAIL",
                    time.time() - started,
                    int(found.group(1)) if found else 0))
    if code != 0:
        report()
        print("\nSTOPPED at {}. Fix the cause before proceeding.".format(path.name))
        sys.exit(1)

report()
print("\nAll {} gates PASS.".format(len(results)))
