#!/usr/bin/env python
"""Run every gate in order and stop at the first failure.

    python .checks/run_all_gates.py
    SKIP_RERUN=1 python .checks/run_all_gates.py   (skip the reproducibility re-run)

A failing gate is not a warning. The phase it guards is not complete, and the
work that depends on it is not to be trusted until the cause is fixed.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

CHECKS = Path(__file__).resolve().parent
ROOT = CHECKS.parent

gates = sorted(CHECKS.glob("gate_[0-9][0-9]_*.py"))
results = []

for path in gates:
    started = time.time()
    completed = subprocess.run([sys.executable, str(path)], cwd=ROOT)
    seconds = time.time() - started
    status = "PASS" if completed.returncode == 0 else "FAIL"
    results.append((path.name, status, seconds))
    if completed.returncode != 0:
        print("\n--- summary ---")
        for name, state, elapsed in results:
            print("  {:<32} {:<5} {:>7.1f}s".format(name, state, elapsed))
        print("\nSTOPPED at {}. Fix the cause before proceeding.".format(path.name))
        sys.exit(1)

print("\n--- summary ---")
for name, state, elapsed in results:
    print("  {:<32} {:<5} {:>7.1f}s".format(name, state, elapsed))
print("\nAll {} gates PASS.".format(len(results)))
