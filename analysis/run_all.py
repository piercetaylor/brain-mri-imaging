#!/usr/bin/env python
"""Run the whole pipeline in order and record every result.

    python analysis/run_all.py

The raw images must already be on disk. Fetch them first with
``python data/download_data.py``. Nothing here reaches the network.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src import s02_headers, s03_qc, s04_patches, s05_splits, s06_model
from src import s07_evaluate, s08_figures
from src.s99_utils import Metrics


STAGES = [
    ("02 headers", s02_headers.extract),
    ("03 quality control", s03_qc.run),
    ("03 de-identification audit", s03_qc.audit_phi),
    ("04 patches", s04_patches.build),
    ("05 splits", s05_splits.build),
    ("06 model", s06_model.build),
    ("07 evaluation", s07_evaluate.evaluate),
    ("08 figures", s08_figures.build),
]


def main() -> int:
    started = time.time()
    timings = []
    for name, function in STAGES:
        stage_started = time.time()
        function()
        timings.append((name, time.time() - stage_started))

    total = time.time() - started
    metrics = Metrics()
    metrics.set("pipeline_seconds", round(total, 1))
    for name, seconds in timings:
        key = "pipeline_seconds_stage_" + name.split()[0]
        metrics.set(key, round(metrics.number(key) + seconds, 1)
                    if key in metrics.values else round(seconds, 1))
    metrics.set("metrics_recorded", len(metrics.values) + 1)
    metrics.save()

    print("\n--- pipeline ---")
    for name, seconds in timings:
        print("  {:<28} {:>7.1f}s".format(name, seconds))
    print("  {:<28} {:>7.1f}s".format("total", total))
    print("{} quantities recorded in {}".format(
        len(metrics.values), config.METRICS.relative_to(config.ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
