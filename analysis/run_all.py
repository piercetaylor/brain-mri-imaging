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
    # Two entries share the number 03, so the seconds of this run are summed
    # by key before they are written. The sum is taken over ``timings`` and not
    # over what the metrics record already holds, because that record is loaded
    # from the previous run and adding to it would make every stage report the
    # total of every run the file has ever seen.
    by_stage: dict[str, float] = {}
    for name, seconds in timings:
        key = "pipeline_seconds_stage_" + name.split()[0]
        by_stage[key] = by_stage.get(key, 0.0) + seconds
    for key, seconds in by_stage.items():
        metrics.set(key, round(seconds, 1))
    # The key is written before the count is taken, because the record is
    # loaded from the previous run and may already hold it. Adding one to a
    # length that already counts the key overstates the record by one, which a
    # run from a cleared results/ never shows and every re-run does.
    metrics.set("metrics_recorded", 0)
    metrics.set("metrics_recorded", len(metrics.values))
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
