#!/usr/bin/env python
"""Phase 6 gate: the pipeline reproduces every recorded quantity from an empty state.

The recorded metrics are set aside, ``results/`` and ``data/interim/`` are
cleared, the pipeline is run again, and the new record is compared key by key
against the old one.

Three kinds of quantity are compared for presence and not for value: the
wall-clock timings, whose keys begin with ``config.RECORD_TIMING_PREFIX``, the
training time, and the two keys that describe the record and the run and not the
data, which are the size of the record itself and the timestamp of the license
verification. The last three are named in ``config.RECORD_VOLATILE_KEYS``. The
exclusion set is read from config and not restated here, so it is the set the
notebook signature and gate 07 use. Everything else must match exactly.

``data/raw/`` is neither cleared nor re-fetched. At 8.45 GB the images are
verified against their recorded digests by gate 01 and not downloaded again, so
this gate rebuilds the analysis from an empty analysis state and not from an
empty repository.

This gate re-runs the whole pipeline and takes several minutes. It is skipped
when the environment variable ``SKIP_RERUN`` is set, and the skip is reported.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys

from gate_lib import ROOT, check, config, finish, gate, metrics, skip, unreachable

gate("gate 06 reproducibility")

TIMING_PREFIX = config.RECORD_TIMING_PREFIX
VOLATILE = set(config.RECORD_VOLATILE_KEYS)


def read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["key"]: row["value"] for row in csv.DictReader(handle)}


before = read(config.METRICS)
check("a metrics record exists to compare against", len(before) > 0,
      "{} quantities".format(len(before)))

if os.environ.get("SKIP_RERUN"):
    skip("pipeline re-run", "SKIP_RERUN is set")
    finish()

backup = config.DATA / "metrics_before_rerun.csv"
shutil.copy(config.METRICS, backup)
manifest_backup = config.DATA / "manifest_before_rerun.csv"
shutil.copy(config.MANIFEST, manifest_backup)

for directory in (config.RESULTS, config.DATA / "interim", config.FIGURES):
    if directory.exists():
        shutil.rmtree(directory)
check("results, interim data and figures were cleared",
      not config.RESULTS.exists() and not (config.DATA / "interim").exists())

# The cohort rule is re-run against the index. A manifest that came out
# differently would mean the subset is not defined by the rule alone.
completed = subprocess.run(
    [sys.executable, "-m", "src.s01_manifest"], cwd=ROOT,
    capture_output=True, text=True)
if completed.returncode != 0:
    shutil.copy(manifest_backup, config.MANIFEST)
    printed = (completed.stderr or completed.stdout or "").strip().splitlines()
    unreachable("the cohort rule reproduces data/manifest.csv byte for byte",
                printed[-1] if printed
                else "stage 01 exited {}".format(completed.returncode))
else:
    check("the cohort rule reproduces data/manifest.csv byte for byte",
          config.MANIFEST.read_bytes() == manifest_backup.read_bytes(),
          "{} bytes".format(config.MANIFEST.stat().st_size))

# The download stage is re-run too, so that the recorded acquisition figures
# are reproduced and not carried over. The images themselves are already on
# disk and are verified, not fetched again.
completed = subprocess.run(
    [sys.executable, str(ROOT / "data" / "download_data.py"), "--verify"],
    cwd=ROOT, capture_output=True, text=True)
check("acquisition re-verifies from an empty record", completed.returncode == 0,
      completed.stderr.strip().splitlines()[-1] if completed.returncode else "")

completed = subprocess.run(
    [sys.executable, str(ROOT / "analysis" / "run_all.py")],
    cwd=ROOT, capture_output=True, text=True)
check("the pipeline runs to completion", completed.returncode == 0,
      completed.stderr.strip().splitlines()[-1] if completed.returncode else
      "{} lines of output".format(len(completed.stdout.splitlines())))
if completed.returncode:
    print(completed.stdout[-4000:])
    finish()

after = read(config.METRICS)
check("no recorded quantity disappeared",
      not (set(before) - set(after)), ", ".join(sorted(set(before) - set(after))))
check("no unrecorded quantity appeared",
      not (set(after) - set(before)), ", ".join(sorted(set(after) - set(before))))

comparable = [k for k in before
              if k in after and not k.startswith(TIMING_PREFIX) and k not in VOLATILE]
differing = [k for k in comparable if before[k] != after[k]]
for key in differing[:10]:
    print("       {}: {} then {}".format(key, before[key], after[key]))
check("every quantity reproduces exactly", not differing,
      "{} of {} compared quantities differ".format(len(differing), len(comparable)))

check("timings were recorded on the second run",
      any(k.startswith(TIMING_PREFIX) for k in after),
      "{}s".format(after.get("pipeline_seconds")))

figures = sorted(config.FIGURES.glob("*.png"))
check("every figure was written again",
      len(figures) == int(metrics().number("figures_written")),
      "{} figures".format(len(figures)))

backup.unlink(missing_ok=True)
manifest_backup.unlink(missing_ok=True)
finish()
