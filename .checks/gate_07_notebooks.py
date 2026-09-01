#!/usr/bin/env python
"""Phase 7 gate: the notebooks display the record that is on disk.

``analysis/01`` and ``analysis/02`` are tracked deliverables. They read
``results/`` and ``figures/`` and compute nothing of their own, so their stored
output belongs to whichever record existed when they last ran. A notebook that
was not re-executed after the pipeline changed shows numbers and pictures that
no longer exist anywhere else in the repository, and nothing else in the suite
reads one.

The signature each notebook prints must equal the signature recomputed here from
``results/metrics.csv``. Every figure the notebooks embed must match the file in
``figures/`` byte for byte. The stored execution counts must run from one without
a gap and no cell may hold an error, so the document is one a fresh kernel
produced end to end.

The signature covers the quantities a re-run reproduces exactly: everything in
the record except the wall-clock timings, whose keys begin with
``config.RECORD_TIMING_PREFIX``, and the keys listed in
``config.RECORD_VOLATILE_KEYS``. The counts are taken from the record on disk on
every run, so this gate holds no copy of them the record could drift away from.
"""

from __future__ import annotations

import base64
import csv
import json
import re
import sys

from gate_lib import ROOT, check, config, finish, gate

sys.path.insert(0, str(ROOT))
from src.s99_utils import record_signature  # noqa: E402

gate("gate 07 notebooks")

NOTEBOOKS = ("01_cohort_quality_control.ipynb", "02_tumor_patch_cnn.ipynb")
# The helper both notebooks display a figure through. The argument names the
# file, so the embedded bytes have a stated counterpart in figures/ and the
# comparison does not have to guess which picture belongs to which cell.
DISPLAYS_FIGURE = re.compile(r"figure\(\s*[\"']([^\"']+)[\"']\s*\)")
SIGNATURE_LINE = re.compile(
    r"^{}\s+([0-9a-f]+)\s*$".format(re.escape(config.RECORD_SIGNATURE_LABEL)),
    re.M)

# --- the record the notebooks are held against -----------------------------
check("the metrics record is present", config.METRICS.exists(),
      str(config.METRICS.relative_to(ROOT)))
if not config.METRICS.exists():
    finish()

with open(config.METRICS, newline="", encoding="utf-8") as handle:
    rows_of_record = list(csv.DictReader(handle))
keys = [row["key"] for row in rows_of_record]
timings = [k for k in keys if k.startswith(config.RECORD_TIMING_PREFIX)]
volatile = [k for k in keys if k in config.RECORD_VOLATILE_KEYS]
compared = len(keys) - len(timings) - len(volatile)
check("the signature covers every quantity a re-run reproduces",
      len(timings) + len(volatile) < len(keys)
      and compared == len(set(keys) - set(timings) - set(volatile)),
      "{} quantities, {} timings, {} volatile, {} compared".format(
          len(keys), len(timings), len(volatile), compared))

signature = record_signature()
check("the signature is the configured width of hexadecimal digits",
      len(signature) == config.RECORD_SIGNATURE_DIGITS
      and all(c in "0123456789abcdef" for c in signature),
      "{} over {} quantities".format(signature, compared))

figures = {path.name: path.read_bytes()
           for path in sorted(config.FIGURES.glob("*.png"))}
written = {row["key"]: row["value"] for row in rows_of_record}
check("the figure directory holds the number of figures the record states",
      len(figures) == int(float(written.get("figures_written", -1))),
      "{} files, {} recorded".format(
          len(figures), written.get("figures_written", "absent")))

# --- each notebook ---------------------------------------------------------
embedded_files: set[str] = set()

for name in NOTEBOOKS:
    path = ROOT / "analysis" / name
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        cells = document["cells"]
        readable = True
    except (OSError, ValueError, KeyError) as error:
        cells, readable = [], False
        check("{} was read as a notebook document".format(name), False,
              str(error))
    if readable:
        check("{} was read as a notebook document".format(name), True,
              "{} cells".format(len(cells)))

    code = [c for c in cells if c.get("cell_type") == "code"]

    uncounted = [i for i, c in enumerate(code)
                 if not isinstance(c.get("execution_count"), int)]
    check("{} stored an execution count on every code cell".format(name),
          bool(code) and not uncounted,
          "{} of {} code cells hold none".format(len(uncounted), len(code)))

    counts = [c.get("execution_count") for c in code]
    check("{} execution counts run from one without a gap".format(name),
          counts == list(range(1, len(code) + 1)),
          "{} code cells, counts {}".format(
              len(code),
              "1 to {}".format(counts[-1]) if counts == list(
                  range(1, len(code) + 1)) else counts))

    errors = [output.get("ename", "error")
              for c in code for output in c.get("outputs", [])
              if output.get("output_type") == "error"]
    check("{} stored no error output".format(name), not errors,
          ", ".join(errors[:5]) if errors else
          "{} outputs".format(sum(len(c.get("outputs", [])) for c in code)))

    # The signature. It is printed as one string in one call, so the label and
    # the digest arrive in the same stream message and a stored output that
    # split them would be a document this gate should not accept.
    printed: list[str] = []
    for cell in code:
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            plain = output.get("data", {}).get("text/plain", "")
            text += "".join(plain) if isinstance(plain, list) else plain
            printed += SIGNATURE_LINE.findall(text)
    check("{} prints the record signature exactly once".format(name),
          len(printed) == 1,
          "{} lines beginning '{}'".format(len(printed),
                                           config.RECORD_SIGNATURE_LABEL))
    check("{} stored the signature of the record on disk".format(name),
          printed == [signature],
          "stored {}, recomputed {}".format(
              printed[0] if printed else "nothing", signature))

    # The figures. Each embedded image is paired with the file the cell that
    # produced it named, and the two are compared byte for byte.
    unnamed, unresolved, differing, matched = [], [], [], 0
    for position, cell in enumerate(code):
        named = DISPLAYS_FIGURE.findall("".join(cell.get("source", [])))
        images = [output["data"]["image/png"]
                  for output in cell.get("outputs", [])
                  if "image/png" in output.get("data", {})]
        if len(images) != len(named):
            unnamed.append("cell {} embeds {} images and names {} files".format(
                position + 1, len(images), len(named)))
            continue
        for filename, payload in zip(named, images):
            embedded_files.add(filename)
            if filename not in figures:
                unresolved.append(filename)
                continue
            if base64.b64decode(payload) != figures[filename]:
                differing.append(filename)
            else:
                matched += 1
    check("{} names a file in figures/ for every image it embeds".format(name),
          not unnamed and not unresolved,
          "; ".join(unnamed + ["{} is not in figures/".format(f)
                               for f in unresolved])
          if (unnamed or unresolved) else
          "{} images, each named".format(matched + len(differing)))
    check("{} embeds each figure exactly as figures/ holds it".format(name),
          not differing,
          ", ".join(sorted(set(differing))) if differing else
          "{} figures match byte for byte".format(matched))

# --- the figures the notebooks leave out -----------------------------------
# A figure the pipeline writes and neither notebook shows is a picture no reader
# of the deliverables ever sees, and a figure that was renamed would land here.
unshown = sorted(set(figures) - embedded_files)
check("every figure written to figures/ is embedded in a notebook", not unshown,
      ", ".join(unshown) if unshown else
      "{} figures across {} notebooks".format(len(figures), len(NOTEBOOKS)))

finish()
