#!/usr/bin/env python
"""Execute both notebooks against the record that is on disk and store the run.

    python analysis/run_notebooks.py

The notebooks display ``results/`` and ``figures/`` and compute nothing of their
own, so their stored output belongs to whichever record existed when they last
ran. Run this after ``analysis/run_all.py``, whenever the record or the figures
change. Gate 07 compares what the notebooks stored against what is on disk and
fails when the two have parted.

Every cell is run in order in a fresh kernel, so the stored execution counts run
from one without a gap and the document is one that was produced end to end.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

NOTEBOOKS = ("01_cohort_quality_control.ipynb", "02_tumor_patch_cnn.ipynb")
# Long enough for the cell that loads the patch array and the trained network.
CELL_TIMEOUT = 900


def main() -> int:
    for name in NOTEBOOKS:
        path = ROOT / "analysis" / name
        started = time.time()
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=CELL_TIMEOUT,
                                kernel_name="python3",
                                resources={"metadata": {"path": str(path.parent)}})
        client.execute()
        nbformat.write(notebook, path)
        print("  {:<34} {:>3} cells in {:>6.1f}s".format(
            name, len(notebook.cells), time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
