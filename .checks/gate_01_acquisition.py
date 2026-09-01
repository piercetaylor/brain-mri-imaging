#!/usr/bin/env python
"""Phase 1 gate: the files analyzed are the files that were published.

Four statements are compared. The digest recorded in ``data/checksums.txt``,
the digest the downloaded series actually has, the license recorded in
``data/manifest.csv``, and the license two independent services report for the
same series today. The last of the four is the only one this repository does
not control.
"""

from __future__ import annotations

import csv
import sys

from gate_lib import (ROOT, NETWORK_ERRORS, check, config, finish, gate,
                      metrics, table, unreachable)

sys.path.insert(0, str(ROOT))
from src.s99_utils import idc_sql, nbia_get, sha256_series  # noqa: E402

gate("gate 01 acquisition")

check("data/manifest.csv is present", config.MANIFEST.exists())
check("data/checksums.txt is present", config.CHECKSUMS.exists())
if not (config.MANIFEST.exists() and config.CHECKSUMS.exists()):
    finish()

with open(config.MANIFEST, newline="", encoding="utf-8") as handle:
    manifest = list(csv.DictReader(handle))

check("the manifest names at least the minimum cohort",
      len({r["patient_id"] for r in manifest}) >= config.COHORT_MIN,
      "{} patients".format(len({r["patient_id"] for r in manifest})))
check("every manifest row records the expected license",
      {r["license"] for r in manifest} == {config.EXPECTED_LICENSE},
      ", ".join(sorted({r["license"] for r in manifest})))
check("every patient carries exactly one label series",
      len([r for r in manifest if r["role"] == "segmentation"])
      == len({r["patient_id"] for r in manifest}),
      "{} label series".format(len([r for r in manifest if r["role"] == "segmentation"])))
check("every label series names an image series in the manifest",
      all(r["labels_series_uid"] in {i["series_uid"] for i in manifest
                                     if i["role"] == "image"}
          for r in manifest if r["role"] == "segmentation"))
# The cohort rule qualifies a series on at least one segment carrying the
# semi-automatic algorithm type. It says nothing about the remaining segments,
# and in this collection every qualifying series also holds automatic ones.
# Stage 4 measures what share of the positive voxels a reviewed segment
# contributed, so the label is checked here for the property the rule asserts.
labels = [r for r in manifest if r["role"] == "segmentation"]
check("every label series carries at least one reviewed segment",
      all(config.CORRECTED_ALGORITHM_TYPE in r["algorithm_type"]
          for r in labels),
      "{} of {} series".format(
          sum(1 for r in labels
              if config.CORRECTED_ALGORITHM_TYPE in r["algorithm_type"]),
          len(labels)))

recorded = {}
for line in config.CHECKSUMS.read_text(encoding="utf-8").splitlines():
    if line.startswith("#") or not line.strip():
        continue
    sha, files, size, uid = line.split()
    recorded[uid] = (sha, int(files), int(size))

check("checksums.txt covers every manifested series",
      set(recorded) == {r["series_uid"] for r in manifest},
      "{} recorded, {} manifested".format(len(recorded), len(manifest)))

missing = [r for r in manifest
           if not (config.RAW / r["patient_id"] / r["role"] / r["series_uid"]).exists()]
check("every manifested series is on disk", not missing,
      "{} absent".format(len(missing)))

if not missing:
    mismatched, counted = [], 0
    for row in manifest:
        directory = config.RAW / row["patient_id"] / row["role"] / row["series_uid"]
        digest, files, size = sha256_series(directory)
        counted += files
        if recorded[row["series_uid"]] != (digest, files, size):
            mismatched.append(row["series_uid"])
    check("every series matches its recorded digest", not mismatched,
          "{} series, {} files".format(len(manifest), counted))
    check("the file count matches the manifest",
          counted == sum(int(r["instance_count"]) for r in manifest),
          "{} on disk, {} manifested".format(
              counted, sum(int(r["instance_count"]) for r in manifest)))

check("raw data is excluded from version control",
      "data/raw/" in (ROOT / ".gitignore").read_text(encoding="utf-8"))

# The checks this repository cannot fake: what the two services say today.
sample = [r["series_uid"] for r in manifest[:20]]
try:
    reported = idc_sql(
        "SELECT DISTINCT license_short_name AS license FROM index "
        "WHERE SeriesInstanceUID IN ({})".format(
            ", ".join("'" + uid + "'" for uid in sample))
    )
except NETWORK_ERRORS as error:
    unreachable("the data commons index reports the expected license", error)
else:
    names = sorted({row.get("license", "the license field is absent")
                    for row in reported})
    check("the data commons index reports the expected license",
          names == [config.EXPECTED_LICENSE], ", ".join(names))

patient = sorted({r["patient_id"] for r in manifest})[0]
try:
    series = nbia_get("getSeries", Collection=config.NBIA_COLLECTION,
                      PatientID=patient)
except NETWORK_ERRORS as error:
    unreachable("the archive that published the images reports CC BY 4.0", error)
else:
    uris = sorted({s.get("LicenseURI", "the LicenseURI field is absent")
                   for s in series})
    check("the archive that published the images reports CC BY 4.0",
          uris == [config.EXPECTED_LICENSE_URI],
          "{}: {}".format(patient, ", ".join(uris)))

recorded_metrics = metrics()
check("the recorded download matches what is on disk",
      int(recorded_metrics.number("download_series")) == len(manifest),
      "{} series recorded".format(recorded_metrics.get("download_series")))

finish()
