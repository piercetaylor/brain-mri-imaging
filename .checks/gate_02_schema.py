#!/usr/bin/env python
"""Phase 2 gate: the header table holds every configured tag, correctly typed.

The pipeline addresses DICOM attributes by group and element number. This gate
re-reads a sample of instances straight from disk and confirms that the values
recorded in the table are the values the files carry.

Typing is checked on the stored strings and not on the objects ``load_headers``
returns. The loader casts every integer and decimal column as it reads, so a
type assertion placed after it restates the cast and passes on any table that
loads at all. The strings are therefore parsed here against the type each tag is
configured to hold, and the cast is checked by comparing the loaded value back
against the string it came from.

The series inventory is compared against ``data/manifest.csv``, which stage 1
wrote from the index and stage 2 never reads. Comparing the inventory against
the header rows it was aggregated from would restate the aggregation.
"""

from __future__ import annotations

import csv
import datetime
import random
import re
import sys

import pydicom

from gate_lib import ROOT, check, config, finish, gate, metrics, table

sys.path.insert(0, str(ROOT))
from src.s02_headers import HEADER_TABLE, load_headers  # noqa: E402

gate("gate 02 schema")

check("the header table exists", HEADER_TABLE.exists(), str(HEADER_TABLE))
if not HEADER_TABLE.exists():
    finish()

with open(HEADER_TABLE, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    columns = list(reader.fieldnames)
    stored = list(reader)
missing_columns = [name for name in config.HEADER_TAGS if name not in columns]
check("every configured tag is a column", not missing_columns,
      ", ".join(missing_columns) if missing_columns else
      "{} tags".format(len(config.HEADER_TAGS)))

recorded = metrics()
check("the row count matches the recorded instance count",
      len(stored) == int(recorded.number("headers_instances")),
      "{} rows, {} recorded".format(len(stored),
                                    recorded.get("headers_instances")))

# --- the stored strings, before the loader casts them ----------------------
# A tag whose column is absent is reported by the failure above. Naming it again
# in each parsing check would turn one defect into six, so an absent column is
# left out below and each check states how many tags it covered.
INTEGER = re.compile(r"^-?\d+$")
DECIMAL = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LIBRARY_LIST = re.compile(r"^\[.*\]$")


def tags_of(kind: str) -> list[str]:
    return [name for name, (_, configured) in config.HEADER_TAGS.items()
            if configured == kind and name in columns]


def offenders(names, predicate) -> list[str]:
    """The first value in each named column the predicate rejects."""
    found = []
    for name in names:
        for row in stored:
            if not predicate(row[name]):
                found.append("{}={!r}".format(name, row[name]))
                break
    return found


def is_date(text: str) -> bool:
    if not ISO_DATE.match(text):
        return False
    try:
        datetime.date.fromisoformat(text)
    except ValueError:
        return False
    return True


integers = tags_of("int")
bad = offenders(integers, lambda v: bool(INTEGER.match(v)))
check("every integer tag holds an integer before coercion", not bad,
      ", ".join(bad) if bad else
      "{} tags over {} rows".format(len(integers), len(stored)))

decimals = tags_of("float")
bad = offenders(decimals, lambda v: bool(DECIMAL.match(v)))
check("every decimal tag holds a decimal number before coercion", not bad,
      ", ".join(bad) if bad else
      "{} tags over {} rows".format(len(decimals), len(stored)))

dates = tags_of("date")
bad = offenders(dates, is_date)
check("every date tag holds a calendar date in ISO form before coercion",
      not bad, ", ".join(bad) if bad else
      "{} tags over {} rows".format(len(dates), len(stored)))

# A multi-valued decimal string is stored in the DICOM form, with a backslash
# between the values. Stage 3 splits on that character, so a value written in
# the reading library's bracketed form would parse to nothing.
multiples = tags_of("decimal_string")
bad = offenders(
    multiples,
    lambda v: bool(v) and all(DECIMAL.match(part) for part in v.split("\\")))
check("every multi-valued decimal tag holds backslash-separated decimals "
      "before coercion", not bad, ", ".join(bad) if bad else
      "{} tags over {} rows".format(len(multiples), len(stored)))

text_tags = tags_of("str")
bad = offenders(text_tags, lambda v: v == v.strip())
check("every text tag is stored without surrounding space before coercion",
      not bad, ", ".join(bad) if bad else
      "{} tags over {} rows".format(len(text_tags), len(stored)))
bad = offenders(text_tags, lambda v: not LIBRARY_LIST.match(v))
check("no text tag holds the reading library's list form", not bad,
      ", ".join(bad) if bad else
      "{} tags over {} rows".format(len(text_tags), len(stored)))

# The cast itself. The loaded value must name the same number as the string it
# was read from, which is what a type assertion after the cast does not say.
rows = load_headers()
numeric = integers + decimals
mismatched = [
    "{} row {}".format(name, position)
    for name in numeric
    for position, (loaded, source) in enumerate(zip(rows, stored))
    if float(loaded[name]) != float(source[name])
]
check("the loaded value names the number its stored string holds",
      not mismatched, ", ".join(mismatched[:5]) if mismatched else
      "{} tags over {} rows".format(len(numeric), len(rows)))

for name in ("patient_id", "modality", "manufacturer", "series_desc",
             "photometric_interpretation"):
    blank = sum(1 for r in rows if not str(r[name]).strip())
    check("{} is populated on every instance".format(name), blank == 0,
          "{} blank".format(blank))

check("row and column counts are positive",
      all(r["rows"] > 0 and r["cols"] > 0 for r in rows))
check("every instance is magnetic resonance",
      {r["modality"] for r in rows} == {"MR"},
      ", ".join(sorted({r["modality"] for r in rows})))

# --- the inventory against the manifest ------------------------------------
# The manifest is stage 1's record of what the index returned and what was
# downloaded. Stage 2 never reads it, so it is an independent statement of which
# series exist, whom they belong to, what produced them, and how many instances
# each holds.
inventory = table("series_inventory")
with open(config.MANIFEST, newline="", encoding="utf-8") as handle:
    manifest = [r for r in csv.DictReader(handle) if r["role"] == "image"]
manifested = {r["series_uid"]: r for r in manifest}
shared = [r for r in inventory if r["series_uid"] in manifested]

check("the inventory names the image series the manifest names",
      {r["series_uid"] for r in inventory} == set(manifested),
      "{} in the inventory, {} manifested".format(len(inventory),
                                                  len(manifested)))
disagreeing = [r["series_uid"] for r in shared
               if int(r["instances"])
               != int(manifested[r["series_uid"]]["instance_count"])]
check("the inventory instance counts match the manifest", not disagreeing,
      "{} of {} series disagree".format(len(disagreeing), len(shared)))
check("the inventory instance counts sum to the manifested total",
      sum(int(r["instances"]) for r in inventory)
      == sum(int(r["instance_count"]) for r in manifest),
      "{} in the inventory, {} manifested".format(
          sum(int(r["instances"]) for r in inventory),
          sum(int(r["instance_count"]) for r in manifest)))
disagreeing = [r["series_uid"] for r in shared
               if r["patient_id"] != manifested[r["series_uid"]]["patient_id"]]
check("the inventory assigns every series to the manifested patient",
      not disagreeing,
      "{} patients, {} of {} series disagree".format(
          len({r["patient_id"] for r in inventory}), len(disagreeing),
          len(shared)))
disagreeing = [r["series_uid"] for r in shared
               if (r["manufacturer"], r["model_name"])
               != (manifested[r["series_uid"]]["manufacturer"],
                   manifested[r["series_uid"]]["model_name"])]
check("the inventory names the manufacturer and model the manifest names",
      not disagreeing,
      "{} scanner combinations, {} of {} series disagree".format(
          len({(r["manufacturer"], r["model_name"]) for r in inventory}),
          len(disagreeing), len(shared)))
disagreeing = [r["series_uid"] for r in shared
               if r["series_desc"] != manifested[r["series_uid"]]["series_desc"]]
check("the inventory carries the manifested series description",
      not disagreeing,
      "{} of {} series disagree".format(len(disagreeing), len(shared)))

# --- the table against the files -------------------------------------------
# Re-read a sample of files and compare against the recorded values. This is the
# check that would catch a table written from anything but the files. Every
# configured tag is compared, dates and decimal strings included, because the
# coercion those pass through is where a value is most likely to be lost.
random.seed(config.SEED)
sample = random.sample(rows, min(40, len(rows)))
disagreements = []
for row in sample:
    matches = list((config.RAW / row["patient_id"] / "image" / row["series_uid"])
                   .glob(row["file"]))
    if not matches:
        disagreements.append(row["file"] + " is absent")
        continue
    dataset = pydicom.dcmread(matches[0], stop_before_pixels=True)
    for name, (tag, kind) in config.HEADER_TAGS.items():
        if name not in columns:
            continue
        element = dataset.get(tag)
        value = "" if element is None else element.value
        if kind == "int":
            agrees = int(value) == row[name]
        elif kind == "float":
            agrees = abs(float(value) - row[name]) < 1e-9
        elif kind == "date":
            text = str(value)
            expected = "{}-{}-{}".format(text[0:4], text[4:6], text[6:8]) \
                if len(text) == 8 else text
            agrees = row[name] == expected
        elif kind == "decimal_string":
            agrees = row[name] == (value.strip() if isinstance(value, str)
                                   else "\\".join(str(v) for v in value))
        else:
            agrees = str(value).strip() == row[name]
        if not agrees:
            disagreements.append("{} {}".format(row["file"], name))
check("re-reading a sample of files reproduces the table",
      not disagreements,
      "{} instances by {} tags checked, {} disagreements".format(
          len(sample), len([n for n in config.HEADER_TAGS if n in columns]),
          len(disagreements)))

finish()
