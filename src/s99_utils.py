"""Shared helpers: remote queries, object downloads, and the metrics record.

Results are written once, to ``results/metrics.csv``, by whichever stage
computes them. Every number quoted in the README or the write-up is read back
out of that file, so a claim in the prose and a claim in the pipeline cannot
drift apart.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import config

_S3_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
_USER_AGENT = "brain-mri-imaging/1.0 (+https://github.com/piercetaylor)"


# --- remote access ---------------------------------------------------------
RETRIES = 6


def _request(url: str, data: bytes | None = None, headers: dict | None = None):
    hdr = {"User-Agent": _USER_AGENT}
    hdr.update(headers or {})
    return urllib.request.Request(url, data=data, headers=hdr)


def _read(url: str, timeout: int = 300) -> bytes:
    """Fetch one URL, retrying on the transient failures a long run will hit."""
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(_request(url), timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            last = error
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("{} failed after {} attempts: {}".format(url, RETRIES, last))


def idc_sql(sql: str, max_rows: int = 5000) -> list[dict]:
    """Run one read-only SELECT against the Imaging Data Commons index."""
    body = json.dumps({"sql": sql, "max_rows": max_rows}).encode()
    req = _request(
        f"{config.IDC_API}/sql",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        payload = json.loads(response.read().decode())
    if payload.get("truncated"):
        raise RuntimeError(
            f"the index returned a truncated result for: {sql[:120]}..."
        )
    return payload["rows"]


def idc_version() -> dict:
    with urllib.request.urlopen(_request(f"{config.IDC_API}/version"), timeout=120) as r:
        return json.loads(r.read().decode())


def nbia_get(endpoint: str, **params) -> Any:
    """Query the NBIA REST service that serves the collection at its source."""
    url = f"{config.NBIA_API}/{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(_request(url), timeout=300) as response:
        return json.loads(response.read().decode())


def s3_list(prefix: str) -> list[tuple[str, int]]:
    """List every object under one series prefix in the public IDC bucket."""
    keys: list[tuple[str, int]] = []
    token: str | None = None
    while True:
        url = f"{config.S3_HOST}/?list-type=2&prefix={prefix}/&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token, safe="")
        root = ET.fromstring(_read(url, timeout=180).decode())
        for node in root.findall("s:Contents", _S3_NS):
            keys.append(
                (node.find("s:Key", _S3_NS).text, int(node.find("s:Size", _S3_NS).text))
            )
        nxt = root.find("s:NextContinuationToken", _S3_NS)
        token = nxt.text if nxt is not None else None
        if not token:
            break
    return sorted(keys)


def s3_download(key: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    tmp.write_bytes(_read(f"{config.S3_HOST}/{key}", timeout=600))
    os.replace(tmp, destination)
    return destination


def parallel(function, items: Sequence, workers: int = 24) -> list:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, items))


# --- digests ---------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_series(directory: Path) -> tuple[str, int, int]:
    """Digest of a whole series: the digests of its files, sorted and hashed.

    Returns the digest, the number of files, and the total size in bytes. The
    file names are IDC object identifiers and are stable, but sorting the
    per-file digests makes the series digest independent of them.
    """
    files = sorted(p for p in directory.glob("*.dcm"))
    parts = sorted(sha256_file(p) for p in files)
    total = sum(p.stat().st_size for p in files)
    outer = hashlib.sha256("\n".join(parts).encode()).hexdigest()
    return outer, len(files), total


# --- the metrics record ----------------------------------------------------
class Metrics:
    """Append-only key/value record of everything the pipeline measured."""

    def __init__(self, path: Path = config.METRICS):
        self.path = path
        self.values: dict[str, str] = {}
        if path.exists():
            with open(path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    self.values[row["key"]] = row["value"]

    def set(self, key: str, value: Any) -> None:
        if isinstance(value, float):
            value = f"{value:.6g}"
        self.values[key] = str(value)

    def update(self, mapping: dict) -> None:
        for key, value in mapping.items():
            self.set(key, value)

    def get(self, key: str) -> str:
        if key not in self.values:
            raise KeyError(f"metric '{key}' was never recorded")
        return self.values[key]

    def number(self, key: str) -> float:
        return float(self.get(key))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["key", "value"])
            for key in sorted(self.values):
                writer.writerow([key, self.values[key]])


def write_table(rows: Iterable[dict], path: Path, columns: Sequence[str] | None = None) -> int:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write an empty table to {path}")
    columns = list(columns or rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def otsu_threshold(values, bins: int = 256) -> float:
    """Otsu's threshold: the intensity that best separates two classes.

    Magnetic resonance volumes hold a large air compartment whose voxels are
    not zero, because reconstruction noise fills it. A test for a non-zero
    value therefore accepts background as tissue. Otsu's method chooses the
    threshold maximising between-class variance and needs no tuned constant.
    """
    import numpy

    values = numpy.asarray(values).reshape(-1)
    counts, edges = numpy.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_below = numpy.cumsum(counts)
    total = weight_below[-1]
    if total == 0:
        return 0.0
    weight_above = total - weight_below
    moment = numpy.cumsum(counts * centers)
    usable = (weight_below > 0) & (weight_above > 0)
    mean_below = numpy.where(usable, moment / numpy.maximum(weight_below, 1), 0.0)
    mean_above = numpy.where(
        usable, (moment[-1] - moment) / numpy.maximum(weight_above, 1), 0.0)
    between = (weight_below / total) * (weight_above / total) * (
        mean_below - mean_above) ** 2
    return float(centers[int(numpy.argmax(numpy.where(usable, between, -1.0)))])


def banner(text: str) -> None:
    print(f"\n=== {text} ===", flush=True)
