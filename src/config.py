"""Paths, constants and the single random seed used across the pipeline.

Every module imports its parameters from here so that the values recorded in
``results/metrics.csv`` can be traced to one place.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

MANIFEST = DATA / "manifest.csv"
CHECKSUMS = DATA / "checksums.txt"
METRICS = RESULTS / "metrics.csv"

SEED = 20251117

# --- data source -----------------------------------------------------------
# The images are the UPENN-GBM collection (Bakas et al. 2022). The tumor
# segmentations are the BAMF annotations contributed to the same collection
# through the Imaging Data Commons AIMI project (Murugesan et al. 2024).
IDC_API = "https://api.imaging.datacommons.cancer.gov/v3"
NBIA_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
S3_BUCKET = "idc-open-data"
S3_HOST = f"https://{S3_BUCKET}.s3.amazonaws.com"

IDC_COLLECTION = "upenn_gbm"
NBIA_COLLECTION = "UPENN-GBM"
IMAGE_DOI = "10.7937/tcia.709x-dn49"
SEGMENTATION_DOI = "10.5281/zenodo.8345959"
EXPECTED_LICENSE = "CC BY 4.0"
EXPECTED_LICENSE_URI = "https://creativecommons.org/licenses/by/4.0/"

# The segmentation series used for labels are the ones a radiologist reviewed
# and corrected; DICOM records that review as a SEMIAUTOMATIC segment
# algorithm type. The labeled series is the axial T1 post-contrast
# acquisition, which is the series the original coursework also selected.
CORRECTED_ALGORITHM_TYPE = "SEMIAUTOMATIC"
LABEL_SERIES_PATTERN = "t1 axial stealth-post%"
TUMOR_SEGMENT_LABELS = ("Necrosis", "Edema", "Enhancing Lesion")

# Deterministic subset: eligible patients are sorted by identifier and at most
# COHORT_MAX are taken. The full collection is 630 patients and 139.4 GB, so
# the whole of it is not moved. The run fails if fewer than COHORT_MIN patients
# are eligible, because a smaller cohort would not support a patient-level
# split.
COHORT_MAX = 50
COHORT_MIN = 30

# --- DICOM header extraction ----------------------------------------------
# Tags are addressed by group and element number, as in the original exercise,
# so that a renamed keyword in a future pydicom release cannot silently change
# what is read.
HEADER_TAGS: dict[str, tuple[int, str]] = {
    "patient_id": (0x00100020, "str"),
    "body_part": (0x00180015, "str"),
    "study_date": (0x00080020, "date"),
    "series_date": (0x00080021, "date"),
    "content_date": (0x00080023, "date"),
    "accession_number": (0x00080050, "str"),
    "modality": (0x00080060, "str"),
    "manufacturer": (0x00080070, "str"),
    "study_desc": (0x00081030, "str"),
    "series_desc": (0x0008103E, "str"),
    "model_name": (0x00081090, "str"),
    "slice_location": (0x00201041, "float"),
    "rows": (0x00280010, "int"),
    "cols": (0x00280011, "int"),
    "photometric_interpretation": (0x00280004, "str"),
    "bits_allocated": (0x00280100, "int"),
}

# Tags that must be absent or empty for the collection to be de-identified.
PHI_TAGS_MUST_BE_EMPTY: dict[str, int] = {
    "PatientBirthDate": 0x00100030,
    "PatientAddress": 0x00101040,
    "PatientTelephoneNumbers": 0x00102154,
    "InstitutionAddress": 0x00080081,
    "ReferringPhysicianName": 0x00080090,
    "PerformingPhysicianName": 0x00081050,
    "OperatorsName": 0x00081070,
    "OtherPatientIDs": 0x00101000,
}
PHI_TAG_PATIENT_NAME = 0x00100010
PHI_TAG_PATIENT_ID = 0x00100020
PHI_TAG_BURNED_IN = 0x00280301
PHI_TAG_IDENTITY_REMOVED = 0x00120062

# --- patch extraction ------------------------------------------------------
PATCH_SIZE = 32
PATCH_STRIDE = 8
# A patch is positive when at least this fraction of its pixels lie inside the
# whole-tumor mask, and negative when none of them do. Patches between the
# two are ambiguous and are discarded.
POSITIVE_TUMOR_FRACTION = 0.5
# A negative patch must be mostly brain, so that the classifier is not
# separating tissue from air.
NEGATIVE_MIN_BRAIN_FRACTION = 0.5
SLICE_MIN_BRAIN_FRACTION = 0.10
MAX_POSITIVE_PATCHES_PER_PATIENT = 500
NEGATIVES_PER_POSITIVE = 2

# --- splits and training ---------------------------------------------------
# Splits are over patients. Patches from one patient overlap and are near
# duplicates of one another, so a patch-level split would place copies of the
# same tissue on both sides of the partition.
SPLIT_FRACTIONS = {"train": 0.6, "validation": 0.15, "test": 0.25}
BATCH_SIZE = 64
MAX_EPOCHS = 20
GRID_EPOCHS = 8
LEARNING_RATE_GRID = (1.0, 0.1)
DROPOUT_GRID = (0.3, 0.5)
