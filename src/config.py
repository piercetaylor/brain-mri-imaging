"""Paths, constants and the random seeds used across the pipeline.

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

# The primary seed. Every headline quantity is produced at this seed, so a
# reader can reproduce it exactly. SEED_LIST holds the seeds the variance
# comparison in stage 6 runs over; the primary seed is the first of them, and
# the run at that seed is the run the headline numbers come from.
SEED = 20251117
SEED_LIST = (20251117, 20251118, 20251119)

# Resamples drawn when the test ROC-AUC is given a patient-level bootstrap
# interval in stage 7. The interval is a percentile interval at this level.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_INTERVAL = 0.95

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

# A segmentation series qualifies when at least one of its segments carries the
# SEMIAUTOMATIC algorithm type, which DICOM uses for a contour a person
# reviewed. The filter is a property of the series and not of every segment in
# it: a qualifying series can still contain segments marked AUTOMATIC, and in
# this collection every one of them does. Stage 4 reads the algorithm type of
# each tumor segment separately and records what share of the positive voxels
# came from a reviewed segment, so the strength of the label is measured and
# not assumed.
CORRECTED_ALGORITHM_TYPE = "SEMIAUTOMATIC"

# The labeled series is a post-contrast T1 acquisition. Six scanner models
# contributed to this collection and each names that acquisition differently,
# so the series is identified by markers and not by one protocol name. A
# description qualifies when it carries a T1 marker and a post-contrast marker
# and carries none of the excluded markers. Matching is case-insensitive.
# Measured against the index on 2026-08-30, this selects eight distinct
# descriptions across six scanner models and rejects the pre-contrast T1, T2
# and FLAIR descriptions that sit beside them in the same studies.
LABEL_REQUIRED_MARKERS = ("t1", "post")
LABEL_EXCLUDED_MARKERS = ("t2", "flair")
TUMOR_SEGMENT_LABELS = ("Necrosis", "Edema", "Enhancing Lesion")

# The tokens a description uses to name which side of the contrast injection a
# series was acquired on. The rule reads the first of them and nothing else, so
# a description carrying neither token names no contrast state and the rule
# cannot admit it however the series was acquired. Measured against the index
# on 2026-08-30, 11 of the 43 rows of the segmented description vocabulary are
# T1 acquisitions of that kind, over 10 distinct descriptions, and
# results/segmented_series_descriptions.csv lists them.
CONTRAST_STATE_MARKERS = ("post", "pre")

# The structural acquisitions the labeled task depends on. Everything else in
# the cohort is a diffusion or perfusion series. The markers cover the T1, T2
# and FLAIR naming of every scanner model in the cohort; a marker tied to one
# vendor's protocol name would silently reclassify another vendor's structural
# series as non-structural.
STRUCTURAL_MARKERS = ("t1", "t2", "flair")

# What the rule admits and what it rejects, as the index spells them. Measured
# against the index on 2026-08-30: 43 description, manufacturer and model
# combinations carry a segmentation with a SEMIAUTOMATIC segment, over 37
# distinct descriptions, and the rule admits 9 of those rows over 8 distinct
# descriptions. The lists below are drawn from that vocabulary and are held in
# one place because gate 00 applies them to the rule directly and gate 01
# applies them to the vocabulary the index returns today, so a description that
# left the collection cannot go on standing as a fixture.
#
# The rejected list carries the rule's boundary as well as its intent. Three of
# its entries are post-contrast T1 acquisitions in name only: their descriptions
# carry no post-contrast token, so the rule cannot see them and excludes them.
# One entry, "Ax T2 POST", is post-contrast and is excluded for being T2, which
# is the rule working as stated.
LABEL_RULE_ADMIT_EXAMPLES = (
    "t1 axial stealth-post : Processed_CaPTk",
    "t1 axial stealth-post_T : Processed_CaPTk",
    "t1 axial stealth-post_TERA: Processed_CaPTk",
    "AX T1 3D POST STEALTH : Processed_CaPTk",
    "AX T1 POST: Processed_CaPTk",
    "AX T1 POST STEALTH: Processed_CaPTk",
    "AXIAL T1 FLOW COMP POST : Processed_CaPTk",
    "POST AX T1  FS FSE: Processed_CaPTk",
)
LABEL_RULE_REJECT_EXAMPLES = (
    "AX T1 PRE : Processed_CaPTk",
    "AX T1 3D PRE: Processed_CaPTk",
    "Ax T2 POST: Processed_CaPTk",
    "t1 axial: Processed_CaPTk",
    "AX T1 MPRAGE ISOTROPIC: Processed_CaPTk",
    "AXIAL T1 MPRAGE BRAIN : Processed_CaPTk",
    "AXIAL T1 BRAIN 3D : Processed_CaPTk",
    "t2_Flair_axial: Processed_CaPTk",
    "AXIAL T2 FLAIR: Processed_CaPTk",
    "T2 SAG SPACE: Processed_CaPTk",
)


def label_series_predicate(column: str) -> str:
    """SQL that is true for a post-contrast T1 series description.

    ``column`` is the qualified name of the description column. The clauses are
    lowercase substring tests, so the rule does not depend on how any one
    scanner capitalized its protocol name.
    """
    clauses = ["LOWER({}) LIKE '%{}%'".format(column, marker)
               for marker in LABEL_REQUIRED_MARKERS]
    clauses += ["LOWER({}) NOT LIKE '%{}%'".format(column, marker)
                for marker in LABEL_EXCLUDED_MARKERS]
    return "(" + " AND ".join(clauses) + ")"


def is_label_series_description(description: str) -> bool:
    """The same rule as :func:`label_series_predicate`, applied in Python."""
    text = str(description).lower()
    return (all(marker in text for marker in LABEL_REQUIRED_MARKERS)
            and not any(marker in text for marker in LABEL_EXCLUDED_MARKERS))


# Deterministic subset. The full collection is 630 patients and 139.4 GB, so
# the whole of it is not moved. When more than COHORT_MAX patients are
# eligible the cap is filled by taking patients from each scanner model in
# turn, in sorted order within a model, so that capping cannot drop the
# scanner models that contribute fewest patients. The run fails if fewer than
# COHORT_MIN patients are eligible, because a smaller cohort would not support
# a patient-level split.
COHORT_MAX = 50
COHORT_MIN = 30

# --- DICOM header extraction ----------------------------------------------
# Tags are addressed by group and element number, as in the original exercise,
# so that a renamed keyword in a future pydicom release cannot silently change
# what is read.
#
# The second member of each pair is the type the extraction coerces to.
# ImageOrientationPatient is the one multi-valued tag here: it holds six
# decimal strings, so neither of the scalar numeric kinds fits it. Coercing it
# with "float" raises TypeError, because float() will not take a six-element
# value, and coercing it with "str" produces the reading library's list
# representation, "[0.999331, 2.54313e-07, ...]", whose brackets and commas are
# not the DICOM form. The "decimal_string" kind joins the values with the
# backslash the standard uses and preserves every digit, which matters because
# the orientation comparison in stage 3 turns on the sixth significant figure.
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
    "image_orientation": (0x00200037, "decimal_string"),
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

# --- image geometry --------------------------------------------------------
# Before it will assemble a series into a volume, highdicom compares the
# ImageOrientationPatient of every slice, and with its orientation_tol argument
# left at None that comparison is an equality test on the stored text. One of
# the 49 labeled series fails it. UPENN-GBM-00452, the only patient on the GE
# MEDICAL SYSTEMS DISCOVERY MR750w, carries five distinct orientation values
# across the 35 slices of "POST AX T1  FS FSE: Processed_CaPTk", divided 20, 9,
# 3, 2 and 1. The other 48 labeled series each carry one value, which is why
# the failure was invisible while the cohort was 41 patients on one scanner.
#
# The largest angular disagreement between a variant and the modal orientation
# of its series, taken over the 49 labeled series and excluding the modal
# variant compared against itself, is 7.77e-15 in 1 minus cosine similarity.
# That is an angle of 1.25e-07 radians, which displaces 3.12e-05 mm across the
# widest field of view in the cohort, the 250.0 mm spanned by the 256 rows at
# 0.9765625 mm of UPENN-GBM-00019. Stage 3 repeats the measurement over all
# 273 image series and finds the same 7.77e-15, because the only other two
# series carrying several orientation values are the AX T1 FSE and AX FLAIR FS
# acquisitions of the same patient, both at 5.11e-15.
#
# The quantity highdicom tests is not that cosine similarity. It is 1 minus the
# raw dot product of two direction cosine vectors whose six-significant-figure
# decimal strings leave them slightly off unit length. The largest unit-norm
# defect over every direction vector in the 49 labeled series is 3.98e-07, and
# in UPENN-GBM-00452 it carries the tested quantity to -7.97e-07, below zero
# and so below any positive tolerance. The sign of that defect follows from how
# a decimal string rounds and not from the geometry, so the tolerance is sized
# above its magnitude and not below it.
#
# ORIENTATION_TOLERANCE is therefore 12.5 times the 7.97e-07 the tested
# quantity reaches and 1.3e9 times the 7.77e-15 of genuine angular
# disagreement. It admits an angle of 4.47e-03 radians, which displaces 1.12 mm
# across that same 250.0 mm field of view, and rejects everything larger, so
# slices genuinely acquired at different angles still raise.
# ORIENTATION_TOLERANCE_MARGIN is the factor gate 03 holds the cohort to, so
# that a deviation growing toward the tolerance fails the gate before it
# reaches it.
ORIENTATION_TOLERANCE = 1e-5
ORIENTATION_TOLERANCE_MARGIN = 10.0

# Stage 4 then asks highdicom to place the segmentation on the image grid.
# Volume.match_geometry holds three quantities to one tolerance whose default
# is 1e-5: the dot product of each pair of direction vectors, the ratio of the
# two spacings along a matched axis, and the translation between the two
# origins expressed in voxels. The translation is the one this cohort fails.
#
# Measured over the 49 image and segmentation pairs, the largest translation
# residual is 1.736e-03 voxels, on the row axis of UPENN-GBM-00452, the GE
# MEDICAL SYSTEMS DISCOVERY MR750w patient. That would move the mask 8.138e-04
# mm along that axis, at the 0.468800 mm in-plane spacing of the series. The
# next largest is 5.195e-05 voxels on UPENN-GBM-00625, a SIEMENS Verio, and the
# cohort median is 5.744e-06 voxels. Five pairs exceed the 1e-5 default: those
# two, UPENN-GBM-00487 and UPENN-GBM-00494 on SIEMENS Avanto scanners at
# 3.506e-05 and 1.317e-05 voxels, and UPENN-GBM-00609 on a SIEMENS SymphonyTim
# at 1.787e-05 voxels. The other 44 pairs sit below 1e-5.
#
# Every one of those residuals is the rounding of a decimal string and none is
# a position the two series disagree on. Three strings carry it, and each can
# be read off the headers. The segmentation of UPENN-GBM-00625 writes
# PixelSpacing as 4.882812e-01 against the image's 0.48828125 mm, a relative
# difference of 1.024e-07, and that pair's spacing ratio residual is 1.024e-07.
# The segmentation of UPENN-GBM-00452 writes SpacingBetweenSlices as
# 4.999992e+00 against the 5.000008 mm the image series describes, a relative
# difference of 3.181e-06, and that pair's spacing ratio residual is 3.181e-06.
# The same patient writes ImagePositionPatient to six significant figures, so
# each of its coordinates near 200 mm is quantized at 1e-03 mm.
#
# That last quantum is where the line falls. Three coordinates each rounded to
# within 5e-04 mm can misplace an origin along a direction by at most
# sqrt(3) * 5e-04 mm, which is 8.66e-04 mm, and the largest displacement the
# cohort holds is the 8.138e-04 mm of UPENN-GBM-00452. It sits inside the bound
# the strings themselves impose, so the series cannot report the position any
# more precisely than the residual, and the residual is representation noise.
# Every other pair is at least an order of magnitude below the bound. A
# displacement outside it would be a position the label and the image genuinely
# disagree on, and admitting one would move the mask off the tissue it
# describes, so the tolerance is not sized to reach it.
#
# The residual and the displacement are each maximized over the three axes
# independently, so the two need not fall on the same axis. On at least two
# of the 49 pairs they do not. The displacement divided by the residual is
# 4.759 on UPENN-GBM-00609 and 3.293 on UPENN-GBM-00494, and neither quotient
# is a spacing this cohort carries. The displacement is the furthest the
# segmentation would move along any one axis, so the bound the coordinate
# strings impose falls on the displacement and not on the residual.
#
# GEOMETRY_TRANSLATION_TOLERANCE is 11.5 times the 1.736e-03 voxels the cohort
# reaches. It admits 8.59e-03 mm at the finest spacing in the cohort, 0.429687
# mm, and 0.120 mm at the coarsest, 6.000004 mm. Either way it is a fiftieth of
# a voxel, so a mask can still only be placed on the voxel nearest the position
# it reports, and a genuine half-voxel offset is 25 times the tolerance and
# raises.
#
# The one argument match_geometry accepts governs the other two comparisons as
# well, and 2e-02 would let a segmentation 11.5 degrees out of plane pass as
# aligned. Stage 4 therefore holds those two to GEOMETRY_DIRECTION_TOLERANCE
# itself, before it calls match_geometry. The largest direction vector defect
# over the 49 pairs is 2.00e-15 and the largest spacing ratio residual is the
# 3.181e-06 of UPENN-GBM-00452, so 1e-04 stands above both while admitting an
# angle of only 1.414e-02 radians, which is 0.81 degrees.
# GEOMETRY_TOLERANCE_MARGIN is the factor gate 04 holds both measurements to,
# so that a residual growing toward either tolerance fails the gate before it
# reaches it. GEOMETRY_DEFAULT_TOLERANCE is the highdicom default the five
# pairs above exceed, kept so the count of them is a comparison against a
# stated number.
GEOMETRY_TRANSLATION_TOLERANCE = 2e-2
GEOMETRY_DIRECTION_TOLERANCE = 1e-4
GEOMETRY_TOLERANCE_MARGIN = 10.0
GEOMETRY_DEFAULT_TOLERANCE = 1e-5

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
# The positive rate the sampling produces, 1 in 1 + NEGATIVES_PER_POSITIVE. It
# is a property of the sampling and not of the tumors: every one of the 49
# patients carries it to six decimal places, while the rate before sampling
# runs from 0.000743 to 0.098428 with a median of 0.018600. The tables that
# report a positive rate name the two apart, so a reader of one row cannot take
# the constructed rate for a measurement.
SAMPLED_POSITIVE_RATE = 1.0 / (1.0 + NEGATIVES_PER_POSITIVE)

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

# --- epoch selection -------------------------------------------------------
# Training runs for MAX_EPOCHS epochs and the weights of one of those epochs
# are kept. Reporting the last epoch is what the previous run did, and the
# training history it wrote measures the cost. Validation ROC-AUC reached
# 0.9629 at epoch 14 and stood at 0.9071 at epoch 20, a fall of 0.0558, and
# epoch 20 was the lowest of the last seven epochs. The score moved by 0.0541
# between epochs 19 and 20 alone, so the epoch training happens to stop on
# carries variation that says nothing about the configuration being reported.
# The original coursework kept one epoch through ModelCheckpoint on validation
# loss and the rebuild dropped that.
#
# EPOCH_SELECTION_MONITOR names the column of results/training_history.csv the
# retained epoch is read from, and EPOCH_SELECTION_MODE says which extreme of
# that column is wanted. The monitored quantity is validation ROC-AUC and not
# validation loss, because ROC-AUC is the quantity this pipeline reports, gates
# and compares its two partitions on. Selecting the epoch on one quantity while
# reporting another opens a gap between the two that nothing here measures.
# Validation loss is kept and not discarded: every run computes it per epoch,
# EPOCH_SELECTION_ALTERNATIVE_MONITOR names it, and stage 6 records the epoch
# it would have chosen and whether the two monitors agree.
#
# The selection applies to every training run in stage 6: the four grid
# configurations, the reported model, the patch-level comparison, and the four
# further runs the seed-variance comparison performs at the two seeds beyond
# the primary one. That covers all six rows of results/seed_variance.csv, two
# of which reuse the reported model and the patch-level comparison. Applying
# the selection to the patient-level partition and not the patch-level one
# would move the leakage comparison by the size of the selection effect, and
# that comparison is the claim the repository rests on.
#
# The validation score at the retained epoch is a maximum over MAX_EPOCHS
# values of a quantity that moved by 0.0541 between two neighboring epochs, so
# it is optimistically biased as an estimate of validation performance. Stage 6
# records it as a selection statistic and names the bias where it writes it.
# The test score does not carry that bias, because the test patients enter no
# part of the selection.
EPOCH_SELECTION_MONITOR = "validation_roc_auc"
EPOCH_SELECTION_MODE = "max"
EPOCH_SELECTION_ALTERNATIVE_MONITOR = "validation_loss"
EPOCH_SELECTION_ALTERNATIVE_MODE = "min"

# --- marking the splits that took part in the selection --------------------
# The retained epoch is the one that maximizes validation ROC-AUC, so every
# validation figure this pipeline writes is read at the epoch that maximized a
# quantity on that same set. At the primary seed the maximum is 0.9629 and the
# last of the 20 epochs stands at 0.9071, so the selection accounts for 0.0558
# of the recorded validation ROC-AUC. The validation precision, recall, F1 and
# confusion counts are read from the same retained weights and carry the same
# selection. The training patches fit the weights and the test patients enter
# no part of the selection, so neither split takes part in it.
#
# SELECTION_PARTICIPATION_COLUMN is the column every table in results/ that
# identifies its rows by split carries, and the two values below are the
# entries it takes. The values are sentences and not flags, because the point
# of the column is that someone reading one CSV alone needs nothing beside it.
# SELECTION_PARTICIPATION_SPLITS names the splits the selection reads.
SELECTION_PARTICIPATION_COLUMN = "took_part_in_epoch_selection"
SELECTION_PARTICIPATION_YES = (
    "yes: the retained epoch maximizes this split's ROC-AUC")
SELECTION_PARTICIPATION_NO = "no"
SELECTION_PARTICIPATION_SPLITS = ("validation",)


# --- the signature the notebooks record ------------------------------------
# The two notebooks display the record and compute nothing of their own, so a
# notebook whose stored output was produced against an older record shows
# numbers that no longer exist. Each notebook prints a digest of the record it
# read, and gate 07 recomputes that digest from results/metrics.csv.
#
# The digest is taken over the quantities that reproduce exactly. Of the 271
# quantities the record holds, 8 are wall-clock timings whose key begins with
# RECORD_TIMING_PREFIX and 3 are listed in RECORD_VOLATILE_KEYS, leaving 260
# that a re-run must return unchanged. Gate 06 compares the same 260 key by key
# after re-running the pipeline from nothing, so the digest is stable exactly
# where that comparison is.
RECORD_TIMING_PREFIX = "pipeline_seconds"
RECORD_VOLATILE_KEYS = ("model_train_seconds", "metrics_recorded",
                        "download_license_verified")
# Half of a SHA-256 digest. 64 bits is short enough to read off a notebook cell
# and long enough that two different records will not collide by accident.
RECORD_SIGNATURE_DIGITS = 16
# The line the notebooks print it on. Gate 07 looks for this prefix in the
# stored output of every code cell.
RECORD_SIGNATURE_LABEL = "record signature"


def selection_participation(split: str) -> str:
    """Whether ``split`` took part in the epoch selection, as a sentence."""
    return (SELECTION_PARTICIPATION_YES
            if split in SELECTION_PARTICIPATION_SPLITS
            else SELECTION_PARTICIPATION_NO)
