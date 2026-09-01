# Brain MRI cohort quality control and tumor patch classification

A quality-control analysis and a classification analysis share one glioblastoma
cohort. The first reads 59,713 DICOM headers and reports what would stop the
images being pooled. The second trains a convolutional network to decide whether
a 32 by 32 patch of an axial T1 post-contrast slice lies inside the tumor.

The analysis was completed as coursework for DATA_SCI 8140, Advanced Methods in
Health Data Science, in **Fall 2025**. This repository rebuilds that work from
the original public dataset with the course scaffolding removed, so that the
methods and the reasoning behind them stand as a record in their own right.

The dataset is not the one the coursework used. That work drew on the TCIA
Brain-Tumor-Progression collection, which carries a restricted license barring
redistribution, and on a set of image patches supplied with no license, citation
or source. Neither can be published. Both were replaced by UPENN-GBM (Bakas et
al. 2021), whose license permits reuse and whose provenance the pipeline
re-checks at the source on every run.

## The question

Before a cohort of medical images can be pooled for modeling, it has to be shown
that pooling is defensible. Using the UPENN-GBM collection (Bakas et al. 2021),
two questions were addressed. The first is what the header of every image says
about the equipment, the pixel grid, the encoding and the de-identification, and
which of those differ across the cohort. The second is how well a small
convolutional network separates tumor from healthy tissue when it is scored on
patients it has never seen.

The two halves share one cohort deliberately. The first establishes what the
second is entitled to assume.

## The data

Bakas, S. et al. *Multi-parametric magnetic resonance imaging (mpMRI) scans for
de novo Glioblastoma (GBM) patients from the University of Pennsylvania Health
System (UPENN-GBM)*. The Cancer Imaging Archive,
[doi:10.7937/TCIA.709X-DN49](https://doi.org/10.7937/TCIA.709X-DN49). Released
under CC BY 4.0.

Tumor labels are the BAMF annotations of that collection (Van Oss et al. 2024),
[doi:10.5281/zenodo.8345959](https://doi.org/10.5281/zenodo.8345959), also
CC BY 4.0. They mark necrosis, edema and enhancing lesion, and are distributed as
DICOM Segmentation objects that name the image series they were drawn on. The
segmentations published by the collection's own authors are NIfTI volumes on a
co-registered grid that does not align with the DICOM images, and the archive
serves them only through a transfer client, so they are not used here.

The images and the labels come from the NCI Imaging Data Commons object store
(Fedorov et al. 2021), which needs no credentials and no download client. The
full collection holds 630 patients and 3,680 MR series, which is more than this
pipeline should move.

The subset follows a rule and not a hand selection. A patient is eligible when a
post-contrast T1 series of that patient carries a segmentation in which at least
one segment is marked as reviewed by a person. The post-contrast T1 series is
identified by markers in the series description. A description qualifies when it
carries both a T1 marker and a post-contrast marker and carries neither a T2 nor
a FLAIR marker, so that the rule does not depend on how any one manufacturer
named its protocol. The rule is emitted as
SQL against index release 24.2.2 and mirrored by a Python predicate the gates
use. It returns 49 eligible patients against a cap of 50, from
`UPENN-GBM-00019` to `UPENN-GBM-00626`: 273 image series, 49 segmentation
series, 59,762 files and 9.44 GB on disk.

Nothing is redistributed here. `data/download_data.py` fetches the manifested
series and digests each one. A series digest is the SHA-256 of its per-file
digests, sorted and joined, so it does not depend on the order files arrive in.
The digests are recorded in `data/checksums.txt` and compared on every run. The
acquisition gate additionally asks two independent services what license they
report for the manifested series today, which is the one check this repository
cannot fake.

## What was found

### The cohort spans six scanner models, and one of them holds seven eighths of it

The 273 image series were acquired on 6 manufacturer and model combinations
across 2 manufacturers, out of the 17 combinations and 3 manufacturers the whole
collection holds. The largest single model accounts for 87.5 percent of the
series. One patient contributed series from two models, and the other 48 each
sit on one. Heterogeneity is present and it is lopsided, and any result measured
on this cohort is dominated by one vendor's output.

An earlier version of this analysis reported a single-scanner cohort and
attributed it to the segmentation filter. That attribution was wrong. The cohort
was selected by matching the series description against one manufacturer's
literal protocol name, and that string is the whole of the loss. The current
marker rule admits 8 distinct descriptions, recorded in
`results/label_series_descriptions.csv`. Three of them begin with the string the
old rule matched and cover 41 of the 49 patients, all on one model. The other
five descriptions carry the remaining 8 patients and bring in the other five
scanner models. A selection rule keyed on one vendor's protocol name silently
produced a cohort that could not support the claim the quality-control half was
built to make. The defect stayed invisible until the rule was decomposed and each
filter was run on its own.

### Geometry differs within nearly every patient

The cohort holds twelve distinct row and column combinations. Of the 49
patients, 48 hold more than one, and one patient holds four. Comparing the first
and the last series a patient contributed, 34 of 49 change image size between
them. Stacking a patient's series into one array without resampling would fail
or silently misalign.

Encoding is consistent. All 273 series are MR, all are MONOCHROME2, and all
allocate 16 bits per pixel. No header tag varies within a series, so treating a
series as the unit of description is safe. Of the 273 series, 77 carry 15
descriptions that name a diffusion or perfusion acquisition, whose contrast is
unrelated to the contrast the segmentation was drawn on. Those are counted here
and excluded from the labeled task, leaving 196 structural series.

### Geometry defects that appear only across manufacturers

Assembling a series into a volume compares the `ImageOrientationPatient` of
every slice. Of the 273 series, 3 carry more than one value, and each carries 5
distinct values. The widest spread sits on a series of 35 slices and differs in
the seventh decimal place. All three belong to the single patient scanned on the
one non-Siemens model. The largest genuine angular disagreement anywhere in the
cohort is
7.772e-15 in cosine similarity, which displaces about 31 nanometers across the
widest field of view. No real reorientation exists in the data, and the volume
assembly is given a tolerance of 1e-05, with the measured deviation recorded per
series and gated.

Placing a segmentation on its image grid requires the translation between the
two origins to fall on a whole number of voxels. Of the 49 image and
segmentation pairs, 5 leave a residual above the 1e-05 library default, and the
cohort median is 5.744e-06 voxels. The largest is 1.7359e-03 voxels on
`UPENN-GBM-00452`, whose mask would move 8.1379e-04 mm along one axis, each
figure the largest of the three axes. Every one of the five sits on a scanner
model the previous single-scanner cohort did not contain.

Each residual was traced to the decimal string that produces it. One
segmentation writes its pixel spacing as `4.882812e-01` against the image's
`0.48828125`. Another writes slice spacing as `4.999992e+00` against the
5.000008 mm the image describes, and that pair's measured spacing ratio residual
is 3.1806e-06. The worst patient writes its image position to six significant
figures, so a coordinate near 200 mm is quantized at 1e-03 mm. Three coordinates
each rounded to within 5e-04 mm can misplace an origin by at most 8.66e-04 mm,
and the largest displacement measured is 8.138e-04 mm, inside that bound. That
bound is the line between representation noise and a real disagreement, and
nothing in the cohort crossed it. The translation tolerance is set at 2e-02
voxels, and the direction and spacing comparisons are held separately to 1e-04.

The orientation defect and the alignment defect are what the quality-control
half exists to find. A pipeline built and tested on one manufacturer's output
carried two latent failures that only a cohort spanning manufacturers could
reveal. The full record is in
`results/qc_findings.csv`, which holds 149 findings, 114 of them flagged.

### De-identification holds, with two gaps worth naming

The audit read one instance from each of the 322 series, and not every instance.
No series carries a populated patient birth date, address, telephone number,
institution address, or referring, performing or operating physician name. Every
patient identifier follows the collection form, and patient name repeats the
identifier and carries no name. No series declares burned-in annotation. That
result is a declaration read from tag (0028,0301) and not an inspection of pixel
data, and the tag carries no value in any of the 322 audited series.

The audit is qualified in two places. The 273 image series declare that patient
identity has been removed, and the 49 segmentation series carry no value for
that tag at all, which is why 322 audited series yield 273 identity-removed
declarations. All 273 image series also carry private vendor tags, up to 43 in
one series. Private tags lie outside the standard dictionary and are not covered
by the checks above, so their contents were not verified.

### A quarter of the positive label was never reviewed, and it is one tissue class

The labels are automated segmentations of which some segments were reviewed and
corrected by a radiologist. Across the cohort 3,022,788 of 3,993,796 tumor
voxels come from a reviewed segment, a share of 0.7569, leaving 0.2431
unreviewed. Per patient the reviewed share runs from 0.2089 to 0.9998 with a
median of 0.7598. No patient is fully reviewed and all 49 carry unreviewed
voxels, with 98 reviewed segments against 49 unreviewed.

The unreviewed voxel count of 971,008 equals the enhancing-lesion segment count
exactly, and there are exactly 49 unreviewed segments, one per patient. The
enhancing-lesion segment is the unreviewed one throughout, and it never overlaps
a reviewed segment. The unreviewed quarter of the positive label is a single
tissue class and not a scatter across the label. Describing these labels as
radiologist-corrected without that qualification would overstate them.

### The classifier separates tumor from tissue, and one seed is not the answer

The network is the architecture the coursework specified: three convolutional
blocks with 16, 16 and 64 filters, batch normalization on the second and third,
max pooling after each, then a 64 unit dense layer, dropout, and one sigmoid
output. It holds 77,585 parameters and trains in 208 seconds on a CPU.

Patients were assigned whole to the training, validation and test sets, 29, 7
and 13 of them. No patient appears in more than one part. Accuracy below is read
at the constructed positive rate of 0.3333 set out later in this section, and
ROC-AUC does not depend on that rate.

| Split | Patients | Patches | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|
| Training | 29 | 38,070 | 0.9962 | 0.9612 |
| Validation | 7 | 6,306 | 0.9629 | 0.8957 |
| Test | 13 | 14,412 | 0.9864 | 0.9519 |

The validation row is not a performance estimate. Training keeps the epoch whose
validation ROC-AUC is highest and restores those weights, so 0.9629 is the
maximum over twenty epochs on the set that maximum was taken over. It is a
selection statistic and it is optimistically biased. The test row is unbiased,
because the test patients take no part in choosing the epoch, the
hyperparameters or the split.

The test ROC-AUC of 0.9864 is the best of three seeds. The same pipeline at
seeds 20251118 and 20251119 scores 0.9479 and 0.9151, for a mean of 0.9498. The
primary seed was fixed before any of these runs and was not selected for its
result, and the headline number is quoted at that seed so it stays traceable to
one run. The patient-level bootstrap interval on the primary seed runs from
0.9763 to 0.9941 over 2,000 resamples at the 95 percent level, and it covers
neither of the other two seeds. That interval describes uncertainty from the
patient sample alone and understates the total. The spread across seeds, 0.9151
to 0.9864, is the more honest summary of what this cohort supports.

### Keeping the best epoch is worth 0.0141, and the rebuild had dropped it

The Fall 2025 coursework saved the weights of the epoch that scored best on the
validation set. The rebuild had dropped that step and reported the final epoch,
which is a departure from the work it reconstructs and was not deliberate. Epoch
selection is now restored and applied uniformly to the four grid points and to
all six seed-variance runs.

The cost of the omission is measured and not asserted. Every one of the six
seed-variance runs is scored at both candidate epochs. At the primary seed the
final epoch scores 0.9723 on the test patients and the retained epoch 14 scores
0.9864, a gain of 0.0141. Across the six runs the gain averages 0.0154 and ranges
from 0.0000 to 0.0277, so the primary seed sits near the middle and is not the
run selection helped most. Reporting it is fair on that evidence.

Selection does not always help. One run of the six retained its final epoch and
gained nothing, at seed 20251119 on the patient-level partition, where the best
validation epoch is epoch 20. The gain averages 0.0139 over the three
patient-level runs and 0.0169 over the three patch-level runs, so it is of
similar size under both partitions and does not explain the difference between
them.

The monitored quantity is validation ROC-AUC, which is what this repository
reports, gates and runs its leakage comparison on. Validation loss was measured
every epoch alongside it, and the two monitors disagree. Validation loss would
have selected epoch 7, whose validation ROC-AUC is 0.9543, against epoch 14 at
0.9629. Validation loss also swings between 0.2709 and 1.5563 across the twenty
epochs, while validation ROC-AUC stays between 0.8076 and 0.9629. The
disagreement is recorded as a metric.

The epoch is chosen on the same 7 validation patients the hyperparameter grid
was chosen on. That set is small, and every quantity read from it carries the
optimism of having been maximized over.

The prevalence the model was trained and scored at is constructed and is not a
property of the anatomy. Tumor occupies a median 2.55 percent of the tissue
volume, and the median patient yields a natural positive rate of 0.0186 under a
grid sweep. Negatives are subsampled to twice the positives for each patient and
positives are capped at 500 per patient, which gives 58,788 patches at a
positive rate of 0.3333. Every precision and accuracy figure reported for this
model is read at that rate, and so is the 0.6667 majority baseline, which
restates the constructed prevalence. At the natural rate of 0.0186 the majority
baseline would be 0.9814, and precision at a fixed threshold would be far lower
than the figures below.

Always predicting the majority class scores 0.6667 on the test patches.
Per-class performance at a threshold of 0.5 is uneven.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Non-tumor | 0.9464 | 0.9836 | 0.9646 | 9,608 |
| Tumor | 0.9643 | 0.8886 | 0.9249 | 4,804 |

The test confusion matrix reads 9,450 true negatives, 158 false positives, 535
false negatives and 4,269 true positives. The model misses 11.1 percent of the
tumor patches and calls 1.6 percent of the healthy ones tumor, so the errors
still fall mostly on the tumor class. The direction reverses on the training
patches, where 1,136 false positives sit against 340 false negatives, so the
asymmetry belongs to the threshold and to the patients it is read on and not to
the model's ranking. An area under the curve of 0.9864 means the scores order
the two classes well. A screening application would move the threshold down and
trade precision for recall.

Performance also varies by patient. Across the 13 test patients the median
ROC-AUC is 0.9942, ranging from 0.9565 to 1.000, and none falls below 0.8.

### Splitting over patches overstates the result at every seed

Patches are cut on an eight pixel stride, so neighbors overlap by three quarters
of their area. Training the same architecture with the same hyperparameters on a
partition that shuffles patches and not patients raises the test ROC-AUC at
every seed: 0.9864 to 0.9907, 0.9479 to 0.9883, and 0.9151 to 0.9925. The
inflation is 0.0043, 0.0404 and 0.0774, a mean of 0.0407 across a spread of
0.0731, and it is positive on 3 of 3 seeds. Under the patch-level partition all
49 patients appear on both sides of the boundary.

The mean and the spread carry this claim, and the primary seed does not. Its
inflation of 0.0043 is the smallest of the three and an eighteenth of the
largest, so quoting it alone would understate the effect badly. The patch-level
scores are also much tighter across seeds than the patient-level ones, 0.9883 to
0.9925 against 0.9151 to 0.9864. Near-duplicate patches on both sides of the
boundary suppress the variation between patients, so the leaky partition returns
a stable high score whatever the seed.

## Limitations

The labels are not a clinical reference standard. An automated method produced
them and a radiologist reviewed part of them, and the unreviewed part is the
whole enhancing-lesion segment of every patient. Agreement between the labels
and an independent expert reading was not measured, so the ceiling on any score
reported here is unknown.

Heterogeneity across 6 scanner models is not balance, and 87.5 percent of the
series come from one model. Magnetic resonance intensity has no absolute unit and
depends on the scanner and the sequence, and per-volume normalization removes
only the scale. Transfer to equipment outside these six models was not tested and
should not be assumed. The labeled series of only 7 patients was acquired on one
of the other five models, which is too few to compare performance across
equipment.

The test set holds 13 patients, which is a small sample, and the reported score
is the best of three seeds. The seed range from 0.9151 to 0.9864 and the
per-patient range from 0.9565 to 1.000 both describe the spread better than the
pooled figure does.

The cohort rule matches tokens in the series description, so it inherits a
dependence on naming convention. A post-contrast T1 acquisition whose description
carries no post-contrast token is not admitted, and the rule accepts the 8
descriptions recorded in `results/label_series_descriptions.csv` and rejects the
rest of the vocabulary the collection uses. That shrinks the cohort and does not
corrupt it, because every admitted series is a post-contrast T1. It is the same
kind of dependence on vendor vocabulary that produced the single-scanner cohort,
held to a narrower scope.

The learning rate and dropout were chosen on the same 7 validation patients that
choose the retained epoch. Nothing read from that set is an estimate of
performance, and the 0.9629 it reports is the largest of the twenty epoch scores
taken over it. A validation set that small, used twice, is a real weakness in a
49-patient study.

Patch classification is not tumor detection. A patch is scored in isolation with
no spatial context, and 197,060 patches that straddled the tumor boundary were
discarded as ambiguous. The task avoids exactly the boundary cases a
segmentation model would have to resolve.

`docs/writeup.md` works through the analysis step by step with the figures.
`docs/references.md` holds the citations.

## Reproducing it

Python 3.13 with the versions pinned in `requirements.txt`.

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
```

```bash
python -m src.s01_manifest && python data/download_data.py && python analysis/run_all.py
```

The first command builds the cohort manifest from the data commons index. The
second downloads 9.44 GB and verifies it against `data/checksums.txt`. The third
runs stages 02 to 08 and touches no network. That run takes 1,674.8 seconds on a
CPU, of which the modeling stage takes 1,458.5. The modeling stage fits the four
point grid and then trains six models, three seeds on each of the two
partitions. The primary seed is 20251117.

Every measured quantity in this README and in `docs/writeup.md` is read out of
`results/`, which the pipeline writes, and none is transcribed from an earlier
run. Most are read from `results/metrics.csv`, and the remainder from the tables
beside it. `seed_variance.csv` carries the per-seed scores, gains and inflations,
and `training_history.csv` the per-epoch losses and validation scores.
`classification_report.csv` carries the per-class support, `model_grid.csv` the
grid, and `scanner_inventory.csv` the equipment counts.
`label_series_descriptions.csv` carries the description vocabulary with its
patient counts, and `qc_findings.csv` the per-series findings. A few figures are
differences or ratios of recorded quantities, and each is written beside the
quantities it is taken from. Three of the recorded quantities describe the
repository and not the data, namely the 286 metrics, the 14 generated figures and
the 149 quality-control findings. The gate tallies below are the gates' own
printed output.

## Layout

```
src/                  pipeline stages, numbered in execution order
analysis/run_all.py   runs stages 02 to 08 and records every result
analysis/*.ipynb      thin notebooks that read results/ and present it
data/                 download script, manifest, checksums
docs/                 write-up with figures, and references
figures/              fourteen generated figures
results/              generated tables and the recorded metrics
```

## How the work was checked

Each stage ends in a gate that exits zero or non-zero and prints what it
verified. The stages are acquisition, schema, quality control, preparation,
modeling and reproducibility, with an environment gate ahead of them. Two gates
stand outside the stages. One compares the stored output of each notebook
against the record it read, and one compares the keys the record holds against
the keys the source files write.

The gates confirm that every downloaded series matches its recorded digest and
that two services still report CC BY 4.0. They re-read a random sample of 40
DICOM files and check that the recorded header table reproduces what the files
actually contain. They hold the measured orientation deviation and the measured
geometry residuals to a stated margin below their tolerances, so a deviation
growing toward a tolerance fails the gate before it reaches it. They confirm
that no patient appears in more than one part of the split, that nothing outside
the pixel array encodes the class, and that the saved model reproduces the
reported test score when it is loaded again. The environment gate scans every
tracked file for the vendor and product names it forbids. Every gate closes by
asserting that no check read a quantity the record does not hold. All nine pass:
environment 24 of 24, acquisition 16 of 16, schema 25 of 25, quality control 23
of 23, preparation 44 of 44, modeling 46 of 46, reproducibility 11 of 11,
notebooks 21 of 21 and record 11 of 11, for 221 checks in all.

A gate reports what it did not run. A service that does not answer fails the
check that needed it, and is skipped only when `ALLOW_OFFLINE` is set, so that a
check cannot turn itself off when the service it tests moves. The suite counts
those skips beside the gate that took them, because a skipped check asserts
nothing.

The reproducibility gate does not run from an empty state. It clears `results/`,
`data/interim/` and `figures/`, re-runs the cohort rule against the index,
re-verifies every downloaded series against `data/checksums.txt` without
re-fetching `data/raw/`, and re-runs stages 02 to 08. Eleven of the 286 recorded
quantities are held out of the comparison: nine wall-clock timings, the count the
record states about itself, and the license verification. The remaining 275 must
reproduce exactly. Rebuilding the analysis from those cleared directories, 0 of
the 275 compared quantities differ, and no key appears or disappears. The claim
it supports is reproduction from a verified local copy of the images, and it
says nothing about a different machine or a
different release of the numerical libraries.

Findings are reported whichever way they came out. The previous cohort came out
single-scanner because of a selection rule written here, and the earlier
attribution of that to the data was wrong. The rebuild had dropped the epoch
selection the coursework used, which cost 0.0141 at the primary seed. A quarter
of the positive label is unreviewed and it is one tissue class. The reported test
score is the best of three seeds, and the bootstrap interval covers neither of
the other two. The primary seed produces the smallest of the three leakage
inflations, so the headline seed is the one least favorable to the repository's
own methodological claim. The segmentation objects do not declare that identity
was removed. All six are recorded above.

## Citation

Work that uses this analysis should cite the data:

> Bakas, S., Sako, C., Akbari, H., Bilello, M., Sotiras, A., Shukla, G. et al.
> (2021). Multi-parametric magnetic resonance imaging (mpMRI) scans for de novo
> Glioblastoma (GBM) patients from the University of Pennsylvania Health System
> (UPENN-GBM). The Cancer Imaging Archive. doi:10.7937/TCIA.709X-DN49

and the labels:

> Van Oss, J., Murugesan, G. K., McCrumb, D. and Soni, R. (2024). Image
> segmentations produced by BAMF under the AIMI Annotations initiative. Zenodo.
> doi:10.5281/zenodo.8345959

Full citations, including the methods and software sources, are in
`docs/references.md`.

## License

Code in this repository is released under the MIT License, in `LICENSE`. The
images and the segmentations are released by their publishers under CC BY 4.0 and
are not redistributed here.
