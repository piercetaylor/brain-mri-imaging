# Brain MRI cohort quality control and tumor patch classification

Two analyses on one glioblastoma cohort. The first reads 54,632 DICOM headers and
reports what would stop the images being pooled. The second trains a
convolutional network to decide whether a 32 by 32 patch of an axial T1
post-contrast slice lies inside the tumor.

The analysis was completed as coursework for DATA_SCI 8140, Advanced Methods in
Health Data Science, in **Fall 2025**. This repository rebuilds that work from
the original public dataset with the course scaffolding removed, so that the
methods and the reasoning behind them stand as a record in their own right.

The dataset is not the one the coursework used. That work drew on the TCIA
Brain-Tumor-Progression collection, which carries a restricted license barring
redistribution, and on a set of 8,710 image patches supplied with no license,
citation or source. Neither can be published. Both were replaced by UPENN-GBM
(Bakas et al. 2021), whose license permits reuse and whose provenance the
pipeline re-checks at the source on every run.

## The question

Before a cohort of medical images can be pooled for modeling, it has to be shown
that pooling is defensible. Using the UPENN-GBM collection (Bakas et al. 2021),
two questions were addressed. The first is what the header of every image says
about the equipment, the pixel grid, the encoding, and the de-identification, and
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

Both images and labels come from the NCI Imaging Data Commons object store
(Fedorov et al. 2021), which needs no credentials and no download client. The
full collection is 630 patients and 139.4 GB, which is more than this pipeline
should move. The subset follows a rule and not a hand selection: keep the
patients whose axial T1 post-contrast series carries a segmentation a radiologist
reviewed and corrected, sort by patient identifier, and cap the count at 50. That
rule returns 41 patients, 276 series and 54,673 instances, 8.45 GB on disk.

Nothing is redistributed here. `data/download_data.py` fetches the manifested
series and digests each one. A series digest is the SHA-256 of its per-file
digests, sorted and joined, so it does not depend on the order files arrive in.
The digests are recorded in `data/checksums.txt` and compared on every run. The
acquisition gate additionally asks two independent services what license they
report for the manifested series today, which is the one check this repository
cannot fake.

## What was found

### The cohort is homogeneous in equipment and heterogeneous in geometry

The whole UPENN-GBM collection was acquired on 17 manufacturer and model
combinations across 3 manufacturers. Every one of the 235 series in this cohort
came from a single SIEMENS TrioTim. The selection rule asked for
radiologist-corrected segmentations and, without that being intended, returned a
single-scanner subset. Any performance measured here therefore says nothing about
transfer to a different scanner, and the cohort is not representative of the
collection it was drawn from.

Geometry is a different matter. Five distinct row and column combinations appear
across the cohort. Every one of the 41 patients holds more than one, and one
patient holds four. Comparing the first and the last series a patient
contributed, as the original exercise did, 31 of 41 patients change image size
between them. Stacking a patient's series into one array without resampling would
fail or silently misalign.

Encoding is consistent. All 235 series are MR, all are MONOCHROME2, and all
allocate 16 bits per pixel. No header tag varies within a series, so treating a
series as the unit of description is safe. The full record is in
`results/qc_findings.csv`, which holds 102 findings, 82 of them flagged.

### De-identification holds, with one gap worth naming

One instance from each of the 276 series was checked. No series carries a
populated patient birth date, address, telephone number, institution address, or
referring, performing or operating physician name. No series declares burned-in
annotation. Every patient identifier follows the collection form, and patient
name repeats the identifier and carries no name.

Two observations qualify that. The 235 image series declare that patient identity
has been removed; the 41 segmentation objects carry no such declaration, because
they were generated later by a separate process. All 235 image series also carry
private vendor tags, up to 43 in one series. Private tags are outside the
standard dictionary and are not covered by the checks above, so their contents
were not verified.

### The classifier separates tumor from tissue, and its threshold is wrong

The network is the architecture the coursework specified: three convolutional
blocks with 16, 16 and 64 filters, batch normalization on the second and third,
max pooling after each, then a 64 unit dense layer, dropout, and one sigmoid
output. It holds 77,585 parameters and trains in 353 seconds on a CPU.

Patients were assigned whole to the training, validation and test sets, 25, 6 and
10 of them. No patient appears in more than one part.

| Split | Patients | Patches | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|
| Training | 25 | 30,294 | 0.9999 | 0.9917 |
| Validation | 6 | 7,791 | 0.9792 | 0.9153 |
| Test | 10 | 10,452 | 0.9731 | 0.8859 |

Always predicting the majority class scores 0.6667 on the test patches. Per-class
performance at a threshold of 0.5 is lopsided.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Non-tumor | 0.856 | 0.997 | 0.921 | 6,968 |
| Tumor | 0.991 | 0.664 | 0.795 | 3,484 |

The test confusion matrix reads 6,947 true negatives, 21 false positives, 1,172
false negatives and 2,312 true positives. The model almost never calls healthy
tissue tumor and misses a third of the tumor patches. That asymmetry is a
property of the threshold and not of the model's ranking: an area under the curve
of 0.9731 means the scores order the two classes well, while 0.5 sits too high on
that score distribution when two thirds of the training patches are negative. A
screening application would move the threshold down and trade precision for
recall. The direction is the opposite of the original coursework, which reported
more false positives than false negatives on a set that was 70 percent positive.

Training ROC-AUC of 0.9999 against 0.9731 on held-out patients is the memorization
the patient-level split is there to expose. Performance also varies by patient.
Across the 10 test patients the median ROC-AUC is 0.9815, ranging from 0.9361 to
1.000, and none falls below 0.8.

### Splitting over patches would have overstated the result

Patches are cut on an eight pixel stride, so neighbors overlap by three quarters
of their area. Training the same architecture with the same hyperparameters on a
partition that shuffles patches and not patients raises the test ROC-AUC from
0.9731 to 0.9929. The 0.0199 difference is the price of a partition that puts
near copies of the same tissue on both sides of the boundary, and every one of
the 41 patients appears on both sides of it.

### A defect found by looking at the data

The first complete run scored 0.9902 on the test patients. That number was wrong.
Tissue had been identified by testing whether a voxel was non-zero, and
reconstruction noise fills the air around the head, so 30 percent of the negative
patches were empty space. The network was partly separating tissue from air.
Tissue is now identified by Otsu's threshold on each volume, which needs no tuned
constant; near-empty negatives fell to 0 percent, and mean patch intensity is
0.289 for negatives against 0.312 for positives, so brightness alone no longer
separates the classes. The honest score is 0.9731. A preparation gate now fails
if more than 1 percent of negative patches are near-empty.

## Limitations

The labels are not a clinical reference standard. They were produced by an
automated method and reviewed by a radiologist; the review covers some segments
and not others, and the enhancing-lesion segment of the first cohort patient is
marked automatic. Agreement between the labels and an independent expert reading
was not measured, so the ceiling on any score reported here is unknown.

The cohort came from one scanner model. Magnetic resonance intensity has no
absolute unit and depends on the scanner and the sequence, and per-volume
normalization removes only the scale. Transfer to other equipment was not tested
and should not be assumed.

Ten test patients is a small sample. The interval around a test ROC-AUC estimated
from 10 patients is wide, and the per-patient range from 0.9361 to 1.000 gives a
better sense of the spread than the pooled figure does.

Patch classification is not tumor detection. A patch is scored in isolation with
no spatial context, and 167,503 patches that straddled the tumor boundary were
discarded as ambiguous. The task avoids exactly the boundary cases a segmentation
model would have to resolve.

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
second downloads 8.45 GB and verifies it against `data/checksums.txt`. The third
runs stages 02 to 08 and touches no network. The pipeline takes 21 to 27
minutes on a CPU depending on machine load, four fifths of it the grid search,
the training run and the patch-level comparison. It is seeded at 20251117.

Every measured quantity in this README and in `docs/writeup.md` is read out of
`results/metrics.csv`, which the pipeline writes, and none is transcribed from an
earlier run. Three remaining counts describe the repository and not the results,
namely the 162 quantities in `results/metrics.csv`, the 14 generated figures, and
the 102 recorded quality-control findings.

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
modeling and reproducibility, with an environment gate ahead of them.

The gates confirm that every downloaded series matches its recorded digest and
that two services still report CC BY 4.0. They re-read a random sample of DICOM
files and check that the recorded header table reproduces what the files
actually contain. They confirm that no patient appears in more than one part of
the split, that nothing outside the pixel array encodes the class, and that the
saved model reproduces the reported test score when it is loaded again. The
environment gate scans every tracked file for the vendor and product names it
forbids. The final gate clears `results/`, `data/interim/` and
`figures/`, re-runs the cohort rule, the verification and the whole pipeline, and
requires every recorded quantity to reproduce exactly.

Findings are reported whichever way they came out. The cohort turned out to be
single-scanner, which weakens the claim the modeling half can make. The
segmentation objects turned out not to declare that identity was removed. The
first run's score turned out to be inflated by a preparation defect. All three
are recorded above.

## Citation

If you use this analysis, cite the data:

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
