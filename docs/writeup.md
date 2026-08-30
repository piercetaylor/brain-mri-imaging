# Project write-up

## Research question

Pooling medical images from many patients into one training set assumes that the
images are comparable. Using the UPENN-GBM collection (Bakas et al. 2021), it was
asked what the DICOM headers of a glioblastoma cohort reveal about the equipment,
the pixel grid, the encoding and the de-identification, and then how well a small
convolutional network separates tumor from healthy tissue on patients it has
never seen. The first question bounds what the second is allowed to claim.

## Acquisition

The full collection holds 630 patients and 139.4 GB. A rule selects the subset:
keep the patients whose axial T1 post-contrast series carries a segmentation that
a radiologist reviewed and corrected, sort by patient identifier, and cap the
count at 50. The rule returns 41 patients, 235 image series and 41 segmentation
series, 54,673 instances and 8.45 GB. Running the rule again against the same
index release returns the same patients, and the reproducibility gate checks that
`data/manifest.csv` comes back byte for byte identical.

Images and labels are pulled from the NCI Imaging Data Commons object store
(Fedorov et al. 2021), which needs no credentials. Each downloaded series is
digested, and the digest is compared against `data/checksums.txt` on every run.
The acquisition gate also asks the data commons index and the archive that
published the images what license they report for these series today; both must
answer CC BY 4.0 before anything proceeds.

The labels are the BAMF annotations (Van Oss et al. 2024) and not the
segmentations the collection's own authors published. The authors' NIfTI volumes
sit on a co-registered grid that does not align with the DICOM images, and the
archive serves them only through a transfer client. The BAMF annotations are
DICOM Segmentation objects that name the image series they were drawn on, which
makes the correspondence checkable.

## Cohort quality control

Every one of the 54,632 image instances was read and sixteen header tags were
extracted by group and element number. Addressing tags by number and not by
keyword means that a renamed keyword in a future release of the reading library
cannot silently change what is read. The schema gate re-reads a random sample of
forty files and confirms that the recorded table reproduces what the files
contain.

The first finding concerns equipment. The whole collection was acquired on 17
manufacturer and model combinations across 3 manufacturers, and every series in
this cohort came from a single SIEMENS TrioTim. The selection rule asked only for
radiologist-corrected segmentations, and the single-scanner cohort is a side
effect of which patients received that review. The consequence is that the
modeling half cannot speak to transfer across scanners.

![Acquisition equipment in the cohort](../figures/fig01_scanner_inventory.png)

The studies also hold acquisitions that are not structural. Ten distinct
descriptions covering 71 of the 235 series are diffusion or perfusion sequences,
whose contrast is unrelated to the contrast the segmentation was drawn on. They
are counted here and excluded from the labeled task.

![Series descriptions present in the cohort](../figures/fig02_series_descriptions.png)

Geometry is where the cohort is genuinely heterogeneous. Five distinct row and
column combinations appear, every one of the 41 patients holds more than one, and
one patient holds four. Comparing the first and the last series each patient
contributed, 31 of 41 change image size between them.

![Image grids across the cohort](../figures/fig03_image_grids.png)

Encoding is uniform: all 235 series are MR, all are MONOCHROME2, and all allocate
16 bits per pixel. Checking each tag for constancy within its series found no
series carrying two values of rows, columns, photometric interpretation, bit
depth, manufacturer or model, so a series is a safe unit of description.

The de-identification audit read one instance from each of the 276 series. No
series carries a populated birth date, address, telephone number, institution
address, or referring, performing or operating physician name, and none declares
burned-in annotation. Every patient identifier follows the collection form and
the patient name repeats it. Two gaps are worth recording. The 41 segmentation
objects carry no declaration that patient identity was removed, because a
separate process generated them later. All 235 image series carry private vendor
tags, up to 43 in one series, whose contents lie outside the standard dictionary
and were not inspected.

## Feature engineering

Each patient's axial T1 post-contrast volume is paired with its segmentation. The
segmentation is stored on its own grid, whose in-plane orientation is the exact
negative of the image orientation, so the two arrays cannot be overlaid directly.
The mask is resampled onto the image geometry through the patient coordinate
system, and the result is checked by eye and by the gate.

![Segmentation resampled onto the image grid](../figures/fig04_mask_alignment.png)

Tissue is separated from air by Otsu's threshold on each volume. This choice
corrects a defect. The first complete run tested whether a voxel was non-zero,
but reconstruction noise fills the air compartment, so 30 percent of the negative
patches were empty space and the network was partly separating tissue from air.
That run reported a test ROC-AUC of 0.9902. After the correction the share of
near-empty negatives is 0 percent, mean patch intensity is 0.289 for negatives
against 0.312 for positives, and the honest test score is 0.9731. A preparation
gate now fails if more than one percent of negative patches are near-empty.

Patches of 32 by 32 pixels are cut on an eight pixel stride. A patch is positive
when at least half its pixels lie inside the union of the necrosis, edema and
enhancing-lesion segments, and negative when none do and at least half its pixels
are tissue. Patches between the two are ambiguous, and 167,503 of them were
discarded.

![Patches drawn from the labeled volumes](../figures/fig05_patch_examples.png)

Tumor occupies a median 2.6 percent of the tissue volume, so a grid sweep is
overwhelmingly negative: the median patient yields 2.0 percent positive patches
before any sampling. Training on that distribution at a fixed threshold would be
uninformative, so negatives are subsampled to twice the positives for each
patient, and positives are capped at 500 per patient. The result is 48,537
patches at 33.3 percent positive. That prevalence is a design choice and not a
property of the anatomy, and both numbers are recorded.

![Tumor is a small part of the brain](../figures/fig06_class_balance.png)

## Splitting

Patches cut on an eight pixel stride overlap their neighbors by three quarters of
their area, so two patches from one patient are near duplicates. Whole patients
are therefore assigned to one part of the split: 25 to training, 6 to validation
and 10 to test, giving 30,294, 7,791 and 10,452 patches. No patient appears in
more than one part, and the modeling gate checks it.

![Patients are assigned whole to one part of the split](../figures/fig07_splits.png)

## Modeling

The architecture is the one the coursework specified: convolutional blocks of 16,
16 and 64 filters with batch normalization on the second and third, max pooling
after each, a 64 unit dense layer, dropout, and a single sigmoid output, for
77,585 parameters.

The original compiled this model with an Adadelta learning rate of zero. A
learning rate scheduler callback then assigned 0.01 at the start of every epoch,
so the network trained at a rate the compile call never set, and the declared
configuration and the effective one disagreed. Here the optimizer is given a real
learning rate at construction, and that rate is chosen by the grid search the
exercise was named after but never ran. Four combinations of learning rate and
dropout were scored on the validation patients, who appear in neither the
training nor the test set.

| Learning rate | Dropout | Validation ROC-AUC |
|---:|---:|---:|
| 1.0 | 0.5 | 0.9723 |
| 0.1 | 0.5 | 0.9388 |
| 0.1 | 0.3 | 0.9344 |
| 1.0 | 0.3 | 0.9308 |

![Grid search, scored on the validation patients](../figures/fig08_grid_search.png)

The spread of 0.0415 between the best and worst combination is large enough that
the choice matters. A learning rate of 1.0 with dropout 0.5 was carried into a
twenty epoch run.

![Training](../figures/fig09_training_history.png)

## Evaluation

On the ten held-out patients the network reaches a ROC-AUC of 0.9731 and an
accuracy of 0.8859, against 0.6667 for always predicting the majority class.

![Held-out patients](../figures/fig10_roc_test.png)

At a threshold of 0.5 the errors are strongly one-sided: 21 false positives
against 1,172 false negatives. Tumor precision is 0.991 and tumor recall is
0.664.

![Test confusion matrix](../figures/fig11_confusion_matrix.png)

The asymmetry belongs to the threshold and not to the ranking. An area under the
curve of 0.9731 says the scores separate the classes well, and 0.5 sits too high
on that score distribution when two thirds of the training patches are negative.
Lowering the threshold would recover recall at the cost of precision, and a
screening use would want that trade. The original coursework reported the
opposite bias, more false positives than false negatives, on a set that was 70
percent positive.

Training ROC-AUC of 0.9999 against 0.9731 on held-out patients shows the model
memorizing its training patients, which is what a patient-level split exists to
reveal. Pooling patches also hides variation between patients: across the ten
test patients the median ROC-AUC is 0.9815 and the range runs from 0.9361 to
1.000.

![Performance is not uniform across held-out patients](../figures/fig12_per_patient.png)

The sampled errors show two distinct failure modes. The false positives sit at
bright interfaces, where skull or vessel meets parenchyma. The false negatives
mix flat low-contrast patches with patches holding part of an enhancing rim, so
no single appearance accounts for them.

![Errors on the held-out patients](../figures/fig14_errors.png)

## What splitting over patches would have reported

Training the same architecture with the same hyperparameters on a partition that
shuffles patches raises the test ROC-AUC from 0.9731 to 0.9929. Every one of the
41 patients then appears on both sides of the boundary, and near copies of the
same tissue are scored as if they were new patients.

![The same model under two partitions](../figures/fig13_split_leakage.png)

The 0.0199 gap is small in absolute terms and large in what it conceals. A
reported score of 0.9929 on this task would be an artifact of the partition.

## Limitations

The labels are not a clinical reference standard. An automated method produced
them and a radiologist reviewed them, but the review covers some segments and not
others. Agreement with an independent expert reading was not measured, so the
ceiling on any score here is unknown.

Every series came from one scanner model. Magnetic resonance intensity carries no
absolute unit, and per-volume normalization removes only the scale, so transfer
to other equipment was not tested and should not be assumed.

Ten test patients is a small sample, and the per-patient range from 0.9361 to
1.000 describes the spread better than the pooled figure does.

Patch classification is not tumor detection. Each patch is scored without spatial
context, and the 167,503 patches that straddled the tumor boundary were discarded
as ambiguous, which removes exactly the cases a segmentation model would have to
resolve.
