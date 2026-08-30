# References

## Data

**Images.** Bakas, S., C. Sako, H. Akbari, M. Bilello, A. Sotiras, G. Shukla,
J. D. Rudie, N. F. Santamaria, A. F. Kazerooni, S. Pati, S. Rathore,
E. Mamourian, S. M. Ha, W. Parker, J. Doshi, U. Baid, M. Bergman, Z. A. Binder,
R. Verma, R. A. Lustig, A. S. Desai, S. J. Bagley, Z. Mourelatos, J. Morrissette,
C. D. Watt, S. Brem, R. L. Wolf, E. R. Melhem, M. P. Nasrallah, S. Mohan,
D. M. O'Rourke and C. Davatzikos (2021). *Multi-parametric magnetic resonance
imaging (mpMRI) scans for de novo Glioblastoma (GBM) patients from the
University of Pennsylvania Health System (UPENN-GBM)*. The Cancer Imaging
Archive. [doi:10.7937/TCIA.709X-DN49](https://doi.org/10.7937/TCIA.709X-DN49).
Released under CC BY 4.0.

The collection is described in Bakas, S. et al. (2022). The University of
Pennsylvania glioblastoma (UPenn-GBM) cohort: advanced MRI, clinical, genomics,
and radiomics. *Scientific Data* 9, 453.
[doi:10.1038/s41597-022-01560-7](https://doi.org/10.1038/s41597-022-01560-7).
The collection holds 630 subjects with pre-operative multi-parametric MRI. Its
own segmentation labels, distributed as NIfTI volumes, cover 611 subjects
automatically and 232 after expert correction, according to
`UPENN-GBM_data_availability.csv` published beside the collection. Those NIfTI
labels are not the labels used here. They sit on a co-registered grid that does
not align with the DICOM images, and the archive serves them only through a
transfer client. The labels used here are the DICOM segmentations described
next.

**Segmentations.** Van Oss, J., G. K. Murugesan, D. McCrumb and R. Soni (2024).
*Image segmentations produced by BAMF under the AIMI Annotations initiative*,
version 2.0.2. Zenodo.
[doi:10.5281/zenodo.8345959](https://doi.org/10.5281/zenodo.8345959) (concept
record; this version is 10.5281/zenodo.13244892). Released under CC BY 4.0.
These segmentations label necrosis, edema and enhancing lesion on the
pre-operative structural series of UPENN-GBM. They are distributed through the
NCI Imaging Data Commons as DICOM Segmentation objects that reference, by
series identifier, the image series they were drawn on. Each segment records
whether it was produced automatically or reviewed and corrected by a
radiologist.

The annotation program is described in Murugesan, G. K., D. McCrumb,
M. Aboian, T. Verma, R. Soni, F. Memon, J. Van Oss, S. Moore, C. Jarosz,
K. Farahani, U. Baid and A. Fedorov (2024). AI-generated annotations dataset
for diverse cancer radiology collections in NCI Image Data Commons.
*Scientific Data* 11, 1165.
[doi:10.1038/s41597-024-03977-8](https://doi.org/10.1038/s41597-024-03977-8).

**Archives.** The images are published by The Cancer Imaging Archive: Clark, K.,
B. Vendt, K. Smith, J. Freymann, J. Kirby, P. Koppel, S. Moore, S. Phillips,
D. Maffitt, M. Pringle, L. Tarbox and F. Prior (2013). The Cancer Imaging
Archive (TCIA): maintaining and operating a public information repository.
*Journal of Digital Imaging* 26(6):1045-1057.
[doi:10.1007/s10278-013-9622-7](https://doi.org/10.1007/s10278-013-9622-7).

Both images and segmentations are mirrored, indexed and served by the NCI
Imaging Data Commons: Fedorov, A., W. J. R. Longabaugh, D. Pot, D. A. Clunie,
S. Pieper, H. J. W. L. Aerts, A. Homeyer, R. Lewis, A. Akbarzadeh, D. Bontempi,
W. Clifford, M. D. Herrmann, H. Höfener, I. Octaviano, C. George, S. Paquette,
J. Petts, D. Marcus and R. Kikinis (2021). NCI Imaging Data Commons. *Cancer
Research* 81(16):4188-4193.
[doi:10.1158/0008-5472.CAN-21-0950](https://doi.org/10.1158/0008-5472.CAN-21-0950).

## Methods

The three tumor sub-regions the segmentations use, and the convention of
treating their union as the whole tumor, follow Menze, B. H. et al. (2015). The
Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). *IEEE Transactions
on Medical Imaging* 34(10):1993-2024.
[doi:10.1109/TMI.2014.2377694](https://doi.org/10.1109/TMI.2014.2377694).

The decision not to make methylation of the MGMT promoter the classification
target rests on Baid, U. et al. (2021). The RSNA-ASNR-MICCAI BraTS 2021
benchmark on brain tumor segmentation and radiogenomic classification.
[arXiv:2107.02314](https://arxiv.org/abs/2107.02314). The radiogenomic task of
that challenge, predicting MGMT status from pre-operative MRI, produced a
winning validation area under the curve near 0.62, and later analyses of the
same data reported performance indistinguishable from chance.

The encoding of segmentations as DICOM objects, and the rules for relating a
segmentation frame to the image frame it was drawn on, are specified in the
DICOM standard, PS3.3 section C.8.20 (Segmentation IOD) and PS3.15 Annex E
(de-identification profiles), National Electrical Manufacturers Association,
<https://www.dicomstandard.org/current>.

## Software

Bridge, C. P., C. Gorman, S. Pieper, S. W. Doyle, J. K. Lennerz, J. Kalpathy-Cramer,
D. A. Clunie, A. Y. Fedorov and M. D. Herrmann (2022). Highdicom: a Python
library for standardized encoding of image annotations and machine learning
model outputs in pathology and radiology. *Journal of Digital Imaging*
35(6):1719-1737.
[doi:10.1007/s10278-022-00683-y](https://doi.org/10.1007/s10278-022-00683-y).
Used to read DICOM Segmentation objects and to resample a segmentation onto the
geometry of the image series it references.

Mason, D. (2011). SU-E-T-33: pydicom, an open source DICOM library. *Medical
Physics* 38(6):3493. [doi:10.1118/1.3611983](https://doi.org/10.1118/1.3611983).

Paszke, A., S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen,
Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Köpf, E. Yang, Z. DeVito,
M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai and
S. Chintala (2019). PyTorch: an imperative style, high-performance deep learning
library. *Advances in Neural Information Processing Systems* 32:8024-8035.

Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *Journal
of Machine Learning Research* 12:2825-2830.

Harris, C. R. et al. (2020). Array programming with NumPy. *Nature*
585:357-362. [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).

Hunter, J. D. (2007). Matplotlib: a 2D graphics environment. *Computing in
Science and Engineering* 9(3):90-95.
[doi:10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55).

## Provenance

The analysis was first carried out as coursework for DATA_SCI 8140, Advanced
Methods in Health Data Science, University of Missouri, in Fall 2025. The
coursework used two datasets that are not reused here. One is under a
restricted license that forbids redistribution; the other arrived with no
license, citation or source recorded. Both were replaced by UPENN-GBM, whose
license permits reuse and whose provenance is checkable at the source on every
run of the pipeline.
