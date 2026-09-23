# Brain MRI cohort quality control and tumor patch classification

This repository audits a glioblastoma MRI cohort and trains a convolutional network to classify tumor and non-tumor patches. The quality-control analysis examines DICOM headers, geometry, scanner variation, and label provenance before the classification results are interpreted. The work originated in DATA_SCI 8140, Advanced Methods in Health Data Science, in Fall 2025.

The images come from UPENN-GBM (Bakas et al. 2021), and the tumor segmentations come from BAMF annotations (Van Oss et al. 2024). Both are distributed under CC BY 4.0 through the NCI Imaging Data Commons. The repository contains a cohort manifest and derived results; it does not redistribute the images or segmentations.

## Findings

The cohort rule selected 49 patients and 322 series. Six scanner models are represented, although one accounts for 87.5% of the series. Across the selected segmentations, 24.31% of tumor voxels belong to an unreviewed enhancing-lesion segment. The labels therefore do not provide a fully reviewed clinical reference standard.

A network classifies 32 × 32 post-contrast T1 patches. On 13 held-out patients, test ROC-AUC was 0.9864 for the primary seed. The three patient-split seeds ranged from 0.9151 to 0.9864. Splitting patches across patients raised ROC-AUC at every seed, because overlapping patches from the same patients crossed the split. The positive rate was constructed by subsampling negative patches, so precision and accuracy here do not describe a natural-prevalence screening setting.

The [full analysis and figures](docs/writeup.md) give the cohort rule, error analysis, quality-control findings, and limitations. The [earlier extended README](docs/legacy-readme.md) retains the detailed pipeline record.

![A post-contrast T1 slice with its tumor mask and the mask aligned to the image grid.](figures/fig04_mask_alignment.png)

## Reproduce and check

Python 3.13 and the versions in [`requirements.txt`](requirements.txt) are required. These commands use Windows PowerShell. The download is approximately 9.44 GB and does not require credentials.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m src.s01_manifest
.venv/Scripts/python data/download_data.py
.venv/Scripts/python analysis/run_all.py
.venv/Scripts/python .checks/run_all_gates.py
```

The manifest step queries the Imaging Data Commons index. The downloader verifies file digests against [`data/checksums.txt`](data/checksums.txt). The analysis then runs the header, quality-control, patch, split, model, evaluation, and figure stages. The gates check acquisition, schema, analysis stages, reproducibility, notebooks, and the recorded results. The committed quantities are in [`results/`](results/), and the two [notebooks](analysis/) present them.

## Citation and license

The data and labels should be cited as Bakas et al. (2021), DOI [10.7937/TCIA.709X-DN49](https://doi.org/10.7937/TCIA.709X-DN49), and Van Oss et al. (2024), DOI [10.5281/zenodo.8345959](https://doi.org/10.5281/zenodo.8345959). Full citations are in [`docs/references.md`](docs/references.md). The code is released under the [MIT License](LICENSE); the source images and labels retain their CC BY 4.0 licenses.
