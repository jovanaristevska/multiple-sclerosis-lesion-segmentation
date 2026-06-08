# MS Lesion Segmentation with LongiSeg

This repository contains the code and configuration for training and evaluating deep learning models for **Multiple Sclerosis (MS) white matter lesion segmentation** from MRI scans.

Two models are implemented and compared:
- **LongiUNet** — longitudinal model using two MRI timepoints (LongiSeg framework)
- **3D U-Net** — cross-sectional model using a single MRI timepoint (nnU-Net framework)

---

## Project Structure

```
LongiSeg_Project/
├── LongiSeg/                          # LongiSeg source code (cloned from MIC-DKFZ)
├── LongiSeg_raw/
│   ├── Dataset001_MSLesions/          # Longitudinal dataset (LongiUNet)
│   │   ├── imagesTr/                  # Training images (FLAIR + T1W, t1 + t2)
│   │   ├── imagesTs/                  # Test images
│   │   ├── labelsTr/                  # Training labels (lesion change masks)
│   │   ├── labelsTs/                  # Test labels
│   │   ├── dataset.json               # Dataset configuration
│   │   ├── patientsTr.json            # Longitudinal patient pairing (train)
│   │   └── patientsTs.json            # Longitudinal patient pairing (test)
│   └── Dataset002_MSLesionsSingleTP/  # Single timepoint dataset (3D U-Net)
│       ├── imagesTr/                  # Training images (FLAIR + T1W)
│       ├── imagesTs/                  # Test images
│       ├── labelsTr/                  # Training labels (lesion masks)
│       ├── labelsTs/                  # Test labels
│       ├── dataset.json               # Dataset configuration
│       ├── patientsTr.json            # Patient pairing (train)
│       └── patientsTs.json            # Patient pairing (test)
├── LongiSeg_preprocessed/             # Preprocessed data (auto-generated)
├── LongiSeg_results/                  # Training results and checkpoints
├── predictions_allfolds/              # LongiUNet inference results (5 folds)
├── predictions_singletp_5folds/       # 3D U-Net inference results (5 folds)
├── logs/                              # SLURM job logs
├── train_all_folds.sh                 # SLURM training script for LongiUNet
├── train_singletp_all_folds.sh        # SLURM training script for 3D U-Net
├── inference_longiseg.sh              # SLURM inference script for LongiUNet
└── inference_singletp.sh              # SLURM inference script for 3D U-Net
```

---

## Datasets

### Dataset 1 — Longitudinal MS MRI Dataset (LongiUNet)
- **Source**: Lesjak et al., Neuroinformatics 2016
- **Patients**: 20 total — 17 training, 3 test
- **Timepoints**: 2 per patient (baseline + follow-up)
- **Modalities**: FLAIR + T1-weighted
- **Labels**: White matter lesion change masks (follow-up only)
- **Task**: Lesion change detection between two timepoints

### Dataset 2 — Single Timepoint MS MRI Dataset (3D U-Net)
- **Source**: Lesjak et al., 3D MR Image Database
- **Patients**: 30 total — 24 training, 6 test
- **Timepoints**: 1 per patient
- **Modalities**: FLAIR + T1-weighted
- **Labels**: White matter lesion masks
- **Task**: Cross-sectional lesion segmentation

---

## Models

### LongiUNet (Longitudinal)
- **Framework**: LongiSeg (extension of nnU-Net)
- **Trainer**: `LongiSegTrainer`
- **Input**: Two MRI scans (current + prior timepoint)
- **Key feature**: Difference Weighting Block (DWB) for temporal feature fusion
- **Training**: 200 epochs, 5-fold cross-validation
- **Inference**: 5-fold ensemble

### 3D U-Net (Cross-sectional)
- **Framework**: LongiSeg / nnU-Net
- **Trainer**: `nnUNetTrainerNoLongi`
- **Input**: Single MRI scan
- **Key feature**: Self-configuring nnU-Net pipeline
- **Training**: 200 epochs, 5-fold cross-validation
- **Inference**: 5-fold ensemble

---

## Results

| Method | DSC↑ | Precision↑ | Recall↑ | F1↑ | Bal.Acc↑ | IoU↑ |
|--------|------|-----------|--------|-----|---------|-----|
| 3D U-Net (one TP) | 0.608 | 0.778 | 0.519 | 0.608 | 0.759 | 0.462 |
| LongiUNet (two TP) | 0.317 | 0.798 | 0.209 | 0.317 | 0.605 | 0.194 |

> **Note**: LongiUNet and 3D U-Net were evaluated on different datasets performing different tasks. LongiUNet detects lesion *changes* between two timepoints, while 3D U-Net segments existing lesions from a single timepoint. Direct comparison is therefore limited.

---

## Requirements

- Python 3.10
- PyTorch 2.2.1
- CUDA (NVIDIA GPU)
- LongiSeg: https://github.com/MIC-DKFZ/LongiSeg

---

## Installation

```bash
# Clone LongiSeg
git clone https://github.com/MIC-DKFZ/LongiSeg.git
cd LongiSeg
pip install -e .

# Set environment variables
export LongiSeg_raw=/path/to/LongiSeg_raw
export LongiSeg_preprocessed=/path/to/LongiSeg_preprocessed
export LongiSeg_results=/path/to/LongiSeg_results
```

---

## Usage

### 1. Dataset Conversion

Convert raw MRI data to LongiSeg format:

```bash
# Longitudinal dataset (Dataset001)
python convert_ms_dataset.py

# Single timepoint dataset (Dataset002)
python convert_singletp_dataset.py
```

### 2. Preprocessing

```bash
# Dataset001 (LongiUNet)
LongiSeg_plan_and_preprocess -d 001 --verify_dataset_integrity

# Dataset002 (3D U-Net)
LongiSeg_plan_and_preprocess -d 002 --verify_dataset_integrity
```

### 3. Training

```bash
# LongiUNet — all 5 folds
sbatch train_all_folds.sh

# 3D U-Net — all 5 folds
sbatch train_singletp_all_folds.sh
```

### 4. Inference

```bash
# LongiUNet
sbatch inference_longiseg.sh

# 3D U-Net
sbatch inference_singletp.sh
```

### 5. Results

```bash
# LongiUNet results
cat predictions_allfolds/longi_summary.json

# 3D U-Net results
cat predictions_singletp_5folds/longi_summary.json
```
---

## Hardware

- **Cluster**: SLURM-managed HPC cluster
- **GPU**: NVIDIA A100-SXM4-80GB
- **Container**: Singularity with PyTorch 2.2.1

---

## References

- Rokuss et al. (2024). *Longitudinal Segmentation of MS Lesions via Temporal Difference Weighting*. MICCAI Workshops.
- Isensee et al. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*. Nature Methods.
- Lesjak et al. (2016). *A Novel Public MR Image Dataset of Multiple Sclerosis Patients With Lesion Segmentation*. Neuroinformatics.
