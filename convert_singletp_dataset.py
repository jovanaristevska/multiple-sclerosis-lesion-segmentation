"""
Dataset Conversion Script for Single Timepoint MS MRI Dataset
Converts raw MS dataset to nnU-Net/LongiSeg format for cross-sectional training.

Input structure:
    dataset_singletp/
        patient01/
            FLAIR.nii.gz
            T1.nii.gz
            lesion.nii.gz
        patient02/
            ...

Output structure:
    LongiSeg_raw/Dataset002_MSLesionsSingleTP/
        dataset.json
        imagesTr/
            ms_001_0000.nii.gz  <- FLAIR (training patients 1-24)
            ms_001_0001.nii.gz  <- T1W
            ...
        imagesTs/
            ms_025_0000.nii.gz  <- FLAIR (test patients 25-30)
            ms_025_0001.nii.gz  <- T1W
            ...
        labelsTr/
            ms_001.nii.gz       <- lesion mask (training)
            ...
        labelsTs/
            ms_025.nii.gz       <- lesion mask (test, backup only)
            ...

Channel convention:
    _0000 -> FLAIR
    _0001 -> T1W

Split:
    Training (cross-validation): patients 01-24
    Test (held out):             patients 25-30

NOTE: No patientsTr.json needed — standard nnU-Net format
"""

import shutil
import json
import numpy as np
import nibabel as nib
from pathlib import Path


# =============================================================================
# CONFIGURATION — edit these paths if needed
# =============================================================================

# Path to your raw single timepoint MS dataset
RAW_DATASET_DIR = Path(r"D:\Documents\MANU\dataset_singletp")

# Path to LongiSeg_raw
LONGISEG_RAW_DIR = Path(r"D:\Desktop\LongiSeg_Project\LongiSeg_raw")

print(f"Looking for dataset in: {RAW_DATASET_DIR}")
print(f"Dataset exists: {RAW_DATASET_DIR.exists()}")

# Dataset ID and name
DATASET_ID   = 2
DATASET_NAME = "MSLesionsSingleTP"

# Patient numbers — zero padded (01, 02 ... 30)
TRAIN_PATIENTS = list(range(1, 25))    # patients 01-24 for training
TEST_PATIENTS  = list(range(25, 31))   # patients 25-30 held out for testing

# =============================================================================


def get_patient_files(patient_dir: Path) -> dict:
    """Return a dict of expected file paths for a patient."""
    return {
        "flair":  patient_dir / "FLAIR.nii.gz",
        "t1":     patient_dir / "T1.nii.gz",
        "lesion": patient_dir / "lesion.nii.gz",
    }


def verify_patient_files(files: dict, patient_name: str) -> bool:
    """Check all required files exist for a patient."""
    all_ok = True
    for key, path in files.items():
        if not path.exists():
            print(f"  [WARNING] Missing file for {patient_name}: {path.name}")
            all_ok = False
    return all_ok


def process_patient(patient_num: int,
                    output_images_dir: Path,
                    output_labels_dir: Path) -> tuple:
    """
    Process a single patient: copy images and label into output dirs.
    Returns (case_id, success)
    """
    patient_name = f"patient{patient_num:02d}"
    patient_dir  = RAW_DATASET_DIR / patient_name
    case_id      = f"ms_{patient_num:03d}"

    print(f"\n  Processing {patient_name} -> {case_id}")

    if not patient_dir.exists():
        print(f"  [ERROR] Patient directory not found: {patient_dir}")
        return case_id, False

    files = get_patient_files(patient_dir)

    if not verify_patient_files(files, patient_name):
        print(f"  [ERROR] Skipping {patient_name} due to missing files.")
        return case_id, False

    # Copy FLAIR as channel 0000
    dst_flair = output_images_dir / f"{case_id}_0000.nii.gz"
    shutil.copy2(str(files["flair"]), str(dst_flair))
    print(f"    {case_id}_0000.nii.gz  <- FLAIR")

    # Copy T1 as channel 0001
    dst_t1 = output_images_dir / f"{case_id}_0001.nii.gz"
    shutil.copy2(str(files["t1"]), str(dst_t1))
    print(f"    {case_id}_0001.nii.gz  <- T1W")

    # Copy lesion mask as label
    dst_label = output_labels_dir / f"{case_id}.nii.gz"
    shutil.copy2(str(files["lesion"]), str(dst_label))
    print(f"    {case_id}.nii.gz       <- lesion mask")

    return case_id, True


def convert_dataset():
    # ------------------------------------------------------------------
    # 1. Set up output directory structure
    # ------------------------------------------------------------------
    dataset_folder = LONGISEG_RAW_DIR / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    images_tr_dir  = dataset_folder / "imagesTr"
    images_ts_dir  = dataset_folder / "imagesTs"
    labels_tr_dir  = dataset_folder / "labelsTr"
    labels_ts_dir  = dataset_folder / "labelsTs"

    for d in [images_tr_dir, images_ts_dir, labels_tr_dir, labels_ts_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {dataset_folder}\n")

    # ------------------------------------------------------------------
    # 2. Process training patients (01-24)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Processing TRAINING patients (01-24)...")
    print("=" * 60)

    train_cases = []

    for patient_num in TRAIN_PATIENTS:
        case_id, success = process_patient(
            patient_num, images_tr_dir, labels_tr_dir
        )
        if success:
            train_cases.append(case_id)

    # ------------------------------------------------------------------
    # 3. Process test patients (25-30)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Processing TEST patients (25-30)...")
    print("=" * 60)

    test_cases = []

    for patient_num in TEST_PATIENTS:
        case_id, success = process_patient(
            patient_num, images_ts_dir, labels_ts_dir
        )
        if success:
            test_cases.append(case_id)

    # ------------------------------------------------------------------
    # 4. Create dataset.json
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Creating dataset.json...")
    print("=" * 60)

    dataset_json = {
        "channel_names": {
            "0": "FLAIR",
            "1": "T1W"
        },
        "labels": {
            "background": 0,
            "lesion": 1
        },
        "numTraining": len(train_cases),
        "file_ending": ".nii.gz",
        "name": DATASET_NAME,
        "description": "Single timepoint MS lesion segmentation. "
                       "2 channels: FLAIR + T1W. "
                       "Cross-sectional baseline for comparison with LongiUNet.",
        "reference": "Single timepoint MS dataset",
        "licence": "CC-BY",
        "tensorImageSize": "3D"
    }

    json_path = dataset_folder / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    print(f"Saved dataset.json: {json_path}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Training cases  : {len(train_cases)} patients -> imagesTr/ + labelsTr/")
    print(f"Test cases      : {len(test_cases)} patients  -> imagesTs/ + labelsTs/")
    print(f"Dataset folder  : {dataset_folder}")
    print("\nChannel layout per case:")
    print("  _0000 -> FLAIR")
    print("  _0001 -> T1W")
    print(f"\nExpected file counts:")
    print(f"  imagesTr/ : {len(train_cases) * 2} files ({len(train_cases)} cases x 2 channels)")
    print(f"  labelsTr/ : {len(train_cases)} files")
    print(f"  imagesTs/ : {len(test_cases) * 2} files")
    print(f"  labelsTs/ : {len(test_cases)} files")
    print("\nNext step — run preprocessing on the cluster:")
    print(f"  LongiSeg_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")


if __name__ == "__main__":
    convert_dataset()