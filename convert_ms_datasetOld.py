"""
Dataset Conversion Script for Longitudinal MS MRI Dataset
Converts raw MS dataset to LongiSeg/nnU-Net format.

Input structure:
    long-MR-MS/
        patient1/
            patient1_brainmask.nii.gz
            patient1_gt.nii.gz
            patient1_study1_FLAIRreg.nii.gz
            patient1_study1_T1Wreg.nii.gz
            patient1_study2_FLAIRreg.nii.gz
            patient1_study2_T1Wreg.nii.gz
        patient2/
            ...

Output structure:
    LongiSeg_raw/Dataset001_MSLesions/
        dataset.json
        imagesTr/
            ms_001_0000.nii.gz  <- study1 FLAIR (training patients)
            ms_001_0001.nii.gz  <- study1 T1W
            ms_001_0002.nii.gz  <- study2 FLAIR
            ms_001_0003.nii.gz  <- study2 T1W
            ...
        imagesTs/
            ms_018_0000.nii.gz  <- study1 FLAIR (test patients)
            ...
        labelsTr/
            ms_001.nii.gz       <- lesion change mask (training)
            ...

Channel convention:
    _0000 -> Study 1 FLAIR
    _0001 -> Study 1 T1W
    _0002 -> Study 2 FLAIR
    _0003 -> Study 2 T1W

Split:
    Training (cross-validation): patients 1-17
    Test (held out):             patients 18, 19, 20
"""

import os
import shutil
import json
import numpy as np
import nibabel as nib
from pathlib import Path


# =============================================================================
# CONFIGURATION — edit these paths if needed
# =============================================================================

# Path to your raw MS dataset
RAW_DATASET_DIR = Path(r"D:\Desktop\long-MR-MS")

# Path to LongiSeg_raw (where the converted dataset will be saved)
LONGISEG_RAW_DIR = Path(r"D:\Desktop\LongiSeg_Project\LongiSeg_raw")

# Dataset ID and name (choose an ID not already taken)
DATASET_ID   = 1           # results in Dataset001_MSLesions
DATASET_NAME = "MSLesions"

# Patient split
TRAIN_PATIENTS = list(range(1, 18))   # patients 1-17 for training/cross-val
TEST_PATIENTS  = [18, 19, 20]         # patients 18-20 held out for testing

# Apply brain mask to images before saving? (recommended)
APPLY_BRAIN_MASK = True

# =============================================================================


def apply_mask(image_path: Path, mask_path: Path) -> nib.Nifti1Image:
    """Load image and zero out non-brain voxels using the brain mask."""
    img     = nib.load(str(image_path))
    mask    = nib.load(str(mask_path))
    img_data  = img.get_fdata(dtype=np.float32)
    mask_data = mask.get_fdata(dtype=np.float32)
    masked    = img_data * (mask_data > 0).astype(np.float32)
    return nib.Nifti1Image(masked, img.affine, img.header)


def copy_or_mask_and_save(src: Path, dst: Path,
                           mask_path: Path, apply_mask_flag: bool):
    """Either apply brain mask and save, or just copy the file."""
    if apply_mask_flag and mask_path is not None and mask_path.exists():
        masked_img = apply_mask(src, mask_path)
        nib.save(masked_img, str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def get_patient_files(patient_dir: Path, patient_name: str):
    """Return a dict of expected file paths for a patient."""
    return {
        "brainmask":    patient_dir / f"{patient_name}_brainmask.nii.gz",
        "gt":           patient_dir / f"{patient_name}_gt.nii.gz",
        "s1_flair":     patient_dir / f"{patient_name}_study1_FLAIRreg.nii.gz",
        "s1_t1w":       patient_dir / f"{patient_name}_study1_T1Wreg.nii.gz",
        "s2_flair":     patient_dir / f"{patient_name}_study2_FLAIRreg.nii.gz",
        "s2_t1w":       patient_dir / f"{patient_name}_study2_T1Wreg.nii.gz",
    }


def verify_patient_files(files: dict, patient_name: str) -> bool:
    """Check all required files exist for a patient."""
    all_ok = True
    for key, path in files.items():
        if not path.exists():
            print(f"  [WARNING] Missing file for {patient_name}: {path.name}")
            all_ok = False
    return all_ok


def convert_dataset():
    # ------------------------------------------------------------------
    # 1. Set up output directory structure
    # ------------------------------------------------------------------
    dataset_folder = LONGISEG_RAW_DIR / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    images_tr_dir  = dataset_folder / "imagesTr"
    images_ts_dir  = dataset_folder / "imagesTs"
    labels_tr_dir  = dataset_folder / "labelsTr"

    for d in [images_tr_dir, images_ts_dir, labels_tr_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {dataset_folder}\n")

    # ------------------------------------------------------------------
    # 2. Process training patients (1-17)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Processing TRAINING patients (1-17)...")
    print("=" * 60)

    train_cases = []

    for patient_num in TRAIN_PATIENTS:
        patient_name = f"patient{patient_num}"
        patient_dir  = RAW_DATASET_DIR / patient_name
        case_id      = f"ms_{patient_num:03d}"

        print(f"\nProcessing {patient_name} -> {case_id}")

        if not patient_dir.exists():
            print(f"  [ERROR] Patient directory not found: {patient_dir}")
            continue

        files = get_patient_files(patient_dir, patient_name)

        if not verify_patient_files(files, patient_name):
            print(f"  [ERROR] Skipping {patient_name} due to missing files.")
            continue

        mask_path = files["brainmask"] if APPLY_BRAIN_MASK else None

        # Copy/save the 4 image channels
        channel_map = {
            "0000": files["s1_flair"],   # Study 1 FLAIR
            "0001": files["s1_t1w"],     # Study 1 T1W
            "0002": files["s2_flair"],   # Study 2 FLAIR
            "0003": files["s2_t1w"],     # Study 2 T1W
        }

        for channel_id, src_path in channel_map.items():
            dst_path = images_tr_dir / f"{case_id}_{channel_id}.nii.gz"
            copy_or_mask_and_save(src_path, dst_path, mask_path, APPLY_BRAIN_MASK)
            print(f"  Saved channel {channel_id}: {dst_path.name}")

        # Copy ground truth label
        label_dst = labels_tr_dir / f"{case_id}.nii.gz"
        shutil.copy2(str(files["gt"]), str(label_dst))
        print(f"  Saved label:           {label_dst.name}")

        train_cases.append(case_id)

    # ------------------------------------------------------------------
    # 3. Process test patients (18-20)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Processing TEST patients (18-20)...")
    print("=" * 60)

    test_cases = []

    for patient_num in TEST_PATIENTS:
        patient_name = f"patient{patient_num}"
        patient_dir  = RAW_DATASET_DIR / patient_name
        case_id      = f"ms_{patient_num:03d}"

        print(f"\nProcessing {patient_name} -> {case_id}")

        if not patient_dir.exists():
            print(f"  [ERROR] Patient directory not found: {patient_dir}")
            continue

        files = get_patient_files(patient_dir, patient_name)

        if not verify_patient_files(files, patient_name):
            print(f"  [ERROR] Skipping {patient_name} due to missing files.")
            continue

        mask_path = files["brainmask"] if APPLY_BRAIN_MASK else None

        channel_map = {
            "0000": files["s1_flair"],
            "0001": files["s1_t1w"],
            "0002": files["s2_flair"],
            "0003": files["s2_t1w"],
        }

        for channel_id, src_path in channel_map.items():
            dst_path = images_ts_dir / f"{case_id}_{channel_id}.nii.gz"
            copy_or_mask_and_save(src_path, dst_path, mask_path, APPLY_BRAIN_MASK)
            print(f"  Saved channel {channel_id}: {dst_path.name}")

        # NOTE: ground truth for test patients is saved separately
        # so it is never accidentally used during training
        gt_backup_dir = dataset_folder / "labelsTs"
        gt_backup_dir.mkdir(exist_ok=True)
        shutil.copy2(str(files["gt"]), str(gt_backup_dir / f"{case_id}.nii.gz"))
        print(f"  Saved test label to labelsTs/ (not used in training)")

        test_cases.append(case_id)

    # ------------------------------------------------------------------
    # 4. Create dataset.json
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Creating dataset.json...")
    print("=" * 60)

    dataset_json = {
        "channel_names": {
            "0": "FLAIR_study1",
            "1": "T1W_study1",
            "2": "FLAIR_study2",
            "3": "T1W_study2"
        },
        "labels": {
            "background": 0,
            "lesion_change": 1
        },
        "numTraining": len(train_cases),
        "file_ending": ".nii.gz",
        "name": DATASET_NAME,
        "description": "Longitudinal MS lesion change segmentation. "
                       "Channels 0-1: Study1 FLAIR+T1W, Channels 2-3: Study2 FLAIR+T1W.",
        "reference": "Lesjak et al., Neuroinformatics 2016",
        "licence": "CC-BY",
        "tensorImageSize": "3D"
    }

    json_path = dataset_folder / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    print(f"Saved: {json_path}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Training cases : {len(train_cases)} patients -> imagesTr/ + labelsTr/")
    print(f"Test cases     : {len(test_cases)} patients -> imagesTs/ + labelsTs/")
    print(f"Dataset folder : {dataset_folder}")
    print("\nChannel layout per case:")
    print("  _0000 -> Study 1 FLAIR")
    print("  _0001 -> Study 1 T1W")
    print("  _0002 -> Study 2 FLAIR")
    print("  _0003 -> Study 2 T1W")
    print("\nNext step — verify dataset integrity:")
    print(f"  LongiSeg_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")


if __name__ == "__main__":
    convert_dataset()
