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
        patientsTr.json      <- LongiSeg-specific: maps patient -> list of case IDs
        imagesTr/
            ms_001_0000.nii.gz  <- study1 FLAIR (training patients 1-17)
            ms_001_0001.nii.gz  <- study1 T1W
            ms_001_0002.nii.gz  <- study2 FLAIR
            ms_001_0003.nii.gz  <- study2 T1W
            ...
        imagesTs/
            ms_018_0000.nii.gz  <- study1 FLAIR (test patients 18-20)
            ...
        labelsTr/
            ms_001.nii.gz       <- lesion change mask (training)
            ...
        labelsTs/
            ms_018.nii.gz       <- lesion change mask (test, backup only)

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

# Dataset ID and name
DATASET_ID   = 1
DATASET_NAME = "MSLesions"

# Patient split
TRAIN_PATIENTS = list(range(1, 18))   # patients 1-17 for training/cross-val
TEST_PATIENTS  = [18, 19, 20]         # patients 18-20 held out for testing

# Apply brain mask to images before saving? (recommended)
APPLY_BRAIN_MASK = True

# =============================================================================


def apply_mask(image_path: Path, mask_path: Path) -> nib.Nifti1Image:
    """Load image and zero out non-brain voxels using the brain mask."""
    img      = nib.load(str(image_path))
    mask     = nib.load(str(mask_path))
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
        "brainmask": patient_dir / f"{patient_name}_brainmask.nii.gz",
        "gt":        patient_dir / f"{patient_name}_gt.nii.gz",
        "s1_flair":  patient_dir / f"{patient_name}_study1_FLAIRreg.nii.gz",
        "s1_t1w":    patient_dir / f"{patient_name}_study1_T1Wreg.nii.gz",
        "s2_flair":  patient_dir / f"{patient_name}_study2_FLAIRreg.nii.gz",
        "s2_t1w":    patient_dir / f"{patient_name}_study2_T1Wreg.nii.gz",
    }


def verify_patient_files(files: dict, patient_name: str) -> bool:
    """Check all required files exist for a patient."""
    all_ok = True
    for key, path in files.items():
        if not path.exists():
            print(f"  [WARNING] Missing file for {patient_name}: {path.name}")
            all_ok = False
    return all_ok


def process_patient(patient_num: int, output_images_dir: Path,
                    output_labels_dir: Path) -> tuple:
    """
    Process a single patient: copy/mask images and label into output dirs.
    Returns (case_id, success).
    """
    patient_name = f"patient{patient_num}"
    patient_dir  = RAW_DATASET_DIR / patient_name
    case_id      = f"ms_{patient_num:03d}"

    print(f"\n  Processing {patient_name} -> {case_id}")

    if not patient_dir.exists():
        print(f"  [ERROR] Patient directory not found: {patient_dir}")
        return case_id, False

    files = get_patient_files(patient_dir, patient_name)

    if not verify_patient_files(files, patient_name):
        print(f"  [ERROR] Skipping {patient_name} due to missing files.")
        return case_id, False

    mask_path = files["brainmask"] if APPLY_BRAIN_MASK else None

    # 4 channels: study1 FLAIR, study1 T1W, study2 FLAIR, study2 T1W
    channel_map = {
        "0000": files["s1_flair"],
        "0001": files["s1_t1w"],
        "0002": files["s2_flair"],
        "0003": files["s2_t1w"],
    }

    for channel_id, src_path in channel_map.items():
        dst_path = output_images_dir / f"{case_id}_{channel_id}.nii.gz"
        copy_or_mask_and_save(src_path, dst_path, mask_path, APPLY_BRAIN_MASK)
        print(f"    Channel {channel_id}: {dst_path.name}")

    label_dst = output_labels_dir / f"{case_id}.nii.gz"
    shutil.copy2(str(files["gt"]), str(label_dst))
    print(f"    Label:          {label_dst.name}")

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
    # 2. Process training patients (1-17)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Processing TRAINING patients (1-17)...")
    print("=" * 60)

    train_cases = []   # list of successful case IDs
    patients_tr = {}   # patientsTr.json content

    for patient_num in TRAIN_PATIENTS:
        case_id, success = process_patient(
            patient_num, images_tr_dir, labels_tr_dir
        )
        if success:
            train_cases.append(case_id)
            # Each patient maps to a list containing their single case ID.
            # This is what LongiSeg's patientsTr.json requires.
            # The value is a LIST because LongiSeg supports patients with
            # more than 2 timepoints — each extra timepoint would be an
            # additional case entry in this list. Since your patients each
            # have exactly 2 timepoints packed into 4 channels of one case,
            # each patient has a single-element list here.
            patients_tr[f"patient{patient_num}"] = [case_id]

    # ------------------------------------------------------------------
    # 3. Process test patients (18-20)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Processing TEST patients (18, 19, 20)...")
    print("=" * 60)

    test_cases = []

    for patient_num in TEST_PATIENTS:
        case_id, success = process_patient(
            patient_num, images_ts_dir, labels_ts_dir
        )
        if success:
            test_cases.append(case_id)

    # ------------------------------------------------------------------
    # 4. Create patientsTr.json  <-- LongiSeg-specific required file
    # ------------------------------------------------------------------
    # Format:
    # {
    #   "patient1":  ["ms_001"],
    #   "patient2":  ["ms_002"],
    #   ...
    #   "patient17": ["ms_017"]
    # }
    patients_tr_path = dataset_folder / "patientsTr.json"
    with open(patients_tr_path, "w") as f:
        json.dump(patients_tr, f, indent=4)

    print(f"\nSaved patientsTr.json: {patients_tr_path}")
    print("  Content preview (first 3 entries):")
    for k, v in list(patients_tr.items())[:3]:
        print(f"    '{k}': {v}")

    # ------------------------------------------------------------------
    # 5. Create dataset.json
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
    print(f"Saved dataset.json: {json_path}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Training cases  : {len(train_cases)} patients -> imagesTr/ + labelsTr/")
    print(f"Test cases      : {len(test_cases)} patients  -> imagesTs/ + labelsTs/")
    print(f"patientsTr.json : {len(patients_tr)} entries")
    print(f"Dataset folder  : {dataset_folder}")
    print("\nFiles in Dataset folder:")
    print("  dataset.json")
    print("  patientsTr.json  <- required by LongiSeg")
    print("  imagesTr/        <- 68 files (17 x 4 channels)")
    print("  imagesTs/        <- 12 files (3 x 4 channels)")
    print("  labelsTr/        <- 17 files")
    print("  labelsTs/        <- 3 files (backup, not used in training)")
    print("\nChannel layout per case:")
    print("  _0000 -> Study 1 FLAIR")
    print("  _0001 -> Study 1 T1W")
    print("  _0002 -> Study 2 FLAIR")
    print("  _0003 -> Study 2 T1W")
    print("\nNext step — run preprocessing:")
    print(f"  LongiSeg_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")


if __name__ == "__main__":
    convert_dataset()
