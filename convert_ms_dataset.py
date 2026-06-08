"""
Dataset Conversion Script for Longitudinal MS MRI Dataset
Converts raw MS dataset to LongiSeg format.

IMPORTANT: LongiSeg requires each timepoint to be a SEPARATE CASE.
Each patient must have at least 2 case entries in patientsTr.json.

Input structure:
    long-MR-MS/
        patient1/
            patient1_brainmask.nii.gz
            patient1_gt.nii.gz           <- lesion change mask (study2 only)
            patient1_study1_FLAIRreg.nii.gz
            patient1_study1_T1Wreg.nii.gz
            patient1_study2_FLAIRreg.nii.gz
            patient1_study2_T1Wreg.nii.gz

Output structure:
    LongiSeg_raw/Dataset001_MSLesions/
        dataset.json
        patientsTr.json
        imagesTr/
            ms_001_t1_0000.nii.gz  <- patient1 study1 FLAIR
            ms_001_t1_0001.nii.gz  <- patient1 study1 T1W
            ms_001_t2_0000.nii.gz  <- patient1 study2 FLAIR
            ms_001_t2_0001.nii.gz  <- patient1 study2 T1W
            ...
        imagesTs/
            ms_018_t1_0000.nii.gz  <- test patients
            ...
        labelsTr/
            ms_001_t1.nii.gz       <- ALL ZEROS (no lesion change at baseline)
            ms_001_t2.nii.gz       <- actual gt lesion change mask
            ...
        labelsTs/
            ms_018_t1.nii.gz       <- zeros
            ms_018_t2.nii.gz       <- gt mask (backup, not used in training)

Channel convention (2 channels per case):
    _0000 -> FLAIR
    _0001 -> T1W

patientsTr.json convention:
    "patient1": ["ms_001_t1", "ms_001_t2"]
    LongiSeg requires len > 1 per patient to build longitudinal pairs

Split:
    Training (cross-validation): patients 1-17
    Test (held out):             patients 18, 19, 20
"""

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

# Path to LongiSeg_raw
LONGISEG_RAW_DIR = Path(r"D:\Desktop\LongiSeg_Project\LongiSeg_raw")

# Dataset ID and name
DATASET_ID   = 1
DATASET_NAME = "MSLesions"

# Patient split
TRAIN_PATIENTS = list(range(1, 18))   # patients 1-17
TEST_PATIENTS  = [18, 19, 20]         # held out

# Apply brain mask to images before saving? (recommended)
APPLY_BRAIN_MASK = True

# =============================================================================


def apply_mask(image_path: Path, mask_path: Path) -> nib.Nifti1Image:
    """Load image and zero out non-brain voxels using the brain mask."""
    img       = nib.load(str(image_path))
    mask      = nib.load(str(mask_path))
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


def create_zero_label(reference_path: Path, output_path: Path):
    """Create an all-zeros label with same shape/affine as reference image."""
    ref     = nib.load(str(reference_path))
    zeros   = np.zeros(ref.shape[:3], dtype=np.uint8)
    zero_img = nib.Nifti1Image(zeros, ref.affine, ref.header)
    nib.save(zero_img, str(output_path))


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


def process_patient(patient_num: int,
                    output_images_dir: Path,
                    output_labels_dir: Path) -> tuple:
    """
    Process a single patient: creates TWO cases (t1 and t2).

    Returns (t1_case_id, t2_case_id, success)

    Case naming:
        ms_001_t1 -> study1 (prior timepoint)
        ms_001_t2 -> study2 (follow-up timepoint)

    Labels:
        ms_001_t1.nii.gz -> all zeros (no lesion change at baseline)
        ms_001_t2.nii.gz -> actual gt lesion change mask
    """
    patient_name = f"patient{patient_num}"
    patient_dir  = RAW_DATASET_DIR / patient_name
    t1_case_id   = f"ms_{patient_num:03d}_t1"
    t2_case_id   = f"ms_{patient_num:03d}_t2"

    print(f"\n  Processing {patient_name} -> {t1_case_id}, {t2_case_id}")

    if not patient_dir.exists():
        print(f"  [ERROR] Patient directory not found: {patient_dir}")
        return t1_case_id, t2_case_id, False

    files = get_patient_files(patient_dir, patient_name)

    if not verify_patient_files(files, patient_name):
        print(f"  [ERROR] Skipping {patient_name} due to missing files.")
        return t1_case_id, t2_case_id, False

    mask_path = files["brainmask"] if APPLY_BRAIN_MASK else None

    # ------------------------------------------------------------------
    # Study 1 (t1) — prior timepoint
    # 2 channels: FLAIR (_0000) and T1W (_0001)
    # ------------------------------------------------------------------
    copy_or_mask_and_save(
        files["s1_flair"],
        output_images_dir / f"{t1_case_id}_0000.nii.gz",
        mask_path, APPLY_BRAIN_MASK
    )
    print(f"    {t1_case_id}_0000.nii.gz  <- study1 FLAIR")

    copy_or_mask_and_save(
        files["s1_t1w"],
        output_images_dir / f"{t1_case_id}_0001.nii.gz",
        mask_path, APPLY_BRAIN_MASK
    )
    print(f"    {t1_case_id}_0001.nii.gz  <- study1 T1W")

    # Label for study1 = all zeros (no lesion change at baseline)
    create_zero_label(
        files["s1_flair"],
        output_labels_dir / f"{t1_case_id}.nii.gz"
    )
    print(f"    {t1_case_id}.nii.gz       <- zeros label (baseline)")

    # ------------------------------------------------------------------
    # Study 2 (t2) — follow-up timepoint
    # 2 channels: FLAIR (_0000) and T1W (_0001)
    # ------------------------------------------------------------------
    copy_or_mask_and_save(
        files["s2_flair"],
        output_images_dir / f"{t2_case_id}_0000.nii.gz",
        mask_path, APPLY_BRAIN_MASK
    )
    print(f"    {t2_case_id}_0000.nii.gz  <- study2 FLAIR")

    copy_or_mask_and_save(
        files["s2_t1w"],
        output_images_dir / f"{t2_case_id}_0001.nii.gz",
        mask_path, APPLY_BRAIN_MASK
    )
    print(f"    {t2_case_id}_0001.nii.gz  <- study2 T1W")

    # Label for study2 = actual lesion change mask
    shutil.copy2(
        str(files["gt"]),
        str(output_labels_dir / f"{t2_case_id}.nii.gz")
    )
    print(f"    {t2_case_id}.nii.gz       <- gt lesion change mask")

    return t1_case_id, t2_case_id, True


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

    train_cases = []   # all case IDs (both t1 and t2)
    patients_tr = {}   # patientsTr.json content

    for patient_num in TRAIN_PATIENTS:
        t1_id, t2_id, success = process_patient(
            patient_num, images_tr_dir, labels_tr_dir
        )
        if success:
            train_cases.extend([t1_id, t2_id])
            # Each patient maps to [t1_case, t2_case]
            # len > 1 satisfies LongiSeg's requirement
            patients_tr[f"patient{patient_num}"] = [t1_id, t2_id]

    # ------------------------------------------------------------------
    # 3. Process test patients (18-20)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Processing TEST patients (18, 19, 20)...")
    print("=" * 60)

    test_cases = []

    for patient_num in TEST_PATIENTS:
        t1_id, t2_id, success = process_patient(
            patient_num, images_ts_dir, labels_ts_dir
        )
        if success:
            test_cases.extend([t1_id, t2_id])

    # ------------------------------------------------------------------
    # 4. Create patientsTr.json
    # ------------------------------------------------------------------
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
            "0": "FLAIR",
            "1": "T1W"
        },
        "labels": {
            "background": 0,
            "lesion_change": 1
        },
        "numTraining": len(train_cases),
        "file_ending": ".nii.gz",
        "name": DATASET_NAME,
        "description": "Longitudinal MS lesion change segmentation. "
                       "Each patient has 2 cases: t1 (baseline, zero label) "
                       "and t2 (follow-up, gt lesion change label). "
                       "2 channels per case: FLAIR + T1W.",
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
    print(f"Training cases  : {len(train_cases)} ({len(TRAIN_PATIENTS)} patients x 2 timepoints)")
    print(f"Test cases      : {len(test_cases)} ({len(TEST_PATIENTS)} patients x 2 timepoints)")
    print(f"patientsTr.json : {len(patients_tr)} patients")
    print(f"Dataset folder  : {dataset_folder}")
    print("\nFiles in imagesTr/:")
    print("  ms_001_t1_0000.nii.gz  <- patient1 study1 FLAIR")
    print("  ms_001_t1_0001.nii.gz  <- patient1 study1 T1W")
    print("  ms_001_t2_0000.nii.gz  <- patient1 study2 FLAIR")
    print("  ms_001_t2_0001.nii.gz  <- patient1 study2 T1W")
    print("  ...")
    print("\nFiles in labelsTr/:")
    print("  ms_001_t1.nii.gz  <- all zeros (baseline)")
    print("  ms_001_t2.nii.gz  <- gt lesion change mask")
    print("  ...")
    print(f"\nExpected file counts:")
    print(f"  imagesTr/ : {len(train_cases) * 2} files ({len(train_cases)} cases x 2 channels)")
    print(f"  labelsTr/ : {len(train_cases)} files")
    print(f"  imagesTs/ : {len(test_cases) * 2} files")
    print(f"  labelsTs/ : {len(test_cases)} files")
    print("\nNext step — run preprocessing on the cluster:")
    print(f"  LongiSeg_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")


if __name__ == "__main__":
    convert_dataset()
