# nako_bids.py
"""
NAKO BIDS Conversion Library
Converts NAKO resting-state fMRI (Siemens 3T 64x64 EPI mosaics) and T1 to BIDS.
"""

import os
import glob
import json
import shutil
import zipfile
import pydicom
import numpy as np
import nibabel as nib
from tqdm import tqdm
from pathlib import Path


def build_affine_from_dicom(ds: pydicom.Dataset) -> np.ndarray:
    """
    Build a NIfTI-compatible affine matrix from Siemens DICOM headers.
    Assumes axial EPI with standard RAS+ orientation.
    """
    try:
        pixel_spacing = ds.PixelSpacing  # [row_spacing, col_spacing]
        slice_thickness = float(getattr(ds, 'SliceThickness', 3.0))
        row_spacing = float(pixel_spacing[0])   # y: posterior → anterior
        col_spacing = float(pixel_spacing[1])   # x: left ← right (radiological)

        affine = np.eye(4)
        affine[0, 0] = -col_spacing  # RAS: flip left-right
        affine[1, 1] = row_spacing   # anterior-posterior
        affine[2, 2] = slice_thickness  # inferior-superior
        return affine
    except Exception as e:
        print(f"⚠️ Could not build affine from DICOM: {e}. Using fallback.")
        return np.diag([-3.125, 3.125, 3.1, 1.0])


def reconstruct_nako_mosaic_fMRI(
    dicom_folder: str,
    output_nii: str = None,
    show_progress: bool = False
) -> tuple[np.ndarray, dict]:
    """
    Reconstructs 4D fMRI from NAKO Siemens EPI mosaic DICOMs (6x6 grid of 64x64 tiles).
    
    Returns:
        fmri_4d: (64, 64, 36, T) float32 array
        metadata: dict with 'first_dcm' (pydicom Dataset), 'n_volumes' (int)
    """
    ROWS, COLS = 6, 6
    TILE_H = TILE_W = 64
    EXPECTED_SHAPE = (384, 384)

    all_files = sorted([
        f for f in os.listdir(dicom_folder)
        if os.path.isfile(os.path.join(dicom_folder, f))
    ])

    if not all_files:
        raise ValueError(f"No files found in DICOM folder: {dicom_folder}")

    valid_files = []
    fmri_volumes = []
    valid_dcms = []

    iterable = tqdm(all_files, desc="Reconstructing", disable=not show_progress) if show_progress else all_files

    for fname in iterable:
        filepath = os.path.join(dicom_folder, fname)
        try:
            ds = pydicom.dcmread(filepath, force=True)
            
            if not hasattr(ds, 'file_meta'):
                ds.file_meta = pydicom.FileMetaDataset()
            if 'TransferSyntaxUID' not in ds.file_meta:
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            
            composite = ds.pixel_array
            
            if composite.shape != EXPECTED_SHAPE:
                continue
                
            tiles = []
            for row in range(ROWS):
                for col in range(COLS):
                    tile = composite[
                        row * TILE_H : (row + 1) * TILE_H,
                        col * TILE_W : (col + 1) * TILE_W
                    ]
                    tiles.append(tile.astype(np.float32))
            
            vol_3d = np.stack(tiles, axis=-1)
            fmri_volumes.append(vol_3d)
            valid_files.append(fname)
            valid_dcms.append(ds)
            
        except Exception:
            continue

    if not valid_files:
        raise ValueError("No valid DICOMs found! Check your folder.")

    fmri_4d = np.stack(fmri_volumes, axis=-1)
    fmri_4d = np.ascontiguousarray(fmri_4d, dtype=np.float32)
    print(f"✅ Reconstructed {len(valid_files)} volumes. Shape: {fmri_4d.shape}")

    if output_nii is not None:
        affine = build_affine_from_dicom(valid_dcms[0])
        tr_sec = float(getattr(valid_dcms[0], 'RepetitionTime', 2000)) / 1000.0
        img = nib.Nifti1Image(fmri_4d, affine)
        img.header.set_xyzt_units(xyz="mm", t="sec")
        img.header.set_zooms((abs(affine[0,0]), abs(affine[1,1]), abs(affine[2,2]), tr_sec))
        nib.save(img, output_nii)
        print(f"🎉 Saved: {output_nii}")

    return fmri_4d, {"first_dcm": valid_dcms[0], "n_volumes": len(valid_files)}


def process_nako_siemens_mosaic_to_bids(
    dicom_folder: str,
    t1_nii_path: str,
    bids_root: str,
    subject_id: str,
    task_name: str = "rest",
    total_readout_time_default: float = 0.035,
    repetition_time_default_ms: float = 2000,
    show_progress: bool = False
):
    """
    Converts NAKO resting-state fMRI from Siemens mosaic DICOMs to BIDS.
    """
    if not os.path.isdir(dicom_folder):
        raise ValueError(f"DICOM folder does not exist: {dicom_folder}")
    if not os.path.isfile(t1_nii_path):
        raise ValueError(f"T1 NIfTI file not found: {t1_nii_path}")

    print("\n=== Step 1: Reconstructing fMRI from Siemens mosaic DICOMs ===")
    fmri_4d, meta = reconstruct_nako_mosaic_fMRI(
        dicom_folder=dicom_folder,
        show_progress=show_progress
    )
    first_dcm = meta["first_dcm"]

    tr_ms = getattr(first_dcm, 'RepetitionTime', repetition_time_default_ms)
    tr_sec = float(tr_ms) / 1000.0

    total_readout_time = getattr(first_dcm, 'TotalReadoutTime', total_readout_time_default)
    if total_readout_time is None:
        total_readout_time = total_readout_time_default

    # ✅ Use DICOM-derived affine
    affine = build_affine_from_dicom(first_dcm)
    dx, dy, dz = abs(affine[0,0]), abs(affine[1,1]), abs(affine[2,2])

    temp_nii = os.path.join(dicom_folder, "temp_bold.nii.gz")
    img = nib.Nifti1Image(fmri_4d, affine)
    img.header.set_xyzt_units(xyz="mm", t="sec")
    img.header.set_zooms((dx, dy, dz, tr_sec))
    nib.save(img, temp_nii)

    print("\n=== Step 2: Writing BIDS structure ===")
    sub_prefix = f"sub-{subject_id}"
    anat_dir = os.path.join(bids_root, sub_prefix, "anat")
    func_dir = os.path.join(bids_root, sub_prefix, "func")
    os.makedirs(anat_dir, exist_ok=True)
    os.makedirs(func_dir, exist_ok=True)

    # Copy T1
    t1_stem = t1_nii_path.replace(".nii.gz", "")
    t1_json_path = t1_stem + ".json"
    shutil.copy(t1_nii_path, os.path.join(anat_dir, f"{sub_prefix}_T1w.nii.gz"))
    if os.path.exists(t1_json_path):
        shutil.copy(t1_json_path, os.path.join(anat_dir, f"{sub_prefix}_T1w.json"))
    else:
        print(f"⚠️ Warning: T1 JSON sidecar not found: {t1_json_path}")

    # Copy fMRI
    bold_nii_name = f"{sub_prefix}_task-{task_name}_bold.nii.gz"
    shutil.copy(temp_nii, os.path.join(func_dir, bold_nii_name))

    # dataset_description.json
    dataset_desc_path = os.path.join(bids_root, "dataset_description.json")
    if not os.path.exists(dataset_desc_path):
        with open(dataset_desc_path, 'w') as f:
            json.dump({
                "Name": "NAKO Resting-State fMRI",
                "BIDSVersion": "1.8.0",
                "DatasetType": "raw",
                "Authors": [
                    "Chair of Informatics for Medical Technologies (CIMT), University of Augsburg",
                    "NAKO Study Group"
                ],
                "Acknowledgements": "Data provided by the NAKO Health Study (https://www.nako.de). Conversion pipeline developed by CIMT.",
                "Funding": ["German Research Foundation (DFG)", "Bavarian State Ministry for Science and the Arts"],
                "License": "CC0",
                "GeneratedBy": [{
                    "Name": "nako_bids",
                    "Version": "1.0",
                    "CodeURL": "https://github.com/cimt-unia/nako_bids"
                }],
                "SourceDatasets": [{
                    "URL": "https://www.nako.de",
                    "Description": "NAKO Gesundheitsstudie (German National Cohort)"
                }]
            }, f, indent=2, ensure_ascii=False)

    # fMRI JSON sidecar
    try:
        inplane_phase = getattr(first_dcm, 'InPlanePhaseEncodingDirection', 'ROW')
        phase_encoding = "j-" if inplane_phase == "ROW" else "i-"
    except Exception as e:
        print(f"⚠️ Could not read InPlanePhaseEncodingDirection: {e}. Using default 'j-'.")
        phase_encoding = "j-"

    def safe_get(attr, default):
        try:
            return float(getattr(first_dcm, attr, default))
        except (ValueError, TypeError):
            return default

    json_data = {
        "TaskName": task_name,
        "RepetitionTime": tr_sec,
        "PhaseEncodingDirection": phase_encoding,
        "TotalReadoutTime": round(total_readout_time, 6),
        "Instructions": "Keep your eyes open, fixate on the cross, and let your mind wander.",
        "TaskDescription": "Resting-state fMRI with eyes open.",
        "CogAtlasID": "https://www.cognitiveatlas.org/task/resting_state",
        "Manufacturer": getattr(first_dcm, 'Manufacturer', 'Siemens'),
        "ManufacturersModelName": getattr(first_dcm, 'ManufacturerModelName', 'Skyra/Prisma'),
        "MagneticFieldStrength": safe_get('MagneticFieldStrength', 3.0),
        "EchoTime": safe_get('EchoTime', 30.0) / 1000.0,
        "FlipAngle": safe_get('FlipAngle', 90.0),
        "PulseSequenceType": "EPI",
        "InstitutionName": getattr(first_dcm, 'InstitutionName', 'NAKO Study Site'),
        "InstitutionalDepartmentName": getattr(first_dcm, 'InstitutionalDepartmentName', 'Radiology'),
        "ScanningSequence": getattr(first_dcm, 'ScanningSequence', 'EP'),
        "SequenceVariant": getattr(first_dcm, 'SequenceVariant', 'SK'),
        "SequenceName": getattr(first_dcm, 'SequenceName', 'ep2d_bold'),
        "MRAcquisitionType": getattr(first_dcm, 'MRAcquisitionType', '2D')
    }
    with open(os.path.join(func_dir, f"{sub_prefix}_task-{task_name}_bold.json"), 'w') as f:
        json.dump(json_data, f, indent=2)

    # README
    readme_path = os.path.join(bids_root, "README")
    if not os.path.exists(readme_path):
        with open(readme_path, 'w') as f:
            f.write(
                "NAKO resting-state fMRI dataset\n"
                "Converted from Siemens 3T EPI mosaic DICOMs (64x64, 6x6 mosaic).\n"
                "Each DICOM = one volume; slices extracted via tiling.\n"
                f"TotalReadoutTime: {total_readout_time:.6f} s (computed from DICOM if available).\n"
                "Note: SliceTiming not included; slice order assumed anatomical (inferior→superior).\n"
            )

    # Cleanup
    if os.path.exists(temp_nii):
        os.remove(temp_nii)

    print(f"\n✅ Successfully processed NAKO subject {subject_id} into BIDS at: {bids_root}")
    print(f"ℹ️ PhaseEncodingDirection: {phase_encoding}")
    print(f"ℹ️ TotalReadoutTime: {total_readout_time:.6f} s")


def process_single_nako_subject_from_zip(
    subject_base_id: str,
    nako_root: str,
    bids_root: str = None,
    temp_dir: str = None
):
    """
    Process a single NAKO subject from ZIP archive.
    """
    if bids_root is None:
        bids_root = os.path.join(nako_root, "BIDS")
    if temp_dir is None:
        temp_dir = os.path.join(nako_root, "temp_unzip")

    bids_sub_id = subject_base_id.replace("_", "")
    anat_dir = os.path.join(bids_root, f"sub-{bids_sub_id}", "anat")
    func_dir = os.path.join(bids_root, f"sub-{bids_sub_id}", "func")

    if (os.path.exists(os.path.join(anat_dir, f"sub-{bids_sub_id}_T1w.nii.gz")) and
        os.path.exists(os.path.join(func_dir, f"sub-{bids_sub_id}_task-rest_bold.nii.gz"))):
        print(f"⏩ Skipping {subject_base_id}: already processed.")
        return

    zip_path = os.path.join(nako_root, "Resting_State_TRA", f"{subject_base_id}_Resting_State_TRA.zip")
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    t1_pattern = os.path.join(nako_root, "T1_3D_SAG_DEFACED_NIFTI", f"{subject_base_id}_T1_3D_SAG_sn*.nii.gz")
    t1_matches = glob.glob(t1_pattern)
    if not t1_matches:
        t1_pattern_nd = os.path.join(nako_root, "T1_3D_SAG_ND_DEFACED_NIFTI", f"{subject_base_id}_T1_3D_SAG_ND_sn*.nii.gz")
        t1_matches = glob.glob(t1_pattern_nd)
        if t1_matches:
            print(f"⚠️ Warning: Using _ND_ T1 for {subject_base_id} (no distortion-corrected version found).")
        else:
            raise FileNotFoundError(f"No T1 image found for {subject_base_id}.")

    t1_nii_path = t1_matches[0]

    extract_to = Path(temp_dir) / f"extract_{subject_base_id}"
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)

    dicom_folder = extract_to / subject_base_id / "Resting_State_TRA"
    if not dicom_folder.exists():
        shutil.rmtree(extract_to, ignore_errors=True)
        raise FileNotFoundError(f"DICOM folder missing in ZIP: {dicom_folder}")

    try:
        process_nako_siemens_mosaic_to_bids(
            dicom_folder=str(dicom_folder),
            t1_nii_path=t1_nii_path,
            bids_root=bids_root,
            subject_id=bids_sub_id,
            task_name="rest",
            show_progress=False
        )
    finally:
        shutil.rmtree(extract_to, ignore_errors=True)


def batch_process_nako_from_zips(nako_root: str):
    """Process all subjects in NAKO dataset."""
    zip_folder = os.path.join(nako_root, "Resting_State_TRA")
    zip_files = sorted(glob.glob(os.path.join(zip_folder, "*_Resting_State_TRA.zip")))
    
    if not zip_files:
        raise FileNotFoundError(f"No ZIP files found in {zip_folder}")

    print(f"🔍 Found {len(zip_files)} subjects. Processing...")
    for zip_path in tqdm(zip_files, desc="Processing subjects", unit="subject"):
        stem = Path(zip_path).stem
        if stem.endswith("_Resting_State_TRA"):
            base_id = "_".join(stem.split("_")[:2])
            process_single_nako_subject_from_zip(base_id, nako_root)


# =============================================================================
# EXAMPLE USAGE (UNCOMMENT TO RUN)
# =============================================================================

# if __name__ == "__main__":
#     NAKO_ROOT = r"C:\full\path\to\NAKO-1048_MRT"
#     process_single_nako_subject_from_zip("130911_30", NAKO_ROOT)