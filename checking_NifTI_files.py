#!/usr/bin/env python3
import zipfile
from pathlib import Path

# Read-only dataset location
ROOT_DIR = Path(r"/Volumes/New Volume/NACC_Matched_Pairs")

# Writable cache for extracted zips
CACHE_DIR = Path.home() / "NACC_unzipped_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_LIST = Path.home() / "valid_nifti_subjects.txt"

def has_nifti_in_dir(dir_path: Path):
    for f in dir_path.rglob("*"):
        if f.is_file() and not f.name.startswith("._"):
            if f.suffix == ".nii" or f.name.endswith(".nii.gz"):
                return True
    return False

def zip_contains_nifti(zip_path: Path):
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                name_lower = name.lower()
                if name_lower.endswith(".nii") or name_lower.endswith(".nii.gz"):
                    return True
    except zipfile.BadZipFile:
        return False
    return False

def extract_zip_to_cache(zip_path: Path, cache_root: Path):
    target_dir = cache_root / zip_path.stem
    if target_dir.exists() and any(target_dir.iterdir()):
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
    except zipfile.BadZipFile:
        print(f"  ⚠ Bad zip: {zip_path.name}")
        return None
    return target_dir

def scan_subject(subj_dir: Path):
    mri_dir = subj_dir / "MRI"
    pet_dir = subj_dir / "PET"
    if not mri_dir.exists() or not pet_dir.exists():
        return False, False

    # ✅ First check original folders (fast, no extraction)
    mri_ok = has_nifti_in_dir(mri_dir)
    pet_ok = has_nifti_in_dir(pet_dir)

    # If already found, no need to unzip
    if mri_ok and pet_ok:
        return True, True

    # Otherwise, check zips (only extract if zip contains NIfTI)
    if not mri_ok:
        for zip_path in mri_dir.glob("*.zip"):
            if zip_path.name.startswith("._"):
                continue
            if not zip_contains_nifti(zip_path):
                continue
            cache_subdir = CACHE_DIR / subj_dir.name / "MRI"
            cache_subdir.mkdir(parents=True, exist_ok=True)
            extracted = extract_zip_to_cache(zip_path, cache_subdir)
            if extracted and has_nifti_in_dir(extracted):
                mri_ok = True
                break

    if not pet_ok:
        for zip_path in pet_dir.glob("*.zip"):
            if zip_path.name.startswith("._"):
                continue
            if not zip_contains_nifti(zip_path):
                continue
            cache_subdir = CACHE_DIR / subj_dir.name / "PET"
            cache_subdir.mkdir(parents=True, exist_ok=True)
            extracted = extract_zip_to_cache(zip_path, cache_subdir)
            if extracted and has_nifti_in_dir(extracted):
                pet_ok = True
                break

    return mri_ok, pet_ok

valid_subjects = []

for subj_dir in sorted(ROOT_DIR.iterdir()):
    if not subj_dir.is_dir() or not subj_dir.name.startswith("NACC"):
        continue

    mri_ok, pet_ok = scan_subject(subj_dir)

    if mri_ok and pet_ok:
        print(f"✅ {subj_dir.name} has NIfTI for both MRI and PET")
        valid_subjects.append(subj_dir.name)
    else:
        print(f"❌ {subj_dir.name} missing NIfTI (MRI={mri_ok}, PET={pet_ok})")

with open(OUTPUT_LIST, "w") as f:
    for s in valid_subjects:
        f.write(s + "\n")

print(f"\nTotal valid NIfTI subjects: {len(valid_subjects)}")
print(f"Saved list to: {OUTPUT_LIST}")
print(f"Cache location: {CACHE_DIR}")