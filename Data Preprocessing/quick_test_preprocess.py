#!/usr/bin/env python3
"""
NACC Quick Test Preprocessing Pipeline
Process first N patients for testing and QA
"""

import os
import gc
import shutil
import subprocess
import zipfile
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
import sys
import nibabel as nib
from scipy import ndimage

class QuickTestPreprocessor:
    """Quick test preprocessing for first N patients"""

    # Maximum volume size in voxels to prevent memory errors
    MAX_VOXELS = 512 * 512 * 512  # ~134 million voxels

    def __init__(self, num_patients=200,
                 input_base="D:\\NACC_Matched_Pairs",
                 output_base="D:\\NACC_Test_Preprocessed_NEW",
                 log_file="quick_test_log.txt"):

        self.num_patients = num_patients
        self.input_base = input_base
        self.output_base = output_base
        self.log_file = log_file
        self.setup_logging()

    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # File handler — force UTF-8
        handler = logging.FileHandler(self.log_file, encoding='utf-8')
        handler.setLevel(logging.INFO)

        # Console handler — UTF-8 with error replacement for Windows
        console_stream = open(sys.stdout.fileno(), mode='w',
                              encoding='utf-8', errors='replace',
                              closefd=False)
        console = logging.StreamHandler(stream=console_stream)
        console.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        console.setFormatter(formatter)

        self.logger.addHandler(handler)
        self.logger.addHandler(console)

    def log(self, message):
        """Log message"""
        self.logger.info(message)

    def _is_resource_fork(self, filename):
        """Check if a file is a macOS resource fork"""
        basename = os.path.basename(filename)
        return basename.startswith('._') or basename.startswith('.')

    def _find_nifti(self, directory):
        """
        Find the best valid NIfTI file in directory tree.
        
        Strategy:
          1. Collect all .nii.gz and .nii candidates (skip resource forks)
          2. Prefer .nii.gz over .nii (dcm2niix default output is .nii.gz)
          3. Validate each candidate with nibabel before returning
          4. Return the first valid one, or None if nothing works
        """
        if not os.path.exists(directory):
            return None

        nii_gz_candidates = []
        nii_candidates = []

        for root, dirs, files in os.walk(directory):
            # Skip resource fork directories
            dirs[:] = [d for d in dirs if not self._is_resource_fork(d)]

            for file in files:
                if self._is_resource_fork(file):
                    continue

                full_path = os.path.join(root, file)
                if file.endswith('.nii.gz'):
                    nii_gz_candidates.append(full_path)
                elif file.endswith('.nii'):
                    nii_candidates.append(full_path)

        # Try .nii.gz first (dcm2niix output), then fall back to .nii
        for candidate in nii_gz_candidates + nii_candidates:
            try:
                # Quick validation: try to read the header only
                img = nib.load(candidate)
                _ = img.header  # force header parse
                return candidate
            except Exception:
                # Not a valid NIfTI file, skip it
                continue

        return None

    def run_pipeline(self):
        """Run quick test pipeline"""
        self.log("=" * 90)
        self.log("QUICK TEST PREPROCESSING PIPELINE")
        self.log("=" * 90)
        self.log(f"Processing first {self.num_patients} patients for testing")
        self.log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("")

        # Get list of patients
        all_patients = sorted([d for d in os.listdir(self.input_base)
                              if os.path.isdir(os.path.join(self.input_base, d))])
        test_patients = all_patients[:self.num_patients]

        self.log(f"Total patients available: {len(all_patients)}")
        self.log(f"Processing: {len(test_patients)} patients")
        self.log("")

        # Create output directory
        os.makedirs(self.output_base, exist_ok=True)

        # ============== STEP 1: EXTRACT ZIPS ==============
        self.log("=" * 90)
        self.log("STEP 1/3: EXTRACTING ZIP FILES")
        self.log("=" * 90)
        self.log("")

        extracted_count = 0
        skipped_count = 0
        for idx, patient_id in enumerate(test_patients, 1):
            patient_path = os.path.join(self.input_base, patient_id)

            for modality in ["MRI", "PET"]:
                mod_path = os.path.join(patient_path, modality)
                if not os.path.exists(mod_path):
                    continue

                zip_files = [f for f in os.listdir(mod_path) if f.endswith('.zip')]
                for zip_file in zip_files:
                    if self._is_resource_fork(zip_file):
                        skipped_count += 1
                        continue

                    zip_path = os.path.join(mod_path, zip_file)
                    extract_dir = os.path.join(mod_path, zip_file.replace('.zip', ''))

                    if not os.path.exists(extract_dir):
                        try:
                            os.makedirs(extract_dir, exist_ok=True)
                            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                zip_ref.extractall(extract_dir)
                            extracted_count += 1
                        except Exception as e:
                            self.log(f"ERROR extracting {patient_id}/{zip_file}: {e}")

            if idx % 10 == 0:
                self.log(f"[{idx}/{len(test_patients)}] Extraction progress: {extracted_count} files extracted")

        self.log(f"Step 1 complete: {extracted_count} ZIP files extracted, {skipped_count} macOS resource forks skipped")
        self.log("")

        # ============== STEP 2: DICOM TO NIFTI ==============
        self.log("=" * 90)
        self.log("STEP 2/3: CONVERTING DICOM TO NIFTI")
        self.log("=" * 90)
        self.log("")

        dcm2niix_cmd = "dcm2niix"

        try:
            r = subprocess.run([dcm2niix_cmd, '--version'],
                            capture_output=True, text=True)
            has_dcm2niix = (r.returncode in (0, 1, 2, 3))
            if has_dcm2niix:
                self.log(f"Using dcm2niix from: {dcm2niix_cmd}")
                self.log((r.stdout or r.stderr).strip())
            else:
                self.log("dcm2niix returned unexpected error.")
        except Exception as e:
            has_dcm2niix = False
            self.log(f"dcm2niix check failed: {e}")
            self.log("Will attempt to find existing NIfTI files instead")
            dcm2niix_cmd = None

        nifti_count = 0
        for idx, patient_id in enumerate(test_patients, 1):
            patient_path = os.path.join(self.input_base, patient_id)

            for modality in ["MRI", "PET"]:
                mod_path = os.path.join(patient_path, modality)
                if not os.path.exists(mod_path):
                    continue

                extracted_dirs = [d for d in os.listdir(mod_path)
                                  if os.path.isdir(os.path.join(mod_path, d))
                                  and not self._is_resource_fork(d)]

                for extracted_dir in extracted_dirs:
                    extracted_path = os.path.join(mod_path, extracted_dir)

                    # Check if valid NIfTI already exists (skip resource forks)
                    nifti_files = [f for f in os.listdir(extracted_path)
                                  if (f.endswith('.nii.gz') or f.endswith('.nii'))
                                  and not self._is_resource_fork(f)]

                    if nifti_files:
                        nifti_count += 1
                        continue

                    if has_dcm2niix:
                        try:
                            subprocess.run([
                                dcm2niix_cmd,
                                '-z', 'y',   # compress output to .nii.gz
                                '-f', extracted_dir,
                                '-o', extracted_path,
                                extracted_path
                            ], capture_output=True, timeout=120)
                            nifti_count += 1
                        except Exception as e:
                            self.log(f"ERROR converting {patient_id}/{modality}/{extracted_dir}: {e}")

            if idx % 10 == 0:
                self.log(f"[{idx}/{len(test_patients)}] Conversion progress: {nifti_count} NIfTI files")

        self.log(f"Step 2 complete: {nifti_count} DICOM to NIfTI conversions")
        self.log("")

        # ============== STEP 3: PREPROCESSING ==============
        self.log("=" * 90)
        self.log("STEP 3/3: PREPROCESSING IMAGES")
        self.log("=" * 90)
        self.log("")

        processed_count = 0
        skipped_invalid = 0
        skipped_no_file = 0

        for idx, patient_id in enumerate(test_patients, 1):
            patient_input = os.path.join(self.input_base, patient_id)
            patient_output = os.path.join(self.output_base, patient_id)
            os.makedirs(patient_output, exist_ok=True)

            # Find and process MRI
            mri_nifti = self._find_nifti(os.path.join(patient_input, "MRI"))
            if mri_nifti:
                try:
                    mri_data = self._load_and_preprocess_mri(mri_nifti)
                    if mri_data is not None:
                        mri_output = os.path.join(patient_output, "mri_processed.npy")
                        np.save(mri_output, mri_data)
                        processed_count += 1
                        del mri_data  # free memory immediately
                        gc.collect()
                    else:
                        skipped_invalid += 1
                except Exception as e:
                    self.log(f"ERROR processing MRI for {patient_id}: {e}")
            else:
                skipped_no_file += 1

            # Find and process PET
            pet_nifti = self._find_nifti(os.path.join(patient_input, "PET"))
            if pet_nifti:
                try:
                    pet_data = self._load_and_preprocess_pet(pet_nifti)
                    if pet_data is not None:
                        pet_output = os.path.join(patient_output, "pet_processed.npy")
                        np.save(pet_output, pet_data)
                        processed_count += 1
                        del pet_data  # free memory immediately
                        gc.collect()
                    else:
                        skipped_invalid += 1
                except Exception as e:
                    self.log(f"ERROR processing PET for {patient_id}: {e}")
            else:
                skipped_no_file += 1

            if idx % 10 == 0:
                self.log(f"[{idx}/{len(test_patients)}] Processing progress: "
                         f"{processed_count} done, {skipped_no_file} no file, "
                         f"{skipped_invalid} invalid")

        self.log(f"Step 3 complete: {processed_count} images preprocessed, "
                 f"{skipped_no_file} missing, {skipped_invalid} invalid/skipped")
        self.log("")

        # ============== SUMMARY ==============
        self.log("=" * 90)
        self.log("QUICK TEST PREPROCESSING COMPLETE")
        self.log("=" * 90)
        self.log("")
        self.log("Summary:")
        self.log(f"   - Patients processed: {len(test_patients)}")
        self.log(f"   - ZIP files extracted: {extracted_count}")
        self.log(f"   - macOS resource forks skipped: {skipped_count}")
        self.log(f"   - DICOM to NIfTI conversions: {nifti_count}")
        self.log(f"   - Images preprocessed: {processed_count}")
        self.log(f"   - No NIfTI found: {skipped_no_file}")
        self.log(f"   - Invalid/skipped: {skipped_invalid}")
        self.log("")
        self.log(f"Output location: {self.output_base}")
        self.log(f"Log file: {self.log_file}")
        self.log("")
        self.log("Next steps:")
        self.log("   1. Review preprocessed images for quality")
        self.log("   2. Check output directory for results")
        self.log("   3. If satisfied, run full preprocessing on all patients")
        self.log("")
        self.log(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 90)

    def _load_and_preprocess_mri(self, nifti_path):
        """Load and preprocess MRI image"""
        try:
            img = nib.load(nifti_path)

            # Check volume size before loading data
            total_voxels = np.prod(img.shape[:3])
            if total_voxels > self.MAX_VOXELS:
                self.log(f"  Skipping oversized volume ({img.shape}): {nifti_path}")
                return None

            # Load as float32 directly to save memory
            data = np.asarray(img.dataobj, dtype=np.float32)

            # If 4D, take first volume
            if data.ndim == 4:
                data = data[:, :, :, 0]

            data[data < 0] = 0

            nonzero = data[data > 0]
            if len(nonzero) < 100:
                return None

            v_min = np.percentile(nonzero, 0.5)
            v_max = np.percentile(nonzero, 99.5)

            # Clip and normalize to [0, 1]
            np.clip(data, v_min, v_max, out=data)
            data = (data - v_min) / (v_max - v_min + 1e-8)

            # Convert to [-1, 1] for diffusion model
            data = 2.0 * data - 1.0

            return data
        except Exception as e:
            self.log(f"Error preprocessing MRI: {e}")
            return None

    def _load_and_preprocess_pet(self, nifti_path):
        """Load and preprocess PET image (robust normalization)"""
        try:
            img = nib.load(nifti_path)

            # Check volume size before loading data
            total_voxels = np.prod(img.shape[:3])
            if total_voxels > self.MAX_VOXELS:
                self.log(f"  Skipping oversized volume ({img.shape}): {nifti_path}")
                return None

            # Load as float32 directly to save memory
            data = np.asarray(img.dataobj, dtype=np.float32)

            # If 4D, take first volume
            if data.ndim == 4:
                data = data[:, :, :, 0]

            data[data < 0] = 0

            nonzero = data[data > 0]
            if len(nonzero) < 100:
                return None

            v_min = np.percentile(nonzero, 1.0)
            v_max = np.percentile(nonzero, 99.0)

            # Clip and normalize to [0, 1]
            np.clip(data, v_min, v_max, out=data)
            data = (data - v_min) / (v_max - v_min + 1e-8)

            # Convert to [-1, 1] for diffusion model
            data = 2.0 * data - 1.0

            return data
        except Exception as e:
            self.log(f"Error preprocessing PET: {e}")
            return None

if __name__ == "__main__":
    preprocessor = QuickTestPreprocessor(num_patients=200)
    preprocessor.run_pipeline()