#!/usr/bin/env python3
"""
NACC Quick Test Preprocessing Pipeline
Process first 50 patients for testing and QA
"""

import os
import shutil
import subprocess
import zipfile
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
import nibabel as nib
from scipy import ndimage

class QuickTestPreprocessor:
    """Quick test preprocessing for first N patients"""

    def __init__(self, num_patients=50,
                 input_base="/Volumes/BACKUP/NACC_Matched_Pairs",
                 output_base="/Volumes/BACKUPh/NACC_Test_Preprocessed",
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

        # File handler
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(logging.INFO)

        # Console handler
        console = logging.StreamHandler()
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

    def run_pipeline(self):
        """Run quick test pipeline"""
        self.log("=" * 90)
        self.log("🚀 QUICK TEST PREPROCESSING PIPELINE")
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
        for idx, patient_id in enumerate(test_patients, 1):
            patient_path = os.path.join(self.input_base, patient_id)

            # Extract MRI files
            mri_path = os.path.join(patient_path, "MRI")
            if os.path.exists(mri_path):
                mri_zip_files = [f for f in os.listdir(mri_path) if f.endswith('.zip')]
                for zip_file in mri_zip_files:
                    zip_path = os.path.join(mri_path, zip_file)
                    extract_dir = os.path.join(mri_path, zip_file.replace('.zip', ''))

                    if not os.path.exists(extract_dir):
                        try:
                            os.makedirs(extract_dir, exist_ok=True)
                            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                zip_ref.extractall(extract_dir)
                            extracted_count += 1
                        except Exception as e:
                            self.log(f"ERROR extracting {patient_id}/{zip_file}: {e}")

            # Extract PET files
            pet_path = os.path.join(patient_path, "PET")
            if os.path.exists(pet_path):
                pet_zip_files = [f for f in os.listdir(pet_path) if f.endswith('.zip')]
                for zip_file in pet_zip_files:
                    zip_path = os.path.join(pet_path, zip_file)
                    extract_dir = os.path.join(pet_path, zip_file.replace('.zip', ''))

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

        self.log(f"✓ Step 1 complete: {extracted_count} ZIP files extracted")
        self.log("")

        # ============== STEP 2: DICOM TO NIFTI ==============
        self.log("=" * 90)
        self.log("STEP 2/3: CONVERTING DICOM TO NIFTI")
        self.log("=" * 90)
        self.log("")

        # Check if dcm2niix is available
        # Check if dcm2niix is available (ignore exit code quirks)
        dcm2niix_cmd = "/usr/local/bin/dcm2niix"  # explicit path

        try:
            # We do NOT use check=True because some builds return non-zero for --version
            r = subprocess.run([dcm2niix_cmd, '--version'],
                            capture_output=True, text=True)
            # Consider it available if the binary ran at all
            has_dcm2niix = (r.returncode in (0, 1, 2, 3))  # accept non-zero as 'ok'
            if has_dcm2niix:
                self.log(f"Using dcm2niix from: {dcm2niix_cmd}")
                self.log((r.stdout or r.stderr).strip())
            else:
                self.log("⚠️  dcm2niix returned unexpected error, but will still try to use it.")
        except Exception as e:
            has_dcm2niix = False
            self.log(f"⚠️  dcm2niix check failed: {e}")
            self.log("⚠️  Will attempt to find existing NIfTI files instead")
            dcm2niix_cmd = None

        nifti_count = 0
        for idx, patient_id in enumerate(test_patients, 1):
            patient_path = os.path.join(self.input_base, patient_id)

            # Process MRI
            mri_path = os.path.join(patient_path, "MRI")
            if os.path.exists(mri_path):
                extracted_dirs = [d for d in os.listdir(mri_path) 
                                if os.path.isdir(os.path.join(mri_path, d))]

                for extracted_dir in extracted_dirs:
                    extracted_path = os.path.join(mri_path, extracted_dir)

                    # Check if NIfTI already exists
                    nifti_files = [f for f in os.listdir(extracted_path) 
                                  if f.endswith('.nii.gz') or f.endswith('.nii')]

                    if nifti_files:
                        nifti_count += 1
                        continue

                    if has_dcm2niix:
                        try:
                            subprocess.run([
                                'dcm2niix',
                                '-f', extracted_dir,
                                '-o', extracted_path,
                                extracted_path
                            ], capture_output=True, timeout=60)
                            nifti_count += 1
                        except Exception as e:
                            self.log(f"ERROR converting {patient_id}/MRI/{extracted_dir}: {e}")

            # Process PET
            pet_path = os.path.join(patient_path, "PET")
            if os.path.exists(pet_path):
                extracted_dirs = [d for d in os.listdir(pet_path) 
                                if os.path.isdir(os.path.join(pet_path, d))]

                for extracted_dir in extracted_dirs:
                    extracted_path = os.path.join(pet_path, extracted_dir)

                    # Check if NIfTI already exists
                    nifti_files = [f for f in os.listdir(extracted_path) 
                                  if f.endswith('.nii.gz') or f.endswith('.nii')]

                    if nifti_files:
                        nifti_count += 1
                        continue

                    if has_dcm2niix:
                        try:
                            subprocess.run([
                                'dcm2niix',
                                '-f', extracted_dir,
                                '-o', extracted_path,
                                extracted_path
                            ], capture_output=True, timeout=60)
                            nifti_count += 1
                        except Exception as e:
                            self.log(f"ERROR converting {patient_id}/PET/{extracted_dir}: {e}")

            if idx % 10 == 0:
                self.log(f"[{idx}/{len(test_patients)}] Conversion progress: {nifti_count} NIfTI files")

        self.log(f"✓ Step 2 complete: {nifti_count} DICOM→NIfTI conversions")
        self.log("")

        # ============== STEP 3: PREPROCESSING ==============
        self.log("=" * 90)
        self.log("STEP 3/3: PREPROCESSING IMAGES")
        self.log("=" * 90)
        self.log("")

        processed_count = 0
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
                        np.save(mri_output, mri_data.astype(np.float32))
                        processed_count += 1
                except Exception as e:
                    self.log(f"ERROR processing MRI for {patient_id}: {e}")

            # Find and process PET
            pet_nifti = self._find_nifti(os.path.join(patient_input, "PET"))
            if pet_nifti:
                try:
                    pet_data = self._load_and_preprocess_pet(pet_nifti)
                    if pet_data is not None:
                        pet_output = os.path.join(patient_output, "pet_processed.npy")
                        np.save(pet_output, pet_data.astype(np.float32))
                        processed_count += 1
                except Exception as e:
                    self.log(f"ERROR processing PET for {patient_id}: {e}")

            if idx % 10 == 0:
                self.log(f"[{idx}/{len(test_patients)}] Processing progress: {processed_count} modalities completed")

        self.log(f"✓ Step 3 complete: {processed_count} images preprocessed")
        self.log("")

        # ============== SUMMARY ==============
        self.log("=" * 90)
        self.log("✅ QUICK TEST PREPROCESSING COMPLETE")
        self.log("=" * 90)
        self.log("")
        self.log("📊 Summary:")
        self.log(f"   • Patients processed: {len(test_patients)}")
        self.log(f"   • ZIP files extracted: {extracted_count}")
        self.log(f"   • DICOM→NIfTI conversions: {nifti_count}")
        self.log(f"   • Images preprocessed: {processed_count}")
        self.log("")
        self.log(f"📁 Output location: {self.output_base}")
        self.log(f"📄 Log file: {self.log_file}")
        self.log("")
        self.log("Next steps:")
        self.log("   1. Review preprocessed images for quality")
        self.log("   2. Check /Volumes/BACKUP/NACC_Test_Preprocessed/ for output")
        self.log("   3. If satisfied, run full preprocessing on all 1,941 patients")
        self.log("")
        self.log(f"⏱️  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 90)

    def _find_nifti(self, directory):
        """Find first NIfTI file in directory tree"""
        if not os.path.exists(directory):
            return None

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.nii.gz') or file.endswith('.nii'):
                    return os.path.join(root, file)
        return None

    def _load_and_preprocess_mri(self, nifti_path):
        """Load and preprocess MRI image"""
        try:
            img = nib.load(nifti_path)
            data = img.get_fdata()

            # Remove background (zero values)
            data[data < 0] = 0

            # Normalize intensity (percentile-based)
            nonzero = data[data > 0]
            if len(nonzero) < 100:  # Skip if too few voxels
                return None

            v_min = np.percentile(nonzero, 0.5)
            v_max = np.percentile(nonzero, 99.5)

            # Clip and normalize to [0, 1]
            normalized = np.clip(data, v_min, v_max)
            normalized = (normalized - v_min) / (v_max - v_min + 1e-8)

            # Convert to [-1, 1] for diffusion model
            normalized = 2 * normalized - 1

            return normalized
        except Exception as e:
            print(f"Error preprocessing MRI: {e}")
            return None

    def _load_and_preprocess_pet(self, nifti_path):
        """Load and preprocess PET image (robust normalization)"""
        try:
            img = nib.load(nifti_path)
            data = img.get_fdata()

            # Remove background
            data[data < 0] = 0

            # Robust normalization for PET (accounts for dose variations)
            nonzero = data[data > 0]
            if len(nonzero) < 100:  # Skip if too few voxels
                return None

            # Use more robust percentiles for PET
            v_min = np.percentile(nonzero, 1.0)
            v_max = np.percentile(nonzero, 99.0)

            # Clip and normalize to [0, 1]
            normalized = np.clip(data, v_min, v_max)
            normalized = (normalized - v_min) / (v_max - v_min + 1e-8)

            # Convert to [-1, 1] for diffusion model
            normalized = 2 * normalized - 1

            return normalized
        except Exception as e:
            print(f"Error preprocessing PET: {e}")
            return None

if __name__ == "__main__":
    # Run with first 50 patients
    preprocessor = QuickTestPreprocessor(num_patients=50)
    preprocessor.run_pipeline()
