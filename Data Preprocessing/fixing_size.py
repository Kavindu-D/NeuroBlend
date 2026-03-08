#!/usr/bin/env python3
"""
NACC Quick Test Preprocessing Pipeline
With proper scan selection, NIfTI-to-SimpleITK conversion, and registration.
"""

import os
import gc
import subprocess
import zipfile
import numpy as np
from datetime import datetime
import logging
import sys
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom as scipy_zoom

TARGET_SHAPE = (160, 192, 160)

# Priority order for PET tracer selection
PET_TRACER_PRIORITY = ["AMYLOID", "FLORBETABEN", "PIB", "FDG", "TAU", "FLORTAUCIPIR",
                       "AV1451", "FLORBETAPIR"]


class QuickTestPreprocessor:
    """Quick test preprocessing with registration"""

    MAX_VOXELS = 512 * 512 * 512

    def __init__(self, num_patients=200,
                 input_base=r"D:\NACC_Matched_Pairs",
                 output_base=r"D:\NACC_Matched_Pairs_500_Preprocessed",
                 log_file="quick_test_registered_log_500.txt"):

        self.num_patients = num_patients
        self.input_base = input_base
        self.output_base = output_base
        self.log_file = log_file
        self.setup_logging()

    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if self.logger.handlers:
            self.logger.handlers.clear()

        handler = logging.FileHandler(self.log_file, encoding='utf-8')
        handler.setLevel(logging.INFO)

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
        self.logger.info(message)

    def _is_resource_fork(self, filename):
        basename = os.path.basename(filename)
        return basename.startswith('._') or basename.startswith('.')

    # ─────────────────────────────────────────────────────────────
    # SMART FILE FINDING (from your full pipeline)
    # ─────────────────────────────────────────────────────────────

    def _find_all_nifti(self, directory):
        """Find ALL valid NIfTI files in directory tree, sorted by size (largest first)."""
        if not os.path.exists(directory):
            return []

        candidates = []
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not self._is_resource_fork(d)]
            for file in files:
                if self._is_resource_fork(file):
                    continue
                if file.endswith('.nii.gz') or file.endswith('.nii'):
                    full_path = os.path.join(root, file)
                    try:
                        img = nib.load(full_path)
                        _ = img.header
                        size = os.path.getsize(full_path)
                        candidates.append((full_path, size))
                    except Exception:
                        continue

        # Sort by file size descending (larger = more likely to be full 3D volume)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates]

    def _choose_mri(self, mri_dir):
        """
        Choose the best MRI scan.
        Priority: MPRAGE > T1/SPGR/IR-FSPGR (non-FLAIR) > anything non-FLAIR > fallback
        """
        all_nii = self._find_all_nifti(mri_dir)
        if not all_nii:
            return None

        # Priority 1: MPRAGE
        for f in all_nii:
            upper = (os.path.basename(f) + os.path.dirname(f)).upper()
            if "MPRAGE" in upper:
                self.log(f"    MRI selected (MPRAGE): {os.path.basename(f)}")
                return f

        # Priority 2: T1-weighted (not FLAIR)
        for f in all_nii:
            upper = (os.path.basename(f) + os.path.dirname(f)).upper()
            if ("T1" in upper or "SPGR" in upper or "IR-FSPGR" in upper) \
                    and "FLAIR" not in upper:
                self.log(f"    MRI selected (T1): {os.path.basename(f)}")
                return f

        # Priority 3: Anything that's NOT FLAIR
        for f in all_nii:
            upper = (os.path.basename(f) + os.path.dirname(f)).upper()
            if "FLAIR" not in upper and "T2" not in upper:
                self.log(f"    MRI selected (non-FLAIR): {os.path.basename(f)}")
                return f

        # Fallback: largest file
        self.log(f"    MRI selected (fallback): {os.path.basename(all_nii[0])}")
        return all_nii[0]

    def _choose_pet(self, pet_dir):
        """
        Choose the best PET scan.
        Priority by tracer type, then largest file.
        """
        all_nii = self._find_all_nifti(pet_dir)
        if not all_nii:
            return None

        # Try each tracer in priority order
        for tracer in PET_TRACER_PRIORITY:
            for f in all_nii:
                upper = (os.path.basename(f) + os.path.dirname(f)).upper()
                if tracer in upper:
                    self.log(f"    PET selected ({tracer}): {os.path.basename(f)}")
                    return f

        # Fallback: largest file
        self.log(f"    PET selected (largest): {os.path.basename(all_nii[0])}")
        return all_nii[0]

    # ─────────────────────────────────────────────────────────────
    # PROPER NIfTI → SimpleITK (handles RAS↔LPS)
    # ─────────────────────────────────────────────────────────────

    def _nifti_to_sitk(self, nifti_path):
        """
        Load NIfTI file and convert to SimpleITK image with correct
        spatial metadata (handles NIfTI RAS to SimpleITK LPS conversion).
        Returns (sitk_image, numpy_data_xyz).
        """
        nib_img = nib.load(nifti_path)
        data = nib_img.get_fdata().astype(np.float32)

        # Handle 4D — average or take first volume
        if data.ndim == 4:
            if data.shape[3] <= 10:
                data = np.mean(data, axis=-1).astype(np.float32)
            else:
                data = data[:, :, :, 0].astype(np.float32)

        affine = nib_img.affine
        spacing = np.abs(nib_img.header.get_zooms()[:3]).tolist()

        # NIfTI stores as (x,y,z) in RAS, SimpleITK expects (z,y,x) in LPS
        data_sitk_order = np.transpose(data, (2, 1, 0))  # (x,y,z) → (z,y,x)
        sitk_img = sitk.GetImageFromArray(data_sitk_order)
        sitk_img.SetSpacing([float(s) for s in spacing])

        # Convert origin from RAS to LPS (flip x and y)
        origin = affine[:3, 3].copy().astype(float)
        origin[0] = -origin[0]
        origin[1] = -origin[1]
        sitk_img.SetOrigin(origin.tolist())

        # Convert direction from RAS to LPS (flip x and y rows)
        rot = affine[:3, :3] / np.array(spacing)
        direction = rot.copy()
        direction[0, :] = -direction[0, :]
        direction[1, :] = -direction[1, :]
        sitk_img.SetDirection(direction.flatten().tolist())

        return sitk_img, data

    # ─────────────────────────────────────────────────────────────
    # REGISTRATION (multi-resolution, more robust)
    # ─────────────────────────────────────────────────────────────

    def _register_pet_to_mri(self, pet_sitk, mri_sitk):
        """Register PET to MRI using multi-resolution mutual information."""
        mri_sitk = sitk.Cast(mri_sitk, sitk.sitkFloat32)
        pet_sitk = sitk.Cast(pet_sitk, sitk.sitkFloat32)

        # Centre alignment initialization
        initial_tx = sitk.CenteredTransformInitializer(
            mri_sitk, pet_sitk,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.25)
        reg.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=300,
            convergenceMinimumValue=1e-7,
            convergenceWindowSize=15
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetShrinkFactorsPerLevel([8, 4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([4.0, 2.0, 1.0, 0.0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg.SetInitialTransform(initial_tx, inPlace=False)

        try:
            final_tx = reg.Execute(mri_sitk, pet_sitk)
            metric = reg.GetMetricValue()
            self.log(f"    Registration converged: metric = {metric:.6f}")
        except Exception as e:
            self.log(f"    Registration failed: {e}, using centre alignment")
            final_tx = initial_tx
            metric = None

        # Resample PET into MRI space
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(mri_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(final_tx)
        pet_reg_sitk = resampler.Execute(pet_sitk)

        # Convert to numpy (z,y,x) → (x,y,z)
        pet_reg_array = sitk.GetArrayFromImage(pet_reg_sitk)
        pet_reg_data = np.transpose(pet_reg_array, (2, 1, 0)).astype(np.float32)

        return pet_reg_data, metric

    # ─────────────────────────────────────────────────────────────
    # NORMALIZATION
    # ─────────────────────────────────────────────────────────────

    def _normalize(self, data, low_pct=1.0, high_pct=99.0):
        """Percentile normalization to [-1, 1]."""
        data[data < 0] = 0
        nonzero = data[data > 0]
        if len(nonzero) < 100:
            return data

        v_min = np.percentile(nonzero, low_pct)
        v_max = np.percentile(nonzero, high_pct)

        np.clip(data, v_min, v_max, out=data)
        data = (data - v_min) / (v_max - v_min + 1e-8)
        data = 2.0 * data - 1.0

        return data

    # ─────────────────────────────────────────────────────────────
    # MAIN PIPELINE
    # ─────────────────────────────────────────────────────────────

    def run_pipeline(self):
        self.log("=" * 90)
        self.log("QUICK TEST PREPROCESSING PIPELINE (WITH REGISTRATION)")
        self.log("=" * 90)
        self.log(f"Processing first {self.num_patients} patients")
        self.log(f"Target shape: {TARGET_SHAPE}")
        self.log(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("")

        all_patients = sorted([d for d in os.listdir(self.input_base)
                              if os.path.isdir(os.path.join(self.input_base, d))])
        test_patients = all_patients[:self.num_patients]

        self.log(f"Total patients available: {len(all_patients)}")
        self.log(f"Processing: {len(test_patients)} patients")
        self.log(f"Output directory: {self.output_base}")
        self.log("")

        os.makedirs(self.output_base, exist_ok=True)

        # ============== STEP 1: EXTRACT ZIPS ==============
        self.log("=" * 90)
        self.log("STEP 1/3: EXTRACTING ZIP FILES")
        self.log("=" * 90)
        self.log("")

        extracted_count = 0
        skipped_forks = 0
        for idx, patient_id in enumerate(test_patients, 1):
            patient_path = os.path.join(self.input_base, patient_id)

            for modality in ["MRI", "PET"]:
                mod_path = os.path.join(patient_path, modality)
                if not os.path.exists(mod_path):
                    continue

                for f in os.listdir(mod_path):
                    if not f.endswith('.zip'):
                        continue
                    if self._is_resource_fork(f):
                        skipped_forks += 1
                        continue

                    zip_path = os.path.join(mod_path, f)
                    extract_dir = os.path.join(mod_path, f.replace('.zip', ''))

                    if not os.path.exists(extract_dir):
                        try:
                            os.makedirs(extract_dir, exist_ok=True)
                            with zipfile.ZipFile(zip_path, 'r') as zr:
                                zr.extractall(extract_dir)
                            extracted_count += 1
                        except Exception as e:
                            self.log(f"ERROR extracting {patient_id}/{f}: {e}")

            if idx % 10 == 0:
                self.log(f"[{idx}/{len(test_patients)}] Extraction: "
                         f"{extracted_count} extracted")

        self.log(f"Step 1 complete: {extracted_count} extracted, "
                 f"{skipped_forks} resource forks skipped")
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
                self.log(f"Using dcm2niix: {dcm2niix_cmd}")
                self.log((r.stdout or r.stderr).strip())
        except Exception as e:
            has_dcm2niix = False
            self.log(f"dcm2niix not found: {e}")

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

                for ed in extracted_dirs:
                    ep = os.path.join(mod_path, ed)
                    existing = [f for f in os.listdir(ep)
                                if (f.endswith('.nii.gz') or f.endswith('.nii'))
                                and not self._is_resource_fork(f)]

                    if existing:
                        nifti_count += 1
                        continue

                    if has_dcm2niix:
                        try:
                            subprocess.run([
                                dcm2niix_cmd, '-z', 'y',
                                '-f', ed, '-o', ep, ep
                            ], capture_output=True, timeout=120)
                            nifti_count += 1
                        except Exception as e:
                            self.log(f"ERROR converting {patient_id}/{modality}/{ed}: {e}")

            if idx % 10 == 0:
                self.log(f"[{idx}/{len(test_patients)}] Conversion: "
                         f"{nifti_count} NIfTI files")

        self.log(f"Step 2 complete: {nifti_count} NIfTI files")
        self.log("")

        # ============== STEP 3: SELECT + REGISTER + PREPROCESS ==============
        self.log("=" * 90)
        self.log("STEP 3/3: SCAN SELECTION + REGISTRATION + PREPROCESSING")
        self.log("=" * 90)
        self.log("")

        processed = 0
        no_mri = 0
        no_pet = 0
        reg_fallbacks = 0
        errors = 0

        for idx, patient_id in enumerate(test_patients, 1):
            patient_input = os.path.join(self.input_base, patient_id)
            patient_output = os.path.join(self.output_base, patient_id)

            # Resume support
            mri_out = os.path.join(patient_output, "mri_processed.npy")
            pet_out = os.path.join(patient_output, "pet_processed.npy")
            if os.path.exists(mri_out) and os.path.exists(pet_out):
                processed += 1
                if idx % 10 == 0:
                    self.log(f"[{idx}/{len(test_patients)}] {patient_id} — "
                             f"already done, skipping")
                continue

            self.log(f"  [{idx}/{len(test_patients)}] {patient_id}:")

            # Smart scan selection
            mri_nifti = self._choose_mri(os.path.join(patient_input, "MRI"))
            pet_nifti = self._choose_pet(os.path.join(patient_input, "PET"))

            if not mri_nifti:
                self.log(f"    No MRI found, skipping")
                no_mri += 1
                continue
            if not pet_nifti:
                self.log(f"    No PET found, skipping")
                no_pet += 1
                continue

            try:
                # Load with proper RAS→LPS conversion
                mri_sitk_img, mri_data = self._nifti_to_sitk(mri_nifti)
                pet_sitk_img, pet_data_raw = self._nifti_to_sitk(pet_nifti)

                self.log(f"    MRI shape: {mri_data.shape}, "
                         f"PET shape: {pet_data_raw.shape}")

                # Register PET to MRI
                pet_reg_data, metric = self._register_pet_to_mri(
                    pet_sitk_img, mri_sitk_img)

                if metric is not None and metric > -0.01:
                    self.log(f"    WARNING: weak registration (metric={metric:.4f})")
                    reg_fallbacks += 1

                # Normalize
                mri_norm = self._normalize(mri_data.copy(), 0.5, 99.5)
                pet_norm = self._normalize(pet_reg_data.copy(), 1.0, 99.0)

                # Resample to target shape
                mri_final = scipy_zoom(
                    mri_norm,
                    np.array(TARGET_SHAPE) / np.array(mri_norm.shape),
                    order=1
                ).astype(np.float32)

                pet_final = scipy_zoom(
                    pet_norm,
                    np.array(TARGET_SHAPE) / np.array(pet_norm.shape),
                    order=1
                ).astype(np.float32)

                # Save
                os.makedirs(patient_output, exist_ok=True)
                np.save(mri_out, mri_final)
                np.save(pet_out, pet_final)
                processed += 1

                self.log(f"    OK — final shape: {TARGET_SHAPE}")

                del mri_data, pet_data_raw, pet_reg_data
                del mri_norm, pet_norm, mri_final, pet_final
                del mri_sitk_img, pet_sitk_img
                gc.collect()

            except Exception as e:
                self.log(f"    ERROR: {e}")
                errors += 1

            if idx % 10 == 0:
                self.log(f"[{idx}/{len(test_patients)}] Progress: "
                         f"{processed} done, {errors} errors")

        self.log("")
        self.log(f"Step 3 complete: {processed} patients preprocessed")
        self.log("")

        # ============== SUMMARY ==============
        self.log("=" * 90)
        self.log("PIPELINE COMPLETE")
        self.log("=" * 90)
        self.log("")
        self.log("Summary:")
        self.log(f"   - Patients processed:        {len(test_patients)}")
        self.log(f"   - ZIP files extracted:        {extracted_count}")
        self.log(f"   - macOS resource forks skip:  {skipped_forks}")
        self.log(f"   - DICOM to NIfTI:             {nifti_count}")
        self.log(f"   - Registered + preprocessed:  {processed}")
        self.log(f"   - Weak registrations:         {reg_fallbacks}")
        self.log(f"   - Missing MRI:                {no_mri}")
        self.log(f"   - Missing PET:                {no_pet}")
        self.log(f"   - Errors:                     {errors}")
        self.log(f"   - Output shape:               {TARGET_SHAPE}")
        self.log("")
        self.log(f"Output: {self.output_base}")
        self.log(f"Log: {self.log_file}")
        self.log(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 90)


if __name__ == "__main__":
    preprocessor = QuickTestPreprocessor(num_patients=200)
    preprocessor.run_pipeline()