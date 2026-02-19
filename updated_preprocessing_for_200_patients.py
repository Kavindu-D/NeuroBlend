# # #!/usr/bin/env python3
# # """
# # MRI–PET Registration & Preprocessing Pipeline (FIXED)

# # Key fixes over the original:
# # 1. Proper NIfTI → SimpleITK conversion (handles RAS↔LPS)
# # 2. Registration on actual images, not masks
# # 3. Better brain masking (Otsu thresholding)
# # 4. Percentile normalization to [-1, 1] for diffusion model
# # 5. Overlay QC to verify alignment
# # """

# # import os
# # from pathlib import Path

# # import nibabel as nib
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import SimpleITK as sitk
# # from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
# # from scipy.ndimage import zoom as scipy_zoom

# # # ---------------------------------------------------------------------
# # # CONFIG
# # # ---------------------------------------------------------------------
# # ROOT_DIR = Path(r"D:\200_Matched_Paired_Testing")
# # OUTPUT_DIR = ROOT_DIR / "preprocessed_final_fixed"
# # OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# # MAX_SUBJECTS = 200  # set to 5 for testing, 200 for full run
# # TARGET_SHAPE = (160, 192, 160)

# # PET_TRACER_PRIORITY = ["AMYLOID", "FLORBETABEN", "PIB", "FDG", "TAU"]

# # # ---------------------------------------------------------------------
# # # HELPERS
# # # ---------------------------------------------------------------------
# # def find_subdirs(dir_path):
# #     return [p for p in dir_path.iterdir() if p.is_dir()]


# # def choose_mri_series(mri_root: Path):
# #     subdirs = find_subdirs(mri_root)
# #     if not subdirs:
# #         return None
# #     mprage = [d for d in subdirs if "MPRAGE" in d.name.upper()]
# #     return mprage[0] if mprage else subdirs[0]


# # def choose_pet_series(pet_root: Path):
# #     subdirs = find_subdirs(pet_root)
# #     if not subdirs:
# #         return None
# #     for key in PET_TRACER_PRIORITY:
# #         for d in subdirs:
# #             if key in d.name.upper():
# #                 return d
# #     return subdirs[0]


# # def find_nii_in_dir(series_dir: Path):
# #     for f in series_dir.iterdir():
# #         if f.is_file() and not f.name.startswith("._"):
# #             if f.suffix == ".nii" or f.name.endswith(".nii.gz"):
# #                 return f
# #     return None


# # # ---------------------------------------------------------------------
# # # PROPER NIFTI → SITK CONVERSION
# # # ---------------------------------------------------------------------
# # def nifti_to_sitk(nifti_path):
# #     """
# #     Load a NIfTI file and convert to SimpleITK image with correct
# #     spatial metadata. Handles the RAS (nibabel) ↔ LPS (SimpleITK)
# #     coordinate flip properly.
# #     """
# #     nib_img = nib.load(str(nifti_path))
# #     data = nib_img.get_fdata().astype(np.float32)

# #     # Handle 4D PET: average across time
# #     if data.ndim == 4:
# #         data = np.mean(data, axis=-1)

# #     affine = nib_img.affine
# #     spacing = np.abs(nib_img.header.get_zooms()[:3]).tolist()

# #     # SimpleITK expects (z, y, x) array ordering
# #     # nibabel gives (x, y, z)
# #     # We transpose and flip axes to go from RAS to LPS
# #     data_sitk_order = np.transpose(data, (2, 1, 0))  # (x,y,z) → (z,y,x)

# #     sitk_img = sitk.GetImageFromArray(data_sitk_order)
# #     sitk_img.SetSpacing([float(spacing[0]), float(spacing[1]), float(spacing[2])])

# #     # Origin: flip x and y signs (RAS → LPS)
# #     origin = affine[:3, 3].copy()
# #     origin[0] = -origin[0]  # R → L
# #     origin[1] = -origin[1]  # A → P
# #     sitk_img.SetOrigin(origin.tolist())

# #     # Direction: flip signs for x and y rows
# #     direction = np.eye(3)
# #     rot = affine[:3, :3] / np.array(spacing)
# #     direction = rot.copy()
# #     direction[0, :] = -direction[0, :]  # flip x row
# #     direction[1, :] = -direction[1, :]  # flip y row
# #     sitk_img.SetDirection(direction.flatten().tolist())

# #     return sitk_img, data


# # # ---------------------------------------------------------------------
# # # BRAIN MASKING (IMPROVED)
# # # ---------------------------------------------------------------------
# # def create_brain_mask_otsu(data):
# #     """
# #     Better brain mask using Otsu thresholding + morphological cleanup.
# #     """
# #     sitk_img = sitk.GetImageFromArray(data)
# #     otsu_filter = sitk.OtsuThresholdImageFilter()
# #     otsu_filter.SetInsideValue(0)
# #     otsu_filter.SetOutsideValue(1)
# #     mask_sitk = otsu_filter.Execute(sitk_img)
# #     mask = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)

# #     # Fill holes and smooth
# #     mask = binary_fill_holes(mask).astype(np.float32)
# #     mask = binary_dilation(mask, iterations=1).astype(np.float32)
# #     mask = binary_erosion(mask, iterations=1).astype(np.float32)
# #     return mask


# # # ---------------------------------------------------------------------
# # # NORMALIZATION (DIFFUSION-MODEL FRIENDLY)
# # # ---------------------------------------------------------------------
# # def normalize_percentile(data, mask, low_pct=1.0, high_pct=99.0):
# #     """
# #     Percentile-based normalization to [-1, 1] range.
# #     Uses only brain voxels for computing percentiles.
# #     """
# #     brain_vals = data[mask > 0]
# #     if len(brain_vals) < 100:
# #         return data

# #     v_low = np.percentile(brain_vals, low_pct)
# #     v_high = np.percentile(brain_vals, high_pct)

# #     # Clip and scale to [0, 1]
# #     normed = np.clip(data, v_low, v_high)
# #     normed = (normed - v_low) / (v_high - v_low + 1e-8)

# #     # Scale to [-1, 1]
# #     normed = normed * 2.0 - 1.0

# #     # Zero out background
# #     normed[mask == 0] = -1.0
# #     return normed


# # # ---------------------------------------------------------------------
# # # REGISTRATION (FIXED)
# # # ---------------------------------------------------------------------
# # def register_pet_to_mri(pet_nii_path, mri_nii_path):
# #     """
# #     Register PET to MRI using SimpleITK with proper coordinate handling.
# #     Uses multi-resolution rigid registration on actual images (not masks).
# #     """
# #     mri_sitk, mri_data = nifti_to_sitk(mri_nii_path)
# #     pet_sitk, pet_data = nifti_to_sitk(pet_nii_path)

# #     # Cast to float32 for registration
# #     mri_sitk = sitk.Cast(mri_sitk, sitk.sitkFloat32)
# #     pet_sitk = sitk.Cast(pet_sitk, sitk.sitkFloat32)

# #     # Step 1: Center alignment
# #     initial_tx = sitk.CenteredTransformInitializer(
# #         mri_sitk,
# #         pet_sitk,
# #         sitk.Euler3DTransform(),
# #         sitk.CenteredTransformInitializerFilter.GEOMETRY
# #     )

# #     # Step 2: Multi-resolution rigid registration on actual images
# #     reg = sitk.ImageRegistrationMethod()

# #     # Mattes mutual information (works well for multi-modal)
# #     reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
# #     reg.SetMetricSamplingStrategy(reg.RANDOM)
# #     reg.SetMetricSamplingPercentage(0.25)

# #     # Optimizer
# #     reg.SetOptimizerAsGradientDescent(
# #         learningRate=1.0,
# #         numberOfIterations=300,
# #         convergenceMinimumValue=1e-7,
# #         convergenceWindowSize=15
# #     )
# #     reg.SetOptimizerScalesFromPhysicalShift()

# #     reg.SetInterpolator(sitk.sitkLinear)

# #     # Multi-resolution pyramid
# #     reg.SetShrinkFactorsPerLevel([8, 4, 2, 1])
# #     reg.SetSmoothingSigmasPerLevel([4.0, 2.0, 1.0, 0.0])
# #     reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# #     reg.SetInitialTransform(initial_tx, inPlace=False)

# #     try:
# #         final_tx = reg.Execute(mri_sitk, pet_sitk)
# #         print(f"    Registration converged: final metric = {reg.GetMetricValue():.6f}")
# #     except Exception as e:
# #         print(f"    Registration failed: {e}, using initial alignment only")
# #         final_tx = initial_tx

# #     # Resample PET into MRI space
# #     resampler = sitk.ResampleImageFilter()
# #     resampler.SetReferenceImage(mri_sitk)
# #     resampler.SetInterpolator(sitk.sitkLinear)
# #     resampler.SetDefaultPixelValue(0)
# #     resampler.SetTransform(final_tx)
# #     pet_reg_sitk = resampler.Execute(pet_sitk)

# #     # Convert back to numpy in (x, y, z) order
# #     pet_reg_array = sitk.GetArrayFromImage(pet_reg_sitk)  # (z, y, x)
# #     pet_reg_data = np.transpose(pet_reg_array, (2, 1, 0))  # → (x, y, z)

# #     return pet_reg_data, mri_data


# # # ---------------------------------------------------------------------
# # # RESIZE TO TARGET SHAPE
# # # ---------------------------------------------------------------------
# # def resample_to_target_shape(data, target_shape):
# #     zoom_factors = np.array(target_shape) / np.array(data.shape)
# #     return scipy_zoom(data, zoom_factors, order=1)


# # # ---------------------------------------------------------------------
# # # QC VISUALIZATION (WITH OVERLAY)
# # # ---------------------------------------------------------------------
# # def save_qc_figure(mri_data, pet_data, subj_id, out_dir: Path):
# #     """
# #     Save QC figure with MRI, PET, and overlay for each view.
# #     """
# #     z = mri_data.shape[2] // 2
# #     y = mri_data.shape[1] // 2
# #     x = mri_data.shape[0] // 2

# #     def to_display(v):
# #         v = v.copy().astype(np.float32)
# #         p1, p99 = np.percentile(v, 1), np.percentile(v, 99)
# #         v = np.clip(v, p1, p99)
# #         v = (v - p1) / (p99 - p1 + 1e-8)
# #         return v

# #     mri_d = to_display(mri_data)
# #     pet_d = to_display(pet_data)

# #     fig, axes = plt.subplots(3, 3, figsize=(12, 12))
# #     views = [
# #         (mri_d[:, :, z].T, pet_d[:, :, z].T, "Axial"),
# #         (mri_d[:, y, :].T, pet_d[:, y, :].T, "Coronal"),
# #         (mri_d[x, :, :].T, pet_d[x, :, :].T, "Sagittal"),
# #     ]

# #     for col, (mri_v, pet_v, name) in enumerate(views):
# #         # Row 0: MRI
# #         axes[0, col].imshow(mri_v, cmap="gray", origin="lower")
# #         axes[0, col].set_title(f"MRI {name}")
# #         axes[0, col].axis("off")

# #         # Row 1: PET
# #         axes[1, col].imshow(pet_v, cmap="hot", origin="lower")
# #         axes[1, col].set_title(f"PET {name}")
# #         axes[1, col].axis("off")

# #         # Row 2: Overlay (MRI gray + PET hot transparent)
# #         axes[2, col].imshow(mri_v, cmap="gray", origin="lower")
# #         axes[2, col].imshow(pet_v, cmap="hot", alpha=0.4, origin="lower")
# #         axes[2, col].set_title(f"Overlay {name}")
# #         axes[2, col].axis("off")

# #     fig.suptitle(f"Subject {subj_id} - REGISTERED + NORMALIZED", fontsize=14)
# #     fig.tight_layout()
# #     out_png = out_dir / f"{subj_id}_qc_registered.png"
# #     fig.savefig(out_png, dpi=150, bbox_inches="tight")
# #     plt.close(fig)
# #     print(f"    Saved QC figure: {out_png}")


# # # ---------------------------------------------------------------------
# # # MAIN
# # # ---------------------------------------------------------------------
# # def main():
# #     subj_dirs = sorted([
# #         d for d in ROOT_DIR.iterdir()
# #         if d.is_dir() and d.name.startswith("NACC")
# #     ])
# #     print(f"Found {len(subj_dirs)} subject folders under {ROOT_DIR}")
# #     subj_dirs = subj_dirs[:MAX_SUBJECTS]
# #     print(f"Processing first {len(subj_dirs)} subjects")

# #     success_count = 0
# #     fail_count = 0

# #     for subj_dir in subj_dirs:
# #         subj_id = subj_dir.name
# #         print(f"\n{'='*60}")
# #         print(f"Subject {subj_id}")
# #         print(f"{'='*60}")

# #         mri_root = subj_dir / "MRI"
# #         pet_root = subj_dir / "PET"
# #         if not mri_root.is_dir() or not pet_root.is_dir():
# #             print("  Missing MRI or PET folder, skipping.")
# #             fail_count += 1
# #             continue

# #         mri_series = choose_mri_series(mri_root)
# #         pet_series = choose_pet_series(pet_root)
# #         if mri_series is None or pet_series is None:
# #             print("  No suitable MRI/PET series, skipping.")
# #             fail_count += 1
# #             continue

# #         print(f"  MRI series: {mri_series.name}")
# #         print(f"  PET series: {pet_series.name}")

# #         mri_nii = find_nii_in_dir(mri_series)
# #         pet_nii = find_nii_in_dir(pet_series)
# #         if mri_nii is None or pet_nii is None:
# #             print("  Could not find NIfTI files, skipping.")
# #             fail_count += 1
# #             continue

# #         # Log raw shapes
# #         mri_raw = nib.load(str(mri_nii)).get_fdata()
# #         pet_raw = nib.load(str(pet_nii)).get_fdata()
# #         print(f"  Raw shapes - MRI: {mri_raw.shape}, PET: {pet_raw.shape}")
# #         del mri_raw, pet_raw

# #         try:
# #             # REGISTER PET to MRI
# #             print("  Registering PET to MRI...")
# #             pet_reg, mri_data = register_pet_to_mri(pet_nii, mri_nii)
# #             print(f"  Post-registration - MRI: {mri_data.shape}, PET: {pet_reg.shape}")

# #             # Brain masks
# #             print("  Creating brain masks (Otsu)...")
# #             mri_mask = create_brain_mask_otsu(mri_data)
# #             pet_mask = create_brain_mask_otsu(pet_reg)

# #             # Normalize to [-1, 1]
# #             print("  Normalizing to [-1, 1]...")
# #             mri_norm = normalize_percentile(mri_data, mri_mask, low_pct=0.5, high_pct=99.5)
# #             pet_norm = normalize_percentile(pet_reg, pet_mask, low_pct=1.0, high_pct=99.0)

# #             # Resize to fixed target shape
# #             print(f"  Resampling to {TARGET_SHAPE}...")
# #             mri_final = resample_to_target_shape(mri_norm, TARGET_SHAPE)
# #             pet_final = resample_to_target_shape(pet_norm, TARGET_SHAPE)

# #             print(f"  Final shapes - MRI: {mri_final.shape}, PET: {pet_final.shape}")

# #             # Save
# #             subj_out_dir = OUTPUT_DIR / subj_id
# #             subj_out_dir.mkdir(exist_ok=True, parents=True)

# #             np.save(subj_out_dir / "mri_processed.npy", mri_final.astype(np.float32))
# #             np.save(subj_out_dir / "pet_processed.npy", pet_final.astype(np.float32))

# #             print(f"  ✓ Saved to {subj_out_dir}")

# #             # QC
# #             save_qc_figure(mri_final, pet_final, subj_id, subj_out_dir)
# #             success_count += 1

# #         except Exception as e:
# #             print(f"  ✗ FAILED: {e}")
# #             fail_count += 1
# #             import traceback
# #             traceback.print_exc()

# #     # Summary
# #     print(f"\n{'='*60}")
# #     print(f"PREPROCESSING COMPLETE")
# #     print(f"{'='*60}")
# #     print(f"  Successful: {success_count}")
# #     print(f"  Failed:     {fail_count}")
# #     print(f"  Output:     {OUTPUT_DIR}")
# #     print(f"  All volumes: {TARGET_SHAPE}")
# #     print(f"  File names:  mri_processed.npy, pet_processed.npy")


# # if __name__ == "__main__":
# #     main()




# #!/usr/bin/env python3
# """
# MRI–PET Registration & Preprocessing Pipeline (FIXED + RESUMABLE)

# Key features:
# 1. Proper NIfTI → SimpleITK conversion (handles RAS↔LPS)
# 2. Registration on actual images, not masks
# 3. Better brain masking (Otsu thresholding)
# 4. Percentile normalization to [-1, 1] for diffusion model
# 5. Overlay QC to verify alignment
# 6. RESUME support — skips already-processed subjects
# 7. Progress log file for tracking
# """

# import os
# import json
# from pathlib import Path
# from datetime import datetime

# import nibabel as nib
# import numpy as np
# import matplotlib.pyplot as plt
# import SimpleITK as sitk
# from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
# from scipy.ndimage import zoom as scipy_zoom

# # ---------------------------------------------------------------------
# # CONFIG
# # ---------------------------------------------------------------------
# ROOT_DIR = Path(r"D:\200_Matched_Paired_Testing")
# OUTPUT_DIR = ROOT_DIR / "preprocessed_final_fixed"
# OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# MAX_SUBJECTS = None  # None = process all subjects
# TARGET_SHAPE = (160, 192, 160)

# PET_TRACER_PRIORITY = ["AMYLOID", "FLORBETABEN", "PIB", "FDG", "TAU"]

# # Resume tracking file
# PROGRESS_FILE = OUTPUT_DIR / "processing_progress.json"

# # ---------------------------------------------------------------------
# # PROGRESS TRACKER (FOR RESUME)
# # ---------------------------------------------------------------------
# class ProgressTracker:
#     """
#     Tracks which subjects have been successfully processed.
#     Saves to a JSON file so processing can resume after a crash.
#     """
#     def __init__(self, progress_file: Path):
#         self.progress_file = progress_file
#         self.data = self._load()

#     def _load(self):
#         if self.progress_file.exists():
#             try:
#                 with open(self.progress_file, "r") as f:
#                     return json.load(f)
#             except (json.JSONDecodeError, IOError):
#                 return {"completed": [], "failed": [], "skipped": []}
#         return {"completed": [], "failed": [], "skipped": []}

#     def _save(self):
#         with open(self.progress_file, "w") as f:
#             json.dump(self.data, f, indent=2)

#     def is_completed(self, subj_id: str) -> bool:
#         return subj_id in self.data["completed"]

#     def mark_completed(self, subj_id: str):
#         if subj_id not in self.data["completed"]:
#             self.data["completed"].append(subj_id)
#         # Remove from failed/skipped if previously there
#         if subj_id in self.data["failed"]:
#             self.data["failed"].remove(subj_id)
#         self._save()

#     def mark_failed(self, subj_id: str, reason: str):
#         entry = {"id": subj_id, "reason": reason, "time": datetime.now().isoformat()}
#         # Remove old failure entry if exists
#         self.data["failed"] = [f for f in self.data["failed"]
#                                 if (f if isinstance(f, str) else f.get("id")) != subj_id]
#         self.data["failed"].append(entry)
#         self._save()

#     def mark_skipped(self, subj_id: str, reason: str):
#         entry = {"id": subj_id, "reason": reason}
#         if not any((s if isinstance(s, str) else s.get("id")) == subj_id
#                     for s in self.data["skipped"]):
#             self.data["skipped"].append(entry)
#             self._save()

#     def summary(self):
#         return {
#             "completed": len(self.data["completed"]),
#             "failed": len(self.data["failed"]),
#             "skipped": len(self.data["skipped"]),
#         }

# # ---------------------------------------------------------------------
# # HELPERS
# # ---------------------------------------------------------------------
# def find_subdirs(dir_path):
#     return [p for p in dir_path.iterdir() if p.is_dir()]


# def choose_mri_series(mri_root: Path):
#     subdirs = find_subdirs(mri_root)
#     if not subdirs:
#         return None
#     mprage = [d for d in subdirs if "MPRAGE" in d.name.upper()]
#     return mprage[0] if mprage else subdirs[0]


# def choose_pet_series(pet_root: Path):
#     subdirs = find_subdirs(pet_root)
#     if not subdirs:
#         return None
#     for key in PET_TRACER_PRIORITY:
#         for d in subdirs:
#             if key in d.name.upper():
#                 return d
#     return subdirs[0]


# def find_nii_in_dir(series_dir: Path):
#     for f in series_dir.iterdir():
#         if f.is_file() and not f.name.startswith("._"):
#             if f.suffix == ".nii" or f.name.endswith(".nii.gz"):
#                 return f
#     return None


# def is_subject_complete(subj_out_dir: Path) -> bool:
#     """
#     Check if a subject has all required output files.
#     This is a filesystem-level check in addition to the progress tracker.
#     """
#     required_files = [
#         subj_out_dir / "mri_processed.npy",
#         subj_out_dir / "pet_processed.npy",
#     ]
#     return all(f.exists() for f in required_files)


# # ---------------------------------------------------------------------
# # PROPER NIFTI → SITK CONVERSION
# # ---------------------------------------------------------------------
# def nifti_to_sitk(nifti_path):
#     """
#     Load a NIfTI file and convert to SimpleITK image with correct
#     spatial metadata. Handles the RAS (nibabel) ↔ LPS (SimpleITK)
#     coordinate flip properly.
#     """
#     nib_img = nib.load(str(nifti_path))
#     data = nib_img.get_fdata().astype(np.float32)

#     # Handle 4D PET: average across time
#     if data.ndim == 4:
#         data = np.mean(data, axis=-1)

#     affine = nib_img.affine
#     spacing = np.abs(nib_img.header.get_zooms()[:3]).tolist()

#     # SimpleITK expects (z, y, x) array ordering
#     data_sitk_order = np.transpose(data, (2, 1, 0))

#     sitk_img = sitk.GetImageFromArray(data_sitk_order)
#     sitk_img.SetSpacing([float(spacing[0]), float(spacing[1]), float(spacing[2])])

#     # Origin: flip x and y signs (RAS → LPS)
#     origin = affine[:3, 3].copy()
#     origin[0] = -origin[0]
#     origin[1] = -origin[1]
#     sitk_img.SetOrigin(origin.tolist())

#     # Direction: flip signs for x and y rows
#     rot = affine[:3, :3] / np.array(spacing)
#     direction = rot.copy()
#     direction[0, :] = -direction[0, :]
#     direction[1, :] = -direction[1, :]
#     sitk_img.SetDirection(direction.flatten().tolist())

#     return sitk_img, data


# # ---------------------------------------------------------------------
# # BRAIN MASKING (IMPROVED)
# # ---------------------------------------------------------------------
# def create_brain_mask_otsu(data):
#     """
#     Better brain mask using Otsu thresholding + morphological cleanup.
#     """
#     sitk_img = sitk.GetImageFromArray(data)
#     otsu_filter = sitk.OtsuThresholdImageFilter()
#     otsu_filter.SetInsideValue(0)
#     otsu_filter.SetOutsideValue(1)
#     mask_sitk = otsu_filter.Execute(sitk_img)
#     mask = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)

#     mask = binary_fill_holes(mask).astype(np.float32)
#     mask = binary_dilation(mask, iterations=1).astype(np.float32)
#     mask = binary_erosion(mask, iterations=1).astype(np.float32)
#     return mask


# # ---------------------------------------------------------------------
# # NORMALIZATION (DIFFUSION-MODEL FRIENDLY)
# # ---------------------------------------------------------------------
# def normalize_percentile(data, mask, low_pct=1.0, high_pct=99.0):
#     """
#     Percentile-based normalization to [-1, 1] range.
#     """
#     brain_vals = data[mask > 0]
#     if len(brain_vals) < 100:
#         return data

#     v_low = np.percentile(brain_vals, low_pct)
#     v_high = np.percentile(brain_vals, high_pct)

#     normed = np.clip(data, v_low, v_high)
#     normed = (normed - v_low) / (v_high - v_low + 1e-8)
#     normed = normed * 2.0 - 1.0
#     normed[mask == 0] = -1.0
#     return normed


# # ---------------------------------------------------------------------
# # REGISTRATION (FIXED)
# # ---------------------------------------------------------------------
# def register_pet_to_mri(pet_nii_path, mri_nii_path):
#     """
#     Register PET to MRI using SimpleITK with proper coordinate handling.
#     """
#     mri_sitk, mri_data = nifti_to_sitk(mri_nii_path)
#     pet_sitk, pet_data = nifti_to_sitk(pet_nii_path)

#     mri_sitk = sitk.Cast(mri_sitk, sitk.sitkFloat32)
#     pet_sitk = sitk.Cast(pet_sitk, sitk.sitkFloat32)

#     # Step 1: Center alignment
#     initial_tx = sitk.CenteredTransformInitializer(
#         mri_sitk, pet_sitk,
#         sitk.Euler3DTransform(),
#         sitk.CenteredTransformInitializerFilter.GEOMETRY
#     )

#     # Step 2: Multi-resolution rigid registration
#     reg = sitk.ImageRegistrationMethod()
#     reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
#     reg.SetMetricSamplingStrategy(reg.RANDOM)
#     reg.SetMetricSamplingPercentage(0.25)

#     reg.SetOptimizerAsGradientDescent(
#         learningRate=1.0,
#         numberOfIterations=300,
#         convergenceMinimumValue=1e-7,
#         convergenceWindowSize=15
#     )
#     reg.SetOptimizerScalesFromPhysicalShift()
#     reg.SetInterpolator(sitk.sitkLinear)

#     reg.SetShrinkFactorsPerLevel([8, 4, 2, 1])
#     reg.SetSmoothingSigmasPerLevel([4.0, 2.0, 1.0, 0.0])
#     reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

#     reg.SetInitialTransform(initial_tx, inPlace=False)

#     try:
#         final_tx = reg.Execute(mri_sitk, pet_sitk)
#         print(f"    Registration converged: final metric = {reg.GetMetricValue():.6f}")
#     except Exception as e:
#         print(f"    Registration failed: {e}, using initial alignment only")
#         final_tx = initial_tx

#     # Resample PET into MRI space
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetReferenceImage(mri_sitk)
#     resampler.SetInterpolator(sitk.sitkLinear)
#     resampler.SetDefaultPixelValue(0)
#     resampler.SetTransform(final_tx)
#     pet_reg_sitk = resampler.Execute(pet_sitk)

#     pet_reg_array = sitk.GetArrayFromImage(pet_reg_sitk)
#     pet_reg_data = np.transpose(pet_reg_array, (2, 1, 0))

#     return pet_reg_data, mri_data


# # ---------------------------------------------------------------------
# # RESIZE TO TARGET SHAPE
# # ---------------------------------------------------------------------
# def resample_to_target_shape(data, target_shape):
#     zoom_factors = np.array(target_shape) / np.array(data.shape)
#     return scipy_zoom(data, zoom_factors, order=1)


# # ---------------------------------------------------------------------
# # QC VISUALIZATION (WITH OVERLAY)
# # ---------------------------------------------------------------------
# def save_qc_figure(mri_data, pet_data, subj_id, out_dir: Path):
#     """
#     Save QC figure with MRI, PET, and overlay for each view.
#     """
#     z = mri_data.shape[2] // 2
#     y = mri_data.shape[1] // 2
#     x = mri_data.shape[0] // 2

#     def to_display(v):
#         v = v.copy().astype(np.float32)
#         p1, p99 = np.percentile(v, 1), np.percentile(v, 99)
#         v = np.clip(v, p1, p99)
#         v = (v - p1) / (p99 - p1 + 1e-8)
#         return v

#     mri_d = to_display(mri_data)
#     pet_d = to_display(pet_data)

#     fig, axes = plt.subplots(3, 3, figsize=(12, 12))
#     views = [
#         (mri_d[:, :, z].T, pet_d[:, :, z].T, "Axial"),
#         (mri_d[:, y, :].T, pet_d[:, y, :].T, "Coronal"),
#         (mri_d[x, :, :].T, pet_d[x, :, :].T, "Sagittal"),
#     ]

#     for col, (mri_v, pet_v, name) in enumerate(views):
#         axes[0, col].imshow(mri_v, cmap="gray", origin="lower")
#         axes[0, col].set_title(f"MRI {name}")
#         axes[0, col].axis("off")

#         axes[1, col].imshow(pet_v, cmap="hot", origin="lower")
#         axes[1, col].set_title(f"PET {name}")
#         axes[1, col].axis("off")

#         axes[2, col].imshow(mri_v, cmap="gray", origin="lower")
#         axes[2, col].imshow(pet_v, cmap="hot", alpha=0.4, origin="lower")
#         axes[2, col].set_title(f"Overlay {name}")
#         axes[2, col].axis("off")

#     fig.suptitle(f"Subject {subj_id} - REGISTERED + NORMALIZED", fontsize=14)
#     fig.tight_layout()
#     out_png = out_dir / f"{subj_id}_qc_registered.png"
#     fig.savefig(out_png, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     print(f"    Saved QC figure: {out_png}")


# # ---------------------------------------------------------------------
# # PROCESS SINGLE SUBJECT
# # ---------------------------------------------------------------------
# def process_subject(subj_dir: Path, tracker: ProgressTracker):
#     """
#     Process a single subject. Returns True if successful.
#     """
#     subj_id = subj_dir.name

#     # Check if already completed (both tracker and filesystem)
#     subj_out_dir = OUTPUT_DIR / subj_id
#     if tracker.is_completed(subj_id) and is_subject_complete(subj_out_dir):
#         print(f"  ✓ Already processed, skipping")
#         return True

#     # If tracker says done but files missing, reprocess
#     if tracker.is_completed(subj_id) and not is_subject_complete(subj_out_dir):
#         print(f"  ⚠ Tracker says done but files missing, reprocessing...")

#     # Check for MRI/PET folders
#     mri_root = subj_dir / "MRI"
#     pet_root = subj_dir / "PET"
#     if not mri_root.is_dir() or not pet_root.is_dir():
#         tracker.mark_skipped(subj_id, "Missing MRI or PET folder")
#         print("  Missing MRI or PET folder, skipping.")
#         return False

#     # Find series
#     mri_series = choose_mri_series(mri_root)
#     pet_series = choose_pet_series(pet_root)
#     if mri_series is None or pet_series is None:
#         tracker.mark_skipped(subj_id, "No suitable MRI/PET series")
#         print("  No suitable MRI/PET series, skipping.")
#         return False

#     print(f"  MRI series: {mri_series.name}")
#     print(f"  PET series: {pet_series.name}")

#     # Find NIfTI files
#     mri_nii = find_nii_in_dir(mri_series)
#     pet_nii = find_nii_in_dir(pet_series)
#     if mri_nii is None or pet_nii is None:
#         tracker.mark_skipped(subj_id, "Could not find NIfTI files")
#         print("  Could not find NIfTI files, skipping.")
#         return False

#     # Log raw shapes
#     mri_raw = nib.load(str(mri_nii)).get_fdata()
#     pet_raw = nib.load(str(pet_nii)).get_fdata()
#     print(f"  Raw shapes - MRI: {mri_raw.shape}, PET: {pet_raw.shape}")
#     del mri_raw, pet_raw

#     # Register
#     print("  Registering PET to MRI...")
#     pet_reg, mri_data = register_pet_to_mri(pet_nii, mri_nii)
#     print(f"  Post-registration - MRI: {mri_data.shape}, PET: {pet_reg.shape}")

#     # Brain masks
#     print("  Creating brain masks (Otsu)...")
#     mri_mask = create_brain_mask_otsu(mri_data)
#     pet_mask = create_brain_mask_otsu(pet_reg)

#     # Normalize
#     print("  Normalizing to [-1, 1]...")
#     mri_norm = normalize_percentile(mri_data, mri_mask, low_pct=0.5, high_pct=99.5)
#     pet_norm = normalize_percentile(pet_reg, pet_mask, low_pct=1.0, high_pct=99.0)

#     # Resize
#     print(f"  Resampling to {TARGET_SHAPE}...")
#     mri_final = resample_to_target_shape(mri_norm, TARGET_SHAPE)
#     pet_final = resample_to_target_shape(pet_norm, TARGET_SHAPE)
#     print(f"  Final shapes - MRI: {mri_final.shape}, PET: {pet_final.shape}")

#     # Save
#     subj_out_dir.mkdir(exist_ok=True, parents=True)
#     np.save(subj_out_dir / "mri_processed.npy", mri_final.astype(np.float32))
#     np.save(subj_out_dir / "pet_processed.npy", pet_final.astype(np.float32))
#     print(f"  ✓ Saved to {subj_out_dir}")

#     # QC
#     save_qc_figure(mri_final, pet_final, subj_id, subj_out_dir)

#     # Mark completed
#     tracker.mark_completed(subj_id)
#     return True


# # ---------------------------------------------------------------------
# # MAIN
# # ---------------------------------------------------------------------
# def main():
#     tracker = ProgressTracker(PROGRESS_FILE)
#     prev_summary = tracker.summary()

#     print(f"{'='*60}")
#     print(f"MRI-PET PREPROCESSING PIPELINE (RESUMABLE)")
#     print(f"{'='*60}")
#     print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"Output dir: {OUTPUT_DIR}")
#     print(f"Progress file: {PROGRESS_FILE}")
#     print(f"Previously completed: {prev_summary['completed']}")
#     print(f"Previously failed: {prev_summary['failed']}")
#     print(f"Previously skipped: {prev_summary['skipped']}")
#     print()

#     # Find all subjects
#     subj_dirs = sorted([
#         d for d in ROOT_DIR.iterdir()
#         if d.is_dir() and d.name.startswith("NACC")
#     ])
#     print(f"Found {len(subj_dirs)} subject folders under {ROOT_DIR}")

#     if MAX_SUBJECTS is not None:
#         subj_dirs = subj_dirs[:MAX_SUBJECTS]
#     print(f"Will process up to {len(subj_dirs)} subjects")

#     # Count how many need processing
#     remaining = [d for d in subj_dirs
#                  if not (tracker.is_completed(d.name) and is_subject_complete(OUTPUT_DIR / d.name))]
#     print(f"Remaining to process: {len(remaining)}")
#     print()

#     success_count = 0
#     fail_count = 0
#     skip_count = 0

#     for idx, subj_dir in enumerate(subj_dirs, 1):
#         subj_id = subj_dir.name

#         # Skip if already done
#         subj_out_dir = OUTPUT_DIR / subj_id
#         if tracker.is_completed(subj_id) and is_subject_complete(subj_out_dir):
#             skip_count += 1
#             continue

#         print(f"\n{'='*60}")
#         print(f"[{idx}/{len(subj_dirs)}] Subject {subj_id}")
#         print(f"{'='*60}")

#         try:
#             ok = process_subject(subj_dir, tracker)
#             if ok:
#                 success_count += 1
#             else:
#                 skip_count += 1
#         except Exception as e:
#             print(f"  ✗ FAILED: {e}")
#             tracker.mark_failed(subj_id, str(e))
#             fail_count += 1
#             import traceback
#             traceback.print_exc()

#         # Print running totals
#         total_done = prev_summary["completed"] + success_count
#         print(f"\n  Progress: {total_done} done | {fail_count} failed | {skip_count} skipped this run")

#     # Final summary
#     final_summary = tracker.summary()
#     print(f"\n{'='*60}")
#     print(f"PREPROCESSING COMPLETE")
#     print(f"{'='*60}")
#     print(f"  End time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"  This run:")
#     print(f"    New successes: {success_count}")
#     print(f"    New failures:  {fail_count}")
#     print(f"    Skipped:       {skip_count}")
#     print(f"  Overall totals:")
#     print(f"    Completed:     {final_summary['completed']}")
#     print(f"    Failed:        {final_summary['failed']}")
#     print(f"    Skipped:       {final_summary['skipped']}")
#     print(f"  Output:          {OUTPUT_DIR}")
#     print(f"  Progress file:   {PROGRESS_FILE}")
#     print(f"  All volumes:     {TARGET_SHAPE}")
#     print()
#     print("To resume after a crash, simply run this script again.")
#     print("Already-processed subjects will be skipped automatically.")


# if __name__ == "__main__":
#     main()




































#!/usr/bin/env python3
"""
MRI–PET Registration & Preprocessing Pipeline
(FIXED + RESUMABLE + AUTO-EXTRACT + DICOM + MULTI-FRAME PET FIX)

Key features:
1. AUTO-EXTRACTS zip files inside MRI/ and PET/ folders
2. AUTO-CONVERTS DICOM → volume using SimpleITK
3. Handles multi-frame PET (picks correct series, reshapes 4D)
4. Proper NIfTI → SimpleITK conversion (handles RAS↔LPS)
5. Registration on actual images, not masks
6. Better brain masking (Otsu thresholding)
7. Percentile normalization to [-1, 1] for diffusion model
8. Overlay QC to verify alignment
9. RESUME support
10. Progress log
"""

import os
import json
import zipfile
import gc
from pathlib import Path
from datetime import datetime

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
from scipy.ndimage import zoom as scipy_zoom

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ROOT_DIR = Path(r"D:\200_Matched_Paired_Testing")
OUTPUT_DIR = ROOT_DIR / "preprocessed_final_fixed"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_SUBJECTS = None
TARGET_SHAPE = (160, 192, 160)
MAX_REASONABLE_SLICES = 400  # anything above this is likely 4D

PET_TRACER_PRIORITY = ["AMYLOID", "FLORBETABEN", "PIB", "FDG", "TAU"]

PROGRESS_FILE = OUTPUT_DIR / "processing_progress.json"

# ---------------------------------------------------------------------
# PROGRESS TRACKER
# ---------------------------------------------------------------------
class ProgressTracker:
    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.data = self._load()

    def _load(self):
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"completed": [], "failed": [], "skipped": []}
        return {"completed": [], "failed": [], "skipped": []}

    def _save(self):
        with open(self.progress_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def is_completed(self, subj_id: str) -> bool:
        return subj_id in self.data["completed"]

    def mark_completed(self, subj_id: str):
        if subj_id not in self.data["completed"]:
            self.data["completed"].append(subj_id)
        self.data["failed"] = [f for f in self.data["failed"]
                                if (f if isinstance(f, str) else f.get("id")) != subj_id]
        self._save()

    def mark_failed(self, subj_id: str, reason: str):
        entry = {"id": subj_id, "reason": reason, "time": datetime.now().isoformat()}
        self.data["failed"] = [f for f in self.data["failed"]
                                if (f if isinstance(f, str) else f.get("id")) != subj_id]
        self.data["failed"].append(entry)
        self._save()

    def mark_skipped(self, subj_id: str, reason: str):
        entry = {"id": subj_id, "reason": reason}
        if not any((s if isinstance(s, str) else s.get("id")) == subj_id
                    for s in self.data["skipped"]):
            self.data["skipped"].append(entry)
            self._save()

    def summary(self):
        return {
            "completed": len(self.data["completed"]),
            "failed": len(self.data["failed"]),
            "skipped": len(self.data["skipped"]),
        }

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def is_subject_complete(subj_out_dir: Path) -> bool:
    return all((subj_out_dir / f).exists()
               for f in ["mri_processed.npy", "pet_processed.npy"])


# ---------------------------------------------------------------------
# ZIP EXTRACTION
# ---------------------------------------------------------------------
def extract_zips_in_dir(dir_path: Path):
    if not dir_path.exists():
        return
    zip_files = [f for f in dir_path.iterdir()
                 if f.is_file() and f.suffix == ".zip" and not f.name.startswith("._")]
    for zf in zip_files:
        extract_dir = dir_path / zf.stem
        if extract_dir.exists() and any(extract_dir.iterdir()):
            continue
        extract_dir.mkdir(exist_ok=True)
        print(f"    Extracting: {zf.name}")
        try:
            with zipfile.ZipFile(zf, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        except zipfile.BadZipFile:
            print(f"    ⚠ Bad zip: {zf.name}")


# ---------------------------------------------------------------------
# DICOM → VOLUME (HANDLES MULTI-SERIES)
# ---------------------------------------------------------------------
def load_dicom_series_as_volume(dicom_dir: Path, max_slices=MAX_REASONABLE_SLICES):
    """
    Read DICOM series. If multiple series exist, try each one
    and pick the one with a reasonable number of slices.
    Handles 4D PET by picking the best single series.
    """
    reader = sitk.ImageSeriesReader()

    # Collect all directories that might contain DICOMs
    search_dirs = [dicom_dir]
    for sub in dicom_dir.rglob("*"):
        if sub.is_dir():
            search_dirs.append(sub)

    # Collect all (directory, series_id, file_count) combinations
    all_series = []
    for sdir in search_dirs:
        try:
            series_ids = reader.GetGDCMSeriesIDs(str(sdir))
            for sid in series_ids:
                files = reader.GetGDCMSeriesFileNames(str(sdir), sid)
                all_series.append((sdir, sid, len(files), files))
        except Exception:
            continue

    if not all_series:
        return None, None

    # Sort: prefer series with slice count in reasonable range (50-400)
    # then by largest count within that range
    reasonable = [s for s in all_series if 20 <= s[2] <= max_slices]
    if reasonable:
        # Pick the one with most slices in reasonable range
        reasonable.sort(key=lambda x: x[2], reverse=True)
        chosen = reasonable[0]
    else:
        # All series are either too small or too large
        # Pick the one closest to typical brain PET/MRI slice count (~150-200)
        all_series.sort(key=lambda x: abs(x[2] - 180))
        chosen = all_series[0]

    sdir, sid, count, file_names = chosen
    print(f"      Selected series: {count} slices (from {sdir.name})")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    try:
        sitk_img = reader.Execute()
    except Exception as e:
        print(f"      ⚠ DICOM read failed: {e}")
        return None, None

    data = sitk.GetArrayFromImage(sitk_img)  # (z, y, x)

    # Check if still 4D-like (z >> x or y)
    if data.shape[0] > max_slices and data.ndim == 3:
        # Likely multiple frames concatenated along z
        # Estimate number of frames
        # Typical brain PET has ~90-200 slices
        # Try to find a divisor that gives reasonable slice count
        z_total = data.shape[0]
        best_nframes = 1
        for nf in range(2, 20):
            if z_total % nf == 0:
                slices_per_frame = z_total // nf
                if 50 <= slices_per_frame <= 300:
                    best_nframes = nf
                    break

        if best_nframes > 1:
            slices_per_frame = z_total // best_nframes
            print(f"      Detected {best_nframes} frames × {slices_per_frame} slices, averaging...")
            data = data.reshape(best_nframes, slices_per_frame, data.shape[1], data.shape[2])
            data = np.mean(data, axis=0)  # average across frames

            # Rebuild sitk image with correct shape
            sitk_img_new = sitk.GetImageFromArray(data)
            orig_spacing = list(sitk_img.GetSpacing())
            sitk_img_new.SetSpacing(orig_spacing)
            sitk_img_new.SetOrigin(sitk_img.GetOrigin())
            sitk_img_new.SetDirection(sitk_img.GetDirection())
            sitk_img = sitk_img_new
        else:
            print(f"      ⚠ Large z-dim ({z_total}) but can't determine frame count, using as-is")

    data = np.transpose(data, (2, 1, 0)).astype(np.float32)  # → (x, y, z)
    return data, sitk_img


# ---------------------------------------------------------------------
# FIND FILES
# ---------------------------------------------------------------------
def find_nii_recursive(search_dir: Path):
    nii_files = []
    for f in search_dir.rglob("*"):
        if f.is_file() and not f.name.startswith("._"):
            if f.suffix == ".nii" or f.name.endswith(".nii.gz"):
                nii_files.append(f)
    nii_files.sort(key=lambda x: x.stat().st_size, reverse=True)
    return nii_files


def find_dicom_dirs(search_dir: Path):
    dcm_dirs = []
    for d in search_dir.rglob("*"):
        if d.is_dir():
            dcm_files = list(d.glob("*.dcm"))
            if len(dcm_files) > 5:
                dcm_dirs.append((d, len(dcm_files)))
    dcm_dirs.sort(key=lambda x: x[1], reverse=True)
    return dcm_dirs


# ---------------------------------------------------------------------
# NIFTI → SITK
# ---------------------------------------------------------------------
def nifti_to_sitk(nifti_path):
    nib_img = nib.load(str(nifti_path))
    data = nib_img.get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = np.mean(data, axis=-1)

    affine = nib_img.affine
    spacing = np.abs(nib_img.header.get_zooms()[:3]).tolist()

    data_sitk_order = np.transpose(data, (2, 1, 0))
    sitk_img = sitk.GetImageFromArray(data_sitk_order)
    sitk_img.SetSpacing([float(s) for s in spacing])

    origin = affine[:3, 3].copy()
    origin[0] = -origin[0]
    origin[1] = -origin[1]
    sitk_img.SetOrigin(origin.tolist())

    rot = affine[:3, :3] / np.array(spacing)
    direction = rot.copy()
    direction[0, :] = -direction[0, :]
    direction[1, :] = -direction[1, :]
    sitk_img.SetDirection(direction.flatten().tolist())

    return sitk_img, data


# ---------------------------------------------------------------------
# CHOOSE MRI VOLUME
# ---------------------------------------------------------------------
def choose_mri_volume(mri_root: Path):
    extract_zips_in_dir(mri_root)

    # Try NIfTI first
    all_nii = find_nii_recursive(mri_root)
    if all_nii:
        for f in all_nii:
            if "MPRAGE" in (f.name + str(f.parent)).upper():
                print(f"    Found MRI NIfTI (MPRAGE): {f.name}")
                sitk_img, data = nifti_to_sitk(f)
                return data, sitk_img
        for f in all_nii:
            u = (f.name + str(f.parent)).upper()
            if "T1" in u and "FLAIR" not in u:
                print(f"    Found MRI NIfTI (T1): {f.name}")
                sitk_img, data = nifti_to_sitk(f)
                return data, sitk_img
        non_flair = [f for f in all_nii
                     if "FLAIR" not in (f.name + str(f.parent)).upper()]
        target = non_flair[0] if non_flair else all_nii[0]
        print(f"    Found MRI NIfTI: {target.name}")
        sitk_img, data = nifti_to_sitk(target)
        return data, sitk_img

    # Try DICOM
    dcm_dirs = find_dicom_dirs(mri_root)
    if dcm_dirs:
        # Priority: MPRAGE
        for d, count in dcm_dirs:
            if "MPRAGE" in str(d).upper():
                print(f"    Found MRI DICOM (MPRAGE): {d.name} ({count} files)")
                data, sitk_img = load_dicom_series_as_volume(d)
                if data is not None:
                    return data, sitk_img

        # Priority: T1, not FLAIR
        for d, count in dcm_dirs:
            u = str(d).upper()
            if ("T1" in u or "SPGR" in u or "IR-FSPGR" in u) and "FLAIR" not in u:
                print(f"    Found MRI DICOM (T1): {d.name} ({count} files)")
                data, sitk_img = load_dicom_series_as_volume(d)
                if data is not None:
                    return data, sitk_img

        # Fallback: largest non-FLAIR
        for d, count in dcm_dirs:
            if "FLAIR" not in str(d).upper():
                print(f"    Found MRI DICOM: {d.name} ({count} files)")
                data, sitk_img = load_dicom_series_as_volume(d)
                if data is not None:
                    return data, sitk_img

        # Last resort
        d, count = dcm_dirs[0]
        print(f"    Found MRI DICOM (fallback): {d.name} ({count} files)")
        data, sitk_img = load_dicom_series_as_volume(d)
        if data is not None:
            return data, sitk_img

    return None, None


# ---------------------------------------------------------------------
# CHOOSE PET VOLUME
# ---------------------------------------------------------------------
def choose_pet_volume(pet_root: Path):
    extract_zips_in_dir(pet_root)

    # Try NIfTI first
    all_nii = find_nii_recursive(pet_root)
    if all_nii:
        for tracer in PET_TRACER_PRIORITY:
            for f in all_nii:
                if tracer in (f.name + str(f.parent)).upper():
                    print(f"    Found PET NIfTI ({tracer}): {f.name}")
                    sitk_img, data = nifti_to_sitk(f)
                    return data, sitk_img
        print(f"    Found PET NIfTI: {all_nii[0].name}")
        sitk_img, data = nifti_to_sitk(all_nii[0])
        return data, sitk_img

    # Try DICOM
    dcm_dirs = find_dicom_dirs(pet_root)
    if dcm_dirs:
        for tracer in PET_TRACER_PRIORITY:
            for d, count in dcm_dirs:
                if tracer in str(d).upper():
                    print(f"    Found PET DICOM ({tracer}): {d.name} ({count} files)")
                    data, sitk_img = load_dicom_series_as_volume(d)
                    if data is not None:
                        return data, sitk_img

        # Fallback: try each dir until one works
        for d, count in dcm_dirs:
            print(f"    Found PET DICOM: {d.name} ({count} files)")
            data, sitk_img = load_dicom_series_as_volume(d)
            if data is not None:
                return data, sitk_img

    return None, None


# ---------------------------------------------------------------------
# BRAIN MASKING
# ---------------------------------------------------------------------
def create_brain_mask_otsu(data):
    sitk_img = sitk.GetImageFromArray(data)
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    mask_sitk = otsu_filter.Execute(sitk_img)
    mask = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)
    mask = binary_fill_holes(mask).astype(np.float32)
    mask = binary_dilation(mask, iterations=1).astype(np.float32)
    mask = binary_erosion(mask, iterations=1).astype(np.float32)
    return mask


# ---------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------
def normalize_percentile(data, mask, low_pct=1.0, high_pct=99.0):
    brain_vals = data[mask > 0]
    if len(brain_vals) < 100:
        return data
    v_low = np.percentile(brain_vals, low_pct)
    v_high = np.percentile(brain_vals, high_pct)
    normed = np.clip(data, v_low, v_high)
    normed = (normed - v_low) / (v_high - v_low + 1e-8)
    normed = normed * 2.0 - 1.0
    normed[mask == 0] = -1.0
    return normed


# ---------------------------------------------------------------------
# REGISTRATION
# ---------------------------------------------------------------------
def register_pet_to_mri_sitk(pet_sitk, mri_sitk):
    mri_sitk = sitk.Cast(mri_sitk, sitk.sitkFloat32)
    pet_sitk = sitk.Cast(pet_sitk, sitk.sitkFloat32)

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
        print(f"    Registration converged: metric = {reg.GetMetricValue():.6f}")
    except Exception as e:
        print(f"    Registration failed: {e}, using center alignment")
        final_tx = initial_tx

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(mri_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_tx)
    pet_reg_sitk = resampler.Execute(pet_sitk)

    pet_reg_array = sitk.GetArrayFromImage(pet_reg_sitk)
    pet_reg_data = np.transpose(pet_reg_array, (2, 1, 0)).astype(np.float32)
    return pet_reg_data


# ---------------------------------------------------------------------
# RESIZE
# ---------------------------------------------------------------------
def resample_to_target_shape(data, target_shape):
    zoom_factors = np.array(target_shape) / np.array(data.shape)
    return scipy_zoom(data, zoom_factors, order=1)


# ---------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------
def save_qc_figure(mri_data, pet_data, subj_id, out_dir: Path):
    z = mri_data.shape[2] // 2
    y = mri_data.shape[1] // 2
    x = mri_data.shape[0] // 2

    def to_display(v):
        v = v.copy().astype(np.float32)
        p1, p99 = np.percentile(v, 1), np.percentile(v, 99)
        v = np.clip(v, p1, p99)
        v = (v - p1) / (p99 - p1 + 1e-8)
        return v

    mri_d = to_display(mri_data)
    pet_d = to_display(pet_data)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    views = [
        (mri_d[:, :, z].T, pet_d[:, :, z].T, "Axial"),
        (mri_d[:, y, :].T, pet_d[:, y, :].T, "Coronal"),
        (mri_d[x, :, :].T, pet_d[x, :, :].T, "Sagittal"),
    ]
    for col, (mri_v, pet_v, name) in enumerate(views):
        axes[0, col].imshow(mri_v, cmap="gray", origin="lower")
        axes[0, col].set_title(f"MRI {name}"); axes[0, col].axis("off")
        axes[1, col].imshow(pet_v, cmap="hot", origin="lower")
        axes[1, col].set_title(f"PET {name}"); axes[1, col].axis("off")
        axes[2, col].imshow(mri_v, cmap="gray", origin="lower")
        axes[2, col].imshow(pet_v, cmap="hot", alpha=0.4, origin="lower")
        axes[2, col].set_title(f"Overlay {name}"); axes[2, col].axis("off")

    fig.suptitle(f"Subject {subj_id} - REGISTERED + NORMALIZED", fontsize=14)
    fig.tight_layout()
    out_png = out_dir / f"{subj_id}_qc_registered.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved QC: {out_png}")


# ---------------------------------------------------------------------
# PROCESS SINGLE SUBJECT
# ---------------------------------------------------------------------
def process_subject(subj_dir: Path, tracker: ProgressTracker):
    subj_id = subj_dir.name
    subj_out_dir = OUTPUT_DIR / subj_id

    if tracker.is_completed(subj_id) and is_subject_complete(subj_out_dir):
        return True

    mri_root = subj_dir / "MRI"
    pet_root = subj_dir / "PET"
    if not mri_root.is_dir() or not pet_root.is_dir():
        tracker.mark_skipped(subj_id, "Missing MRI or PET folder")
        print("  Missing MRI or PET folder, skipping.")
        return False

    # MRI
    print("  Loading MRI...")
    mri_data, mri_sitk_img = choose_mri_volume(mri_root)
    if mri_data is None:
        tracker.mark_skipped(subj_id, "No MRI volume found")
        print("  ✗ No MRI volume found, skipping.")
        return False
    print(f"  MRI shape: {mri_data.shape}")

    # PET
    print("  Loading PET...")
    pet_data, pet_sitk_img = choose_pet_volume(pet_root)
    if pet_data is None:
        tracker.mark_skipped(subj_id, "No PET volume found")
        print("  ✗ No PET volume found, skipping.")
        return False
    print(f"  PET shape: {pet_data.shape}")

    # Handle 4D
    if pet_data.ndim == 4:
        print(f"  Averaging 4D PET ({pet_data.shape}) → 3D...")
        pet_data = np.mean(pet_data, axis=-1).astype(np.float32)

    # Sanity check shapes
    if any(s > MAX_REASONABLE_SLICES for s in pet_data.shape):
        print(f"  ⚠ PET shape still too large: {pet_data.shape}, attempting reshape...")
        # Try to reshape: assume concatenated frames along largest dim
        max_dim = np.argmax(pet_data.shape)
        total = pet_data.shape[max_dim]
        for nf in range(2, 30):
            if total % nf == 0:
                per_frame = total // nf
                if 50 <= per_frame <= 300:
                    print(f"    Reshaping: {nf} frames × {per_frame} slices, averaging...")
                    if max_dim == 0:
                        pet_data = pet_data.reshape(nf, per_frame, pet_data.shape[1], pet_data.shape[2])
                    elif max_dim == 1:
                        pet_data = pet_data.reshape(pet_data.shape[0], nf, per_frame, pet_data.shape[2])
                    else:
                        pet_data = pet_data.reshape(pet_data.shape[0], pet_data.shape[1], nf, per_frame)
                    pet_data = np.mean(pet_data, axis=max_dim if max_dim == 0 else max_dim).astype(np.float32)

                    # Rebuild sitk image
                    pet_sitk_order = np.transpose(pet_data, (2, 1, 0))
                    pet_sitk_img = sitk.GetImageFromArray(pet_sitk_order)
                    pet_sitk_img.SetSpacing(pet_sitk_img.GetSpacing())
                    print(f"    New PET shape: {pet_data.shape}")
                    break

    if any(s > MAX_REASONABLE_SLICES for s in pet_data.shape):
        tracker.mark_failed(subj_id, f"PET shape too large after reshape: {pet_data.shape}")
        print(f"  ✗ PET shape still too large: {pet_data.shape}, skipping.")
        return False

    # Register
    print("  Registering PET → MRI...")
    pet_reg = register_pet_to_mri_sitk(pet_sitk_img, mri_sitk_img)
    print(f"  Registered PET shape: {pet_reg.shape}")

    # Masks
    print("  Brain masks (Otsu)...")
    mri_mask = create_brain_mask_otsu(mri_data)
    pet_mask = create_brain_mask_otsu(pet_reg)

    # Normalize
    print("  Normalizing [-1, 1]...")
    mri_norm = normalize_percentile(mri_data, mri_mask, low_pct=0.5, high_pct=99.5)
    pet_norm = normalize_percentile(pet_reg, pet_mask, low_pct=1.0, high_pct=99.0)

    # Resize
    print(f"  Resampling → {TARGET_SHAPE}...")
    mri_final = resample_to_target_shape(mri_norm, TARGET_SHAPE)
    pet_final = resample_to_target_shape(pet_norm, TARGET_SHAPE)
    print(f"  Final: MRI {mri_final.shape}, PET {pet_final.shape}")

    # Save
    subj_out_dir.mkdir(exist_ok=True, parents=True)
    np.save(subj_out_dir / "mri_processed.npy", mri_final.astype(np.float32))
    np.save(subj_out_dir / "pet_processed.npy", pet_final.astype(np.float32))
    print(f"  ✓ Saved to {subj_out_dir}")

    # QC
    save_qc_figure(mri_final, pet_final, subj_id, subj_out_dir)
    tracker.mark_completed(subj_id)

    # Free memory
    del mri_data, pet_data, pet_reg, mri_norm, pet_norm, mri_final, pet_final
    del mri_mask, pet_mask, mri_sitk_img, pet_sitk_img
    gc.collect()

    return True


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    tracker = ProgressTracker(PROGRESS_FILE)
    prev = tracker.summary()

    print(f"{'='*60}")
    print(f"MRI-PET PIPELINE (RESUME + EXTRACT + DICOM + 4D FIX)")
    print(f"{'='*60}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Previously completed: {prev['completed']}")
    print(f"Previously failed: {prev['failed']}")
    print()

    subj_dirs = sorted([d for d in ROOT_DIR.iterdir()
                        if d.is_dir() and d.name.startswith("NACC")])
    print(f"Found {len(subj_dirs)} subjects")
    if MAX_SUBJECTS is not None:
        subj_dirs = subj_dirs[:MAX_SUBJECTS]

    remaining = [d for d in subj_dirs
                 if not (tracker.is_completed(d.name) and is_subject_complete(OUTPUT_DIR / d.name))]
    print(f"Remaining: {len(remaining)}")
    print()

    success = fail = skip = 0

    for idx, subj_dir in enumerate(subj_dirs, 1):
        subj_id = subj_dir.name
        if tracker.is_completed(subj_id) and is_subject_complete(OUTPUT_DIR / subj_id):
            skip += 1
            continue

        print(f"\n{'='*60}")
        print(f"[{idx}/{len(subj_dirs)}] {subj_id}")
        print(f"{'='*60}")

        try:
            ok = process_subject(subj_dir, tracker)
            if ok:
                success += 1
            else:
                skip += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            tracker.mark_failed(subj_id, str(e))
            fail += 1
            import traceback
            traceback.print_exc()

        total = prev["completed"] + success
        print(f"\n  Progress: {total} done | {fail} failed | {skip} skipped")

    final = tracker.summary()
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"  New: {success} ok, {fail} fail")
    print(f"  Total completed: {final['completed']}")
    print(f"  Total failed: {final['failed']}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"\nRun again to resume.")


if __name__ == "__main__":
    main()