#!/usr/bin/env python3
"""
Generate QC PNG images for preprocessed NACC data.
Reads mri_processed.npy and pet_processed.npy from each patient folder
and creates a 3x3 QC figure (Axial/Coronal/Sagittal × MRI/PET/Overlay).

Handles MRI and PET having different shapes by slicing each at its own
centre and displaying them side-by-side (no overlay when shapes differ).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
PREPROCESSED_DIR = Path(r"D:\NACC_Matched_Pairs_500_Preprocessed")


# ──────────────────────────────────────���──────────────────────────────
# QC FIGURE
# ─────────────────────────────────────────────────────────────────────
def to_display(volume):
    """Rescale a volume to [0, 1] for display using percentile clipping."""
    v = volume.copy().astype(np.float32)
    p1 = np.percentile(v, 1)
    p99 = np.percentile(v, 99)
    v = np.clip(v, p1, p99)
    v = (v - p1) / (p99 - p1 + 1e-8)
    return v


def get_centre_slices(data):
    """Get centre slices for axial, coronal, sagittal views."""
    x = data.shape[0] // 2
    y = data.shape[1] // 2
    z = data.shape[2] // 2

    axial = data[:, :, z].T        # x-y plane
    coronal = data[:, y, :].T      # x-z plane
    sagittal = data[x, :, :].T     # y-z plane

    return axial, coronal, sagittal


def save_qc_figure(mri_data, pet_data, subj_id, out_dir):
    """
    Create a QC figure.

    If MRI and PET have the SAME shape:
      3×3 grid — Row 0: MRI, Row 1: PET, Row 2: Overlay

    If they have DIFFERENT shapes:
      2×3 grid — Row 0: MRI (sliced at its own centre),
                 Row 1: PET (sliced at its own centre)
      (No overlay since the slices don't correspond spatially)
    """
    same_shape = (mri_data.shape == pet_data.shape)

    mri_d = to_display(mri_data)
    pet_d = to_display(pet_data)

    # Get centre slices independently for each volume
    mri_axial, mri_coronal, mri_sagittal = get_centre_slices(mri_d)
    pet_axial, pet_coronal, pet_sagittal = get_centre_slices(pet_d)

    mri_views = [mri_axial, mri_coronal, mri_sagittal]
    pet_views = [pet_axial, pet_coronal, pet_sagittal]
    view_names = ["Axial", "Coronal", "Sagittal"]

    if same_shape:
        # 3×3: MRI / PET / Overlay
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))

        for col in range(3):
            # Row 0 — MRI
            axes[0, col].imshow(mri_views[col], cmap="gray", origin="lower")
            axes[0, col].set_title(f"MRI {view_names[col]}")
            axes[0, col].axis("off")

            # Row 1 — PET
            axes[1, col].imshow(pet_views[col], cmap="hot", origin="lower")
            axes[1, col].set_title(f"PET {view_names[col]}")
            axes[1, col].axis("off")

            # Row 2 — Overlay
            axes[2, col].imshow(mri_views[col], cmap="gray", origin="lower")
            axes[2, col].imshow(pet_views[col], cmap="hot", alpha=0.4, origin="lower")
            axes[2, col].set_title(f"Overlay {view_names[col]}")
            axes[2, col].axis("off")

        fig.suptitle(
            f"Subject {subj_id} — Preprocessed QC (Registered)\n"
            f"Shape: {mri_data.shape}",
            fontsize=14
        )

    else:
        # 2×3: MRI / PET (no overlay — different shapes)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        for col in range(3):
            # Row 0 — MRI
            axes[0, col].imshow(mri_views[col], cmap="gray", origin="lower")
            axes[0, col].set_title(f"MRI {view_names[col]}")
            axes[0, col].axis("off")

            # Row 1 — PET
            axes[1, col].imshow(pet_views[col], cmap="hot", origin="lower")
            axes[1, col].set_title(f"PET {view_names[col]}")
            axes[1, col].axis("off")

        fig.suptitle(
            f"Subject {subj_id} — Preprocessed QC (Not Registered)\n"
            f"MRI: {mri_data.shape}  |  PET: {pet_data.shape}",
            fontsize=14
        )

    fig.tight_layout()

    out_png = out_dir / f"{subj_id}_qc.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_png


# ─────────────────────────────────────────────────────────────────────
# MRI-ONLY / PET-ONLY QC (when only one modality is available)
# ─────────────────────────────────────────────────────────────────────
def save_single_modality_qc(data, modality_name, subj_id, out_dir):
    """Generate a 1×3 QC figure for a single modality."""
    d = to_display(data)
    axial, coronal, sagittal = get_centre_slices(d)
    views = [axial, coronal, sagittal]
    view_names = ["Axial", "Coronal", "Sagittal"]

    cmap = "gray" if modality_name == "MRI" else "hot"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for col in range(3):
        axes[col].imshow(views[col], cmap=cmap, origin="lower")
        axes[col].set_title(f"{modality_name} {view_names[col]}")
        axes[col].axis("off")

    fig.suptitle(
        f"Subject {subj_id} — {modality_name} Only\n"
        f"Shape: {data.shape}",
        fontsize=14
    )
    fig.tight_layout()

    out_png = out_dir / f"{subj_id}_qc.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("GENERATING QC IMAGES FOR PREPROCESSED DATA")
    print("=" * 70)
    print(f"Source directory : {PREPROCESSED_DIR}")
    print(f"Start time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if not PREPROCESSED_DIR.exists():
        print(f"ERROR: Directory not found: {PREPROCESSED_DIR}")
        sys.exit(1)

    # Gather all patient folders
    patient_dirs = sorted([
        d for d in PREPROCESSED_DIR.iterdir()
        if d.is_dir() and d.name.startswith("NACC")
    ])

    print(f"Total patient folders found: {len(patient_dirs)}")
    print()

    generated = 0
    skipped_missing = 0
    skipped_exists = 0
    errors = 0
    shape_mismatches = 0

    for idx, patient_dir in enumerate(patient_dirs, 1):
        subj_id = patient_dir.name
        mri_path = patient_dir / "mri_processed.npy"
        pet_path = patient_dir / "pet_processed.npy"

        has_mri = mri_path.exists()
        has_pet = pet_path.exists()

        # Skip if neither exists
        if not has_mri and not has_pet:
            print(f"  [{idx}/{len(patient_dirs)}] {subj_id} — SKIP (no .npy files)")
            skipped_missing += 1
            continue

        # Skip if QC already exists
        qc_path = patient_dir / f"{subj_id}_qc.png"
        if qc_path.exists():
            skipped_exists += 1
            if idx % 50 == 0:
                print(f"  [{idx}/{len(patient_dirs)}] {subj_id} — already exists, skipping")
            continue

        try:
            if has_mri and has_pet:
                # Both modalities available
                mri_data = np.load(mri_path)
                pet_data = np.load(pet_path)

                if mri_data.shape != pet_data.shape:
                    shape_mismatches += 1

                save_qc_figure(mri_data, pet_data, subj_id, patient_dir)

                del mri_data, pet_data

            elif has_mri:
                mri_data = np.load(mri_path)
                save_single_modality_qc(mri_data, "MRI", subj_id, patient_dir)
                del mri_data

            else:
                pet_data = np.load(pet_path)
                save_single_modality_qc(pet_data, "PET", subj_id, patient_dir)
                del pet_data

            generated += 1

            if idx % 10 == 0 or idx == 1:
                print(f"  [{idx}/{len(patient_dirs)}] {subj_id} — saved QC PNG")

        except Exception as e:
            print(f"  [{idx}/{len(patient_dirs)}] {subj_id} — ERROR: {e}")
            errors += 1

    # ─────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("QC IMAGE GENERATION COMPLETE")
    print("=" * 70)
    print()
    print(f"  Total patient folders   : {len(patient_dirs)}")
    print(f"  QC images generated     : {generated}")
    print(f"  Already existed (skip)  : {skipped_exists}")
    print(f"  Missing .npy (skip)     : {skipped_missing}")
    print(f"  Shape mismatches        : {shape_mismatches}")
    print(f"  Errors                  : {errors}")
    print()
    print(f"  Output location: {PREPROCESSED_DIR}")
    print(f"  End time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()