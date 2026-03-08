#!/usr/bin/env python3
"""
Generate QC PNG images for preprocessed NACC data.
Creates a 2×3 QC figure matching clinical radiology style:

    Row 0:  AXIAL T1-W   |  CORONAL T1-W   |  SAGITTAL T1-W
    Row 1:  AXIAL T2-W   |  CORONAL T2-W   |  SAGITTAL T2-W

Labels are placed at the bottom of each panel (like the reference image).
Handles cases where only one modality is available.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
PREPROCESSED_DIR = Path(r"D:\Organized_New_Only\MCI\MCI_Preprocessed")

# Modality labels (change these if your files represent other contrasts)
TOP_LABEL = "T1-W"       # Label for mri_processed.npy  (top row)
BOTTOM_LABEL = "T2-W"    # Label for pet_processed.npy  (bottom row)


# ─────────────────────────────────────────────────────────────────────
# UTILITIES
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
    """
    Get centre slices for axial, coronal, sagittal views.
    Returns (axial, coronal, sagittal) — each is a 2D array.
    """
    cx = data.shape[0] // 2
    cy = data.shape[1] // 2
    cz = data.shape[2] // 2

    axial    = data[:, :, cz].T      # x-y plane
    coronal  = data[:, cy, :].T      # x-z plane
    sagittal = data[cx, :, :].T      # y-z plane

    return axial, coronal, sagittal


# ─────────────────────────────────────────────────────────────────────
# QC FIGURE — 2×3 (two modalities)
# ─────────────────────────────────────────────────────────────────────
def save_qc_figure(top_data, bottom_data, subj_id, out_dir,
                   top_label=TOP_LABEL, bottom_label=BOTTOM_LABEL):
    """
    Create a 2×3 QC figure in the style of the reference image.

    Top row:    Axial / Coronal / Sagittal  — top_label  (e.g. T1-W)
    Bottom row: Axial / Coronal / Sagittal  — bottom_label (e.g. T2-W)

    Labels appear at the bottom-centre of each panel.
    """
    view_names = ["AXIAL", "CORONAL", "SAGITTAL"]

    top_d = to_display(top_data)
    bot_d = to_display(bottom_data)

    top_slices = get_centre_slices(top_d)      # (axial, coronal, sagittal)
    bot_slices = get_centre_slices(bot_d)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8),
                             facecolor='black')
    fig.subplots_adjust(wspace=0.05, hspace=0.12)

    for col in range(3):
        # ── Top row (T1-W) ──
        ax_top = axes[0, col]
        ax_top.imshow(top_slices[col], cmap="gray", origin="lower",
                      aspect="equal")
        ax_top.set_facecolor('black')
        ax_top.axis("off")

        # Label at the bottom of the panel
        ax_top.text(
            0.5, 0.02,
            f"{view_names[col]} {top_label}",
            transform=ax_top.transAxes,
            fontsize=13, fontweight='bold',
            color='white',
            ha='center', va='bottom',
            bbox=dict(boxstyle='square,pad=0.15',
                      facecolor='black', alpha=0.6, edgecolor='none')
        )

        # ── Bottom row (T2-W) ──
        ax_bot = axes[1, col]
        ax_bot.imshow(bot_slices[col], cmap="gray", origin="lower",
                      aspect="equal")
        ax_bot.set_facecolor('black')
        ax_bot.axis("off")

        ax_bot.text(
            0.5, 0.02,
            f"{view_names[col]} {bottom_label}",
            transform=ax_bot.transAxes,
            fontsize=13, fontweight='bold',
            color='white',
            ha='center', va='bottom',
            bbox=dict(boxstyle='square,pad=0.15',
                      facecolor='black', alpha=0.6, edgecolor='none')
        )

    # Optional: add subject ID as a subtle top title
    fig.suptitle(
        f"Subject: {subj_id}",
        fontsize=11, color='white', y=0.98
    )

    out_png = out_dir / f"{subj_id}_qc.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor='black', edgecolor='none')
    plt.close(fig)

    return out_png


# ─────────────────────────────────────────────────────────────────────
# SINGLE MODALITY QC — 1×3
# ─────────────────────────────────────────────────────────────────────
def save_single_modality_qc(data, modality_label, subj_id, out_dir):
    """Generate a 1×3 QC figure for a single modality."""
    view_names = ["AXIAL", "CORONAL", "SAGITTAL"]

    d = to_display(data)
    slices = get_centre_slices(d)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4),
                             facecolor='black')
    fig.subplots_adjust(wspace=0.05)

    for col in range(3):
        ax = axes[col]
        ax.imshow(slices[col], cmap="gray", origin="lower", aspect="equal")
        ax.set_facecolor('black')
        ax.axis("off")

        ax.text(
            0.5, 0.02,
            f"{view_names[col]} {modality_label}",
            transform=ax.transAxes,
            fontsize=13, fontweight='bold',
            color='white',
            ha='center', va='bottom',
            bbox=dict(boxstyle='square,pad=0.15',
                      facecolor='black', alpha=0.6, edgecolor='none')
        )

    fig.suptitle(
        f"Subject: {subj_id}  ({modality_label} only)",
        fontsize=11, color='white', y=0.98
    )

    out_png = out_dir / f"{subj_id}_qc.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor='black', edgecolor='none')
    plt.close(fig)
    return out_png


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("GENERATING QC IMAGES  (T1-W / T2-W style)")
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
        mri_path = patient_dir / "mri_processed.npy"    # T1-W (top row)
        pet_path = patient_dir / "pet_processed.npy"     # T2-W (bottom row)

        has_mri = mri_path.exists()
        has_pet = pet_path.exists()

        # Skip if neither modality is available
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
                # ── Both modalities: 2×3 figure ──
                mri_data = np.load(mri_path)
                pet_data = np.load(pet_path)

                if mri_data.shape != pet_data.shape:
                    shape_mismatches += 1

                save_qc_figure(
                    top_data=mri_data,
                    bottom_data=pet_data,
                    subj_id=subj_id,
                    out_dir=patient_dir,
                    top_label=TOP_LABEL,
                    bottom_label=BOTTOM_LABEL
                )
                del mri_data, pet_data

            elif has_mri:
                mri_data = np.load(mri_path)
                save_single_modality_qc(mri_data, TOP_LABEL, subj_id, patient_dir)
                del mri_data

            else:
                pet_data = np.load(pet_path)
                save_single_modality_qc(pet_data, BOTTOM_LABEL, subj_id, patient_dir)
                del pet_data

            generated += 1

            if idx % 10 == 0 or idx == 1:
                print(f"  [{idx}/{len(patient_dirs)}] {subj_id} — ✓ saved QC PNG")

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