"""
Alzheimer's Disease Classification - Gradio Web UI

Upload an MRI scan (required) and optionally a PET scan.
If no PET is provided, a synthetic PET is generated using the simple 3D U-Net
trained in simple_mri2pet.ipynb.
The classification model predicts CN/MCI/AD with Grad-CAM explainability.

Usage:
    python app.py

Place model checkpoints in the checkpoints/ directory:
    checkpoints/classification_best.pt   -- AD classification model
    checkpoints/simple_mri2pet.pt        -- simple 3D U-Net for PET synthesis
"""

import os
import sys
import time

import numpy as np
import torch
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add app directory to path for imports
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from preprocessing import (
    preprocess_for_model, load_volume, normalize_volume,
    resize_volume, center_crop_3d, ORIG_SHAPE, CROP_SHAPE,
    _is_already_normalized,
)
from models import load_classification_model
from inference import (
    generate_synthetic_pet,
    run_classification,
    generate_xai_visualizations,
    load_simple_unet,
    CLASS_NAMES,
    CLASS_DESCRIPTIONS,
)

# ============================================================
# Configuration
# ============================================================

CHECKPOINT_DIR      = os.path.join(APP_DIR, 'checkpoints')
CLASSIFICATION_CKPT = os.path.join(CHECKPOINT_DIR, 'best_fold_4.pt')
SYNTHESIS_CKPT      = os.path.join(CHECKPOINT_DIR, 'simple_mri2pet.pt')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VALID_EXTENSIONS = {
    '.nii', '.nii.gz',
    '.npy', '.npz',
    '.dcm', '.dicom', '.ima',
    '.tif', '.tiff',
    '.png', '.jpg', '.jpeg', '.bmp', '.webp',
}

# ============================================================
# Model Loading (at startup)
# ============================================================

classification_model = None
pet_unet             = None   # simple 3D U-Net  (replaces diffusion_model)


def load_models():
    """Load both models at startup. Returns a status string."""
    global classification_model, pet_unet

    status = []

    # --- Classification model ---
    if os.path.exists(CLASSIFICATION_CKPT):
        try:
            classification_model, _ = load_classification_model(CLASSIFICATION_CKPT, DEVICE)
            status.append(f"Classification model loaded ({DEVICE})")
        except Exception as e:
            status.append(f"WARNING: Failed to load classification model: {e}")
    else:
        status.append(
            f"WARNING: Classification checkpoint not found at {CLASSIFICATION_CKPT}")

    # --- Simple 3D U-Net for PET synthesis ---
    if os.path.exists(SYNTHESIS_CKPT):
        try:
            pet_unet = load_simple_unet(SYNTHESIS_CKPT, DEVICE)
            status.append(f"PET synthesis U-Net loaded ({DEVICE})")
        except Exception as e:
            pet_unet = None
            status.append(
                f"WARNING: Failed to load PET synthesis model from {SYNTHESIS_CKPT}: {e}\n"
                "Synthetic PET will be unavailable. "
                "You can still run classification by uploading a real PET scan."
            )
    else:
        status.append(
            f"WARNING: PET synthesis checkpoint not found at {SYNTHESIS_CKPT} "
            "(synthetic PET unavailable)")

    return '\n'.join(status)


# ============================================================
# Main Classification Pipeline
# ============================================================

def classify(mri_file, pet_file, progress=gr.Progress()):
    """Main classification pipeline triggered by the Classify button."""

    # --- Validate inputs ---
    if mri_file is None:
        raise gr.Error("Please upload an MRI scan.")

    if classification_model is None:
        raise gr.Error(
            "Classification model not loaded. Place checkpoint in checkpoints/")

    mri_path = mri_file.name if hasattr(mri_file, 'name') else str(mri_file)

    # --- Preprocess MRI ---
    progress(0.05, desc="Loading MRI...")
    mri_tensor, mri_full, mri_info = preprocess_for_model(mri_path)

    input_notes = []
    if mri_info.get('is_2d_input'):
        input_notes.append(
            f"[MRI] 2D image detected ({mri_info['format']}): replicated to "
            "pseudo-3D volume. For best accuracy use a 3D NIfTI or NPY scan."
        )

    # --- Handle PET ---
    is_real_pet    = False
    pet_source_msg = ""

    if pet_file is not None:
        # Real PET provided by the user
        progress(0.1, desc="Loading PET...")
        pet_path = pet_file.name if hasattr(pet_file, 'name') else str(pet_file)
        pet_tensor, pet_full, pet_info = preprocess_for_model(pet_path)
        pet_confidence = 1.0
        is_real_pet    = True
        pet_source_msg = (
            f"Real PET scan uploaded ({pet_info['format']}) "
            f"(confidence: {pet_confidence})"
        )
        if pet_info.get('is_2d_input'):
            input_notes.append(
                f"[PET] 2D image detected ({pet_info['format']}): replicated to pseudo-3D."
            )

    else:
        # Generate synthetic PET with the simple 3D U-Net
        if pet_unet is None:
            raise gr.Error(
                "No PET uploaded and PET synthesis model not loaded. "
                "Either upload a PET scan or place simple_mri2pet.pt in checkpoints/"
            )

        progress(0.10, desc="Generating synthetic PET with 3D U-Net...")
        t0 = time.time()

        # Patch count for progress bar
        # (volume padded to multiples of 64, step=32 → ~5³=125 patches typical)
        _patch_total = [1]  # mutable for closure

        def pet_progress(current, total):
            _patch_total[0] = total
            frac = 0.10 + 0.60 * (current / max(total, 1))
            progress(frac, desc=f"Synthesising PET: patch {current}/{total}")

        pet_full = generate_synthetic_pet(
            mri_full,
            pet_unet,
            device=DEVICE,
            patch_size=64,
            overlap=0.5,
            progress_callback=pet_progress,
        )

        elapsed = time.time() - t0

        # pet_full is (H, W, D) — same axis order as mri_full from
        # preprocess_for_model, so center_crop_3d receives consistent input.
        # The [-1,1] → [0,1] rescaling and (H,W,D)→(D,H,W) transpose for
        # the U-Net are handled internally by generate_synthetic_pet.
        pet_cropped = center_crop_3d(pet_full, CROP_SHAPE)
        pet_tensor  = torch.from_numpy(pet_cropped).unsqueeze(0).unsqueeze(0)
        pet_confidence = 0.3
        is_real_pet    = False
        pet_source_msg = (
            f"Synthetic PET generated by 3D U-Net in {elapsed:.0f}s "
            f"(confidence: {pet_confidence})"
        )

    if input_notes:
        pet_source_msg = pet_source_msg + "\n\n" + "\n".join(input_notes)

    # --- Classification ---
    progress(0.75, desc="Running classification...")
    results = run_classification(
        mri_tensor, pet_tensor, classification_model,
        pet_confidence, device=DEVICE, mc_samples=10,
    )

    # --- XAI Visualizations ---
    progress(0.85, desc="Generating explainability visualizations...")
    figures = generate_xai_visualizations(
        mri_tensor, pet_tensor, classification_model,
        pet_confidence, results['predicted_class'],
        mri_full, pet_full, is_real_pet, device=DEVICE,
    )

    # --- Format outputs ---
    progress(0.95, desc="Preparing results...")

    label_dict = {
        CLASS_DESCRIPTIONS[i]: float(results['probabilities'][i])
        for i in range(len(CLASS_NAMES))
    }

    prob_fig = _create_probability_chart(results['probabilities'])

    mmse_text = (
        f"Predicted MMSE: {results['mmse_predicted']:.1f} "
        f"+/- {results['mmse_std']:.1f} / 30"
    )

    uncertainty_text = (
        f"Predictive Entropy:   {results['entropy']:.4f}\n"
        f"Epistemic Uncertainty: {results['mutual_info']:.4f}\n"
        f"PET Confidence:        {pet_confidence}"
    )

    progress(1.0, desc="Done!")

    return (
        label_dict,
        prob_fig,
        mmse_text,
        uncertainty_text,
        pet_source_msg,
        figures['brain_slices'],
        figures['gradcam'],
        figures['attention'],
    )


def _create_probability_chart(probabilities):
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars   = ax.barh(CLASS_NAMES, probabilities, color=colors, edgecolor='black')
    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{prob:.3f}', va='center', fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Probability')
    ax.set_title('Classification Probabilities')
    fig.tight_layout()
    return fig


# ============================================================
# Gradio Interface
# ============================================================

def build_interface():
    with gr.Blocks(title="AD Classification System") as demo:

        gr.Markdown(
            "# Alzheimer's Disease Classification System\n"
            "Upload an **MRI scan** (required). Optionally upload a **PET scan**. "
            "If no PET is provided, a synthetic PET is generated using the "
            "simple 3D U-Net (simple_mri2pet.pt).\n\n"
            "**Supported formats:**\n"
            "- **3D volumes** (best accuracy): NIfTI (`.nii`, `.nii.gz`), "
            "NumPy (`.npy`, `.npz`)\n"
            "- **DICOM** (single slice): `.dcm`\n"
            "- **TIFF**: `.tif`, `.tiff` (multi-frame stacks treated as 3D)\n"
            "- **2D images**: `.png`, `.jpg`, `.bmp`, `.webp` "
            "— converted to pseudo-3D; results are approximate"
        )

        with gr.Row():
            # --- Inputs ---
            with gr.Column(scale=1):
                mri_input = gr.File(
                    label="MRI Scan (required)",
                    file_types=[
                        '.nii', '.nii.gz', '.npy', '.npz',
                        '.dcm', '.dicom', '.ima',
                        '.tif', '.tiff',
                        '.png', '.jpg', '.jpeg', '.bmp', '.webp',
                    ],
                    type='filepath',
                )
                pet_input = gr.File(
                    label="PET Scan (optional — synthesised if omitted)",
                    file_types=[
                        '.nii', '.nii.gz', '.npy', '.npz',
                        '.dcm', '.dicom', '.ima',
                        '.tif', '.tiff',
                        '.png', '.jpg', '.jpeg', '.bmp', '.webp',
                    ],
                    type='filepath',
                )
                classify_btn = gr.Button("Classify", variant="primary", size="lg")
                pet_source   = gr.Textbox(label="PET Source", interactive=False)

            # --- Results ---
            with gr.Column(scale=1):
                classification_label = gr.Label(
                    label="Classification Result", num_top_classes=3)
                prob_chart       = gr.Plot(label="Class Probabilities")
                mmse_output      = gr.Textbox(label="MMSE Prediction", interactive=False)
                uncertainty_output = gr.Textbox(
                    label="Uncertainty Estimation", interactive=False, lines=3)

        # --- Visualization Tabs ---
        with gr.Tabs():
            with gr.TabItem("Brain Slices"):
                brain_plot = gr.Plot(label="Brain Volume Slices")
            with gr.TabItem("Grad-CAM"):
                gradcam_plot = gr.Plot(label="Grad-CAM Heatmaps")
            with gr.TabItem("Cross-Attention"):
                attention_plot = gr.Plot(label="Cross-Attention Maps")

        classify_btn.click(
            fn=classify,
            inputs=[mri_input, pet_input],
            outputs=[
                classification_label,
                prob_chart,
                mmse_output,
                uncertainty_output,
                pet_source,
                brain_plot,
                gradcam_plot,
                attention_plot,
            ],
        )

        gr.Markdown(
            f"*Running on: {DEVICE} | "
            f"Classification model: {'loaded' if classification_model else 'not found'} | "
            f"PET synthesis U-Net: {'loaded' if pet_unet else 'not found'}*"
        )

    return demo


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("Loading models...")
    print(load_models())
    print()

    print("Starting Gradio interface...")
    demo = build_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )