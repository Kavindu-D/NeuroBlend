"""
Inference pipeline orchestration.

Ties together preprocessing, simple-UNet PET synthesis, classification,
uncertainty estimation, and XAI visualization generation.

PET synthesis uses the simple 3D U-Net trained in simple_mri2pet.ipynb,
with sliding-window inference to stay within GPU memory limits.
"""

import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocessing import (
    preprocess_for_model, load_volume, normalize_volume,
    resize_volume, center_crop_3d, ORIG_SHAPE, CROP_SHAPE,
    _is_already_normalized,
)
from models.xai import GradCAM3D, IntegratedGradients3D, extract_cross_attention_maps

CLASS_NAMES = ['CN', 'MCI', 'AD']
CLASS_DESCRIPTIONS = [
    'CN (Cognitively Normal)',
    'MCI (Mild Cognitive Impairment)',
    'AD (Alzheimer\'s Disease)',
]


# ============================================================
# Simple 3D U-Net  (must match simple_mri2pet.ipynb exactly)
# ============================================================

def _unet_block(ic, oc):
    return nn.Sequential(
        nn.Conv3d(ic, oc, 3, padding=1), nn.BatchNorm3d(oc), nn.ReLU(inplace=True),
        nn.Conv3d(oc, oc, 3, padding=1), nn.BatchNorm3d(oc), nn.ReLU(inplace=True),
    )


class SimpleMRI2PETUNet(nn.Module):
    """3-level 3D U-Net — identical to the one in simple_mri2pet.ipynb."""

    def __init__(self, base=32):
        super().__init__()
        self.e1  = _unet_block(1, base)
        self.e2  = _unet_block(base,   base * 2)
        self.e3  = _unet_block(base * 2, base * 4)
        self.bn  = _unet_block(base * 4, base * 8)
        self.d3  = _unet_block(base * 8 + base * 4, base * 4)
        self.d2  = _unet_block(base * 4 + base * 2, base * 2)
        self.d1  = _unet_block(base * 2 + base,     base)
        self.out = nn.Conv3d(base, 1, 1)
        self.pool = nn.MaxPool3d(2)
        self.up   = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        x  = self.bn(self.pool(s3))
        x  = self.d3(torch.cat([self.up(x), s3], 1))
        x  = self.d2(torch.cat([self.up(x), s2], 1))
        x  = self.d1(torch.cat([self.up(x), s1], 1))
        return torch.sigmoid(self.out(x))


def load_simple_unet(checkpoint_path, device):
    """Load SimpleMRI2PETUNet weights from a .pt file.

    Returns:
        model (SimpleMRI2PETUNet) in eval mode
    """
    model = SimpleMRI2PETUNet().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    # Checkpoint may be bare state_dict or wrapped dict
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    model.load_state_dict(state)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Simple U-Net loaded from {checkpoint_path}  ({n/1e6:.1f}M params on {device})")
    return model


# ============================================================
# Synthetic PET Generation — sliding-window 3D U-Net
# ============================================================

def generate_synthetic_pet(mri_full_volume, pet_unet, device='cpu',
                            patch_size=64, overlap=0.5,
                            progress_callback=None):
    """Generate a full 3D synthetic PET volume from MRI using the simple U-Net.

    Uses a sliding-window strategy so VRAM usage never exceeds one patch
    at a time, regardless of the full-volume size.

    Args:
        mri_full_volume : numpy array (D, H, W) — normalised MRI volume
        pet_unet        : SimpleMRI2PETUNet in eval mode
        device          : torch device
        patch_size      : cube edge in voxels (must match training, default 64)
        overlap         : fraction of patch overlap for blending (default 0.5)
        progress_callback: callable(current_patch, total_patches)

    Returns:
        numpy array (D, H, W) dtype float32 — synthetic PET in [0, 1]
    """
    gc.collect()
    if str(device) == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    pet_unet.eval()

    # ------------------------------------------------------------------
    # Fix 1 — Normalization range
    #   preprocessing.py normalises to [-1, 1] for the classification model,
    #   but simple_mri2pet.ipynb was trained on [0, 1] (percentile clip then
    #   divide). If we pass [-1, 1] data the sigmoid output saturates and you
    #   get a flat orange image. Rescale here before entering the U-Net.
    # ------------------------------------------------------------------
    mri_input = mri_full_volume.copy().astype(np.float32)
    if mri_input.min() < -0.1:                    # detect [-1, 1] range
        mri_input = (mri_input + 1.0) / 2.0       # → [0, 1]
    mri_input = np.clip(mri_input, 0.0, 1.0)      # safety clamp

    # ------------------------------------------------------------------
    # Fix 2 — Axis order
    #   preprocess_for_model stores volumes as (H, W, D) matching
    #   ORIG_SHAPE = (160, 192, 160).  The training .npy files were saved as
    #   (D, H, W) so patches are indexed mri[d:d+p, h:h+p, w:w+p].
    #   Transpose to (D, H, W) before inference, then back afterwards so the
    #   returned volume is still (H, W, D) for the visualisation code.
    # ------------------------------------------------------------------
    mri_input = mri_input.transpose(2, 0, 1)      # (H, W, D) → (D, H, W)

    D, H, W = mri_input.shape
    p    = patch_size
    step = max(1, int(p * (1 - overlap)))

    # Smooth Hanning blend weight — avoids hard seams at patch boundaries
    win1d = np.hanning(p).astype(np.float32) + 1e-6
    win3d = win1d[:, None, None] * win1d[None, :, None] * win1d[None, None, :]

    # Pad so every dimension covers a whole number of patches
    pad_d = (p - D % p) % p
    pad_h = (p - H % p) % p
    pad_w = (p - W % p) % p
    mri_p = np.pad(mri_input,
                   ((0, pad_d), (0, pad_h), (0, pad_w)),
                   mode='reflect')
    PD, PH, PW = mri_p.shape
    out_p = np.zeros((PD, PH, PW), dtype=np.float32)
    wgt_p = np.zeros((PD, PH, PW), dtype=np.float32)

    def _positions(dim):
        pos = list(range(0, dim - p + 1, step)) or [0]
        if pos[-1] + p < dim:
            pos.append(dim - p)
        return pos

    xs = _positions(PD)
    ys = _positions(PH)
    zs = _positions(PW)

    total   = len(xs) * len(ys) * len(zs)
    current = 0

    with torch.no_grad():
        for x in xs:
            for y in ys:
                for z in zs:
                    chunk = mri_p[x:x+p, y:y+p, z:z+p]
                    t     = (torch.from_numpy(chunk)
                             .unsqueeze(0).unsqueeze(0)
                             .to(device))
                    pred  = pet_unet(t)[0, 0].cpu().numpy()
                    out_p[x:x+p, y:y+p, z:z+p] += pred  * win3d
                    wgt_p[x:x+p, y:y+p, z:z+p] += win3d

                    current += 1
                    if progress_callback:
                        progress_callback(current, total)

    gc.collect()
    if str(device) == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Trim padding, get result in (D, H, W)
    result_dhw = (out_p / (wgt_p + 1e-8))[:D, :H, :W].astype(np.float32)

    # Transpose back to (H, W, D) — same axis order as mri_full from preprocessing.
    result_hwv = result_dhw.transpose(1, 2, 0)   # (D, H, W) -> (H, W, D)

    # Fix 3 — Output range
    # The U-Net sigmoid outputs [0, 1], but everything downstream expects [-1, 1]:
    #   • _create_brain_slices_figure: vmin=-1, vmax=1 — with [0,1] input,
    #     background (value=0) maps to the MIDDLE of the hot colormap = orange,
    #     not black. That is why the whole image looks orange.
    #   • Classification model was trained on real PET normalised to [-1, 1].
    #   • center_crop_3d pads with -1.0, correct only for [-1, 1] data.
    # Rescale [0,1] -> [-1,1] so one convention is used everywhere.
    return (result_hwv * 2.0 - 1.0).astype(np.float32)


# ============================================================
# Classification with MC Dropout Uncertainty
# ============================================================

def run_classification(mri_tensor, pet_tensor, classification_model,
                       pet_confidence, device='cpu', mc_samples=10):
    """Run classification with MC Dropout uncertainty estimation.

    Args:
        mri_tensor          : (1, 1, D, H, W) center-cropped MRI
        pet_tensor          : (1, 1, D, H, W) center-cropped PET
        classification_model: ADClassificationModel
        pet_confidence      : float (1.0 for real PET, 0.3 for synthetic)
        device              : torch device
        mc_samples          : number of MC Dropout forward passes

    Returns:
        dict with predicted_class, class_name, probabilities, mmse_predicted,
        mmse_std, entropy, mutual_info
    """
    mri      = mri_tensor.to(device)
    pet      = pet_tensor.to(device)
    pet_conf = torch.tensor([pet_confidence], dtype=torch.float32, device=device)

    # MC Dropout inference
    classification_model.eval()
    classification_model.enable_mc_dropout()

    all_probs = []
    all_mmse  = []

    with torch.no_grad():
        for _ in range(mc_samples):
            cls_logits, mmse_pred, attn_weights = classification_model(mri, pet, pet_conf)
            probs = F.softmax(cls_logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_mmse.append(mmse_pred.squeeze(-1).cpu().numpy())

    classification_model.eval()   # restore normal eval mode

    all_probs = np.stack(all_probs, axis=0)   # (mc_samples, 1, num_classes)
    all_mmse  = np.stack(all_mmse,  axis=0)   # (mc_samples, 1)

    mean_probs      = all_probs.mean(axis=0)[0]          # (num_classes,)
    predicted_class = int(mean_probs.argmax())

    # Predictive entropy (total uncertainty)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))

    # Mutual information (epistemic uncertainty)
    expected_entropy = -np.mean(
        np.sum(all_probs[:, 0, :] * np.log(all_probs[:, 0, :] + 1e-10), axis=1))
    mutual_info = entropy - expected_entropy

    # MMSE prediction (denormalised to 0-30 scale)
    mmse_mean = float(all_mmse.mean()) * 30.0
    mmse_std  = float(all_mmse.std())  * 30.0

    return {
        'predicted_class': predicted_class,
        'class_name':      CLASS_DESCRIPTIONS[predicted_class],
        'probabilities':   mean_probs,
        'mmse_predicted':  mmse_mean,
        'mmse_std':        mmse_std,
        'entropy':         float(entropy),
        'mutual_info':     float(mutual_info),
    }


# ============================================================
# XAI Visualization Generation
# ============================================================

def generate_xai_visualizations(mri_tensor, pet_tensor, classification_model,
                                 pet_confidence, predicted_class,
                                 mri_full_volume, pet_full_volume,
                                 is_real_pet, device='cpu'):
    """Generate all XAI visualizations as matplotlib figures.

    Args:
        mri_tensor          : (1, 1, D, H, W) cropped tensor
        pet_tensor          : (1, 1, D, H, W) cropped tensor
        classification_model: model
        pet_confidence      : float
        predicted_class     : int
        mri_full_volume     : numpy (D, H, W) for display
        pet_full_volume     : numpy (D, H, W) for display
        is_real_pet         : bool
        device              : torch device

    Returns:
        dict of matplotlib Figure objects:
            'brain_slices', 'gradcam', 'attention'
    """
    mri      = mri_tensor.to(device)
    pet      = pet_tensor.to(device)
    pet_conf = torch.tensor([pet_confidence], dtype=torch.float32, device=device)

    figures = {}

    # --- Brain Slices ---
    figures['brain_slices'] = _create_brain_slices_figure(
        mri_full_volume, pet_full_volume, is_real_pet)

    # --- Grad-CAM ---
    gradcam_mri = GradCAM3D(classification_model, classification_model.mri_backbone.layer4)
    heatmap_mri, _, probs = gradcam_mri.generate(
        mri, pet, target_class=predicted_class, pet_conf=pet_conf)

    heatmap_pet = None
    if is_real_pet:
        gradcam_pet = GradCAM3D(classification_model, classification_model.pet_backbone.layer4)
        heatmap_pet, _, _ = gradcam_pet.generate(
            mri, pet, target_class=predicted_class, pet_conf=pet_conf)
        gradcam_pet.remove_hooks()

    gradcam_mri.remove_hooks()

    mri_cropped_np = mri_tensor[0, 0].numpy()
    pet_cropped_np = pet_tensor[0, 0].numpy()

    figures['gradcam'] = _create_gradcam_figure(
        mri_cropped_np, heatmap_mri, pet_cropped_np, heatmap_pet,
        predicted_class, probs, is_real_pet)

    # --- Cross-Attention Maps ---
    attn_received, attn_given = extract_cross_attention_maps(
        classification_model, mri, pet, pet_conf, CROP_SHAPE)

    figures['attention'] = _create_attention_figure(
        mri_cropped_np, attn_received, attn_given)

    gc.collect()
    return figures


# ============================================================
# Figure builders (unchanged from original)
# ============================================================

def _create_brain_slices_figure(mri_vol, pet_vol, is_real_pet):
    D, H, W = mri_vol.shape
    n_cols  = 2 if (is_real_pet or pet_vol is not None) else 1
    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 12))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    views = [
        ('Axial',    lambda v: v[:, :, W // 2]),
        ('Coronal',  lambda v: v[:, H // 2, :]),
        ('Sagittal', lambda v: v[D // 2, :, :]),
    ]

    for row, (name, slicer) in enumerate(views):
        axes[row, 0].imshow(slicer(mri_vol), cmap='gray', vmin=-1, vmax=1)
        axes[row, 0].set_title(f'{name} - MRI')
        axes[row, 0].axis('off')

        if n_cols > 1 and pet_vol is not None:
            pet_label = 'PET (Real)' if is_real_pet else 'PET (Synthetic)'
            axes[row, 1].imshow(slicer(pet_vol), cmap='hot', vmin=-1, vmax=1)
            axes[row, 1].set_title(f'{name} - {pet_label}')
            axes[row, 1].axis('off')

    fig.suptitle('Brain Volume Slices', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def _create_gradcam_figure(mri_vol, heatmap_mri, pet_vol, heatmap_pet,
                            predicted_class, probs, is_real_pet):
    D, H, W  = mri_vol.shape
    has_pet  = is_real_pet and heatmap_pet is not None
    n_cols   = 4 if has_pet else 2
    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 12))

    views = [
        ('Axial',    lambda v: v[:, :, W // 2]),
        ('Coronal',  lambda v: v[:, H // 2, :]),
        ('Sagittal', lambda v: v[D // 2, :, :]),
    ]

    for row, (name, slicer) in enumerate(views):
        axes[row, 0].imshow(slicer(mri_vol), cmap='gray', vmin=-1, vmax=1)
        axes[row, 0].set_title(f'{name} - MRI')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(slicer(mri_vol), cmap='gray', vmin=-1, vmax=1)
        axes[row, 1].imshow(slicer(heatmap_mri), cmap='jet', alpha=0.4, vmin=0, vmax=1)
        axes[row, 1].set_title(f'{name} - MRI Grad-CAM')
        axes[row, 1].axis('off')

        if has_pet:
            axes[row, 2].imshow(slicer(pet_vol), cmap='hot', vmin=-1, vmax=1)
            axes[row, 2].set_title(f'{name} - PET')
            axes[row, 2].axis('off')

            axes[row, 3].imshow(slicer(pet_vol), cmap='hot', vmin=-1, vmax=1)
            axes[row, 3].imshow(slicer(heatmap_pet), cmap='jet', alpha=0.4, vmin=0, vmax=1)
            axes[row, 3].set_title(f'{name} - PET Grad-CAM')
            axes[row, 3].axis('off')

    prob_str = ' | '.join(
        [f'{CLASS_NAMES[i]}: {probs[i]:.3f}' for i in range(len(CLASS_NAMES))])
    pet_info = ('MRI + Real PET' if has_pet
                else 'MRI only (synthetic PET not shown in Grad-CAM)')
    fig.suptitle(
        f'Grad-CAM | Predicted: {CLASS_DESCRIPTIONS[predicted_class]}\n'
        f'Probabilities: {prob_str}\nModalities: {pet_info}',
        fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def _create_attention_figure(mri_vol, attn_received, attn_given):
    D, H, W   = mri_vol.shape
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    views = [
        ('Axial',    lambda v: v[:, :, W // 2]),
        ('Coronal',  lambda v: v[:, H // 2, :]),
        ('Sagittal', lambda v: v[D // 2, :, :]),
    ]

    for row, (name, slicer) in enumerate(views):
        axes[row, 0].imshow(slicer(mri_vol), cmap='gray', vmin=-1, vmax=1)
        axes[row, 0].set_title(f'{name} - MRI')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(slicer(attn_received), cmap='viridis', vmin=0, vmax=1)
        axes[row, 1].set_title(f'{name} - Attention Received')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(slicer(mri_vol), cmap='gray', vmin=-1, vmax=1)
        axes[row, 2].imshow(slicer(attn_received), cmap='magma', alpha=0.5, vmin=0, vmax=1)
        axes[row, 2].set_title(f'{name} - MRI + Attention')
        axes[row, 2].axis('off')

    fig.suptitle('Cross-Attention Maps (MRI-PET Fusion)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig