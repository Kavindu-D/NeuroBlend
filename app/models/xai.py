"""
Explainability (XAI) methods for the classification model.

Provides 3D Grad-CAM, Integrated Gradients, and cross-attention map extraction.
Extracted from AD_Classification_Model.ipynb.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import zoom as scipy_zoom


# ============================================================
# 3D Grad-CAM
# ============================================================

class GradCAM3D:
    """3D Grad-CAM for volumetric feature extractors.
    Hooks into a target layer and computes class-discriminative heatmaps.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []

        self._hooks.append(
            target_layer.register_forward_hook(self._save_activation))
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, mri, pet, target_class=None, pet_conf=None):
        """Generate Grad-CAM heatmap.

        Args:
            mri: (1, 1, D, H, W)
            pet: (1, 1, D, H, W)
            target_class: int or None (uses predicted class)
            pet_conf: (1,) pet confidence tensor, or None

        Returns:
            heatmap: (D, H, W) normalized [0, 1]
            predicted_class: int
            class_probs: (num_classes,)
        """
        self.model.eval()
        mri.requires_grad_(False)
        pet.requires_grad_(False)

        cls_logits, _, _ = self.model(mri, pet, pet_conf)
        probs = F.softmax(cls_logits, dim=1)
        predicted_class = cls_logits.argmax(dim=1).item()

        if target_class is None:
            target_class = predicted_class

        self.model.zero_grad()
        score = cls_logits[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2, 3))

        heatmap = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]

        heatmap = F.relu(heatmap)

        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        heatmap = F.interpolate(
            heatmap, size=mri.shape[2:], mode='trilinear', align_corners=False)
        heatmap = heatmap.squeeze().cpu().numpy()

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap, predicted_class, probs[0].detach().cpu().numpy()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ============================================================
# Integrated Gradients
# ============================================================

class IntegratedGradients3D:
    """Integrated Gradients for 3D volumetric models.
    Per-voxel attribution by accumulating gradients along a path from baseline to input.
    """

    def __init__(self, model):
        self.model = model

    def generate(self, mri, pet, target_class=None, pet_conf=None, n_steps=30):
        """Compute IG attributions for MRI.

        Args:
            mri, pet: (1, 1, D, H, W)
            target_class: class to explain (None = predicted)
            pet_conf: (1,) PET confidence tensor
            n_steps: number of interpolation steps

        Returns:
            ig_mri: (D, H, W) MRI attribution map [0, 1]
            predicted_class: int
            class_probs: (num_classes,)
        """
        self.model.eval()
        baseline_mri = torch.zeros_like(mri)

        with torch.no_grad():
            cls_logits, _, _ = self.model(mri, pet, pet_conf)
            probs = F.softmax(cls_logits, dim=1)
            predicted_class = cls_logits.argmax(dim=1).item()

        if target_class is None:
            target_class = predicted_class

        mri_grads_sum = torch.zeros_like(mri)

        for step in range(n_steps + 1):
            alpha = step / n_steps
            interp_mri = (baseline_mri + alpha * (mri - baseline_mri)).clone()
            interp_mri.requires_grad_(True)

            cls_logits, _, _ = self.model(interp_mri, pet, pet_conf)
            score = cls_logits[0, target_class]

            self.model.zero_grad()
            score.backward(retain_graph=False)

            mri_grads_sum += interp_mri.grad.detach()

        ig_mri = (mri - baseline_mri).detach() * (mri_grads_sum / (n_steps + 1))
        ig_mri = ig_mri.squeeze().abs().cpu().numpy()
        ig_mri = (ig_mri - ig_mri.min()) / (ig_mri.max() - ig_mri.min() + 1e-8)

        return ig_mri, predicted_class, probs[0].detach().cpu().numpy()


# ============================================================
# Cross-Attention Map Extraction
# ============================================================

def extract_cross_attention_maps(model, mri, pet, pet_confidence, crop_shape):
    """Extract and upsample cross-attention maps from the fusion module.

    Args:
        model: ADClassificationModel
        mri: (1, 1, D, H, W) tensor
        pet: (1, 1, D, H, W) tensor
        pet_confidence: (1,) tensor
        crop_shape: tuple (D, H, W) of the input volume size

    Returns:
        attn_received: (D, H, W) normalized [0, 1] - how much each token is attended to
        attn_given: (D, H, W) normalized [0, 1] - how much each token attends to others
    """
    model.eval()

    with torch.no_grad():
        mri_pooled, mri_spatial = model.mri_backbone(mri)
        pet_pooled, pet_spatial = model.pet_backbone(pet)
        fused, attn_weights = model.fusion(mri_spatial, pet_spatial, pet_confidence)

    spatial_shape = mri_spatial.shape[2:]
    attn = attn_weights[0].cpu().numpy()

    attn_received = attn.mean(axis=0).reshape(spatial_shape)
    attn_given = attn.mean(axis=1).reshape(spatial_shape)

    zoom_factors = [s / a for s, a in zip(crop_shape, spatial_shape)]

    attn_received_up = scipy_zoom(attn_received, zoom_factors, order=1)
    attn_received_up = (attn_received_up - attn_received_up.min()) / \
                       (attn_received_up.max() - attn_received_up.min() + 1e-8)

    attn_given_up = scipy_zoom(attn_given, zoom_factors, order=1)
    attn_given_up = (attn_given_up - attn_given_up.min()) / \
                    (attn_given_up.max() - attn_given_up.min() + 1e-8)

    return attn_received_up, attn_given_up
