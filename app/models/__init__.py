"""
Model loading helpers.

Handles loading classification and diffusion model checkpoints.
"""

import torch
from .classification import build_classification_model
from .diffusion import build_diffusion_model, GaussianDiffusion, CrossAttnUNet


def load_classification_model(checkpoint_path, device='cpu'):
    """Load classification model from a .pt checkpoint.

    Args:
        checkpoint_path: path to best_model.pt from classification training
        device: torch device

    Returns:
        model: ADClassificationModel in eval mode
        metadata: dict with epoch, best_bal_acc, etc.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain 'model_state_dict'."
        )

    state = ckpt['model_state_dict']

    # Infer optional clinical branch from checkpoint shapes.
    cls_input_dim = int(state['cls_head.0.weight'].shape[1]) if 'cls_head.0.weight' in state else 512
    clinical_dim = 0
    clinical_embed_dim = 0
    if any(k.startswith('clinical_encoder.') for k in state.keys()):
        if 'clinical_encoder.0.weight' in state:
            clinical_dim = int(state['clinical_encoder.0.weight'].shape[1])
        clinical_embed_dim = max(0, cls_input_dim - 512)

    model = build_classification_model(
        clinical_dim=clinical_dim,
        clinical_embed_dim=clinical_embed_dim if clinical_dim > 0 else 0,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, ckpt


def load_diffusion_model(checkpoint_path, device='cpu'):
    """Load diffusion model from a .pt checkpoint.

    Uses EMA weights for inference (better synthesis quality).
    Auto-detects architecture (ConditionalUNet vs CrossAttnUNet) from checkpoint keys.

    Args:
        checkpoint_path: path to best_model.pt from diffusion training
        device: torch device

    Returns:
        model: ConditionalUNet or CrossAttnUNet in eval mode
        diffusion: GaussianDiffusion instance
        metadata: dict with epoch, best_ssim, etc.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' not in ckpt and 'ema_state_dict' not in ckpt and 'ema_shadow' not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain diffusion weights "
            "(expected 'model_state_dict', 'ema_state_dict', or 'ema_shadow')."
        )

    # Get the state dict (prefer EMA weights)
    weights = ckpt.get('ema_state_dict', ckpt.get('ema_shadow', ckpt.get('model_state_dict', {})))

    # Strip _orig_mod. prefix if present (from torch.compile)
    clean_weights = {}
    for k, v in weights.items():
        clean_k = k.replace('_orig_mod.', '')
        clean_weights[clean_k] = v
    weights = clean_weights

    # Detect architecture from checkpoint keys
    is_cross_attn = 'cond_encoder.layers.0.0.weight' in weights or 'input_conv.weight' in weights
    is_conditional = 'conv_in.weight' in weights

    if is_cross_attn:
        # CrossAttnUNet architecture (checkpoint_epoch_196.pt style)
        base_ch = 64
        ch_mult = (1, 2, 4, 8)
        time_emb_dim = 64
        dropout = 0.1

        model = CrossAttnUNet(
            in_channels=1, out_channels=1,
            base_ch=base_ch, ch_mult=ch_mult,
            num_res_enc=2, num_res_dec=3,
            time_emb_dim=time_emb_dim, dropout=dropout,
        )
        model.load_state_dict(weights)

    elif is_conditional:
        # Original ConditionalUNet architecture
        base_ch = 64
        ch_mult = (1, 2, 4, 8)
        num_res_blocks = 2
        attn_levels = (2, 3)
        time_emb_dim = 256
        dropout = 0.1

        if 'cfg' in ckpt:
            saved_cfg = ckpt['cfg']
            base_ch = saved_cfg.get('BASE_CH', base_ch)
            ch_mult_raw = saved_cfg.get('CH_MULT', ch_mult)
            ch_mult = tuple(ch_mult_raw) if isinstance(ch_mult_raw, list) else ch_mult_raw
            num_res_blocks = saved_cfg.get('NUM_RES_BLOCKS', num_res_blocks)
            attn_levels_raw = saved_cfg.get('ATTN_LEVELS', attn_levels)
            attn_levels = tuple(attn_levels_raw) if isinstance(attn_levels_raw, list) else attn_levels_raw
            time_emb_dim = saved_cfg.get('TIME_EMB_DIM', time_emb_dim)
            dropout = saved_cfg.get('DROPOUT', dropout)

        conv_in_w = weights.get('conv_in.weight')
        in_channels = int(conv_in_w.shape[1]) if conv_in_w is not None else 5

        model = build_diffusion_model(
            base_ch=base_ch, ch_mult=ch_mult,
            num_res_blocks=num_res_blocks, attn_levels=attn_levels,
            time_emb_dim=time_emb_dim, dropout=dropout,
            in_channels=in_channels,
        )
        model.load_state_dict(weights)

    else:
        if any(k.startswith('mri_backbone.') for k in weights.keys()):
            raise ValueError(
                f"Checkpoint {checkpoint_path} appears to be a classification model, "
                "not a diffusion model."
            )
        raise ValueError(
            f"Checkpoint {checkpoint_path} has unknown architecture "
            "(expected 'conv_in.weight' or 'cond_encoder' keys)."
        )

    model.to(device)
    model.eval()

    # Read T from checkpoint config (default 500 for new models, 1000 for legacy)
    T = 500
    if 'cfg' in ckpt:
        T = ckpt['cfg'].get('NUM_TIMESTEPS', 500)
    if 'config' in ckpt and ckpt['config']:
        T = ckpt['config'].get('NUM_TIMESTEPS', T)
    diffusion = GaussianDiffusion(T=T, schedule='cosine')

    return model, diffusion, ckpt
