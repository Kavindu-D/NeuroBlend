"""
MRI-to-PET Conditional Diffusion Model Architecture.

Conditional U-Net with DDIM sampling and classifier-free guidance.
Extracted from MRI_to_PET_Diffusion_Model.ipynb for inference use.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ============================================================
# Building Blocks
# ============================================================

def _num_groups(channels, max_groups=32):
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.drop(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            out = attn @ v
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


# ============================================================
# ResBlock variant for CrossAttnUNet (uses time_mlp naming)
# ============================================================

class ResBlockV2(nn.Module):
    """ResBlock with time_mlp naming (for CrossAttnUNet checkpoint compatibility)."""
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2),
        )
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        t_out = self.time_mlp(t_emb)[:, :, None, None]
        scale, shift = t_out.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = F.silu(self.norm2(h))
        h = self.drop(h)
        h = self.conv2(h)
        return h + self.skip(x)


class CrossAttention(nn.Module):
    """Cross-attention between x and conditioning features (combined KV projection).
    Handles different spatial sizes between x and cond by flattening to sequences.
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm_x = nn.GroupNorm(_num_groups(channels), channels)
        self.norm_cond = nn.GroupNorm(_num_groups(channels), channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.kv = nn.Conv2d(channels, channels * 2, 1)  # Combined K and V
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, cond):
        B, C, H, W = x.shape
        _, _, Hc, Wc = cond.shape

        x_norm = self.norm_x(x)
        cond_norm = self.norm_cond(cond)

        # Q from x: (B, heads, H*W, head_dim)
        q = self.q(x_norm).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        # KV from cond: (B, heads, Hc*Wc, head_dim)
        kv = self.kv(cond_norm).reshape(B, 2, self.num_heads, self.head_dim, Hc * Wc)
        k = kv[:, 0].permute(0, 1, 3, 2)  # (B, heads, Hc*Wc, head_dim)
        v = kv[:, 1].permute(0, 1, 3, 2)  # (B, heads, Hc*Wc, head_dim)

        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            out = attn @ v

        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(out)


class CondEncoder(nn.Module):
    """Separate encoder for MRI conditioning with progressive downsampling."""
    def __init__(self, in_ch=3, channels=(64, 128, 256, 512)):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_ch = in_ch
        for ch in channels:
            self.layers.append(nn.Sequential(
                nn.Conv2d(prev_ch, ch, 3, stride=2, padding=1),
                nn.GroupNorm(_num_groups(ch), ch),
                nn.SiLU(),
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.GroupNorm(_num_groups(ch), ch),
                nn.SiLU(),
            ))
            prev_ch = ch

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


class CrossAttnUNet(nn.Module):
    """Conditional U-Net with separate condition encoder and cross-attention.

    This architecture matches checkpoint_epoch_196.pt exactly:
    - Separate cond_encoder for MRI (3ch -> 64 -> 128 -> 256 -> 512)
    - Cross-attention in the middle block (combined KV)
    - 2 res blocks per encoder level, 3 per decoder level
    - Every decoder block receives a skip connection concatenation
    - time_mlp naming convention with scale+shift modulation
    """
    def __init__(self, in_channels=1, out_channels=1, base_ch=64,
                 ch_mult=(1, 2, 4, 8), num_res_enc=2, num_res_dec=3,
                 time_emb_dim=64, dropout=0.1):
        super().__init__()
        channels = [base_ch * m for m in ch_mult]  # [64, 128, 256, 512]
        n_levels = len(channels)
        t_dim = time_emb_dim * 4  # 64 -> 256

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        # Condition encoder (MRI)
        self.cond_encoder = CondEncoder(in_ch=3, channels=channels)

        # Input projection (noisy PET only)
        self.input_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Encoder (downs) - 2 blocks per level
        self.downs = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        # Build encoder: the exact structure from checkpoint
        # downs[0]: 64->64, downs[1]: 64->64
        # downs[2]: 64->128, downs[3]: 128->128
        # downs[4]: 128->256, downs[5]: 256->256
        # downs[6]: 256->512, downs[7]: 512->512
        encoder_in_out = [
            (64, 64), (64, 64),      # Level 0
            (64, 128), (128, 128),   # Level 1
            (128, 256), (256, 256),  # Level 2
            (256, 512), (512, 512),  # Level 3
        ]
        for in_ch, out_ch in encoder_in_out:
            self.downs.append(ResBlockV2(in_ch, out_ch, t_dim, dropout))

        for ch in channels[:-1]:  # downsample after levels 0, 1, 2
            self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))

        # Middle
        mid = channels[-1]  # 512
        self.mid_block1 = ResBlockV2(mid, mid, t_dim, dropout)
        self.mid_attn = SelfAttention(mid, num_heads=8)
        self.mid_cross = CrossAttention(mid, num_heads=8)
        self.mid_block2 = ResBlockV2(mid, mid, t_dim, dropout)

        # Decoder (ups) - 3 blocks per level, each receives skip concat
        self.ups = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        # Build decoder: exact structure from checkpoint
        # The skip pattern: each block gets concatenated input
        decoder_in_out = [
            # Level 3 (output 512)
            (1024, 512),  # ups[0]: 512 (mid) + 512 (skip from downs[7])
            (1024, 512),  # ups[1]: 512 + 512 (skip from downs[6])
            (768, 512),   # ups[2]: 512 + 256 (skip from downs[5])
            # Level 2 (output 256)
            (768, 256),   # ups[3]: 512 + 256 (skip from downs[4])
            (512, 256),   # ups[4]: 256 + 256 (skip from downs[3] is 128? no, needs recheck)
            (384, 256),   # ups[5]: 256 + 128 (skip)
            # Level 1 (output 128)
            (384, 128),   # ups[6]: 256 + 128
            (256, 128),   # ups[7]: 128 + 128
            (192, 128),   # ups[8]: 128 + 64
            # Level 0 (output 64)
            (192, 64),    # ups[9]: 128 + 64
            (128, 64),    # ups[10]: 64 + 64
            (128, 64),    # ups[11]: 64 + 64
        ]
        for in_ch, out_ch in decoder_in_out:
            self.ups.append(ResBlockV2(in_ch, out_ch, t_dim, dropout))

        for ch in reversed(channels[1:]):  # upsample after level 3, 2, 1
            self.up_samples.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))

        # Output
        self.final_norm = nn.GroupNorm(_num_groups(channels[0]), channels[0])
        self.final_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1)

    def forward(self, x, t, cond, x0_prev=None):
        """x: (B,1,H,W) noisy PET, t: (B,) timesteps, cond: (B,3,H,W) MRI."""
        # Time embedding
        t_emb = self.time_mlp(t)

        # Encode condition (MRI) - get multi-scale features for cross-attention and skips
        cond_features = self.cond_encoder(cond)
        # cond_features: [64@80x96, 128@40x48, 256@20x24, 512@10x12]

        # Input projection
        h = self.input_conv(x)

        # Encoder - save skip from every block
        skips = []
        down_idx = 0
        for lv in range(4):  # 4 levels
            for _ in range(2):  # 2 blocks per level
                h = self.downs[down_idx](h, t_emb)
                skips.append(h)
                down_idx += 1
            if lv < 3:  # downsample after levels 0, 1, 2
                h = self.down_samples[lv](h)
        # skips: [64@160x192, 64@160x192, 128@80x96, 128@80x96,
        #         256@40x48, 256@40x48, 512@20x24, 512@20x24]

        # Middle with cross-attention to deepest condition features
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_cross(h, cond_features[-1])  # cross-attn with 512@10x12
        h = self.mid_block2(h, t_emb)

        # Decoder - skip pattern: 2 encoder skips + 1 cond skip per level
        up_idx = 0

        # Level 3 (h at 20x24): use downs[7], downs[6], cond[2]
        h = torch.cat([h, skips[7]], dim=1)  # 512+512=1024
        h = self.ups[up_idx](h, t_emb); up_idx += 1
        h = torch.cat([h, skips[6]], dim=1)  # 512+512=1024
        h = self.ups[up_idx](h, t_emb); up_idx += 1
        h = torch.cat([h, cond_features[2]], dim=1)  # 512+256=768
        h = self.ups[up_idx](h, t_emb); up_idx += 1
        h = self.up_samples[0](h)  # upsample 20x24 -> 40x48

        # Level 2 (h at 40x48): use downs[5], downs[4], cond[1]
        h = torch.cat([h, skips[5]], dim=1)  # 512+256=768
        h = self.ups[up_idx](h, t_emb); up_idx += 1
        h = torch.cat([h, skips[4]], dim=1)  # 256+256=512
        h = self.ups[up_idx](h, t_emb); up_idx += 1
        h = torch.cat([h, cond_features[1]], dim=1)  # 256+128=384
        h = self.ups[up_idx](h, t_emb); up_idx += 1
        h = self.up_samples[1](h)  # upsample 40x48 -> 80x96

        # Level 1 (h at 80x96): use downs[3], downs[2], cond[0]
        h = torch.cat([h, skips[3]], dim=1)  # 256+128=384
        h = self.ups[up_idx](h, t_emb); up_idx += 1
        h = torch.cat([h, skips[2]], dim=1)  # 128+128=256
        h = self.ups[up_idx](h, t_emb); up_idx += 1
        h = torch.cat([h, cond_features[0]], dim=1)  # 128+64=192
        h = self.ups[up_idx](h, t_emb); up_idx += 1
        h = self.up_samples[2](h)  # upsample 80x96 -> 160x192

        # Level 0 (h at 160x192): use downs[1], downs[0], downs[0] (reuse)
        h = torch.cat([h, skips[1]], dim=1)  # 128+64=192
        h = self.ups[up_idx](h, t_emb); up_idx += 1
        h = torch.cat([h, skips[0]], dim=1)  # 64+64=128
        h = self.ups[up_idx](h, t_emb); up_idx += 1
        h = torch.cat([h, skips[0]], dim=1)  # 64+64=128 (reuse skip[0])
        h = self.ups[up_idx](h, t_emb); up_idx += 1

        return self.final_conv(F.silu(self.final_norm(h)))


# ============================================================
# Conditional U-Net (original architecture)
# ============================================================

class ConditionalUNet(nn.Module):
    """Conditional U-Net for MRI->PET diffusion with self-conditioning.
    Input:  cat(noisy_pet[1ch], mri_cond[3ch], x0_prev[1ch]) = 5 channels
    Output: predicted noise [1ch]
    """

    def __init__(self, in_channels=5, out_channels=1, base_ch=64,
                 ch_mult=(1, 2, 4, 8), num_res_blocks=2,
                 attn_levels=(2, 3), time_emb_dim=256, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        channels = [base_ch * m for m in ch_mult]
        n_levels = len(channels)
        t_dim = time_emb_dim * 4

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Encoder
        self.encoder = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_ch = channels[0]
        for lv in range(n_levels):
            ch = channels[lv]
            blocks = nn.ModuleList()
            blocks.append(ResBlock(prev_ch, ch, t_dim, dropout))
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(ch, ch, t_dim, dropout))
            if lv in attn_levels:
                blocks.append(SelfAttention(ch, max(1, ch // 64)))
            self.encoder.append(blocks)
            prev_ch = ch
            if lv < n_levels - 1:
                self.downsamples.append(Downsample(ch))

        # Bottleneck
        mid = channels[-1]
        self.mid_block1 = ResBlock(mid, mid, t_dim, dropout)
        self.mid_attn = SelfAttention(mid, max(1, mid // 64))
        self.mid_block2 = ResBlock(mid, mid, t_dim, dropout)

        # Decoder
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        prev_ch = mid
        for lv in reversed(range(n_levels)):
            ch = channels[lv]
            blocks = nn.ModuleList()
            blocks.append(ResBlock(prev_ch + ch, ch, t_dim, dropout))
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(ch, ch, t_dim, dropout))
            if lv in attn_levels:
                blocks.append(SelfAttention(ch, max(1, ch // 64)))
            self.decoder.append(blocks)
            if lv > 0:
                self.upsamples.append(Upsample(ch))
            prev_ch = ch

        self.out_norm = nn.GroupNorm(_num_groups(channels[0]), channels[0])
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1)

    def forward(self, x, t, cond, x0_prev=None):
        """x: (B,1,H,W) noisy PET, t: (B,) timesteps, cond: (B,3,H,W) MRI,
           x0_prev: (B,1,H,W) previous x0 prediction for self-conditioning (or None/zeros)."""
        if self.in_channels >= 5:
            if x0_prev is None:
                x0_prev = torch.zeros_like(x)
            h = self.conv_in(torch.cat([x, cond, x0_prev], dim=1))  # 1+3+1 = 5 channels
        else:
            h = self.conv_in(torch.cat([x, cond], dim=1))  # 1+3 = 4 channels
        t_emb = self.time_embed(t)

        skips = []
        for lv, blocks in enumerate(self.encoder):
            for block in blocks:
                h = block(h, t_emb) if isinstance(block, ResBlock) else block(h)
            skips.append(h)
            if lv < len(self.downsamples):
                h = self.downsamples[lv](h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        up_idx = 0
        for blocks in self.decoder:
            h = torch.cat([h, skips.pop()], dim=1)
            for block in blocks:
                h = block(h, t_emb) if isinstance(block, ResBlock) else block(h)
            if up_idx < len(self.upsamples):
                h = self.upsamples[up_idx](h)
                up_idx += 1

        return self.out_conv(F.silu(self.out_norm(h)))


# ============================================================
# Diffusion Process
# ============================================================

def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class GaussianDiffusion:
    def __init__(self, T=500, schedule='cosine'):
        self.T = T
        if schedule == 'cosine':
            betas = cosine_beta_schedule(T)
        else:
            betas = torch.linspace(1e-4, 0.02, T)

        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_ac = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_ac = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_ac.to(x0.device)[t].view(-1, 1, 1, 1)
        sm = self.sqrt_1m_ac.to(x0.device)[t].view(-1, 1, 1, 1)
        return sa * x0 + sm * noise

    def predict_x0(self, x_t, t, eps_pred):
        sa = self.sqrt_ac.to(x_t.device)[t].view(-1, 1, 1, 1)
        sm = self.sqrt_1m_ac.to(x_t.device)[t].view(-1, 1, 1, 1)
        return ((x_t - sm * eps_pred) / sa.clamp(min=1e-8)).clamp(-1, 1)

    @torch.no_grad()
    def ddim_sample(self, model, cond, shape, steps=50, eta=0.0, cfg_scale=1.0,
                    progress_callback=None):
        """DDIM sampling with optional classifier-free guidance and self-conditioning.
        CPU-compatible (no autocast).
        """
        device = cond.device
        B = shape[0]
        times = torch.linspace(self.T - 1, 0, steps + 1).long()
        x = torch.randn(shape, device=device)
        x0_prev = torch.zeros(shape, device=device)  # self-conditioning init

        for i in range(len(times) - 1):
            t_now = times[i].item()
            t_next = times[i + 1].item()
            t_b = torch.full((B,), t_now, device=device, dtype=torch.long)

            if cfg_scale > 1.0:
                x_double = torch.cat([x, x], dim=0)
                t_double = torch.cat([t_b, t_b], dim=0)
                c_double = torch.cat([cond, torch.zeros_like(cond)], dim=0)
                x0_prev_double = torch.cat([x0_prev, x0_prev], dim=0)
                eps_double = model(x_double, t_double, c_double, x0_prev_double)
                eps_cond, eps_uncond = eps_double.chunk(2, dim=0)
                eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps = model(x, t_b, cond, x0_prev)

            a_t = self.alphas_cumprod.to(device)[t_now]
            a_next = self.alphas_cumprod.to(device)[max(t_next, 0)]

            x0_pred = ((x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)).clamp(-1, 1)
            x0_prev = x0_pred  # feed to next step as self-conditioning

            if t_next <= 0:
                x = x0_pred
            else:
                sigma = eta * torch.sqrt((1 - a_next) / (1 - a_t) * (1 - a_t / a_next))
                dir_xt = torch.sqrt(1 - a_next - sigma ** 2) * eps
                x = torch.sqrt(a_next) * x0_pred + dir_xt
                if eta > 0:
                    x = x + sigma * torch.randn_like(x)

            if progress_callback:
                progress_callback(i + 1, len(times) - 1)

        return x


# ============================================================
# Factory
# ============================================================

def build_diffusion_model(base_ch=64, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
                           attn_levels=(2, 3), time_emb_dim=256, dropout=0.1,
                           in_channels=5):
    return ConditionalUNet(
        in_channels=in_channels, out_channels=1,
        base_ch=base_ch, ch_mult=ch_mult,
        num_res_blocks=num_res_blocks,
        attn_levels=tuple(attn_levels),
        time_emb_dim=time_emb_dim, dropout=dropout,
    )
