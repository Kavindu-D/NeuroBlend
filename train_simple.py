#!/usr/bin/env python3
"""
MRI → PET 2D Conditional Diffusion Model - COMPLETE TRAINING PIPELINE
For your 50-patient quick test dataset

This version:
- Handles 3D MRI and 3D *or 4D* PET (time dimension averaged if present)
- Does NOT require MRI/PET shapes to match beforehand
- Resizes slices independently to 128x128
- Trains a 2D conditional diffusion model on axial slices
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# ---------------- CONFIG ----------------
DATA_ROOT = "/Volumes/BACKUP/NACC_Test_Preprocessed"  # your quick test output
IMG_SIZE = 128  # resize all slices to 128x128
BATCH_SIZE = 8
T = 500  # fewer diffusion steps for speed
EPOCHS = 20
LR = 1e-4
SAVE_DIR = "./mri2pet_diffusion_ckpts_simple"

# ---------------- DATASET ----------------
class MRIPETSliceDataset(Dataset):
    """
    Loads MRI and PET volumes.
    - PET can be 3D (H,W,D) or 4D (H,W,D,T); if 4D we average over time.
    - MRI and PET are each independently resized in-plane to IMG_SIZE x IMG_SIZE.
    - We then take axial slices and keep slices where either MRI or PET has signal.
    """
    def __init__(self, root_dir: str, img_size: int = 128,
                 min_nonzero_pixels: int = 100):
        self.root = Path(root_dir)
        self.img_size = img_size
        self.min_nonzero_pixels = min_nonzero_pixels
        self.samples = []

        print(f"Scanning {root_dir} for MRI/PET pairs (simple independent resize)...")
        patient_dirs = [d for d in sorted(self.root.iterdir()) if d.is_dir()]
        for patient_dir in tqdm(patient_dirs, desc="Patients"):
            mri_path = patient_dir / "mri_processed.npy"
            pet_path = patient_dir / "pet_processed.npy"
            if not (mri_path.exists() and pet_path.exists()):
                continue

            try:
                mri = np.load(mri_path)  # expect 3D
                pet = np.load(pet_path)  # may be 3D or 4D

                # If PET is 4D (H,W,D,T) → average across time
                if pet.ndim == 4:
                    pet = pet.mean(axis=-1)

                if mri.ndim != 3 or pet.ndim != 3:
                    print(f"Skipping {patient_dir.name}: MRI{mri.shape}, PET{pet.shape}")
                    continue

                Hm, Wm, Dm = mri.shape
                Hp, Wp, Dp = pet.shape

                # Resize in-plane independently to IMG_SIZE x IMG_SIZE
                # MRI: (Hm, Wm, Dm) → (IMG_SIZE, IMG_SIZE, Dm)
                mri_resized = zoom(
                    mri,
                    (self.img_size / Hm, self.img_size / Wm, 1.0),
                    order=1
                )
                # PET: (Hp, Wp, Dp) → (IMG_SIZE, IMG_SIZE, Dp)
                pet_resized = zoom(
                    pet,
                    (self.img_size / Hp, self.img_size / Wp, 1.0),
                    order=1
                )

                # Use number of slices = min(Dm, Dp) to be safe
                D = min(mri_resized.shape[2], pet_resized.shape[2])

                for k in range(D):
                    mri_slice = mri_resized[:, :, k]
                    pet_slice = pet_resized[:, :, k]

                    # Skip slices that are almost empty in both
                    if (np.count_nonzero(mri_slice) < self.min_nonzero_pixels and
                        np.count_nonzero(pet_slice) < self.min_nonzero_pixels):
                        continue

                    self.samples.append((mri_slice.astype(np.float32),
                                         pet_slice.astype(np.float32)))
            except Exception as e:
                print(f"Error loading {patient_dir.name}: {e}")

        print(f"✓ Found {len(self.samples)} valid slices from {len(patient_dirs)} patients")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mri_slice, pet_slice = self.samples[idx]
        # add channel dimension
        mri_t = torch.from_numpy(mri_slice).unsqueeze(0)  # [1,H,W]
        pet_t = torch.from_numpy(pet_slice).unsqueeze(0)  # [1,H,W]
        return mri_t, pet_t

# ---------------- UTILITIES ----------------
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    """
    Sinusoidal timestep embedding.
    timesteps: (B,)
    returns: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / (half - 1)
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1), mode="constant")
    return emb

# ---------------- SIMPLE U-NET ----------------
# class SimpleUNet2D(nn.Module):
#     """
#     Minimal 2D conditional U-Net:
#     - Input: [noisy_pet, mri_cond] → 2 channels
#     - Output: predicted noise (1 channel)
#     """
#     def __init__(self, base_ch: int = 64, time_emb_dim: int = 128):
#         super().__init__()
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_emb_dim, base_ch),
#             nn.SiLU(),
#             nn.Linear(base_ch, base_ch),
#         )

#         # Encoder
#         self.down1 = nn.Sequential(
#             nn.Conv2d(2, base_ch, 3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.pool1 = nn.MaxPool2d(2)

#         self.down2 = nn.Sequential(
#             nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.pool2 = nn.MaxPool2d(2)

#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1),
#             nn.ReLU(inplace=True),
#         )

#         # Decoder
#         self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
#         self.dec2 = nn.Sequential(
#             nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
#             nn.ReLU(inplace=True),
#         )

#         self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
#         self.dec1 = nn.Sequential(
#             nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
#             nn.ReLU(inplace=True),
#         )

#         self.out = nn.Conv2d(base_ch, 1, 1)

#     def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, mri_cond: torch.Tensor) -> torch.Tensor:
#         # x_noisy: [B,1,H,W], mri_cond: [B,1,H,W]
#         h = torch.cat([x_noisy, mri_cond], dim=1)  # [B,2,H,W]
#         t_emb = timestep_embedding(t, 128)
#         t_emb = self.time_mlp(t_emb)[:, :, None, None]

#         # Down
#         d1 = self.down1(h)
#         p1 = self.pool1(d1)

#         d2 = self.down2(p1)
#         p2 = self.pool2(d2)

#         b = self.bottleneck(p2)
#         b = b + t_emb.expand_as(b)  # inject time

#         # Up
#         u2 = self.up2(b)
#         u2 = torch.cat([u2, d2], dim=1)
#         u2 = self.dec2(u2)

#         u1 = self.up1(u2)
#         u1 = torch.cat([u1, d1], dim=1)
#         u1 = self.dec1(u1)

#         out = self.out(u1)
#         return out
class SimpleUNet2D(nn.Module):
    """
    Minimal 2D conditional U-Net:
    - Input: [noisy_pet, mri_cond] → 2 channels
    - Output: predicted noise (1 channel)
    """
    def __init__(self, base_ch: int = 64, time_emb_dim: int = 128):
        super().__init__()
        # time embedding will map 128-dim → 256 channels (base_ch*4, matches bottleneck)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, base_ch * 4),
            nn.SiLU(),
            nn.Linear(base_ch * 4, base_ch * 4),
        )

        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(2, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck: channels = base_ch*4 = 256
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, mri_cond: torch.Tensor) -> torch.Tensor:
        # x_noisy: [B,1,H,W], mri_cond: [B,1,H,W]
        h = torch.cat([x_noisy, mri_cond], dim=1)  # [B,2,H,W]

        # time embedding → match bottleneck channels (base_ch*4 = 256)
        t_emb = timestep_embedding(t, 128)                 # [B,128]
        t_emb = self.time_mlp(t_emb)[:, :, None, None]     # [B,256,1,1]

        # Down
        d1 = self.down1(h)          # [B,64,H,W]
        p1 = self.pool1(d1)         # [B,64,H/2,W/2]

        d2 = self.down2(p1)         # [B,128,H/2,W/2]
        p2 = self.pool2(d2)         # [B,128,H/4,W/4]

        b = self.bottleneck(p2)     # [B,256,H/4,W/4]
        b = b + t_emb               # time conditioning, same channels

        # Up
        u2 = self.up2(b)            # [B,128,H/2,W/2]
        u2 = torch.cat([u2, d2], dim=1)   # [B,256,H/2,W/2]
        u2 = self.dec2(u2)          # [B,128,H/2,W/2]

        u1 = self.up1(u2)           # [B,64,H,W]
        u1 = torch.cat([u1, d1], dim=1)   # [B,128,H,W]
        u1 = self.dec1(u1)          # [B,64,H,W]

        out = self.out(u1)          # [B,1,H,W]
        return out


# ---------------- DIFFUSION PROCESS ----------------
def setup_diffusion(T: int = 500):
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod

betas, alphas, alphas_cumprod = setup_diffusion(T)

def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    Forward diffusion: q(x_t | x_0)
    x0: [B,1,H,W], t: [B]
    """
    device = x0.device
    sqrt_ac = torch.sqrt(alphas_cumprod).to(device)[t].view(-1, 1, 1, 1)
    sqrt_1mac = torch.sqrt(1.0 - alphas_cumprod).to(device)[t].view(-1, 1, 1, 1)
    return sqrt_ac * x0 + sqrt_1mac * noise

# ---------------- TRAINING ----------------
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for mri, pet in tqdm(dataloader, desc="Train", leave=False):
        mri = mri.to(device)
        pet = pet.to(device)

        B = pet.size(0)
        t = torch.randint(0, T, (B,), device=device).long()
        noise = torch.randn_like(pet)

        x_t = q_sample(pet, t, noise)
        pred_noise = model(x_t, t, mri)

        loss = torch.nn.functional.mse_loss(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# ---------------- SAMPLING ----------------
@torch.no_grad()
def sample_one(model, mri_slice: torch.Tensor, steps: int = 50):
    """
    Simple DDPM sampling for one slice.
    mri_slice: [1,1,H,W]
    """
    model.eval()
    device = next(model.parameters()).device
    mri_slice = mri_slice.to(device)

    x = torch.randn_like(mri_slice)  # start from noise

    for i in reversed(range(steps)):
        t = torch.tensor([i], device=device).long()
        eps = model(x, t, mri_slice)
        alpha_t = alphas[i].to(device)
        alpha_cum = alphas_cumprod[i].to(device)
        beta_t = betas[i].to(device)

        sqrt_one_minus = torch.sqrt(1 - alpha_cum)
        mean = (1/torch.sqrt(alpha_t)) * (x - (beta_t / sqrt_one_minus) * eps)

        if i > 0:
            z = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = mean + sigma_t * z
        else:
            x = mean

    return x.cpu()

# ---------------- MAIN ----------------
def main(args):
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MRIPETSliceDataset(DATA_ROOT, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0, pin_memory=True)

    model = SimpleUNet2D(base_ch=64, time_emb_dim=128).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_loss = float("inf")
    for epoch in range(EPOCHS):
        loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }, os.path.join(SAVE_DIR, "best_model.pt"))

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, f"model_epoch_{epoch+1:03d}.pt"))

    print("Training complete. Best model saved at:", os.path.join(SAVE_DIR, "best_model.pt"))

    # Optional: quick demo on one slice
    try:
        root = Path(DATA_ROOT)
        patient_dir = sorted(root.iterdir())[0]
        mri = np.load(patient_dir / "mri_processed.npy")
        k = mri.shape[2] // 2
        mri_slice = mri[:, :, k]
        mri_resized = zoom(mri_slice, (IMG_SIZE / mri_slice.shape[0],
                                       IMG_SIZE / mri_slice.shape[1]), order=1)
        mri_t = torch.from_numpy(mri_resized).float().unsqueeze(0).unsqueeze(0)
        syn_pet = sample_one(model, mri_t, steps=50)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(mri_resized, cmap="gray")
        plt.title("MRI (condition)")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(syn_pet.squeeze().numpy(), cmap="hot")
        plt.title("Synthetic PET")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("simple_demo_synthesis.png", dpi=150)
        print("Saved simple_demo_synthesis.png")
    except Exception as e:
        print("Demo synthesis failed:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # you can add args later if needed
    args = parser.parse_args()
    
    main(args)
