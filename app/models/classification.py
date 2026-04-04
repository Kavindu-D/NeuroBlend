"""
Alzheimer's Disease Classification Model Architecture.

Dual-branch 3D ResNet-18 + Asymmetric Cross-Attention Fusion + Multi-Task Heads.
Extracted from AD_Classification_Model.ipynb for inference use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 3D ResNet-18 Feature Extractor
# ============================================================

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet3D(nn.Module):
    """3D ResNet-18 backbone for single-channel volumetric input.
    Returns both the pooled feature vector and spatial feature maps (for Grad-CAM).
    """

    def __init__(self, in_channels=1, num_blocks=(2, 2, 2, 2)):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv3d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride):
        layers = [BasicBlock3D(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock3D(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        spatial = self.layer4(x)
        pooled = self.avgpool(spatial).flatten(1)
        return pooled, spatial


# ============================================================
# Asymmetric Cross-Attention Fusion
# ============================================================

class AsymmetricCrossAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.mri_to_pet = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.pet_to_mri = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_mri = nn.LayerNorm(dim)
        self.norm_pet = nn.LayerNorm(dim)

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, mri_tokens, pet_tokens, pet_confidence=None):
        if pet_confidence is not None:
            pc = pet_confidence.view(-1, 1, 1)
            scaled_pet = pet_tokens * pc
        else:
            scaled_pet = pet_tokens

        mri_attended, _ = self.mri_to_pet(
            query=mri_tokens, key=scaled_pet, value=scaled_pet)
        mri_attended = self.norm_mri(mri_tokens + mri_attended)

        pet_attended, attn_w = self.pet_to_mri(
            query=scaled_pet, key=mri_tokens, value=mri_tokens)
        pet_attended = self.norm_pet(scaled_pet + pet_attended)

        mri_pool = mri_attended.mean(dim=1)
        pet_pool = pet_attended.mean(dim=1)
        raw_gate = self.gate(torch.cat([mri_pool, pet_pool], dim=-1))

        if pet_confidence is not None:
            pc = pet_confidence.view(-1, 1)
            gate = raw_gate * pc + (1.0 - pc)
        else:
            gate = raw_gate

        fused = gate.unsqueeze(1) * mri_attended + (1 - gate.unsqueeze(1)) * pet_attended
        fused = self.norm_out(fused + self.ffn(fused))

        return fused, attn_w


class AsymmetricCrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            AsymmetricCrossAttentionLayer(feature_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, mri_spatial, pet_spatial, pet_confidence=None):
        B, C = mri_spatial.shape[:2]
        mri_tokens = mri_spatial.flatten(2).permute(0, 2, 1)
        pet_tokens = pet_spatial.flatten(2).permute(0, 2, 1)

        attn_weights = None
        for layer in self.layers:
            mri_tokens, attn_weights = layer(mri_tokens, pet_tokens, pet_confidence)
            pet_tokens = mri_tokens

        fused = mri_tokens.permute(0, 2, 1)
        fused = self.pool(fused).squeeze(-1)

        return fused, attn_weights


# ============================================================
# Full Multi-Task Classification Model
# ============================================================

class ADClassificationModel(nn.Module):
    def __init__(self, num_classes=3, feature_dim=512, dropout=0.3,
                 fusion_heads=8, fusion_layers=2, fusion_dropout=0.1,
                 clinical_dim=0, clinical_embed_dim=64):
        super().__init__()

        self.mri_backbone = ResNet3D(in_channels=1)
        self.pet_backbone = ResNet3D(in_channels=1)

        self.fusion = AsymmetricCrossAttentionFusion(
            feature_dim=feature_dim, num_heads=fusion_heads,
            num_layers=fusion_layers, dropout=fusion_dropout)

        self.clinical_dim = int(clinical_dim or 0)
        self.clinical_embed_dim = int(clinical_embed_dim or 0)
        self.has_clinical_branch = self.clinical_dim > 0 and self.clinical_embed_dim > 0

        if self.has_clinical_branch:
            self.clinical_encoder = nn.Sequential(
                nn.Linear(self.clinical_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, self.clinical_embed_dim),
                nn.ReLU(inplace=True),
            )
            classifier_input_dim = feature_dim + self.clinical_embed_dim
        else:
            classifier_input_dim = feature_dim

        self.cls_head = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, mri, pet, pet_confidence=None, clinical=None):
        mri_pooled, mri_spatial = self.mri_backbone(mri)
        pet_pooled, pet_spatial = self.pet_backbone(pet)

        fused, attn_weights = self.fusion(mri_spatial, pet_spatial, pet_confidence)

        if self.has_clinical_branch:
            if clinical is None:
                clinical = torch.zeros(
                    (mri.shape[0], self.clinical_dim),
                    device=mri.device,
                    dtype=mri.dtype,
                )
            clin_embed = self.clinical_encoder(clinical)
            fused = torch.cat([fused, clin_embed], dim=1)

        cls_logits = self.cls_head(fused)
        mmse_pred = self.reg_head(fused)

        return cls_logits, mmse_pred, attn_weights

    def enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


# ============================================================
# Temperature Scaling
# ============================================================

class TemperatureScaling(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, logits):
        return logits / self.temperature


# ============================================================
# Factory
# ============================================================

def build_classification_model(num_classes=3, feature_dim=512, dropout=0.3,
                                fusion_heads=8, fusion_layers=2, fusion_dropout=0.1,
                                clinical_dim=0, clinical_embed_dim=64):
    return ADClassificationModel(
        num_classes=num_classes,
        feature_dim=feature_dim,
        dropout=dropout,
        fusion_heads=fusion_heads,
        fusion_layers=fusion_layers,
        fusion_dropout=fusion_dropout,
        clinical_dim=clinical_dim,
        clinical_embed_dim=clinical_embed_dim,
    )
