import torch
import torch.nn as nn
import os

class PretrainedResNetEncoder(nn.Module):
    def __init__(self, pretrained_model_path, output_dim=512, freeze_backbone=True):
        super().__init__()
        # Load the pre-trained Net1D model
        self.backbone = torch.load(pretrained_model_path, map_location="cpu", weights_only=False)
        
        # Freeze backbone parameters
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
        # The output dimension of backbone's last stage (before dense layer) is 1024
        # We project it to 512 dimensions
        self.projection_head = nn.Linear(1024, output_dim)
        
    def forward(self, x):
        # x shape: [Batch_Size, 1, 4000] or [Batch_Size, 4000]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out = x
        
        # first conv
        out = self.backbone.first_conv(out)
        if getattr(self.backbone, 'use_bn', False):
            out = self.backbone.first_bn(out)
        out = self.backbone.first_activation(out)
        
        # stages
        for i_stage in range(self.backbone.n_stages):
            net = self.backbone.stage_list[i_stage]
            out = net(out)
            
        # Global average pooling
        out = out.mean(-1) # shape: (batch, 1024)
        
        # Projection to 512 dims
        out = self.projection_head(out) # shape: (batch, 512)
        return out


class SequenceAwareEncoder(nn.Module):
    def __init__(self, input_dim=4000, latent_dim=512):
        super().__init__()
        # Search for a pre-trained model file in common paths
        potential_paths = [
            "AGENT/04_DATA/scripts/model.pth",
            "model.pth",
            "BDI-flow-test/resnet1d/trained_model/model.pth",
            "../resnet1d/trained_model/model.pth"
        ]
        self.pretrained_encoder = None
        for path in potential_paths:
            if os.path.exists(path):
                try:
                    print(f"[INFO] Pre-trained weights found at '{path}'. Loading...")
                    self.pretrained_encoder = PretrainedResNetEncoder(path, output_dim=latent_dim, freeze_backbone=True)
                    break
                except Exception as e:
                    print(f"[WARN] Failed to load pre-trained weights from '{path}': {str(e)}")
                    self.pretrained_encoder = None
                    
        if self.pretrained_encoder is None:
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
                nn.GroupNorm(4, 16),
                nn.GELU(),
                nn.Conv1d(16, 64, kernel_size=7, stride=2, padding=3),
                nn.GroupNorm(16, 64),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(128)
            )
            self.fc_latent = nn.Linear(64 * 128, latent_dim)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        if self.pretrained_encoder is not None:
            return self.pretrained_encoder(x)
            
        # Fallback to native Conv1D
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        latent = self.fc_latent(features)
        return latent

class SpectrumDecoder(nn.Module):
    def __init__(self, latent_dim=512, output_dim=4000):
        super().__init__()
        self.fc_expand = nn.Linear(latent_dim, 64 * 128)
        self.reconstruct = nn.Sequential(
            nn.ConvTranspose1d(64, 16, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(16, 1, kernel_size=15, stride=2, padding=7, output_padding=1)
        )
        # Linear adjustment layer to match exact output resolution
        self.out_layer = nn.Linear(512, output_dim)

    def forward(self, latent):
        x = self.fc_expand(latent)
        x = x.view(x.size(0), 64, 128)
        x = self.reconstruct(x)
        x = self.out_layer(x)
        return x # Output shape: [Batch_Size, 1, 4000]

class LocalizedPatchEBM(nn.Module):
    def __init__(self, num_points=4000):
        super().__init__()
        # Small networks optimized to calculate local structural energy criteria
        # 4000 // 3 = 1333
        self.patch_net = nn.Sequential(
            nn.Linear(1333, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, aligned_spectrum):
        """
        Slices reconstructed 4000-point vector into 3 distinct operational zones:
        Aliphatic (0.5-3.0 ppm), Carbohydrate (3.0-5.5 ppm), and Aromatic (5.5-9.0 ppm).
        """
        if len(aligned_spectrum.shape) == 2:
            aligned_spectrum = aligned_spectrum.unsqueeze(1)
        # Assumes uniform ppm slicing for structural simplification
        patch_size = aligned_spectrum.size(2) // 3
        patch1 = aligned_spectrum[:, :, 0:patch_size].squeeze(1)
        patch2 = aligned_spectrum[:, :, patch_size:2*patch_size].squeeze(1)
        patch3 = aligned_spectrum[:, :, 2*patch_size:3*patch_size].squeeze(1)
        
        energy1 = self.patch_net(patch1)
        energy2 = self.patch_net(patch2)
        energy3 = self.patch_net(patch3)
        
        # Mathematical Formulation: Global Energy = Sum of weighted localized energy patches
        global_energy = 0.4 * energy1 + 0.4 * energy2 + 0.2 * energy3
        return global_energy
