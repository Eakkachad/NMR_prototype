import torch
import torch.nn as nn

class SequenceAwareEncoder(nn.Module):
    def __init__(self, input_dim=4000, latent_dim=512):
        super().__init__()
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
        # x shape: [Batch_Size, 1, 4000]
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
