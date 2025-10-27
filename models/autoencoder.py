# models/autoencoder.py
"""
Convolutional Autoencoder for 28x28 grayscale inputs (MNIST).

Usage:
    from models.autoencoder import Autoencoder
    model = Autoencoder(latent_dim=64)

API:
    - encode(x) -> z
    - decode(z) -> x_recon (in [0,1] via Sigmoid)
    - forward(x, return_latent=False) -> x_recon or (x_recon, z)
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AEConfig:
    in_channels: int = 1
    img_size: int = 28
    latent_dim: int = 64
    base_channels: int = 32  # width of the first conv block


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 64, in_channels: int = 1):
        super().__init__()
        c = AEConfig(in_channels=in_channels, latent_dim=latent_dim)

        # ----- Encoder -----
        self.enc_conv1 = nn.Conv2d(c.in_channels, c.base_channels, kernel_size=3, padding=1)  # -> [B, 32, 28, 28]
        self.enc_conv2 = nn.Conv2d(c.base_channels, c.base_channels * 2, kernel_size=3, padding=1)  # -> [B, 64, 14, 14]
        self.pool = nn.MaxPool2d(2)  # 28->14->7

        # After two pools: [B, 64, 7, 7] = 3136 features
        self.enc_flat_dim = (c.base_channels * 2) * 7 * 7
        self.fc_mu = nn.Linear(self.enc_flat_dim, c.latent_dim)

        # ----- Decoder -----
        self.fc_dec = nn.Linear(c.latent_dim, self.enc_flat_dim)
        self.dec_deconv1 = nn.ConvTranspose2d(c.base_channels * 2, c.base_channels, kernel_size=2, stride=2)  # 7->14
        self.dec_deconv2 = nn.ConvTranspose2d(c.base_channels, c.base_channels // 2, kernel_size=2, stride=2)  # 14->28
        self.dec_conv_out = nn.Conv2d(c.base_channels // 2, c.in_channels, kernel_size=3, padding=1)

        self._init_weights()

    # --- Public API ---
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.enc_conv1(x))           # [B, 32, 28, 28]
        h = self.pool(F.relu(self.enc_conv2(h)))  # [B, 64, 14, 14] -> pool -> [B, 64, 7, 7]
        h = self.pool(h)
        h = h.view(h.size(0), -1)               # [B, 3136]
        z = self.fc_mu(h)                       # [B, latent_dim]
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc_dec(z))                                  # [B, 3136]
        h = h.view(h.size(0), 64, 7, 7)                              # [B, 64, 7, 7]
        h = F.relu(self.dec_deconv1(h))                              # [B, 32, 14, 14]
        h = F.relu(self.dec_deconv2(h))                              # [B, 16, 28, 28]
        x_recon = torch.sigmoid(self.dec_conv_out(h))                # [B, 1, 28, 28] in [0,1]
        return x_recon

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        z = self.encode(x)
        x_recon = self.decode(z)
        return (x_recon, z) if return_latent else x_recon

    # --- Utils ---
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Conv2d) and m is self.dec_conv_out:
                # Output conv followed by sigmoid: use smaller init to avoid saturation
                nn.init.xavier_uniform_(m.weight, gain=1.0)
