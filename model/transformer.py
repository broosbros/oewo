import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super(PositionalEncoding2D, self).__init__()
        self.channels = channels

    def forward(self, tensor):
        batch_size, ch, height, width = tensor.shape
        pos_x = torch.arange(width, device=tensor.device).float()
        pos_y = torch.arange(height, device=tensor.device).float()

        dim_t = torch.arange(0, self.channels, step=2, device=tensor.device).float()
        dim_t = 10000 ** (dim_t / self.channels)

        pos_x = pos_x.unsqueeze(1) / dim_t
        pos_y = pos_y.unsqueeze(1) / dim_t

        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(1)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(1)

        pos = torch.cat(
            (
                pos_y.unsqueeze(1).repeat(1, width, 1),
                pos_x.unsqueeze(0).repeat(height, 1, 1),
            ),
            dim=-1,
        ).permute(2, 0, 1)
        pos = pos.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        return tensor + pos[:, :ch]


class LearnableUpsample(nn.Module):
    def __init__(self, channels, scale_factor):
        super(LearnableUpsample, self).__init__()
        self.conv = nn.Conv2d(channels, channels * scale_factor**2, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class FrequencyDecompositionVisionTransformer(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=768, scale_factor=2):
        super(FrequencyDecompositionVisionTransformer, self).__init__()
        self.scale_factor = scale_factor
        self.hidden_dim = hidden_dim

        # Load pre-trained ViT-B/16 model
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Replace the classification head with a custom head
        self.vit.heads = nn.Sequential(
            nn.Linear(self.vit.hidden_dim, 14 * 14 * hidden_dim), nn.ReLU()
        )

        # Freeze ViT parameters except for the new head
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.vit.heads.parameters():
            param.requires_grad = True

        # Learnable upsampling layers
        self.upsample1 = LearnableUpsample(hidden_dim, 2)
        self.upsample2 = LearnableUpsample(hidden_dim, 2)
        self.upsample3 = LearnableUpsample(hidden_dim, 2)
        self.upsample4 = LearnableUpsample(hidden_dim, 2)

        # Projection layers for low and high frequency
        self.low_freq_proj = nn.Conv2d(
            hidden_dim, input_channels, kernel_size=5, padding=2
        )
        self.high_freq_proj = nn.Conv2d(
            hidden_dim, input_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        x_upsampled = F.interpolate(
            x, size=(224, 224), mode="bilinear", align_corners=False
        )

        x = self.vit(x_upsampled)

        x = x.reshape(x.size(0), self.hidden_dim, 14, 14)

        x = self.upsample1(x)  # 14x14 -> 28x28
        x = self.upsample2(x)  # 28x28 -> 56x56
        x = self.upsample3(x)  # 56x56 -> 112x112
        x = self.upsample4(x)  # 112x112 -> 224x224
        x = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)

        low_freq = self.low_freq_proj(x)
        high_freq = self.high_freq_proj(x)

        return low_freq, high_freq


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for scale_factor in [2, 3, 4, 5, 6, 8]:
        model = FrequencyDecompositionVisionTransformer(
            input_channels=3, scale_factor=scale_factor
        ).to(device)
        x = torch.randn(1, 3, 224, 224).to(device)  # 224x224 input

        with torch.no_grad():
            low_freq, high_freq = model(x)

        print(f"\nScale factor: {scale_factor}")
        print(f"Input shape: {x.shape}")
        print(f"Low frequency output shape: {low_freq.shape}")
        print(f"High frequency output shape: {high_freq.shape}")
