import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_L_16_Weights, vit_l_16  # Using a larger ViT model


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
        self.conv = nn.Conv2d(channels, channels * scale_factor**2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class FrequencyDecompositionVisionTransformer(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=1024, scale_factor=2):
        super(FrequencyDecompositionVisionTransformer, self).__init__()
        self.scale_factor = scale_factor
        self.hidden_dim = hidden_dim
        self.vit_output_size = 32  # ViT-L/16 outputs a 32x32 feature map for 512x512 input

        # Load pre-trained ViT-L/16 model
        self.vit = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)

        # Modify the patch embedding layer to handle 512x512 input
        self.vit.patch_embed = nn.Conv2d(3, 1024, kernel_size=16, stride=16)

        # Replace the classification head with a custom head
        self.vit.heads = nn.Sequential(
            nn.Linear(
                self.vit.hidden_dim,
                self.vit_output_size * self.vit_output_size * hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim * self.vit_output_size * self.vit_output_size, hidden_dim),
            nn.ReLU(),
        )

        # Freeze ViT parameters except for the new head and patch embedding
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.vit.heads.parameters():
            param.requires_grad = True
        for param in self.vit.patch_embed.parameters():
            param.requires_grad = True

        # Calculate the number of upsampling layers needed
        self.num_upsample_layers = self.calculate_upsample_layers()

        # Learnable upsampling layers
        self.upsample_layers = nn.ModuleList(
            [LearnableUpsample(hidden_dim, 2) for _ in range(self.num_upsample_layers)]
        )

        # Additional convolution layers after ViT
        self.extra_conv_layers = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Projection layers for low and high frequency
        self.low_freq_proj = nn.Conv2d(
            hidden_dim, input_channels, kernel_size=5, padding=2
        )
        self.high_freq_proj = nn.Conv2d(
            hidden_dim, input_channels, kernel_size=3, padding=1
        )

    def calculate_upsample_layers(self):
        target_size = 512 * self.scale_factor
        current_size = self.vit_output_size
        num_layers = 0
        while current_size < target_size:
            current_size *= 2
            num_layers += 1
        return num_layers

    def forward(self, x):
        target_size = 512 * self.scale_factor

        x = self.vit(x)
        x = x.reshape(
            x.size(0), self.hidden_dim, self.vit_output_size, self.vit_output_size
        )

        x = self.extra_conv_layers(x)

        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)

        if x.size(-1) != target_size:
            x = F.interpolate(
                x, size=(target_size, target_size), mode="bilinear", align_corners=False
            )

        # Project to low and high frequency components
        low_freq = self.low_freq_proj(x)
        high_freq = self.high_freq_proj(x)

        return low_freq, high_freq


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for scale_factor in [2, 3, 4]:
        model = FrequencyDecompositionVisionTransformer(
            input_channels=3, scale_factor=scale_factor
        ).to(device)
        x = torch.randn(1, 3, 512, 512).to(device)  # 512x512 input

        with torch.no_grad():
            low_freq, high_freq = model(x)

        print(f"\nScale factor: {scale_factor}")
        print(f"Input shape: {x.shape}")
        print(f"Low frequency output shape: {low_freq.shape}")
        print(f"High frequency output shape: {high_freq.shape}")
