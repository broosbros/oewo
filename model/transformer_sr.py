import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_L_16_Weights, vit_l_16


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class LearnableUpsample(nn.Module):
    def __init__(self, channels, scale_factor):
        super(LearnableUpsample, self).__init__()
        self.conv = nn.Conv2d(
            channels, channels * scale_factor**2, kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return self.relu(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.conv_query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv_key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv_value = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.conv_query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.conv_key(x).view(batch_size, -1, H * W)
        value = self.conv_value(x).view(batch_size, -1, H * W)

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        return self.gamma * out + x


class EnhancedVisionTransformerSR(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=1024, scale_factor=2):
        super(EnhancedVisionTransformerSR, self).__init__()
        self.scale_factor = scale_factor
        self.hidden_dim = hidden_dim
        self.vit_output_size = (
            32  # ViT-L/16 outputs a 32x32 feature map for 512x512 input
        )

        # Initial feature extraction
        self.initial_conv = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.initial_residual = ResidualBlock(64)

        # Load pre-trained ViT-L/16 model
        self.vit = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)

        # Modify the patch embedding layer to handle LR input
        self.vit.patch_embed = nn.Conv2d(64, 1024, kernel_size=16, stride=16)

        # Replace the classification head with a custom head
        self.vit.heads = nn.Sequential(
            nn.Linear(
                self.vit.hidden_dim,
                self.vit_output_size * self.vit_output_size * hidden_dim,
            ),
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

        # Learnable upsampling layers with residual and attention blocks
        self.upsample_blocks = nn.ModuleList()
        for _ in range(self.num_upsample_layers):
            self.upsample_blocks.append(
                nn.Sequential(
                    LearnableUpsample(hidden_dim, 2),
                    ResidualBlock(hidden_dim),
                    AttentionBlock(hidden_dim),
                )
            )

        # Final convolution to produce the HR image
        self.final_conv = nn.Conv2d(
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
        target_size = x.size(-1) * self.scale_factor

        # Initial feature extraction
        x = self.initial_conv(x)
        x = self.initial_residual(x)

        # ViT processing
        x = self.vit(x)

        x = x.reshape(
            x.size(0), self.hidden_dim, self.vit_output_size, self.vit_output_size
        )

        # Upsampling with residual and attention blocks
        for upsample_block in self.upsample_blocks:
            x = upsample_block(x)

        if x.size(-1) != target_size:
            x = F.interpolate(
                x, size=(target_size, target_size), mode="bilinear", align_corners=False
            )

        # Final convolution to produce the HR image
        hr_output = self.final_conv(x)

        return hr_output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for scale_factor in [2, 3, 4]:
        model = EnhancedVisionTransformerSR(
            input_channels=3, scale_factor=scale_factor
        ).to(device)
        lr_size = 512 // scale_factor
        x = torch.randn(1, 3, lr_size, lr_size).to(device)  # LR input

        with torch.no_grad():
            hr_output = model(x)

        print(f"\nScale factor: {scale_factor}")
        print(f"Input shape (LR): {x.shape}")
        print(f"Output shape (HR): {hr_output.shape}")
