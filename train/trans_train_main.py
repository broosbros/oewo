import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from pytorch_msssim import ms_ssim, ssim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.transformer_alt import FrequencyDecompositionVisionTransformer


class CustomDataset(Dataset):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        lr_dir = os.path.join(self.base_dir, "LR")
        hr_dir = os.path.join(self.base_dir, "HR")
        hr_low_freq_dir = os.path.join(self.base_dir, "HR_low_freq")
        hr_high_freq_dir = os.path.join(self.base_dir, "HR_high_freq")

        if not all(
            os.path.exists(d)
            for d in [lr_dir, hr_dir, hr_low_freq_dir, hr_high_freq_dir]
        ):
            raise FileNotFoundError(
                "All required directories must exist in the base directory."
            )

        image_files = [
            f for f in os.listdir(lr_dir) if f.endswith((".jpg", ".png", ".jpeg"))
        ]

        for f in image_files:
            image_paths.append(
                {
                    "lr": os.path.join(lr_dir, f),
                    "hr": os.path.join(hr_dir, f),
                    "hr_low_freq": os.path.join(hr_low_freq_dir, f),
                    "hr_high_freq": os.path.join(hr_high_freq_dir, f),
                }
            )

        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        paths = self.image_paths[idx]
        lr_image = self.load_image(paths["lr"])
        hr_image = self.load_image(paths["hr"])
        hr_low_freq = self.load_image(paths["hr_low_freq"])
        hr_high_freq = self.load_image(paths["hr_high_freq"])
        return lr_image, hr_image, hr_low_freq, hr_high_freq

    @staticmethod
    def load_image(path):
        return transforms.ToTensor()(Image.open(path).convert("RGB"))


def save_image(tensor, path):
    tensor = tensor.cpu().detach()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    image = transforms.ToPILImage()(tensor)
    image.save(path)


class FrequencyAwareLoss(nn.Module):
    def __init__(self, alpha=0.84, beta=0.5):
        super(FrequencyAwareLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-8  # Small constant to avoid division by zero

    def forward(
        self, pred, target, pred_low_freq, pred_high_freq, gt_low_freq, gt_high_freq
    ):
        # MS-SSIM loss for overall prediction
        ms_ssim_loss = 1 - ms_ssim(pred, target, data_range=1.0, size_average=True)

        # L1 loss for low and high frequency components
        l1_low_freq = F.l1_loss(pred_low_freq, gt_low_freq)
        l1_high_freq = F.l1_loss(pred_high_freq, gt_high_freq)

        # Normalize losses
        total_loss = ms_ssim_loss + l1_low_freq + l1_high_freq
        ms_ssim_loss_norm = ms_ssim_loss / (total_loss + self.epsilon)
        l1_low_freq_norm = l1_low_freq / (total_loss + self.epsilon)
        l1_high_freq_norm = l1_high_freq / (total_loss + self.epsilon)

        # Combine normalized losses
        combined_loss = self.alpha * ms_ssim_loss_norm + (1 - self.alpha) * (
            self.beta * l1_low_freq_norm + (1 - self.beta) * l1_high_freq_norm
        )

        # Scale the combined loss to maintain the original magnitude
        scaled_loss = combined_loss * total_loss.detach()

        return scaled_loss, ms_ssim_loss, l1_low_freq, l1_high_freq


def train(model, dataloader, num_epochs, criterion, optimizer, output_dir, log_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writer = SummaryWriter(log_dir)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ms_ssim_loss = 0.0
        epoch_l1_low_freq = 0.0
        epoch_l1_high_freq = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (lr_img, hr_img, gt_low_freq, gt_high_freq) in enumerate(
            progress_bar
        ):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            gt_low_freq, gt_high_freq = gt_low_freq.to(device), gt_high_freq.to(device)

            optimizer.zero_grad()

            pred_low_freq, pred_high_freq = model(lr_img)
            pred_sr = pred_low_freq + pred_high_freq

            loss, ms_ssim_loss, l1_low_freq, l1_high_freq = criterion(
                pred_sr,
                hr_img,
                pred_low_freq,
                pred_high_freq,
                gt_low_freq,
                gt_high_freq,
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_ms_ssim_loss += ms_ssim_loss.item()
            epoch_l1_low_freq += l1_low_freq.item()
            epoch_l1_high_freq += l1_high_freq.item()

            # Calculate PSNR and SSIM
            with torch.no_grad():
                mse = F.mse_loss(pred_sr, hr_img)
                psnr = 10 * torch.log10(1 / mse)
                epoch_psnr += psnr.item()
                epoch_ssim += ssim(
                    pred_sr, hr_img, data_range=1.0, size_average=True
                ).item()

            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "PSNR": f"{psnr.item():.2f}",
                    "SSIM": f"{ssim(pred_sr, hr_img, data_range=1.0, size_average=True).item():.4f}",
                }
            )

            if batch_idx % 10 == 0:
                save_image(
                    lr_img[0],
                    os.path.join(output_dir, f"lr_img_{epoch+1}_{batch_idx}.png"),
                )
                save_image(
                    hr_img[0],
                    os.path.join(output_dir, f"hr_img_{epoch+1}_{batch_idx}.png"),
                )
                save_image(
                    pred_sr[0],
                    os.path.join(output_dir, f"pred_sr_{epoch+1}_{batch_idx}.png"),
                )
                save_image(
                    pred_low_freq[0],
                    os.path.join(
                        output_dir, f"pred_low_freq_{epoch+1}_{batch_idx}.png"
                    ),
                )
                save_image(
                    pred_high_freq[0],
                    os.path.join(
                        output_dir, f"pred_high_freq_{epoch+1}_{batch_idx}.png"
                    ),
                )
                save_image(
                    gt_low_freq[0],
                    os.path.join(output_dir, f"gt_low_freq_{epoch+1}_{batch_idx}.png"),
                )
                save_image(
                    gt_high_freq[0],
                    os.path.join(output_dir, f"gt_high_freq_{epoch+1}_{batch_idx}.png"),
                )

            writer.add_scalar(
                "Loss/Total", loss.item(), epoch * len(dataloader) + batch_idx
            )
            writer.add_scalar(
                "Loss/MS-SSIM", ms_ssim_loss.item(), epoch * len(dataloader) + batch_idx
            )
            writer.add_scalar(
                "Loss/L1_Low_Freq",
                l1_low_freq.item(),
                epoch * len(dataloader) + batch_idx,
            )
            writer.add_scalar(
                "Loss/L1_High_Freq",
                l1_high_freq.item(),
                epoch * len(dataloader) + batch_idx,
            )
            writer.add_scalar(
                "Metrics/PSNR", psnr.item(), epoch * len(dataloader) + batch_idx
            )
            writer.add_scalar(
                "Metrics/SSIM",
                ssim(pred_sr, hr_img, data_range=1.0, size_average=True).item(),
                epoch * len(dataloader) + batch_idx,
            )

        avg_loss = epoch_loss / len(dataloader)
        avg_ms_ssim_loss = epoch_ms_ssim_loss / len(dataloader)
        avg_l1_low_freq = epoch_l1_low_freq / len(dataloader)
        avg_l1_high_freq = epoch_l1_high_freq / len(dataloader)
        avg_psnr = epoch_psnr / len(dataloader)
        avg_ssim = epoch_ssim / len(dataloader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Avg Loss: {avg_loss:.4f}, "
            f"Avg MS-SSIM Loss: {avg_ms_ssim_loss:.4f}, "
            f"Avg L1 Low Freq: {avg_l1_low_freq:.4f}, "
            f"Avg L1 High Freq: {avg_l1_high_freq:.4f}, "
            f"Avg PSNR: {avg_psnr:.2f}, "
            f"Avg SSIM: {avg_ssim:.4f}"
        )

        writer.add_scalar("Epoch/Loss", avg_loss, epoch)
        writer.add_scalar("Epoch/MS-SSIM Loss", avg_ms_ssim_loss, epoch)
        writer.add_scalar("Epoch/L1 Low Freq", avg_l1_low_freq, epoch)
        writer.add_scalar("Epoch/L1 High Freq", avg_l1_high_freq, epoch)
        writer.add_scalar("Epoch/PSNR", avg_psnr, epoch)
        writer.add_scalar("Epoch/SSIM", avg_ssim, epoch)

        # Save model checkpoint
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"model_checkpoint_epoch_{epoch+1}.pth"),
        )

    writer.close()


# Main execution
if __name__ == "__main__":
    input_dir = os.path.expanduser("/scope-workspaceuser3/processed_ffhq_small")
    output_dir = os.path.expanduser("/scope-workspaceuser3/outputs_small")
    log_dir = os.path.expanduser("/scope-workspaceuser3/logs_small")

    dataset = CustomDataset(input_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    model = FrequencyDecompositionVisionTransformer(input_channels=3, scale_factor=2)
    criterion = FrequencyAwareLoss(alpha=0.84, beta=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 50
    train(model, dataloader, num_epochs, criterion, optimizer, output_dir, log_dir)
