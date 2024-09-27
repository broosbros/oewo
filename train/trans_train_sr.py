import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from pytorch_msssim import ms_ssim, ssim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from model.transformer_alt import (
    EnhancedVisionTransformerSR,
)  # Assuming this is the correct import


class CustomDataset(Dataset):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        lr_dir = os.path.join(self.base_dir, "LR")
        hr_dir = os.path.join(self.base_dir, "HR")

        if not all(os.path.exists(d) for d in [lr_dir, hr_dir]):
            raise FileNotFoundError(
                "LR and HR directories must exist in the base directory."
            )

        image_files = [
            f for f in os.listdir(lr_dir) if f.endswith((".jpg", ".png", ".jpeg"))
        ]

        for f in image_files:
            image_paths.append(
                {
                    "lr": os.path.join(lr_dir, f),
                    "hr": os.path.join(hr_dir, f),
                }
            )

        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        paths = self.image_paths[idx]
        lr_image = self.load_image(paths["lr"])
        hr_image = self.load_image(paths["hr"])
        return lr_image, hr_image

    @staticmethod
    def load_image(path):
        return transforms.ToTensor()(Image.open(path).convert("RGB"))


def save_image(tensor, path):
    tensor = tensor.cpu().detach()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    image = transforms.ToPILImage()(tensor)
    image.save(path)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.84):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        mse_loss = nn.MSELoss()(pred, target)
        ms_ssim_loss = 1 - ms_ssim(pred, target, data_range=1.0, size_average=True)
        return self.alpha * ms_ssim_loss + (1 - self.alpha) * mse_loss


def train(model, dataloader, num_epochs, criterion, optimizer, output_dir, log_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writer = SummaryWriter(log_dir)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (lr_img, hr_img) in enumerate(progress_bar):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            optimizer.zero_grad()

            sr_img = model(lr_img)

            loss = criterion(sr_img, hr_img)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate PSNR and SSIM
            with torch.no_grad():
                mse = nn.MSELoss()(sr_img, hr_img)
                psnr = 10 * torch.log10(1 / mse)
                epoch_psnr += psnr.item()
                epoch_ssim += ssim(
                    sr_img, hr_img, data_range=1.0, size_average=True
                ).item()

            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "PSNR": f"{psnr.item():.2f}",
                    "SSIM": f"{ssim(sr_img, hr_img, data_range=1.0, size_average=True).item():.4f}",
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
                    sr_img[0],
                    os.path.join(output_dir, f"sr_img_{epoch+1}_{batch_idx}.png"),
                )

            writer.add_scalar(
                "Loss/Total", loss.item(), epoch * len(dataloader) + batch_idx
            )
            writer.add_scalar(
                "Metrics/PSNR", psnr.item(), epoch * len(dataloader) + batch_idx
            )
            writer.add_scalar(
                "Metrics/SSIM",
                ssim(sr_img, hr_img, data_range=1.0, size_average=True).item(),
                epoch * len(dataloader) + batch_idx,
            )

        avg_loss = epoch_loss / len(dataloader)
        avg_psnr = epoch_psnr / len(dataloader)
        avg_ssim = epoch_ssim / len(dataloader)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Avg Loss: {avg_loss:.4f}, "
            f"Avg PSNR: {avg_psnr:.2f}, "
            f"Avg SSIM: {avg_ssim:.4f}"
        )

        writer.add_scalar("Epoch/Loss", avg_loss, epoch)
        writer.add_scalar("Epoch/PSNR", avg_psnr, epoch)
        writer.add_scalar("Epoch/SSIM", avg_ssim, epoch)

        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"model_checkpoint_epoch_{epoch+1}.pth"),
        )

    writer.close()


# Main execution
if __name__ == "__main__":
    input_dir = os.path.expanduser("~/dataset")
    output_dir = os.path.expanduser("~/outputs")
    log_dir = os.path.expanduser("~/logs")

    dataset = CustomDataset(input_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    model = EnhancedVisionTransformerSR(input_channels=3, scale_factor=2)
    criterion = CombinedLoss(alpha=0.84)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 50
    train(model, dataloader, num_epochs, criterion, optimizer, output_dir, log_dir)
