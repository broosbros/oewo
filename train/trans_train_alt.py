import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from pytorch_msssim import ssim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from model.transformer_alt import FrequencyDecompositionVisionTransformer
except ImportError:
    raise ImportError(
        "Please make sure to implement `FrequencyDecompositionVisionTransformer` in model/transformer.py"
    )


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

        if not (os.path.exists(lr_dir) and os.path.exists(hr_dir) and os.path.exists(hr_low_freq_dir) and os.path.exists(hr_high_freq_dir)):
            raise FileNotFoundError(
                "The directories 'LR', 'HR', 'HR_low_freq', and 'HR_high_freq' must exist in the base directory."
            )

        image_files = [f for f in os.listdir(lr_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

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


class LaplacianPyramid(nn.Module):
    def __init__(self, levels):
        super(LaplacianPyramid, self).__init__()
        self.levels = levels

    def forward(self, img):
        pyramids = [img]
        current_img = img
        for _ in range(self.levels - 1):
            down = F.avg_pool2d(current_img, kernel_size=2, stride=2)
            up = F.interpolate(down, scale_factor=2, mode='bilinear', align_corners=False)
            laplacian = current_img - up
            pyramids.append(laplacian)
            current_img = down
        pyramids.append(current_img)  # last level
        return pyramids


def laplacian_pyramid_loss(pred, target, levels=3):
    laplacian_pyramid = LaplacianPyramid(levels)
    pred_pyramids = laplacian_pyramid(pred)
    target_pyramids = laplacian_pyramid(target)
    loss = 0
    for pred_lap, target_lap in zip(pred_pyramids, target_pyramids):
        loss += F.mse_loss(pred_lap, target_lap)
    return loss


def save_image(tensor, path):
    image = transforms.ToPILImage()(tensor.cpu())
    image.save(path)

def fft_loss(pred, target):
    pred_fft = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    pred_phase = torch.angle(pred_fft)
    target_phase = torch.angle(target_fft)

    magnitude_loss = F.mse_loss(pred_mag, target_mag)
    phase_loss = F.mse_loss(pred_phase, target_phase)

    return magnitude_loss + phase_loss

def uncertainty_based_loss(pred_low_freq, pred_high_freq, gt_low_freq, gt_high_freq):
    uncertainty_low = torch.abs(pred_low_freq - gt_low_freq)
    uncertainty_high = torch.abs(pred_high_freq - gt_high_freq)

    mean_uncertainty_low = torch.mean(uncertainty_low, dim=(2, 3))
    mean_uncertainty_high = torch.mean(uncertainty_high, dim=(2, 3))

    var_uncertainty_low = torch.var(uncertainty_low, dim=(2, 3))
    var_uncertainty_high = torch.var(uncertainty_high, dim=(2, 3))

    combined_uncertainty_low = mean_uncertainty_low + torch.sqrt(var_uncertainty_low)
    combined_uncertainty_high = mean_uncertainty_high + torch.sqrt(var_uncertainty_high)

    gt_low_magnitude = torch.mean(torch.abs(gt_low_freq), dim=(2, 3))
    gt_high_magnitude = torch.mean(torch.abs(gt_high_freq), dim=(2, 3))
    total_magnitude = (
        gt_low_magnitude + gt_high_magnitude + 1e-6
    )  # Avoid division by zero
    weight_low = gt_low_magnitude / total_magnitude
    weight_high = gt_high_magnitude / total_magnitude

    weighted_uncertainty_low = combined_uncertainty_low * weight_low
    weighted_uncertainty_high = combined_uncertainty_high * weight_high

    total_uncertainty = torch.mean(weighted_uncertainty_low + weighted_uncertainty_high)

    kl_div_low = F.kl_div(
        F.log_softmax(pred_low_freq, dim=1),
        F.softmax(gt_low_freq, dim=1),
        reduction="batchmean",
    )
    kl_div_high = F.kl_div(
        F.log_softmax(pred_high_freq, dim=1),
        F.softmax(gt_high_freq, dim=1),
        reduction="batchmean",
    )

    epsilon = 1e-6
    uncertainty_loss = -torch.log(total_uncertainty + epsilon)
    kl_loss = kl_div_low + kl_div_high

    total_loss = uncertainty_loss + 0.5 * kl_loss

    return total_loss

def ssim_loss(pred, target):
    ssim_value = ssim(pred, target, data_range=1.0, size_average=True)
    return 1 - ssim_value
    
def train(model, dataloader, num_epochs, criterion, optimizer, output_dir, log_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writer = SummaryWriter(log_dir)

    for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (lr_img, hr_img, gt_low_freq, gt_high_freq) in tqdm(
            enumerate(dataloader), desc="Batch", unit="Batches"
        ):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)
            gt_low_freq = gt_low_freq.to(device)
            gt_high_freq = gt_high_freq.to(device)

            optimizer.zero_grad()

            pred_low_freq, pred_high_freq = model(lr_img)

            loss_sr = 1 - ssim_loss(pred_low_freq + pred_high_freq, hr_img)
            loss_low_freq = fft_loss(pred_low_freq, gt_low_freq)
            loss_high_freq = fft_loss(pred_high_freq, gt_high_freq)
            uncertainty_loss = uncertainty_based_loss(
                pred_low_freq, pred_high_freq, gt_low_freq, gt_high_freq
            )
            pyramid_loss = laplacian_pyramid_loss(
                pred_low_freq + pred_high_freq, hr_img
            )

            total_loss = loss_sr + loss_low_freq + loss_high_freq + uncertainty_loss + pyramid_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            batch_count += 1

            writer.add_scalar("Loss/SR Loss", loss_sr.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar("Loss/Low Freq Loss", loss_low_freq.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar("Loss/High Freq Loss", loss_high_freq.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar("Loss/Uncertainty Loss", uncertainty_loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar("Loss/Pyramid Loss", pyramid_loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar("Loss/Total Loss", total_loss.item(), epoch * len(dataloader) + batch_idx)

            if batch_idx % 10 == 0:
                for i in range(lr_img.size(0)):
                    save_image(lr_img[i], os.path.join(output_dir, f"lr_{epoch}_{batch_idx}_{i}.png"))
                    save_image(hr_img[i], os.path.join(output_dir, f"hr_{epoch}_{batch_idx}_{i}.png"))
                    save_image(pred_low_freq[i], os.path.join(output_dir, f"pred_low_freq_{epoch}_{batch_idx}_{i}.png"))
                    save_image(pred_high_freq[i], os.path.join(output_dir, f"pred_high_freq_{epoch}_{batch_idx}_{i}.png"))

        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")
        writer.add_scalar("Loss/Average Epoch Loss", avg_epoch_loss, epoch)

    writer.close()


def main():
	input_dir = os.path.expanduser("/scope-workspaceuser3/processed_ffhq")
	output_dir = os.path.expanduser("/scope-workspaceuser3/outputs")
	log_dir = os.path.expanduser("/scope-workspaceuser3/logs")
	dataset = CustomDataset(input_dir)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

	model = FrequencyDecompositionVisionTransformer(input_channels=3, scale_factor=2)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.0001)

	num_epochs = 2
	train(model, dataloader, num_epochs, criterion, optimizer, output_dir, log_dir)


if __name__ == "__main__":
    main()
