import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

try:
    from model.transformer import FrequencyDecompositionVisionTransformer
except ImportError:
    raise ImportError(
        "Please make sure to implement `FrequencyDecompositionVisionTransformer` in model/transformer.py"
    )

class CustomDataset(Dataset):
    def __init__(self, lr_images, hr_images, hr_low_freq_maps, hr_high_freq_maps):
        self.lr_images = lr_images
        self.hr_images = hr_images
        self.hr_low_freq_maps = hr_low_freq_maps
        self.hr_high_freq_maps = hr_high_freq_maps

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        return (
            self.lr_images[idx],
            self.hr_images[idx],
            self.hr_low_freq_maps[idx],
            self.hr_high_freq_maps[idx],
        )

def load_image(path):
    return transforms.ToTensor()(Image.open(path))

def load_dataset(base_dir):
    lr_images, hr_images, hr_low_freq_maps, hr_high_freq_maps = [], [], [], []

    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            lr_dir = os.path.join(subdir_path, 'LR')
            hr_dir = os.path.join(subdir_path, 'HR')
            hr_low_freq_dir = os.path.join(subdir_path, 'HR_low_freq')
            hr_high_freq_dir = os.path.join(subdir_path, 'HR_high_freq')

            def load_images_from_dir(directory, target_list):
                if os.path.isdir(directory):
                    image_files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))]
                    if image_files:
                        print(f"Loading {len(image_files)} images from {directory}.")
                        images = [load_image(os.path.join(directory, f)) for f in image_files]
                        target_list.extend(images)
                    else:
                        print(f"No images found in {directory}.")
                else:
                    print(f"Directory {directory} does not exist.")

            load_images_from_dir(lr_dir, lr_images)
            load_images_from_dir(hr_dir, hr_images)
            load_images_from_dir(hr_low_freq_dir, hr_low_freq_maps)
            load_images_from_dir(hr_high_freq_dir, hr_high_freq_maps)

    if not (lr_images and hr_images and hr_low_freq_maps and hr_high_freq_maps):
        raise ValueError("One or more of the image lists are empty. Please check the dataset directory.")

    lr_images = torch.stack(lr_images) if lr_images else torch.empty(0)
    hr_images = torch.stack(hr_images) if hr_images else torch.empty(0)
    hr_low_freq_maps = torch.stack(hr_low_freq_maps) if hr_low_freq_maps else torch.empty(0)
    hr_high_freq_maps = torch.stack(hr_high_freq_maps) if hr_high_freq_maps else torch.empty(0)

    return lr_images, hr_images, hr_low_freq_maps, hr_high_freq_maps

def save_image(tensor, path):
    img = transforms.ToPILImage()(tensor.cpu())
    img.save(path)

def inference(model, dataloader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (lr_img, _, _, _) in enumerate(dataloader):
            lr_img = lr_img.to(device)
            pred_low_freq, pred_high_freq = model(lr_img)

            for j in range(lr_img.size(0)):
                lr_img_np = lr_img[j].cpu().numpy().transpose(1, 2, 0)
                pred_img_np = (pred_low_freq[j] + pred_high_freq[j]).cpu().numpy().transpose(1, 2, 0)

                lr_img_pil = transforms.ToPILImage()(lr_img[j].cpu())
                pred_img_pil = transforms.ToPILImage()(pred_img_np)

                lr_img_pil.save(os.path.join(output_dir, f"lr_image_{i*4 + j}.png"))
                pred_img_pil.save(os.path.join(output_dir, f"pred_image_{i*4 + j}.png"))

if __name__ == "__main__":
    input_dir = 'data'
    lr_images, _, _, _ = load_dataset(input_dir)

    dataset = CustomDataset(lr_images, torch.empty_like(lr_images), torch.empty_like(lr_images), torch.empty_like(lr_images))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = FrequencyDecompositionVisionTransformer(input_channels=3, scale_factor=2)
    model.load_state_dict(torch.load("freq_decomp_transformer.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_dir = 'output'
    inference(model, dataloader, device, output_dir)
    print("Inference completed and images saved.")
