import os
import concurrent.futures
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_frequency_maps(img_tensor):
    img_array = img_tensor.cpu().numpy().transpose(1, 2, 0)
    low_freq = gaussian_filter(img_array, sigma=2)
    high_freq = img_array - low_freq
    low_freq_tensor = torch.tensor(low_freq.transpose(2, 0, 1)).to(device)
    high_freq_tensor = torch.tensor(high_freq.transpose(2, 0, 1)).to(device)
    return low_freq_tensor, high_freq_tensor

def process_single_image(args):
    filename, input_dir, output_dir, target_size_hr, target_size_lr = args
    img_path = os.path.join(input_dir, filename)
    
    with Image.open(img_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize HR image if it's not already 1024x1024
        if img.size != target_size_hr:
            img_hr = img.resize(target_size_hr, Image.Resampling.BICUBIC)
        else:
            img_hr = img

        img_hr.save(os.path.join(output_dir, "HR", filename))

        # Create and save LR image
        img_lr = img_hr.resize(target_size_lr, Image.Resampling.BICUBIC)
        img_lr.save(os.path.join(output_dir, "LR", filename))

        # Create frequency maps
        img_tensor_hr = transforms.ToTensor()(img_hr).to(device)
        img_tensor_lr = transforms.ToTensor()(img_lr).to(device)

        hr_low_freq, hr_high_freq = create_frequency_maps(img_tensor_hr)
        lr_low_freq, lr_high_freq = create_frequency_maps(img_tensor_lr)

        transforms.ToPILImage()(hr_low_freq.cpu()).save(
            os.path.join(output_dir, "HR_low_freq", filename)
        )
        transforms.ToPILImage()(hr_high_freq.cpu()).save(
            os.path.join(output_dir, "HR_high_freq", filename)
        )
        transforms.ToPILImage()(lr_low_freq.cpu()).save(
            os.path.join(output_dir, "LR_low_freq", filename)
        )
        transforms.ToPILImage()(lr_high_freq.cpu()).save(
            os.path.join(output_dir, "LR_high_freq", filename)
        )

def process_images_multithreaded(
    input_dir,
    output_dir,
    max_workers=None,
    target_size_hr=(1024, 1024),
    target_size_lr=(512, 512),
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir in ["HR", "LR", "HR_low_freq", "HR_high_freq", "LR_low_freq", "LR_high_freq"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    filenames = [
        f for f in os.listdir(input_dir) if f.endswith((".jpg", ".png", ".jpeg"))
    ]
    filenames.sort()  # Ensure files are processed in order

    args_list = [(filename, input_dir, output_dir, target_size_hr, target_size_lr) for filename in filenames]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(process_single_image, args_list))

    print("Processing complete!")

class CustomDataset(Dataset):
    def __init__(
        self, base_dir, target_size_hr=(1024, 1024), target_size_lr=(512, 512)
    ):
        self.base_dir = base_dir
        self.target_size_hr = target_size_hr
        self.target_size_lr = target_size_lr
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        hr_dir = os.path.join(self.base_dir, "HR")
        return [
            os.path.join(hr_dir, f)
            for f in sorted(os.listdir(hr_dir))
            if f.endswith((".jpg", ".png", ".jpeg"))
        ]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        hr_path = self.file_list[idx]
        filename = os.path.basename(hr_path)

        lr_path = os.path.join(self.base_dir, "LR", filename)
        hr_low_freq_path = os.path.join(self.base_dir, "HR_low_freq", filename)
        hr_high_freq_path = os.path.join(self.base_dir, "HR_high_freq", filename)
        lr_low_freq_path = os.path.join(self.base_dir, "LR_low_freq", filename)
        lr_high_freq_path = os.path.join(self.base_dir, "LR_high_freq", filename)

        hr_image = load_image(hr_path, self.target_size_hr)
        lr_image = load_image(lr_path, self.target_size_lr)
        hr_low_freq_image = load_image(hr_low_freq_path, self.target_size_hr)
        hr_high_freq_image = load_image(hr_high_freq_path, self.target_size_hr)
        lr_low_freq_image = load_image(lr_low_freq_path, self.target_size_lr)
        lr_high_freq_image = load_image(lr_high_freq_path, self.target_size_lr)

        return (
            lr_image,
            hr_image,
            hr_low_freq_image,
            hr_high_freq_image,
            lr_low_freq_image,
            lr_high_freq_image,
        )

def load_image(path, target_size):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.Resampling.BICUBIC)
    return transforms.ToTensor()(img).to(device)

def print_image_dimensions(
    lr_image, hr_image, hr_low_freq, hr_high_freq, lr_low_freq, lr_high_freq
):
    def get_dimensions(tensor):
        return tensor.shape[1:]

    lr_dims = get_dimensions(lr_image)
    hr_dims = get_dimensions(hr_image)
    hr_low_freq_dims = get_dimensions(hr_low_freq)
    hr_high_freq_dims = get_dimensions(hr_high_freq)
    lr_low_freq_dims = get_dimensions(lr_low_freq)
    lr_high_freq_dims = get_dimensions(lr_high_freq)

    print(f"LR Image dimensions: {lr_dims}")
    print(f"HR Image dimensions: {hr_dims}")
    print(f"HR Low Frequency dimensions: {hr_low_freq_dims}")
    print(f"HR High Frequency dimensions: {hr_high_freq_dims}")
    print(f"LR Low Frequency dimensions: {lr_low_freq_dims}")
    print(f"LR High Frequency dimensions: {lr_high_freq_dims}")

if __name__ == "__main__":
    input_dir = "data"
    output_dir = "data"

    process_images_multithreaded(input_dir, output_dir)

    dataset = CustomDataset(output_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        (
            lr_images,
            hr_images,
            hr_low_freq_images,
            hr_high_freq_images,
            lr_low_freq_images,
            lr_high_freq_images,
        ) = batch

        print_image_dimensions(
            lr_images[0],
            hr_images[0],
            hr_low_freq_images[0],
            hr_high_freq_images[0],
            lr_low_freq_images[0],
            lr_high_freq_images[0],
        )
        break
