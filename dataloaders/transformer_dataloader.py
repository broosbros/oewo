import os

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


def process_images_in_batches(
    input_dir,
    output_dir,
    batch_size=10,
    scale_factor=2,
    target_size_hr=(448, 448),
    target_size_lr=(224, 224),
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        if os.path.isdir(subdir_path):
            hr_dir = os.path.join(output_dir, subdir, "HR")
            lr_dir = os.path.join(output_dir, subdir, "LR")
            hr_low_freq_dir = os.path.join(output_dir, subdir, "HR_low_freq")
            hr_high_freq_dir = os.path.join(output_dir, subdir, "HR_high_freq")
            lr_low_freq_dir = os.path.join(output_dir, subdir, "LR_low_freq")
            lr_high_freq_dir = os.path.join(output_dir, subdir, "LR_high_freq")

            for dir_path in [
                hr_dir,
                lr_dir,
                hr_low_freq_dir,
                hr_high_freq_dir,
                lr_low_freq_dir,
                lr_high_freq_dir,
            ]:
                os.makedirs(dir_path, exist_ok=True)

            filenames = [
                f
                for f in os.listdir(subdir_path)
                if f.endswith((".jpg", ".png", ".jpeg"))
            ]

            for i in range(0, len(filenames), batch_size):
                batch_filenames = filenames[i : i + batch_size]

                for filename in batch_filenames:
                    img_path = os.path.join(subdir_path, filename)
                    with Image.open(img_path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")

                        # Resize HR image
                        img_hr = img.resize(target_size_hr, Image.BICUBIC)
                        img_hr.save(os.path.join(hr_dir, filename))

                        # Create and save LR image
                        img_lr = img.resize(target_size_lr, Image.BICUBIC)
                        img_lr.save(os.path.join(lr_dir, filename))

                        # Create frequency maps
                        img_tensor_hr = transforms.ToTensor()(img_hr).to(device)
                        img_tensor_lr = transforms.ToTensor()(img_lr).to(device)

                        hr_low_freq, hr_high_freq = create_frequency_maps(img_tensor_hr)
                        lr_low_freq, lr_high_freq = create_frequency_maps(img_tensor_lr)

                        transforms.ToPILImage()(hr_low_freq.cpu()).save(
                            os.path.join(hr_low_freq_dir, filename)
                        )
                        transforms.ToPILImage()(hr_high_freq.cpu()).save(
                            os.path.join(hr_high_freq_dir, filename)
                        )
                        transforms.ToPILImage()(lr_low_freq.cpu()).save(
                            os.path.join(lr_low_freq_dir, filename)
                        )
                        transforms.ToPILImage()(lr_high_freq.cpu()).save(
                            os.path.join(lr_high_freq_dir, filename)
                        )

    print("Processing complete!")


class CustomDataset(Dataset):
    def __init__(self, base_dir, target_size_hr=(448, 448), target_size_lr=(224, 224)):
        self.base_dir = base_dir
        self.target_size_hr = target_size_hr
        self.target_size_lr = target_size_lr
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        file_list = []
        for subdir in os.listdir(self.base_dir):
            subdir_path = os.path.join(self.base_dir, subdir)
            if os.path.isdir(subdir_path):
                hr_dir = os.path.join(subdir_path, "HR")
                if os.path.isdir(hr_dir):
                    file_list.extend(
                        [
                            os.path.join(hr_dir, f)
                            for f in os.listdir(hr_dir)
                            if f.endswith((".jpg", ".png", ".jpeg"))
                        ]
                    )
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        hr_path = self.file_list[idx]
        lr_path = hr_path.replace("HR", "LR")
        hr_low_freq_path = hr_path.replace("HR", "HR_low_freq")
        hr_high_freq_path = hr_path.replace("HR", "HR_high_freq")
        lr_low_freq_path = hr_path.replace("HR", "LR_low_freq")
        lr_high_freq_path = hr_path.replace("HR", "LR_high_freq")

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
    img = img.resize(target_size, Image.BICUBIC)
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
    input_dir = os.path.expanduser("/scope-workspaceuser3/gt_db")
    output_dir =  os.path.expanduser("/scope-workspaceuser3/gt_db_processed")

    process_images_in_batches(input_dir, output_dir, batch_size=4)

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
