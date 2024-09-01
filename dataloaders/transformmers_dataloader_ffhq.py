import os
import concurrent.futures
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from tqdm import tqdm
import time

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
    
    try:
        with Image.open(img_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            if img.size != target_size_hr:
                img_hr = img.resize(target_size_hr, Image.Resampling.BICUBIC)
            else:
                img_hr = img.copy()

            img_hr.save(os.path.join(output_dir, "HR", filename))

            img_lr = img_hr.resize(target_size_lr, Image.Resampling.BICUBIC)
            img_lr.save(os.path.join(output_dir, "LR", filename))

            img_tensor_hr = transforms.ToTensor()(img_hr).to(device)
            img_tensor_lr = transforms.ToTensor()(img_lr).to(device)

            hr_low_freq, hr_high_freq = create_frequency_maps(img_tensor_hr)
            lr_low_freq, lr_high_freq = create_frequency_maps(img_tensor_lr)

            transforms.ToPILImage()(hr_low_freq.cpu()).save(os.path.join(output_dir, "HR_low_freq", filename))
            transforms.ToPILImage()(hr_high_freq.cpu()).save(os.path.join(output_dir, "HR_high_freq", filename))
            transforms.ToPILImage()(lr_low_freq.cpu()).save(os.path.join(output_dir, "LR_low_freq", filename))
            transforms.ToPILImage()(lr_high_freq.cpu()).save(os.path.join(output_dir, "LR_high_freq", filename))

        return True
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return False

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

    filenames = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    total_files = len(filenames)
    
    args_list = [(filename, input_dir, output_dir, target_size_hr, target_size_lr) for filename in filenames]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_single_image, args_list), total=total_files, desc="Processing images"))
    
    successful = sum(results)
    print(f"Processing complete. Successfully processed {successful}/{total_files} images.")

class LargeDatasetIterator(IterableDataset):
    def __init__(self, base_dir, target_size_hr=(1024, 1024), target_size_lr=(512, 512)):
        self.base_dir = base_dir
        self.target_size_hr = target_size_hr
        self.target_size_lr = target_size_lr
        self.hr_dir = os.path.join(base_dir, "HR")
        self.file_list = [f for f in os.listdir(self.hr_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        self.total_files = len(self.file_list)

    def __iter__(self):
        for filename in self.file_list:
            hr_path = os.path.join(self.hr_dir, filename)
            lr_path = os.path.join(self.base_dir, "LR", filename)
            hr_low_freq_path = os.path.join(self.base_dir, "HR_low_freq", filename)
            hr_high_freq_path = os.path.join(self.base_dir, "HR_high_freq", filename)
            lr_low_freq_path = os.path.join(self.base_dir, "LR_low_freq", filename)
            lr_high_freq_path = os.path.join(self.base_dir, "LR_high_freq", filename)

            try:
                hr_image = load_image(hr_path, self.target_size_hr)
                lr_image = load_image(lr_path, self.target_size_lr)
                hr_low_freq_image = load_image(hr_low_freq_path, self.target_size_hr)
                hr_high_freq_image = load_image(hr_high_freq_path, self.target_size_hr)
                lr_low_freq_image = load_image(lr_low_freq_path, self.target_size_lr)
                lr_high_freq_image = load_image(lr_high_freq_path, self.target_size_lr)

                yield (lr_image, hr_image, hr_low_freq_image, hr_high_freq_image, lr_low_freq_image, lr_high_freq_image)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")

def load_image(path, target_size):
    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(target_size, Image.Resampling.BICUBIC)
        return transforms.ToTensor()(img).to(device)

def process_dataset(dataloader, total_batches):
    update_interval = max(1, total_batches // 100)  # Update progress about 100 times
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        print("Starting processing")

        if i % update_interval == 0:
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / (i + 1)) * total_batches
            estimated_remaining_time = estimated_total_time - elapsed_time
            
            print(f"Processed {i}/{total_batches} batches "
                  f"({i/total_batches:.2%}) - "
                  f"Elapsed: {elapsed_time:.2f}s, "
                  f"Estimated remaining: {estimated_remaining_time:.2f}s")

    total_time = time.time() - start_time
    print(f"Dataset processing complete! Total time: {total_time:.2f}s")

if __name__ == "__main__":
    input_dir = "~/scope-workspaceuser3/ffhq/images1024x1024"
    output_dir = "~/scope-workspaceuser3/processed_ffhq"

    process_images_multithreaded(input_dir, output_dir)

    dataset = LargeDatasetIterator(output_dir)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=25)


    process_dataset(dataloader, dataset.total_files)
