import multiprocessing
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from tqdm import tqdm


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

        return True
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return False


def process_images_multiprocess(
    input_dir,
    output_dir,
    num_processes=None,
    target_size_hr=(1024, 1024),
    target_size_lr=(512, 512),
    num_images=None,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir in ["HR", "LR"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    filenames = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if num_images is not None:
        filenames = filenames[:num_images]

    total_files = len(filenames)

    args_list = [
        (filename, input_dir, output_dir, target_size_hr, target_size_lr)
        for filename in filenames
    ]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_image, args_list),
                total=total_files,
                desc="Processing images",
            )
        )

    successful = sum(results)
    print(
        f"Processing complete. Successfully processed {successful}/{total_files} images."
    )


class LargeDatasetIterator(IterableDataset):
    def __init__(
        self,
        base_dir,
        target_size_hr=(1024, 1024),
        target_size_lr=(512, 512),
        num_images=None,
    ):
        self.base_dir = base_dir
        self.target_size_hr = target_size_hr
        self.target_size_lr = target_size_lr
        self.hr_dir = os.path.join(base_dir, "HR")
        self.file_list = [
            f
            for f in os.listdir(self.hr_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        if num_images is not None:
            self.file_list = self.file_list[:num_images]

        self.total_files = len(self.file_list)

    def __iter__(self):
        for filename in self.file_list:
            hr_path = os.path.join(self.hr_dir, filename)
            lr_path = os.path.join(self.base_dir, "LR", filename)

            try:
                hr_image = load_image(hr_path, self.target_size_hr)
                lr_image = load_image(lr_path, self.target_size_lr)

                yield (lr_image, hr_image)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")


def load_image(path, target_size):
    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(target_size, Image.Resampling.BICUBIC)
        return transforms.ToTensor()(img)


def process_dataset(dataloader, total_batches):
    update_interval = max(1, total_batches // 100)  # Update progress about 100 times
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        if i % update_interval == 0:
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / (i + 1)) * total_batches
            estimated_remaining_time = estimated_total_time - elapsed_time

            print(
                f"Processed {i}/{total_batches} batches "
                f"({i/total_batches:.2%}) - "
                f"Elapsed: {elapsed_time:.2f}s, "
                f"Estimated remaining: {estimated_remaining_time:.2f}s"
            )

    total_time = time.time() - start_time
    print(f"Dataset processing complete! Total time: {total_time:.2f}s")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    input_dir = os.path.expanduser("/scope-workspaceuser3/ffhq/images1024x1024")
    output_dir = os.path.expanduser("/scope-workspaceuser3/processed_ffhq")
    nuim_images = 500

    process_images_multiprocess(input_dir, output_dir, num_images=num_images)

    dataset = LargeDatasetIterator(output_dir, num_images=num_images)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1)

    process_dataset(dataloader, dataset.total_files)
