import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from model.transformer import FrequencyDecompositionVisionTransformer


def load_image(path):
    """Load and preprocess the input image."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Ensure image size matches model input
            transforms.ToTensor(),
        ]
    )(Image.open(path)).unsqueeze(0)


def save_image(tensor, filename):
    """Save the tensor as an image file."""
    tensor = tensor.squeeze().cpu().clamp(0, 1)  # Ensure tensor values are in [0, 1]
    img = transforms.ToPILImage()(tensor)
    img.save(filename)


def plot_images(lr_img, low_freq, high_freq, combined):
    """Plot and save the comparison of images."""

    def convert_to_image(tensor):
        tensor = (
            tensor.squeeze().cpu().clamp(0, 1)
        )  # Ensure tensor values are in [0, 1]
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)  # Convert grayscale to RGB if needed
        return transforms.ToPILImage()(tensor)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs[0, 0].imshow(convert_to_image(lr_img))
    axs[0, 0].set_title("LR Input")
    axs[0, 1].imshow(convert_to_image(low_freq))
    axs[0, 1].set_title("Low Frequency")
    axs[1, 0].imshow(convert_to_image(high_freq))
    axs[1, 0].set_title("High Frequency")
    axs[1, 1].imshow(convert_to_image(combined))
    axs[1, 1].set_title("Combined Output")
    plt.tight_layout()
    plt.savefig("output_comparison.png")
    plt.close()


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = FrequencyDecompositionVisionTransformer(input_channels=3, scale_factor=2)
    model.load_state_dict(
        torch.load("freq_decomp_transformer.pth", map_location=device)
    )
    model.to(device)
    model.eval()

    # Load and preprocess the input image
    input_image_path = "data/s01/LR/01.jpg"
    lr_img = load_image(input_image_path).to(device)

    # Inference
    with torch.no_grad():
        low_freq, high_freq = model(lr_img)

    # Combine low and high frequency projections
    combined_output = low_freq + high_freq

    # Debug: Print tensor ranges
    print(f"LR Image Min: {lr_img.min()}, Max: {lr_img.max()}")
    print(f"Low Frequency Min: {low_freq.min()}, Max: {low_freq.max()}")
    print(f"High Frequency Min: {high_freq.min()}, Max: {high_freq.max()}")
    print(f"Combined Output Min: {combined_output.min()}, Max: {combined_output.max()}")

    # Save output images
    save_image(lr_img, "lr_input.png")
    save_image(low_freq, "low_freq_output.png")
    save_image(high_freq, "high_freq_output.png")
    save_image(combined_output, "combined_output.png")

    # Plot and save comparison
    plot_images(lr_img, low_freq, high_freq, combined_output)

    print("Inference completed. Output images saved.")


if __name__ == "__main__":
    main()
