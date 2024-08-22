# Setting Up a Virtual Python Environment with Poetry

This guide will walk you through setting up a virtual Python environment using Poetry, a dependency manager for this projects.

## Prerequisites

Before you begin, make sure you have the following installed:
- Python 3.12 or greater
- Poetry (You can install Poetry using the instructions from [here](https://python-poetry.org/docs/#installation))

## Steps to Setup Virtual Environment and Install Dependencies

1. **Clone the repository:**

   ```bash
   git clone https://github.ecodesamsung.com/viraj-shah2021/Guided-Super-Resolution-VIT.git
   cd Guided-Super-Resolution-VIT
   ```

2. **Create a virtual Environment with poetry:**

   ```bash
   poetry env use python3.12
   ```

3. **Install dependencies:**

   ```bash
   poetry install
   ```

**Additionally, you can activate the environment separately if needed using `poetry shell`**

## Folder structure

```
root
--- model
    transformer.py
    unet.py
    unet_cuda.py
--- train
    transformer_train.py
```

## File descriptions

### Frequency Learnable Transformer (model/transformer.py)

**PositionalEncoding2D**: This class adds positional encodings to the input tensor. Positional encodings are used to give the model information about the position of pixels in the image. This is particularly useful in models where the sequence of input data (like in transformers) matters, but the input does not inherently carry positional information (like images).

**LearnableUpsample**: This class performs upsampling on the input tensor. Upsampling is the process of increasing the spatial resolution (height and width) of the input images. It uses a series of transposed convolutional layers (sometimes called deconvolutional layers) to increase the resolution of the input tensor. This class is designed to be learnable, meaning it can optimize its upsampling filters during the training process.

**FrequencyDecompositionTransformer**: This class represents a neural network model that combines the concepts of transformers with frequency decomposition. It first embeds the input image using a convolutional layer, applies positional encoding, and then processes the data through a transformer encoder. The output of the transformer is then upsampled using the LearnableUpsample class. Finally, it separates the output into low and high-frequency components using two different convolutional layers. This model is designed for tasks that benefit from analyzing images in both spatial and frequency domains, potentially improving performance on tasks like image super-resolution or segmentation.

### Unet (model/unet.py)

**PositionalEncoding**
Implements a positional encoding layer that adds information about the position of items in a sequence. This is particularly useful in models where the sequence order matters but is not inherently captured, such as in transformer models.
- Inputs: A tensor representing noise levels.
- Outputs: A tensor of the same shape as the input, enriched with positional information through a combination of sine and cosine functions.


**FeatureWiseAffine**
Applies a feature-wise affine transformation to the input tensor. This can be used to modulate features individually across channels.
- Inputs: An input tensor and a noise embedding tensor.
- Outputs: The transformed tensor, where each feature/channel can be scaled and shifted based on the noise embedding.
 - Key Operations: Depending on the use_affine_level flag, it either adds noise directly to the input or applies a scaling (gamma) and shifting (beta) operation to each feature.

**ResSE**
Implements a Residual Squeeze-and-Excitation (ResSE) block, which enhances the representational capacity of a network by adaptively recalibrating channel-wise feature responses.
- Inputs: An input tensor.
- Outputs: The transformed tensor, where channel-wise features are adaptively recalibrated based on global information.
- Key Operations: Performs global average pooling to squeeze global spatial information into a channel descriptor, then uses this descriptor to excite specific channels adaptively through a simple gating mechanism with a sigmoid activation.