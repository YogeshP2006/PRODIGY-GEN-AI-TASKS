# Stable Diffusion Image Generation

This notebook demonstrates how to use the Stable Diffusion model to generate images from text prompts. It covers:

*   **Environment Setup**: Verifying PyTorch and CUDA availability, and installing necessary libraries like `diffusers`, `transformers`, and `accelerate`.
*   **Model Loading**: Loading a pre-trained Stable Diffusion model (v1-5) from Hugging Face onto a GPU.
*   **Image Generation**: Generating images using various text prompts, adjusting inference steps, and utilizing `torch.Generator` for reproducible results.

## How to Run

1.  **Install Dependencies**: Run the cell containing `!pip install -q diffusers transformers accelerate safetensors`.
2.  **Load Model**: Execute the cells to load the Stable Diffusion pipeline.
3.  **Generate Images**: Modify the `prompt` variable in the image generation cells and run them to create new images.
4.  **Reproducibility**: Use `torch.Generator` with a fixed seed to get the same image output for a given prompt and parameters.

## Libraries Used

*   `torch`: PyTorch deep learning framework.
*   `diffusers`: Hugging Face library for state-of-art diffusion models.
*   `transformers`: Hugging Face library for pre-trained models.
*   `accelerate`: Hugging Face library for easily running PyTorch models on different hardware.
