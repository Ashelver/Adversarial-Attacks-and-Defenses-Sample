import torch
import matplotlib.pyplot as plt
from SVHN_ViT_Train import SVHN_ViTClassifier

def add_gaussian_noise_to_edges(image, noise_std=0.1):
    """
    Add Gaussian noise to the 2-pixel wide edge of each 16x16 patch in the input image.
    The corners of the patches will not receive double noise.
    
    Parameters:
    - image (torch.Tensor): Input image with shape (batch_size, channels, height, width).
    - noise_std (float): Standard deviation of the Gaussian noise to be added to the edges.
    
    Returns:
    - torch.Tensor: Tensor with noise added to the edges of each 16x16 patch.
    """
    batch_size, channels, height, width = image.shape



    # Ensure the image is on the correct device (GPU or CPU)
    device = image.device

    # Generate Gaussian noise
    noise = torch.normal(mean=0.0, std=noise_std, size=image.shape).to(device)

    tensor = image.clone().to(device)
    
    # Process each 16x16 patch
    for i in range(0, height, 16):
        for j in range(0, width, 16):
            # Define the 16x16 patch
            patch = tensor[:, :, i:i+16, j:j+16]

            # Generate Gaussian noise for edges (2 pixels wide)
            top_edge_noise = noise[:, :, i:i+2, j:j+16]   # Top edge
            bottom_edge_noise = noise[:, :, i+14:i+16, j:j+16]  # Bottom edge
            left_edge_noise = noise[:, :, i:i+16, j:j+2]   # Left edge
            right_edge_noise = noise[:, :, i:i+16, j+14:j+16]  # Right edge

            # Add Gaussian noise to the 2-pixel wide edges (excluding corners)
            patch[:, :, 0:2, :] += top_edge_noise[:, :, 0:2, :]
            patch[:, :, -2:, :] += bottom_edge_noise[:, :, -2:, :]
            patch[:, :, :, 0:2] += left_edge_noise[:, :, :, 0:2]
            patch[:, :, :, -2:] += right_edge_noise[:, :, :, -2:]

            # Correct the corners where noise has been added twice
            patch[:, :, 0:2, 0:2] -= top_edge_noise[:, :, 0:2, 0:2]  # Top-left corner
            patch[:, :, 0:2, -2:] -= top_edge_noise[:, :, 0:2, -2:]  # Top-right corner
            patch[:, :, -2:, 0:2] -= bottom_edge_noise[:, :, -2:, 0:2]  # Bottom-left corner
            patch[:, :, -2:, -2:] -= bottom_edge_noise[:, :, -2:, -2:]  # Bottom-right corner

            # Replace the original patch with the modified patch
            tensor[:, :, i:i+16, j:j+16] = patch

    return tensor


def get_examples(MODEL):
    test_loader = MODEL.test_loader
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    noisy_images = add_gaussian_noise_to_edges(images)
    
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(3):
        label = labels[i]
        image = images[i]
        noisy_image = noisy_images[i]

        image = MODEL.inv_normalize(image)  # Anti-normalize
        noisy_image = MODEL.inv_normalize(noisy_image).cpu()
        # Convert image formats
        original_image = image.permute(1, 2, 0)  # From (C, H, W) to (H, W, C)
        noisy_image = noisy_image.permute(1, 2, 0)

        # Original
        axes[i].imshow(original_image)
        axes[i].set_title(f"Original Image (Label: {label.item()})")
        axes[i].axis('off')

        axes[i + 3].imshow(noisy_image.clamp(0, 255))
        axes[i + 3].set_title(f"Noisy (Label: {label.item()})")
        axes[i + 3].axis('off')

    fig.suptitle(f"Adding Edges noise")
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    classifier = SVHN_ViTClassifier(batch_size=64, lr=3e-4, epochs=30)
    classifier.load_model() # path to your model
    get_examples(classifier)