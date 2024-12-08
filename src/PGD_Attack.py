# coding: UTF-8
import torch

# PGD Definition
def PGD(MODEL, images, labels, epsilon=0.001, num_steps=1):
    # Set model to eval mode
    model = MODEL.model
    model.eval()

    # Criterion and optimizer
    criterion = MODEL.criterion
    optimizer = MODEL.optimizer
    
    # Move the data to the device
    images = images.to(MODEL.device)
    labels = labels.to(MODEL.device)

    # Initialize perturbed images with the original images
    perturbed_images = images.clone().detach()
    perturbed_images.requires_grad = True 

    # PGD iterative steps
    for step in range(num_steps):
        # Forward pass
        outputs = model(perturbed_images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Calculate the gradient
        data_grad = perturbed_images.grad.data

        # Update perturbed images with a small step
        perturbed_images = perturbed_images.data + 0.005 * data_grad.sign()

        # Ensure perturbations are within the epsilon-ball
        delta = torch.clamp(perturbed_images - images, min=-epsilon, max=epsilon)
        perturbed_images = (images + delta).detach()
        perturbed_images.requires_grad = True

    return perturbed_images.detach()