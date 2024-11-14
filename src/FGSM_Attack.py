# coding: UTF-8
import torch

# FGSM Definition
def FGSM(MODEL, images, labels, epsilon=0.001, num_steps=None):
    # Set model to eval mode
    model = MODEL.model
    model.eval()

    # Criterion and optimizer
    criterion = MODEL.criterion
    optimizer = MODEL.optimizer
    
    # Move the data to device
    images = images.to(MODEL.device)
    labels = labels.to(MODEL.device)
    images.requires_grad = True 

    # Forwarding
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backwarding
    optimizer.zero_grad()
    loss.backward()
    
    # Get perturbed images
    data_grad = images.grad.data
    perturbed_images = images.data + epsilon * data_grad.sign()
    return perturbed_images