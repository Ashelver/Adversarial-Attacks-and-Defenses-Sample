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


def FGSM_ONLY_GRADIENT(MODEL, images, labels, epsilon=None, num_steps=None):
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
    
    # Get grad images
    data_grad = images.grad.data
    return data_grad.sign()


def FGSM_GRADIENT_MASK_IMAGES(MODEL, images, labels, epsilon=None, num_steps=None):
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

    # Get the sign of the grad
    data_grad_sign = data_grad.sign()


    # 1. Remove positive signs
    images_with_sign1_as_0 = images.data.clone()
    images_with_sign1_as_0[data_grad_sign == 1] = 0

    # 2. Remove negative signs
    images_with_sign_neg1_as_0 = images.data.clone()
    images_with_sign_neg1_as_0[data_grad_sign == -1] = 0

    return images_with_sign1_as_0, images_with_sign_neg1_as_0 



def FGSM_GRADIENT_SENSITIVITY_IMAGES(MODEL, images, labels, epsilon=None, num_steps=None):
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

    # The sensitivity of the pixels (abs of gradients summed across channels)
    sensitivity = data_grad.abs().sum(dim=1)  # Sum across channels
    sensitivity_expanded = sensitivity.unsqueeze(1).repeat(1, 3, 1, 1)
    
    # For each image, set the largest half of the sensitivity values to 1
    for i in range(sensitivity_expanded.size(0)):  # Iterate over the batch
        sensitivity_image = sensitivity_expanded[i]  # Shape: (3, 224, 224)
        
        # Flatten the sensitivity_image
        flat_sensitivity = sensitivity_image.view(-1)  # Flatten to (3 * 224 * 224,)
        
        # Sort sensitivity values and find the threshold
        _, sorted_idx = torch.sort(flat_sensitivity, descending=True)
        threshold_idx = int(flat_sensitivity.size(0) / 4)  # Halfway point
        threshold = flat_sensitivity[sorted_idx[threshold_idx]]

        # Set the largest half to 1 in the flattened version
        flat_sensitivity[flat_sensitivity >= threshold] = 1
        
        # Reshape it back to the original image shape
        sensitivity_image = flat_sensitivity.view(sensitivity_image.shape)  # Reshape back to (3, 224, 224)
        
        # Update the sensitivity_expanded tensor
        sensitivity_expanded[i] = sensitivity_image

    return sensitivity_expanded