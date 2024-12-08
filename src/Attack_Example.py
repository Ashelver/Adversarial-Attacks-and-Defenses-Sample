import matplotlib.pyplot as plt
from SVHN_ViT_Train import SVHN_ViTClassifier
from FGSM_Attack import FGSM, FGSM_ONLY_GRADIENT, FGSM_GRADIENT_MASK_IMAGES, FGSM_GRADIENT_SENSITIVITY_IMAGES
from PGD_Attack import PGD

def get_attack_examples(MODEL, attack_algorithm, epsilon=0.001, num_steps=1):
    test_loader = MODEL.test_loader
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    perturbed_images = attack_algorithm(MODEL, images, labels, epsilon=epsilon, num_steps=num_steps)
    
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(3):
        label = labels[i]
        image = images[i]
        perturbed_image = perturbed_images[i]

        image = MODEL.inv_normalize(image)  # Anti-normalize
        perturbed_image = MODEL.inv_normalize(perturbed_image).cpu()
        # Convert image formats
        original_image = image.permute(1, 2, 0)  # From (C, H, W) to (H, W, C)
        perturbed_image = perturbed_image.permute(1, 2, 0)

        # Original
        axes[i].imshow(original_image)
        axes[i].set_title(f"Original Image (Label: {label.item()})")
        axes[i].axis('off')

        axes[i + 3].imshow(perturbed_image.clamp(0, 255))
        axes[i + 3].set_title(f"Perturbed Image (Label: {label.item()})")
        axes[i + 3].axis('off')

    fig.suptitle(f"{attack_algorithm.__name__} perturbed images with epsilon = {epsilon}")
    plt.tight_layout()
    plt.show()


def get_noise_examples(MODEL, attack_algorithm, epsilon=None, num_steps=1):
    test_loader = MODEL.test_loader
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    perturbed_images = attack_algorithm(MODEL, images, labels, epsilon=epsilon, num_steps=num_steps)
    
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(3):
        label = labels[i]
        image = images[i]
        perturbed_image = perturbed_images[i]

        image = MODEL.inv_normalize(image)  # Anti-normalize
        perturbed_image = MODEL.inv_normalize(perturbed_image).cpu()
        # Convert image formats
        original_image = image.permute(1, 2, 0)  # From (C, H, W) to (H, W, C)
        perturbed_image = perturbed_image.permute(1, 2, 0)

        # Original
        axes[i].imshow(original_image)
        axes[i].set_title(f"Original Image (Label: {label.item()})")
        axes[i].axis('off')

        axes[i + 3].imshow(perturbed_image.clamp(0, 255))
        axes[i + 3].set_title(f"Gradient (Label: {label.item()})")
        axes[i + 3].axis('off')

    fig.suptitle(f"{attack_algorithm.__name__}")
    plt.tight_layout()
    plt.show()


def get_sign_examples(MODEL, attack_algorithm, epsilon=None, num_steps=1):
    test_loader = MODEL.test_loader
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Generate remove_positive and remove_negative using the attack algorithm
    remove_positive, remove_negative = attack_algorithm(MODEL, images, labels, epsilon=epsilon, num_steps=num_steps)
    
    # Create a 3x3 grid for displaying images
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(3):
        label = labels[i]
        
        # Denormalize images
        original_image = MODEL.inv_normalize(images[i]).permute(1, 2, 0).cpu()  # Original image
        image_remove_positive = MODEL.inv_normalize(remove_positive[i]).permute(1, 2, 0).cpu()  # Image with positive gradients removed
        image_remove_negative = MODEL.inv_normalize(remove_negative[i]).permute(1, 2, 0).cpu()  # Image with negative gradients removed

        # Display original image
        axes[i].imshow(original_image.clamp(0, 1))
        axes[i].set_title(f"Original (Label: {label.item()})")
        axes[i].axis('off')

        # Display image with positive gradients removed
        axes[i + 3].imshow(image_remove_positive.clamp(0, 1))
        axes[i + 3].set_title(f"Remove Positive (Label: {label.item()})")
        axes[i + 3].axis('off')

        # Display image with negative gradients removed
        axes[i + 6].imshow(image_remove_negative.clamp(0, 1))
        axes[i + 6].set_title(f"Remove Negative (Label: {label.item()})")
        axes[i + 6].axis('off')

    # Add a title for the entire figure
    fig.suptitle(f"Attack: {attack_algorithm.__name__}", fontsize=16)
    plt.tight_layout()
    plt.show()

def get_sensitivity_examples(MODEL, attack_algorithm, epsilon=None, num_steps=1):
    test_loader = MODEL.test_loader
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    perturbed_images = attack_algorithm(MODEL, images, labels, epsilon=epsilon, num_steps=num_steps)
    
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(3):
        label = labels[i]
        image = images[i]
        perturbed_image = perturbed_images[i]

        image = MODEL.inv_normalize(image)  # Anti-normalize
        perturbed_image = MODEL.inv_normalize(perturbed_image).cpu()
        # Convert image formats
        original_image = image.permute(1, 2, 0)  # From (C, H, W) to (H, W, C)
        perturbed_image = perturbed_image.permute(1, 2, 0)

        # Original
        axes[i].imshow(original_image)
        axes[i].set_title(f"Original Image (Label: {label.item()})")
        axes[i].axis('off')

        axes[i + 3].imshow(perturbed_image.clamp(0, 255))
        axes[i + 3].set_title(f"Gradient Sensitivity (Label: {label.item()})")
        axes[i + 3].axis('off')

    fig.suptitle(f"{attack_algorithm.__name__}")
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    classifier = SVHN_ViTClassifier(batch_size=64, lr=3e-4, epochs=15)
    # classifier.train()
    classifier.load_model() # path to your model
    epsilon_values = [0.0, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1]
    get_noise_examples(classifier, FGSM_ONLY_GRADIENT)
    get_sign_examples(classifier, FGSM_GRADIENT_MASK_IMAGES)
    get_sensitivity_examples(classifier, FGSM_GRADIENT_SENSITIVITY_IMAGES)
    for epsilon in epsilon_values:
        get_attack_examples(classifier, FGSM, epsilon=epsilon)
    for epsilon in epsilon_values:
        get_attack_examples(classifier, PGD, epsilon=epsilon, num_steps=5)

