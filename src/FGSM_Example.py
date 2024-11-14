import matplotlib.pyplot as plt
from SVHN_ViT_Train import SVHN_ViTClassifier
from FGSM_Attack import FGSM

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

        # FGSM 
        axes[i + 3].imshow(perturbed_image.clamp(0, 255))
        axes[i + 3].set_title(f"Perturbed Image (Label: {label.item()})")
        axes[i + 3].axis('off')

    fig.suptitle(f"{attack_algorithm.__name__} perturbed images with epsilon = {epsilon}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    classifier = SVHN_ViTClassifier(batch_size=64, lr=3e-4, epochs=15)
    # classifier.train()
    classifier.load_model('../models/ViT.pt')
    epsilon_values = [0.0, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1]
    for epsilon in epsilon_values:
        get_attack_examples(classifier, FGSM, epsilon=epsilon)