import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from SVHN_ViT_Train import SVHN_ViTClassifier
from FGSM_Attack import FGSM

def adversarial_test(MODEL, attack_algorithm, epsilon=0.001, num_steps=1):
    model = MODEL.model
    model.eval()
    correct, total = 0, 0
    test_loader = MODEL.test_loader

    for images, labels in tqdm(test_loader, desc=f"{attack_algorithm.__name__} Testing", leave=False):
        images, labels = images.to(MODEL.device), labels.to(MODEL.device)
        perturbed_images = attack_algorithm(MODEL, images, labels, epsilon=epsilon, num_steps=num_steps)

        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":
    classifier = SVHN_ViTClassifier(batch_size=64, lr=3e-4, epochs=15)
    # classifier.train()
    classifier.load_model('../models/FGSMT_ViT.pth-epoch-7.pt')

    # Different epsilon values
    epsilon_values = [0.0, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1]
    accuracies = []

    # Evaluating
    for epsilon in epsilon_values:
        accuracy = adversarial_test(classifier, FGSM, epsilon=epsilon)
        accuracies.append(accuracy)
        print(f'Epsilon: {epsilon}, Accuracy: {accuracy:.2f}%')

    # Drawing
    plt.plot(epsilon_values, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Effect of Epsilon on Model Accuracy %s' %FGSM.__name__)
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show()