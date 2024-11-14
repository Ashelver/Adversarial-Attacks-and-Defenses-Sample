from tqdm import tqdm
import torch
import time
from SVHN_ViT_Train import SVHN_ViTClassifier


class FGSM_SVHN_ViTClassifier(SVHN_ViTClassifier):
    def FGSM_train(self, epsilon, alpha=0.5):
        self.model.train()
        for epoch in range(self.epochs):
            start_time = time.time()
            save_checkpoints = self.save_path + '-epoch-' + str(epoch) + '.pt'
            running_loss = 0.0
            correct, total = 0, 0
            with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}') as pbar:
                for i, (images, labels) in enumerate(pbar):
                    images, labels = images.to(self.device), labels.to(self.device)
                    images.requires_grad = True

                    # Normal forwarding and backwarding
                    self.optimizer.zero_grad()
                    outputs_normal = self.model(images)
                    loss_normal = self.criterion(outputs_normal, labels)
                    loss_normal.backward(retain_graph=True)


                    # FGSM forward and backwarding
                    data_grad = images.grad.data
                    perturbed_images = images.data + epsilon * data_grad.sign()
                    outputs_perturbed = self.model(perturbed_images)
                    loss_perturbed = self.criterion(outputs_perturbed, labels)

                    # Get total loss from two losses
                    total_loss = alpha * loss_normal + (1-alpha)*loss_perturbed

                    # Total backwarding and updating 
                    self.optimizer.zero_grad()
                    total_loss.backward(retain_graph=False)
                    self.optimizer.step()

                    # Loss and Acurracy
                    running_loss += total_loss.item()
                    _, predicted = torch.max(outputs_normal.data, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_loss = running_loss / len(self.train_loader)
            accuracy = 100 * correct / total
            end_time = time.time()
            epoch_duration = end_time - start_time
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Epoch duration: {epoch_duration:.2f}s")
            torch.save(self.model.state_dict(), save_checkpoints)


# Execute for training
if __name__ == "__main__":
    classifier = FGSM_SVHN_ViTClassifier(batch_size=64, lr=3e-4, epochs=15)
    classifier.set_path('../models/FGSMT_ViT')
    classifier.FGSM_train(epsilon=0.1,alpha=0.5)
    classifier.test()