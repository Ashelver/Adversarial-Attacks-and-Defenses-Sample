from tqdm import tqdm
import torch
import time
from SVHN_ViT_Train import SVHN_ViTClassifier
from Noise_adding import add_gaussian_noise_to_edges

class EDGES_NOISE_SVHN_ViTClassifier(SVHN_ViTClassifier):
    def EDGES_NOISE_train(self, noise_std=0.1):
        self.model.train()
        for epoch in range(self.epochs):
            start_time = time.time()
            save_checkpoints = self.save_path + '-epoch-' + str(epoch + 1) + '.pt'
            running_loss = 0.0
            correct, total = 0, 0
            with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}') as pbar:
                for i, (images, labels) in enumerate(pbar):
                    images, labels = images.to(self.device), labels.to(self.device)
                    noisy_images = add_gaussian_noise_to_edges(images, noise_std)

                    #---------------Normal
                    # Forwarding
                    self.optimizer.zero_grad()
                    outputs_normal = self.model(images)
                    loss_normal = self.criterion(outputs_normal, labels)

                    # Backwarding
                    loss_normal.backward()
                    self.optimizer.step()

                    #---------------Noisy
                    # Forwarding
                    self.optimizer.zero_grad()
                    outputs_noisy = self.model(noisy_images)
                    loss_noisy = self.criterion(outputs_noisy, labels)

                    # Backwarding
                    loss_noisy.backward()
                    self.optimizer.step()

                    # Loss and Acurracy
                    running_loss += (loss_normal.item() + loss_noisy.item())
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
    classifier = EDGES_NOISE_SVHN_ViTClassifier(batch_size=64, lr=3e-4, epochs=30)
    classifier.set_path('../models/EDGEST_ViT')
    classifier.EDGES_NOISE_train(noise_std=0.1)
    classifier.test()