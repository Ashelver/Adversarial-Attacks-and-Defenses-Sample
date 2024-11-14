# coding: UTF-8
import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time

class SVHN_ViTClassifier:
    def __init__(self, batch_size=64, lr=3e-4, epochs=10, device=None, save_path='../models/ViT'):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path =  save_path

        # Data preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Anti-normalize
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )

        # Load data
        self.train_loader, self.test_loader = self.load_data()

        # initialize
        self.model = self.initialize_model()

        # Loss function, Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        print('Using device:', self.device)

    def set_path(self,path):
        self.save_path = path

    def load_data(self):
        train_dataset = datasets.SVHN(root='../data', split='train', download=True, transform=self.transform)
        test_dataset = datasets.SVHN(root='../data', split='test', download=True, transform=self.transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def initialize_model(self):
        model = timm.create_model('vit_small_patch16_224', pretrained=True)
        model.head = nn.Linear(model.head.in_features, 10)
        model = model.to(self.device)
        return model

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            start_time = time.time()
            save_checkpoints = self.save_path + '-epoch-' + str(epoch) + '.pt'
            running_loss = 0.0
            correct, total = 0, 0
            with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}') as pbar:
                for i, (images, labels) in enumerate(pbar):
                    images, labels = images.to(self.device), labels.to(self.device)

                    # Forwarding
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    # Backwarding
                    loss.backward()
                    self.optimizer.step()

                    # Loss and Acurracy
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_loss = running_loss / len(self.train_loader)
            accuracy = 100 * correct / total
            end_time = time.time()
            epoch_duration = end_time - start_time
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Epoch duration: {epoch_duration:.2f}s")
            torch.save(self.model.state_dict(), save_checkpoints)

    def test(self):
        self.model.eval()
        correct, total = 0, 0
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(self.test_loader)
        accuracy = 100 * correct / total
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print("Load",path,"successful!")

# Execute for training
if __name__ == "__main__":
    classifier = SVHN_ViTClassifier(batch_size=64, lr=3e-4, epochs=15)
    classifier.train()
    classifier.test()