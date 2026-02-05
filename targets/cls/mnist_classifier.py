import os
from pathlib import Path

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms


BASE_DIR = os.environ.get("WRITABLE_DIR", Path(__file__).parent)
MODEL_DIR = Path(BASE_DIR) / "pretrained_models"
DATA_DIR = Path(BASE_DIR) / "dataset" / "mnist"


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x, return_features=False):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 64 * 7 * 7)

        x = self.relu(self.fc1(x))
        if return_features:
            return x

        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Data transformation
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ]
    )

    # Load Data
    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize Model, Loss, and Optimizer
    model = MNISTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    wandb.init(project="mnist-classifier")

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = torch.bernoulli(data)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} "
                    f"[{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                    f"Loss: {loss.item():.6f}"
                )

        # Validation phase
        model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = torch.bernoulli(data)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)
        print(
            f"\nEnd of Epoch {epoch+1}: Test set: Average loss: {test_loss:.4f}, "
            f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
        )
        wandb.log(
            {
                "train_loss": running_loss / len(train_loader),
                "test_loss": test_loss,
                "accuracy": accuracy,
            }
        )

    # Save the model
    save_path = f"{MODEL_DIR}/mnist-cls/mnist_01input_cls.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train()
