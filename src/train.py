import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import get_model


def train():
    device = torch.device("cpu")  # CPU training

    # Load data
    train_loader, val_loader, classes = get_dataloaders(
        data_dir="data/chest_xray",
        batch_size=16
    )

    # -----------------------------
    # Compute class distribution
    # -----------------------------
    num_normal = 0
    num_pneumonia = 0

    for _, labels in train_loader:
        num_normal += (labels == 0).sum().item()
        num_pneumonia += (labels == 1).sum().item()

    print(f"Train NORMAL samples: {num_normal}")
    print(f"Train PNEUMONIA samples: {num_pneumonia}")

    # Positive class = pneumonia (label 1)
    pos_weight = torch.tensor([num_normal / num_pneumonia])

    # -----------------------------
    # Model, loss, optimizer
    # -----------------------------
    model = get_model().to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.4f}")

    # -----------------------------
    # Save model
    # -----------------------------
    torch.save(model.state_dict(), "models/pneumonia_model.pth")
    print("Model saved to models/pneumonia_model.pth")


if __name__ == "__main__":
    train()
