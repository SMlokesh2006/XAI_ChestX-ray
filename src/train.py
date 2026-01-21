import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import get_model

def train():
    device = torch.device("cpu")  # CPU training

    train_loader, val_loader, classes = get_dataloaders(
        data_dir="data/chest_xray",
        batch_size=16
    )

    model = get_model().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5

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

    torch.save(model.state_dict(), "models/pneumonia_model.pth")
    print("Model saved to models/pneumonia_model.pth")

if __name__ == "__main__":
    train()
