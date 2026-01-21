import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from dataset import get_dataloaders
from model import get_model

def evaluate():
    device = torch.device("cpu")

    _, val_loader, _ = get_dataloaders(
        data_dir="data/chest_xray",
        batch_size=16
    )

    model = get_model().to(device)
    model.load_state_dict(torch.load("models/pneumonia_model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Validation Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    evaluate()
