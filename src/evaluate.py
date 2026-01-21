import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from dataset import get_dataloaders
from model import get_model

def evaluate():
    device = torch.device("cpu")

    # Load test data instead of val
    _, test_loader, _ = get_dataloaders(
        data_dir="data/chest_xray",
        batch_size=16
    )

    model = get_model().to(device)
    model.load_state_dict(
        torch.load("models/pneumonia_model.pth", map_location=device)
    )
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=["NORMAL", "PNEUMONIA"]
    )

    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    evaluate()
