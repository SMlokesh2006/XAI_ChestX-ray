import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

LABELS = [
    "Pneumonia",
    "Cardiomegaly",
    "Pleural Effusion",
    "Edema",
    "Consolidation"
]

# Load model architecture
model = models.densenet121(weights=None)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, len(LABELS))

# Load trained weights
model_path = Path(__file__).resolve().parent / "model" / "chexpert_densenet121_partial.pth"
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

model.load_state_dict(torch.load(str(model_path), map_location="cpu"))

model.eval()
print("Model loaded successfully.")
