import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

# -----------------------------
# Labels
# -----------------------------
LABELS = [
    "Pneumonia",
    "Cardiomegaly",
    "Pleural Effusion",
    "Edema",
    "Consolidation"
]

# -----------------------------
# Load model
# -----------------------------
model = models.densenet121(weights=None)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, len(LABELS))

model.load_state_dict(
    torch.load("model/chexpert_densenet121_partial.pth", map_location="cpu")
)
model.eval()

print("Model loaded.")

# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img_path = "sample_images/sample1.jpeg"
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# -----------------------------
# Forward pass
# -----------------------------
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.sigmoid(outputs)[0]

print("\nPredictions:")
for label, prob in zip(LABELS, probs):
    print(f"{label}: {prob.item():.3f}")

# -----------------------------
# Grad-CAM
# -----------------------------
target_layer = model.features[-1]

cam = GradCAM(
    model=model,
    target_layers=[target_layer]
)

# Choose the top predicted class
top_class = probs.argmax().item()
targets = [ClassifierOutputTarget(top_class)]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)[0]

# -----------------------------
# Visualization
# -----------------------------
rgb_img = np.array(image.resize((224, 224))) / 255.0
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original X-ray")
plt.imshow(rgb_img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Grad-CAM: {LABELS[top_class]}")
plt.imshow(cam_image)
plt.axis("off")

plt.tight_layout()
plt.show()
