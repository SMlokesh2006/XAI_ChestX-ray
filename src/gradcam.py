import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

# -----------------------------
# Load model
# -----------------------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("models/pneumonia_model.pth", map_location="cpu"))
model.eval()

# -----------------------------
# Load image
# -----------------------------
img_path = "data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
image = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensor = transform(image).unsqueeze(0)

# -----------------------------
# Model prediction
# -----------------------------
with torch.no_grad():
    logit = model(input_tensor)
    prob = torch.sigmoid(logit).item()

# -----------------------------
# Grad-CAM
# -----------------------------
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

grayscale_cam = cam(input_tensor=input_tensor)[0]  # (H, W)

# -----------------------------
# EXPLANATION PART (NEW)
# -----------------------------

# Overall importance
cam_mean = grayscale_cam.mean()

# Region-wise attention (upper vs lower lungs)
h = grayscale_cam.shape[0]
upper_attention = grayscale_cam[:h//2, :].mean()
lower_attention = grayscale_cam[h//2:, :].mean()

print("\n--- Model Explanation ---")
print(f"Prediction: Pneumonia")
print(f"Confidence score: {prob:.3f}")
print(f"Grad-CAM mean activation: {cam_mean:.3f}")
print(f"Upper lung attention: {upper_attention:.3f}")
print(f"Lower lung attention: {lower_attention:.3f}")

if prob > 0.8:
    confidence_text = "High confidence prediction"
elif prob > 0.5:
    confidence_text = "Moderate confidence prediction"
else:
    confidence_text = "Low confidence prediction"

if lower_attention > upper_attention:
    region_text = "Model focuses more on lower lung regions, which is typical for pneumonia."
else:
    region_text = "Model focuses more on upper lung regions."

print("\nExplanation:")
print(f"- {confidence_text}")
print(f"- {region_text}")

# -----------------------------
# Visualization
# -----------------------------
rgb_img = np.array(image.resize((224, 224))) / 255.0
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Original X-ray")
plt.imshow(rgb_img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Grad-CAM (Pneumonia)")
plt.imshow(cam_image)
plt.axis("off")

plt.show()
