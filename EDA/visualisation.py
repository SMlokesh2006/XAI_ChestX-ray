import os
import matplotlib.pyplot as plt
from PIL import Image
import random

data_dir = "data/chest_xray/train"

classes = ["NORMAL", "PNEUMONIA"]
counts = []

for cls in classes:
    counts.append(len(os.listdir(os.path.join(data_dir, cls))))

plt.bar(classes, counts)
plt.title("Class Distribution (Train Set)")
plt.ylabel("Number of Images")
plt.show()

print(dict(zip(classes, counts)))

fig, axes = plt.subplots(2, 3, figsize=(9, 6))

for row, cls in enumerate(classes):
    images = os.listdir(os.path.join(data_dir, cls))
    for col in range(3):
        img_path = os.path.join(data_dir, cls, random.choice(images))
        img = Image.open(img_path).convert("L")
        axes[row, col].imshow(img, cmap="gray")
        axes[row, col].axis("off")
        axes[row, col].set_title(cls)

plt.suptitle("Sample Chest X-ray Images")
plt.show()

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}")
    for cls in classes:
        path = f"data/chest_xray/{split}/{cls}"
        print(cls, len(os.listdir(path)))
