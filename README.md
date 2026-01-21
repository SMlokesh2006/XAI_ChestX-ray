# XAI_ChestX-ray
# Explainable AI for Pneumonia Detection from Chest X-ray Images

## 📌 Project Overview
This project implements a deep learning–based system to detect **Pneumonia** from chest X-ray images and explains the model’s predictions using **Grad-CAM** (Gradient-weighted Class Activation Mapping).

The goal is not only to achieve accurate classification but also to **provide visual explanations** highlighting the lung regions that influence the model’s decisions, improving transparency and trust in medical AI systems.

---

## 🗂 Dataset
**Chest X-ray Images (Pneumonia)** dataset is used.

### Dataset Structure
data/chest_xray/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
├── val/
│ ├── NORMAL/
│ └── PNEUMONIA/
└── test/
├── NORMAL/
└── PNEUMONIA/


- Binary classification task
- Images are grayscale chest X-rays
- Dataset is known to have small validation and test splits

---

## 🔍 Exploratory Data Analysis (EDA)
The following analyses were performed:
- Class distribution visualization
- Sample image visualization (NORMAL vs PNEUMONIA)
- Image size variability analysis
- Pixel intensity distribution

EDA revealed a clean dataset with visually distinguishable pneumonia patterns.

---

## 🧠 Model Architecture
- **Backbone:** ResNet-18 (pretrained on ImageNet)
- **Modification:** Final fully connected layer replaced for binary classification
- **Loss Function:** Binary Cross-Entropy with Logits
- **Optimizer:** Adam
- **Training Device:** CPU

---

## 🏋️ Training Details
- Image size: `224 × 224`
- Batch size: `16`
- Epochs: `5`
- Data augmentation: Random horizontal flip
- Normalization: ImageNet mean & standard deviation

### Training Loss
Loss decreased smoothly across epochs, indicating stable learning.

---

## 📊 Evaluation Results

### Validation Set
- Accuracy: **100%**
- Validation size: **16 images**

### Test Set
- Accuracy: **100%**
- Test size: **16 images**
- Confusion Matrix:
[[8 0]
[0 8]]


### ⚠️ Important Note on Results
The validation and test sets are **very small**, which can lead to optimistic accuracy scores.  
Therefore, accuracy values should be interpreted as **strong initial performance rather than guaranteed generalization**.

---

## 🔍 Explainability with Grad-CAM
Grad-CAM was applied to visualize important regions contributing to Pneumonia predictions.

### Observations:
- Heatmaps focus primarily on lung regions
- No strong activation on irrelevant areas (borders, text)
- Supports clinical plausibility of model decisions

Explainability is treated as a **core contribution** of this project rather than raw accuracy.

---

## 📁 Project Structure
XAI_Pneumonia_Project/
├── data/
│ └── chest_xray/
├── src/
│ ├── dataset.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│ └── gradcam.py
├── models/
│ └── pneumonia_model.pth
├── results/
│ └── gradcam_outputs/
├── notebooks/
│ └── eda.ipynb
├── README.md
└── requirements.txt


---

## ▶️ How to Run

### 1. Train the Model
```bash
python src/train.py
python src/evaluate.py
python src/gradcam.py
```

### Dependencies

Python 3.10+

PyTorch

Torchvision

NumPy

Matplotlib

OpenCV

scikit-learn

pytorch-grad-cam

### Limitations

Small validation and test datasets

CPU-only training

Results may not generalize to unseen hospital data

### Future Work

Training on larger, more diverse datasets

Cross-validation

Multi-class disease classification

Deployment as a web application

Clinical expert validation

### Conclusion

This proje  ct demonstrates that explainable deep learning can effectively detect pneumonia from chest X-ray images while providing meaningful visual explanations. Despite dataset size limitations, Grad-CAM visualizations support the reliability and interpretability of the model.