# XAI_ChestX-ray
**Explainable Deep Learning for Pneumonia Detection from Chest X-ray Images**

## 📌 Project Overview

This project develops a deep learning–based pneumonia detection system from chest X-ray images and integrates Explainable AI (XAI) using **Grad-CAM** / **Grad-CAM++** to visualize the lung regions influencing model predictions.

The primary objective is not only accurate classification but also transparent and clinically interpretable decision-making, which is essential for trustworthy medical AI.

---

## 🗂 Dataset

The project uses the **Chest X-ray Pneumonia** dataset containing labeled radiographic images.

### Dataset Structure
```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### Key Characteristics
- **Binary classification**: NORMAL vs PNEUMONIA
- **Type**: Grayscale medical images
- **Small validation set** → risk of optimistic accuracy
- **Larger unseen test evaluation** used for realistic performance

---

## 🔍 Exploratory Data Analysis (EDA)

Performed analyses include:
- Class distribution visualization
- Sample image inspection
- Image size consistency check
- Pixel-intensity distribution study

**EDA confirmed:**
- Clean dataset structure
- Visible pneumonia-related opacity patterns
- Potential class imbalance, motivating weighted training

---

## 🧠 Model Architecture

- **Backbone**: ResNet-18 pretrained on ImageNet
- **Modification**: Final fully connected layer adapted for binary classification
- **Loss Function**:
    - Binary Cross-Entropy with Logits
    - Class-weighted loss to address imbalance
- **Optimizer**: Adam
- **Training Device**: CPU

---

## 🏋️ Training Strategy

### Hyperparameters
- **Image size**: 224 × 224
- **Batch size**: 16
- **Epochs**: 20
- **Learning rate**: 3 × 10⁻⁵

### Data Augmentation (critical for generalization & XAI)
- Random resized crop
- Small rotation
- Brightness/contrast jitter
- Horizontal flip

These augmentations prevent shortcut learning and improve Grad-CAM interpretability.

---

## 📊 Evaluation Results

### Realistic Large Test Evaluation
- **Test samples**: 624 images
- **Accuracy**: ≈ 91%
- Balanced performance across classes

### Classification Summary
- **NORMAL recall**: Improved significantly after class weighting
- **PNEUMONIA recall**: Remained very high (near-perfect sensitivity)
- **Macro-F1**: ≈ 0.90 → strong balanced performance

This confirms good generalization compared to the misleading initial 100% accuracy from the tiny validation split.

---

## 🔍 Explainability with Grad-CAM / Grad-CAM++

Explainable AI is a core contribution of this project.

### Method
- **Grad-CAM** applied to the final convolutional layer of ResNet-18
- **Grad-CAM++** used for sharper localization of pneumonia regions

### Observations
- Heatmaps focus primarily on lung fields, not borders or artifacts
- Attention aligns with clinically plausible pneumonia regions
- Stronger augmentation and longer training produced clearer, localized explanations

### Interpretation
These results demonstrate that the model:
1. Learns meaningful medical features, not dataset shortcuts
2. Provides transparent visual justification for predictions
3. Supports trustworthy AI in medical imaging

---

## 📁 Project Structure

```
XAI_ChestX-ray/
├── data/
│   └── chest_xray/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
├── models/
│   └── pneumonia_model.pth
├── results/
│   └── gradcam_outputs/
├── notebooks/
│   └── eda.ipynb
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run

### 1. Create Virtual Environment

> **IMPORTANT**: For this project, please install all packages **manually** using the provided commands. Do not rely on automated scripts or global installations.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train, Evaluate, Explain
```bash
python src/train.py
python src/evaluate.py
python src/gradcam.py
```

### 4. Deactivate Environment
```bash
deactivate
```

---

## 📦 Dependencies

- Python 3.10 – 3.12
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- OpenCV
- scikit-learn
- grad-cam

(All versions listed in `requirements.txt`.)

---

## ⚠️ Limitations

- Dataset size is relatively small
- CPU-only training limits experimentation
- External clinical validation not performed
- Performance may vary across hospitals or imaging devices

---

## 🚀 Future Work

- Cross-validation for robustness
- Multi-disease chest X-ray classification
- Web-based deployment for clinical demo
- Radiologist-guided evaluation of explanations

---

## 🧾 Conclusion

This project demonstrates that **explainable deep learning** can effectively detect pneumonia from chest X-ray images while providing clinically meaningful visual explanations.

Through class-weighted training, strong augmentation, and Grad-CAM-based interpretability, the final model achieves realistic generalization (**~91% accuracy**) and improved transparency—highlighting the importance of trustworthy AI in healthcare.