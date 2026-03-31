# XAI_ChestX-ray
**Explainable Deep Learning for Pneumonia Detection from Chest X-ray Images**

## 📌 Project Overview

This project develops a deep learning–based pneumonia detection system from chest X-ray images and integrates Explainable AI (XAI) using **Grad-CAM** / **Grad-CAM++** to visualize the lung regions influencing model predictions.

The primary objective is to provide **transparent and clinically interpretable decision-making** through a modern web interface, making it a trustworthy tool for medical AI.

---

## 🚀 Web Application

The project now includes a full-stack web application:
- **Frontend**: Built with **React (Vite)** and **Tailwind CSS**. Features a professional medical-grade UI, drag-and-drop uploads, and interactive heatmap toggles.
- **Backend**: **Flask** API serving the PyTorch model and performing real-time Grad-CAM++ inference.

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
- **Evaluation**: Trained on weighted classes to handle imbalance; evaluated on a large unseen test set.

---

## 🧠 Model & Explainability

- **Architecture**: **ResNet-18** (pretrained on ImageNet), fine-tuned for binary classification.
- **Explainability**: **Grad-CAM++** is used to generate heatmaps that highlight the regions of the X-ray that led to the prediction (e.g., lung opacities).

---

## 📁 Project Structure

```
XAI_ChestX-ray/
├── app.py                  # Flask Backend API
├── frontend/               # React Frontend Application
│   ├── src/
│   ├── public/
│   └── package.json
├── data/
│   └── chest_xray/
├── EDA/                    # Exploratory Data Analysis
│   ├── results/
│   └── visualisation.py
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_large.py   # Alternative evaluation script
│   ├── gradcam.py
│   └── explainability_utils.py
├── models/
│   └── pneumonia_model.pth
├── results/
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run

### 1. Backend (Flask API)

Open a terminal in the project root:

```bash
# 1. Create & Activate Virtual Environment (if not done)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Install Python Dependencies
pip install -r requirements.txt

# 3. Start the Server
python app.py
```
*Server runs at `http://localhost:5000`*

### 2. Frontend (React App)

Open a **new** terminal in the `frontend/` directory:

```bash
cd frontend

# 1. Install Node Dependencies
npm install

# 2. Start Development Server
npm run dev
```
*Application opens at `http://localhost:5173`*

---

## 📦 Dependencies

### Backend (Python)
- Flask, Flask-CORS
- PyTorch, Torchvision
- NumPy, Matplotlib, OpenCV, Pillow
- Grad-CAM, scikit-learn, tqdm

### Frontend (Node.js)
- React, Vite
- Tailwind CSS
- Framer Motion, Lucide React, Axios

---

## ⚠️ Limitations
- Dataset size is relatively small.
- CPU-only training limits experimentation speed (though inference is fast).
- External clinical validation not performed.

---

## 🧾 Conclusion
This project demonstrates that **explainable deep learning** can effectively detect pneumonia (~91% accuracy) and provide clinically meaningful visual explanations via a user-friendly web interface.