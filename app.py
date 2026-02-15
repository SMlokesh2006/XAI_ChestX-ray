
import os
import sys

# Ensure src/ is in the path so we can import modules from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import numpy as np
import cv2
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
from torchvision import transforms

# Imports from existing project structure
# We need to make sure we can import these correctly. 
# If src/gradcam.py has 'import explainability_utils', adding src to sys.path helps.
from src.model import get_model
from src.explainability_utils import generate_cam

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model
model = None
MODEL_PATH = os.path.join("models", "pneumonia_model.pth")

def load_trained_model():
    """Validates and loads the trained model."""
    global model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model()
        
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}. Using random weights (for testing only).")
            
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        # Initialize generic model just to prevent crash if file is missing/corrupt
        model = get_model()
        model.eval()

# Load model on startup
load_trained_model()

def transform_image(image_bytes):
    """Preprocesses the image for the ResNet model."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # Same transform as in src/gradcam.py
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    return image, input_tensor

def array_to_base64_img(img_array):
    """Converts a numpy image (BGR or RGB) to base64 string."""
    # Ensure it's in 0-255 range and uint8
    if img_array.dtype != np.uint8:
        # Assuming normalized 0-1 if float
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
            
    # Convert to RGB if needed provided it is in BGR (OpenCV default)
    # But usually we handle RGB in PIL. 
    # Let's stick effectively to RGB if we produce it via PIL/Matplotlib or BGR if CV2.
    
    img_pil = Image.fromarray(img_array)
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Pneumonia Detection API is running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        # preprocess
        img_bytes = file.read()
        original_image, input_tensor = transform_image(img_bytes)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            
        prediction_class = "PNEUMONIA" if prob > 0.5 else "NORMAL"
        confidence = prob if prob > 0.5 else 1 - prob
        
        # Generate Grad-CAM
        # Target layer for ResNet18 usually layer4[-1]
        target_layer = [model.layer4[-1]]
        
        # generate_cam returns grayscale CAM (224, 224)
        grayscale_cam = generate_cam(model, input_tensor, target_layer, method="gradcam++", target_category=0)
        
        # Resize original image to 224x224 for overlay
        orig_resized = np.array(original_image.resize((224, 224)))
        
        # Create heatmap overlay
        from pytorch_grad_cam.utils.image import show_cam_on_image
        
        # show_cam_on_image expects float32 image [0,1]
        rgb_img_float = orig_resized.astype(np.float32) / 255.0
        visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
        
        # Convert visualization to base64
        heatmap_b64 = array_to_base64_img(visualization)
        
        # Generated isolated heatmap (for sidebar/toggle view)
        # simple jet colormap on the grayscale cam
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        heatmap_only_b64 = array_to_base64_img(heatmap_colored)
        
        # Explanation Text
        explanation = []
        explanation.append(f"Model is {confidence*100:.1f}% confident this is {prediction_class}.")
        
        # Simple region logic (Upper vs Lower lung)
        h = grayscale_cam.shape[0]
        upper_attention = grayscale_cam[:h//2, :].mean()
        lower_attention = grayscale_cam[h//2:, :].mean()
        
        if prediction_class == "PNEUMONIA":
            if lower_attention > upper_attention:
                explanation.append("The model focuses significantly on the lower lung regions, which typically indicates consolidated opacities common in pneumonia.")
            else:
                 explanation.append("The model detected anomalies in the upper lung regions.")
        else:
            explanation.append("No significant pathological patterns were detected.")
            
        return jsonify({
            "prediction": prediction_class,
            "confidence": confidence * 100,
            "heatmap": heatmap_b64,
            "heatmap_only": heatmap_only_b64,
            "explanation": " ".join(explanation)
        })

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
