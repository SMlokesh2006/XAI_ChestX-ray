import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def get_cam_method(method_name, model, target_layers):
    """Factory to get the CAM method object."""
    methods = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "scorecam": ScoreCAM,
        "eigencam": EigenCAM
    }
    return methods[method_name](model=model, target_layers=target_layers)

def generate_cam(model, input_tensor, target_layers, method="gradcam++", target_category=None):
    """Generates the CAM heatmap for a given input tensor."""
    cam_method = get_cam_method(method, model, target_layers)
    targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
    
    # Generate CAM
    grayscale_cam = cam_method(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0, :]

def get_bounding_box(grayscale_cam, threshold=0.5):
    """Generates a bounding box around the hotspot region."""
    mask = np.uint8(grayscale_cam > threshold)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, w, h)

def calculate_faithfulness(model, input_tensor, grayscale_cam, target_class_index):
    """
    Calculates the faithfulness of the explanation by masking the most 
    important regions and checking the prediction drop.
    """
    # 1. Get original prediction
    with torch.no_grad():
        org_out = model(input_tensor)
        org_prob = torch.sigmoid(org_out).item()

    # 2. Create mask (parameters can be tuned)
    # Mask top 20% of the activation
    threshold = np.percentile(grayscale_cam, 80)
    mask = torch.tensor(grayscale_cam < threshold).float()
    
    # 3. Mask the input
    # Unsqueeze to match (1, 3, 224, 224)
    # mask is currently (224, 224) -> (1, 1, 224, 224)
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.to(input_tensor.device)
    
    masked_input = input_tensor * mask
    
    # 4. Get new prediction
    with torch.no_grad():
        new_out = model(masked_input)
        new_prob = torch.sigmoid(new_out).item()
        
    return org_prob, new_prob, (org_prob - new_prob)

def visualize_results(image_rgb, cams, boxes, faithfulness_scores, save_path="results/explanation.png"):
    """
    Visualizes the original image, different CAM methods, and bounding boxes.
    cams: dict of {method_name: grayscale_cam}
    boxes: dict of {method_name: input for cv2.rectangle}
    """
    num_methods = len(cams)
    plt.figure(figsize=(5 * (num_methods + 1), 5))
    
    # Original
    plt.subplot(1, num_methods + 1, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")
    
    for i, (name, cam) in enumerate(cams.items()):
        # Overlay heatmap
        visualization = show_cam_on_image(image_rgb, cam, use_rgb=True)
        
        # Draw bounding box if exists
        if boxes.get(name):
            x, y, w, h = boxes[name]
            # Scale to image size (assuming cam is normalized 1x1, but here it's already resized? 
            # Actually pytorch-grad-cam usually returns size of input tensor spatial dims if passed)
            # In our case, we usually get it same size as input.
            # But let's act on the visualization image directly which is 224x224
            cv2.rectangle(visualization, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        plt.subplot(1, num_methods + 1, i + 2)
        plt.imshow(visualization)
        
        # Add faithfulness info to title
        score = faithfulness_scores.get(name)
        title = f"{name.upper()}"
        if score:
            title += f"\nDrop: {score:.3f}"
            
        plt.title(title)
        plt.axis("off")
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Explanation saved to {save_path}")
    plt.show()
