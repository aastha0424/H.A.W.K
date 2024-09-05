# model.py

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Resize to the input size expected by the model
    transforms.ToTensor(),           # Convert to tensor
])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
    return image

def load_model():
    model = torch.hub.load('Z-Zheng/Changen', 'changestar_1x96', backbone_name='mitb1',
                              pretrained=True, dataset_name='s2looking', force_reload=True)
    return model

def compare_images(model, image1_path, image2_path, device='cpu'):
    # Load and preprocess images
    t1_image = transform(load_image(image1_path)).unsqueeze(0)  # Add batch dimension
    t2_image = transform(load_image(image2_path)).unsqueeze(0)   # Add batch dimension
    bi_images = torch.cat([t1_image, t2_image], dim=1)  # [b, tc, h, w]

    # Ensure model and input are on the same device
    model = model.to(device)
    bi_images = bi_images.to(device)

    # Make predictions
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        predictions = model(bi_images)
        change_prob = predictions['change_prediction']  # [b, 1, h, w]

    # Convert tensor to numpy array and squeeze the batch and channel dimensions
    change_prob_np = change_prob.detach().cpu().numpy().squeeze()  # [h, w]

    # Threshold the change probability map
    change_mask = (change_prob_np > 0.000007).astype(np.uint8)  # Binary mask

    # Detect contours on the change mask
    contours, _ = cv2.findContours(change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return change_prob_np, contours
