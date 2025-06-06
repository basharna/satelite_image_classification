import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class FlareDetectorModel(nn.Module):
    """
    CNN model for flare detection in satellite images.
    Binary classification: flare present (1) or not present (0).
    """
    def __init__(self, in_channels=3, img_size=256):
        super(FlareDetectorModel, self).__init__()
        
        # Store image size for visualization
        self.img_size = img_size
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size of the flattened features dynamically
        # After 4 pooling layers (each dividing dimensions by 2):
        # img_size -> img_size/2 -> img_size/4 -> img_size/8 -> img_size/16
        feature_size = img_size // 16
        self.flat_features = 128 * feature_size * feature_size
        
        # Classification layers
        # Calculate the input features for the first fully connected layer
        self.fc1 = nn.Linear(self.flat_features, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 1)  # Binary classification
        
        # Initialize feature maps dictionary for visualization
        self.feature_maps = {}
        
    def forward(self, x):
        # Feature extraction
        x1 = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x2 = self.pool2(F.relu(self.bn2(self.conv2(x1))))
        x3 = self.pool3(F.relu(self.bn3(self.conv3(x2))))
        x4 = self.pool4(F.relu(self.bn4(self.conv4(x3))))
        
        # Save feature maps for visualization
        self.feature_maps = {
            'conv1': x1,
            'conv2': x2,
            'conv3': x3,
            'conv4': x4
        }
        
        # Flatten
        batch_size = x4.size(0)
        x_flat = x4.view(batch_size, -1)
        
        # Classification
        x_fc1 = F.relu(self.fc1(x_flat))
        x_fc1 = self.dropout1(x_fc1)
        x_fc2 = F.relu(self.fc2(x_fc1))
        x_fc2 = self.dropout2(x_fc2)
        x_out = self.fc3(x_fc2)
        
        # Clamp the output logits to prevent extreme values that cause numerical instability
        # This limits the logits to a reasonable range while still allowing confident predictions
        x_out = torch.clamp(x_out, min=-10.0, max=10.0)
        
        # Return logits (no sigmoid here as we'll use BCEWithLogitsLoss)
        return x_out
        
    def visualize_flare(self, image_tensor, original_image=None, confidence=None):
        """
        Visualize the detected flare using a simplified Grad-CAM technique
        
        Args:
            image_tensor: Input tensor that was used for prediction
            original_image: Optional PIL image to draw on. If None, will use the tensor
            confidence: Confidence score of the prediction (0-1)
            
        Returns:
            PIL Image with flare area highlighted
        """
        # Store current model mode to restore it later
        was_training = self.training
        
        # Set model to eval mode for visualization
        self.eval()
        
        # Make sure we have a batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Get the device
        device = next(self.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # We'll use a simpler approach that doesn't require gradients
        # First, get the feature maps from the last convolutional layer
        with torch.no_grad():
            # Forward pass to get feature maps
            _ = self(image_tensor)
            
            # Get the feature maps from the last convolutional layer
            feature_maps = self.feature_maps['conv4']
            
            # Get the weights from the last fully connected layer
            weights = self.fc3.weight.data
            
            # Reshape weights to match feature maps
            weights = weights.view(weights.size(0), -1, 1, 1)
            
            # Apply weights to feature maps to get the class activation map
            cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
            
            # Apply ReLU to focus on features that have a positive influence on the class
            cam = F.relu(cam)
            
            # Normalize the CAM
            cam = F.interpolate(cam, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Convert to numpy and prepare for visualization
        cam = cam.squeeze().cpu().numpy()
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        # If original image is provided, use it; otherwise convert tensor to image
        if original_image is not None:
            img_np = np.array(original_image)
        else:
            # Denormalize the tensor
            img = image_tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            img_np = np.uint8(255 * img)
        
        # Resize image to match heatmap size if needed
        if img_np.shape[:2] != (self.img_size, self.img_size):
            img_np = cv2.resize(img_np, (self.img_size, self.img_size))
        
        # Convert to BGR for OpenCV if needed
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Overlay heatmap on image
        result_img = cv2.addWeighted(img_np, 0.7, heatmap, 0.3, 0)
        
        # Convert back to RGB for PIL
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Ignore activations at the edges (10% border)
        border_size = int(self.img_size * 0.1)
        center_cam = cam.copy()
        # Set border regions to 0
        center_cam[:border_size, :] = 0  # Top
        center_cam[-border_size:, :] = 0  # Bottom
        center_cam[:, :border_size] = 0  # Left
        center_cam[:, -border_size:] = 0  # Right
        
        # Only highlight if there's a significant activation in the center
        max_activation = center_cam.max()
        
        # Set a threshold for significant activation (0.5)
        if max_activation > 0.5:
            # Find the brightest point in the center region
            y, x = np.unravel_index(np.argmax(center_cam), center_cam.shape)
            
            # Draw a circle around the brightest point (flare)
            cv2.circle(result_img, (x, y), 20, (0, 255, 0), 3)
            
            # Add text label with confidence if available
            if confidence is not None:
                label = f"Flare Detected ({confidence:.2f})"
            else:
                label = "Flare Detected"
                
            cv2.putText(result_img, label, (x - 60, y - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert back to PIL Image
        result_pil = Image.fromarray(result_img)
        
        # Restore original model mode
        if was_training:
            self.train()
        
        return result_pil
    
    def predict_with_visualization(self, image_tensor, original_image=None):
        """
        Make a prediction and return both the prediction and visualization
        
        Args:
            image_tensor: Input tensor
            original_image: Optional original PIL image
            
        Returns:
            tuple: (prediction, visualization_image)
        """
        # Store current model mode
        was_training = self.training
        
        # Make prediction
        self.eval()
        
        # Ensure we have a batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Get device
        device = next(self.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # Forward pass for prediction
        with torch.no_grad():
            output = self(image_tensor)
            confidence = torch.sigmoid(output).item()
            # Flare is label 1, so we check if prediction is 1 (> 0.5)
            prediction = confidence > 0.5
        
        # Create visualization regardless of prediction to show what the model is looking at
        try:
            visualization = self.visualize_flare(image_tensor, original_image, confidence)
        except Exception as e:
            print(f"Warning: Visualization failed with error: {e}")
            # Create a simple placeholder image if visualization fails
            if original_image is not None:
                visualization = original_image.copy()
            else:
                # Create a blank image with text
                img = np.ones((224, 224, 3), dtype=np.uint8) * 255
                if prediction:
                    cv2.putText(img, f"Flare Detected ({confidence:.2f})", (50, 112), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(img, f"No Flare ({confidence:.2f})", (50, 112), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                visualization = Image.fromarray(img)
        
        # Restore original model mode
        if was_training:
            self.train()
            
        return prediction, visualization
