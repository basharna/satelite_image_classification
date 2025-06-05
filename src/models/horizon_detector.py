import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class HorizonDetectorModel(nn.Module):
    """
    CNN model for horizon detection in satellite images.
    Binary classification: horizon present (1) or not present (0).
    Includes visualization capability to draw dots along the detected horizon line.
    """
    def __init__(self, in_channels=3, img_size=256):
        super(HorizonDetectorModel, self).__init__()
        
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
        # First fully connected layer will be initialized in the forward pass
        self.fc1 = None
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
        
        # Initialize fc1 if it's the first forward pass
        if self.fc1 is None:
            in_features = x_flat.size(1)
            self.fc1 = nn.Linear(in_features, 512).to(x.device)
            # Also update fc2 and fc3 to match the device
            self.fc2 = self.fc2.to(x.device)
            self.fc3 = self.fc3.to(x.device)
        
        # Classification
        x_fc1 = F.relu(self.fc1(x_flat))
        x_fc1 = self.dropout1(x_fc1)
        x_fc2 = F.relu(self.fc2(x_fc1))
        x_fc2 = self.dropout2(x_fc2)
        x_out = self.fc3(x_fc2)
        
        # Return logits (no sigmoid here as we'll use BCEWithLogitsLoss)
        return x_out
        
    def visualize_horizon(self, image_tensor, original_image=None):
        """
        Visualize the detected horizon line using Grad-CAM technique
        
        Args:
            image_tensor: Input tensor that was used for prediction
            original_image: Optional PIL image to draw on. If None, will use the tensor
            
        Returns:
            PIL Image with dots drawn along the detected horizon line
        """
        # Ensure model is in eval mode
        self.eval()
        
        # Make sure we have a batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        # Get the device
        device = next(self.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # Create a copy of the tensor that requires grad
        with torch.enable_grad():
            image_tensor_with_grad = image_tensor.detach().clone().requires_grad_(True)
            
            # Forward pass with gradient tracking
            self.zero_grad()
            output = self(image_tensor_with_grad)
            
            # Create one-hot encoding for the target class (horizon = 1)
            one_hot = torch.zeros((1, 1), device=device)
            one_hot.fill_(1.0)  # Focus on the positive class (horizon = 1)
            
            # Backward pass to get gradients
            output.backward(gradient=one_hot)
            
            # Get the feature maps from the last convolutional layer
            feature_maps = self.feature_maps['conv4']
            
            # Use gradients of the feature maps if available, otherwise use a simpler approach
            if hasattr(feature_maps, 'grad') and feature_maps.grad is not None:
                gradients = feature_maps.grad
            else:
                # Alternative approach: use the feature maps directly
                # This is less accurate but will work when gradients are not available
                gradients = torch.ones_like(feature_maps)
            
            # Calculate weights based on gradients
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            
            # Apply weights to feature maps
            cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
            cam = F.relu(cam)  # Apply ReLU to focus on positive contributions
            
            # Normalize CAM
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
        
        # Resize CAM to original image size
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cv2.resize(cam, (self.img_size, self.img_size))
        
        # Convert to heatmap
        heatmap = np.uint8(255 * cam)
        
        # Get original image
        if original_image is None:
            # Convert tensor to numpy array
            img = image_tensor.squeeze().detach().cpu().numpy()
            # Denormalize
            img = np.transpose(img, (1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1) * 255
            img = img.astype(np.uint8)
        else:
            # Resize original image if needed
            img = np.array(original_image.resize((self.img_size, self.img_size)))
        
        # Apply edge detection to find the horizon line
        # Use a lower threshold to detect more edges
        edges = cv2.Canny(heatmap, 30, 100)
        
        # Create a copy of the image to draw on
        result_img = img.copy()
        
        # Find contours (potential horizon lines)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours are found or they're too small, use a different approach
        if not contours or max(contours, key=cv2.contourArea, default=np.array([[[0, 0]]])).shape[0] < 5:
            # Use thresholding on the heatmap to find horizon line
            _, thresh = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If we still don't have good contours, create a simple horizontal line
        if not contours or max(contours, key=cv2.contourArea, default=np.array([[[0, 0]]])).shape[0] < 5:
            # Create a simple horizontal line in the middle of the image where heatmap is brightest
            row_sums = np.sum(heatmap, axis=1)
            horizon_y = np.argmax(row_sums)
            
            # Create a synthetic contour for the horizon line
            synthetic_contour = []
            for x in range(0, self.img_size, 10):
                synthetic_contour.append([[x, horizon_y]])
            contours = [np.array(synthetic_contour)]
        
        # Draw dots along the detected horizon line
        if contours:
            # Find the longest contour (likely to be the horizon)
            longest_contour = max(contours, key=cv2.contourArea)
            
            # Draw the contour for debugging
            cv2.drawContours(result_img, [longest_contour], -1, (0, 0, 255), 1)
            
            # Get the bounding box of the contour to ensure we span the entire width
            x, y, w, h = cv2.boundingRect(longest_contour)
            
            # Create evenly spaced dots across the entire width of the image
            num_dots = 30  # Increase number of dots
            
            # Determine the min and max x-coordinates to ensure we cover the full horizon
            min_x = max(0, x - 10)  # Extend slightly beyond the detected contour
            max_x = min(self.img_size, x + w + 10)  # Extend slightly beyond the detected contour
            
            # Place dots at regular intervals across the full width
            for dot_x in np.linspace(min_x, max_x, num_dots, dtype=int):
                # Find the corresponding y-coordinate on the contour
                # For each x position, find the closest point on the contour
                distances = [np.sqrt((pt[0][0] - dot_x)**2) for pt in longest_contour]
                if distances:
                    closest_idx = np.argmin(distances)
                    point = longest_contour[closest_idx][0]
                    # Draw larger, more visible green dots
                    cv2.circle(result_img, (dot_x, point[1]), 7, (0, 255, 0), -1)  # Green dots with larger radius
        
        # Convert back to PIL Image
        result_pil = Image.fromarray(result_img)
        
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
        # Make prediction
        self.eval()
        
        # Ensure we have a batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        with torch.no_grad():
            output = self(image_tensor)
            # Horizon is label 1, so we check if prediction is 1 (> 0.5)
            prediction = torch.sigmoid(output).item() > 0.5
        
        # Create visualization if horizon is detected (prediction = True means label 1)
        visualization = None
        if prediction:
            visualization = self.visualize_horizon(image_tensor, original_image)
        
        return prediction, visualization
