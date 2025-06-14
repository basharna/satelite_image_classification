import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class QualityDetectorModel(nn.Module):
    """
    CNN model for image quality detection in satellite images.
    Binary classification: good quality (1) or bad quality (0).
    """
    def __init__(self, in_channels=3, img_size=256):
        super(QualityDetectorModel, self).__init__()
        
        # Store image size for visualization
        self.img_size = img_size
        
        # Feature extraction layers with slightly more capacity
        # for assessing multiple quality aspects
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size of the flattened features dynamically
        # After 4 pooling layers (each dividing dimensions by 2):
        # img_size -> img_size/2 -> img_size/4 -> img_size/8 -> img_size/16
        feature_size = img_size // 16
        self.flat_features = 256 * feature_size * feature_size
        
        # Classification layers
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
        x_out = torch.clamp(x_out, min=-10.0, max=10.0)
        
        # Return logits (no sigmoid here as we'll use BCEWithLogitsLoss)
        return x_out
        
    def visualize_quality(self, image_tensor, original_image=None, confidence=None):
        """
        Visualize areas affecting quality assessment using Grad-CAM
        
        Args:
            image_tensor: Input tensor that was used for prediction
            original_image: Optional PIL image to draw on. If None, will use the tensor
            confidence: Confidence score of the prediction (0-1)
            
        Returns:
            PIL Image with quality indicators highlighted
        """
        # Store current model mode to restore it later
        was_training = self.training
        
        # Set model to eval mode for visualization
        self.eval()
        
        # Make sure we have a batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Prepare image for visualization
        if original_image is None:
            # Convert tensor to numpy image for visualization
            img_np = image_tensor.detach().cpu().numpy()[0]
            # Convert from CHW to HWC format and rescale from [-1,1] to [0,255]
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = (img_np * 0.5 + 0.5) * 255
            img_np = img_np.astype(np.uint8)
        else:
            # Use the provided original image
            img_np = np.array(original_image)
            
        # Use the last convolutional layer for feature visualization
        target_layer = 'conv4'
        
        # Get the feature map from the target layer
        if target_layer not in self.feature_maps:
            # If feature map isn't available, do a forward pass to get it
            with torch.no_grad():
                _ = self(image_tensor)
                
        # Get feature activations
        feature_map = self.feature_maps[target_layer].detach().cpu().numpy()[0]
        
        # Create a heatmap from feature maps by averaging across channels
        # This is a simplified version of Grad-CAM
        heatmap = np.mean(feature_map, axis=0)
        
        # Normalize heatmap to [0, 1]
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
        
        # Resize heatmap to match input image size
        heatmap = cv2.resize(heatmap, (self.img_size, self.img_size))
        
        # Apply colormap to create RGB visualization (red zones = issues, green zones = good)
        heatmap_colored = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Custom colorization based on prediction
        if confidence is not None and confidence > 0.5:  # Good quality prediction
            # Highlight good areas in green (areas with high activations)
            heatmap_colored[:,:,1] = (heatmap * 255).astype(np.uint8)
        else:  # Bad quality prediction
            # Highlight problematic areas in red (areas with high activations)
            heatmap_colored[:,:,2] = (heatmap * 255).astype(np.uint8)
        
        # Ensure image has the right format for OpenCV operations
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Overlay heatmap on image
        result_img = cv2.addWeighted(img_np, 0.7, heatmap_colored, 0.3, 0)
        
        # Convert back to RGB for PIL
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Add quality assessment text
        quality_text = ""
        if confidence is not None:
            if confidence > 0.5:
                quality_text = f"Good Quality ({confidence:.2f})"
                cv2.putText(result_img, quality_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                quality_text = f"Bad Quality ({1-confidence:.2f})"
                cv2.putText(result_img, quality_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
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
            # Good quality is label 1, so we check if prediction is 1 (> 0.5)
            prediction = confidence > 0.5
        
        # Create visualization regardless of prediction to show what the model is looking at
        try:
            visualization = self.visualize_quality(image_tensor, original_image, confidence)
        except Exception as e:
            print(f"Warning: Visualization failed with error: {e}")
            # Create a simple placeholder image if visualization fails
            if original_image is not None:
                visualization = original_image.copy()
            else:
                # Create a blank image with text
                img = np.ones((224, 224, 3), dtype=np.uint8) * 255
                if prediction:
                    cv2.putText(img, f"Good Quality ({confidence:.2f})", (50, 112), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(img, f"Bad Quality ({1-confidence:.2f})", (50, 112), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                visualization = Image.fromarray(img)
        
        # Restore original model mode
        if was_training:
            self.train()
            
        return prediction, visualization
