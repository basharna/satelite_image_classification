import torch
import torch.nn as nn
import torch.nn.functional as F

class HorizonDetectorModel(nn.Module):
    """
    CNN model for horizon detection in satellite images.
    Binary classification: horizon present (1) or not present (0).
    """
    def __init__(self, in_channels=3):
        super(HorizonDetectorModel, self).__init__()
        
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
        
        # Calculate the size of the flattened features
        # For 256x256 input, after 4 pooling layers (each dividing dimensions by 2):
        # 256 -> 128 -> 64 -> 32 -> 16
        # So the feature map size is 16x16 with 128 channels
        self.flat_features = 128 * 16 * 16
        
        # Classification layers
        self.fc1 = nn.Linear(self.flat_features, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 1)  # Binary classification
        
    def forward(self, x):
        # Feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, self.flat_features)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        # Return logits (no sigmoid here as we'll use BCEWithLogitsLoss)
        return x
