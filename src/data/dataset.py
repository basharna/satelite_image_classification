import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class SatelliteImageDataset(Dataset):
    """
    Dataset class for satellite images
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images organized in class folders
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.transform = transform
        self.classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        # Explicitly map classes to indices according to PRD requirements:
        # horizon = 1, no_horizon = 0
        self.class_to_idx = {}
        for cls_name in self.classes:
            if cls_name == 'horizon':
                self.class_to_idx[cls_name] = 1  # Horizon visible = 1
            elif cls_name == 'no_horizon':
                self.class_to_idx[cls_name] = 0  # No horizon = 0
            else:
                # For any other classes, assign sequential indices starting from 2
                # This ensures compatibility with other datasets if needed
                self.class_to_idx[cls_name] = len(self.class_to_idx) + 2
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(data_dir, batch_size=32, img_size=256, num_workers=4):
    """
    Create data loaders for training, validation, and testing
    
    Args:
        data_dir (string): Base directory with train, val, test subdirectories
        batch_size (int): Batch size for data loaders
        img_size (int): Size to resize images to
        num_workers (int): Number of workers for data loading
        
    Returns:
        dict: Dictionary containing train, val, and test data loaders
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),  # Increased rotation range from 10 to 30 degrees
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SatelliteImageDataset(
        os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = SatelliteImageDataset(
        os.path.join(data_dir, 'val'),
        transform=val_test_transform
    )
    
    test_dataset = SatelliteImageDataset(
        os.path.join(data_dir, 'test'),
        transform=val_test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
