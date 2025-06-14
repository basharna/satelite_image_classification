import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
# Configure matplotlib to use 'Agg' backend for non-interactive environments
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision.utils import make_grid, save_image

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.quality_detector import QualityDetectorModel
from src.data.dataset import get_data_loaders
from src.utils.common import set_seed, get_device, create_dir

class DiagnosticBCEWithLogitsLoss(nn.Module):
    """Wrapper around BCEWithLogitsLoss to provide diagnostic information"""
    def __init__(self):
        super(DiagnosticBCEWithLogitsLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, outputs, targets):
        # Store diagnostic information
        with torch.no_grad():
            # Check for NaN values
            if torch.isnan(outputs).any():
                print("WARNING: NaN values detected in model outputs!")
                
            # Check for extreme values
            max_abs_output = torch.max(torch.abs(outputs)).item()
            if max_abs_output > 20:
                print(f"WARNING: Extreme output values detected: {max_abs_output:.2f}")
        
        # Calculate the actual loss
        loss = self.bce_loss(outputs, targets)
        
        # Check if loss is negative (which shouldn't happen with BCE)
        if loss.item() < 0:
            print(f"ERROR: Negative loss value detected: {loss.item()}")
            # Provide more details to debug
            print(f"Outputs shape: {outputs.shape}, min: {outputs.min().item()}, max: {outputs.max().item()}")
            print(f"Targets shape: {targets.shape}, min: {targets.min().item()}, max: {targets.max().item()}")
            
            # Force a reasonable loss value to prevent training collapse
            loss = torch.clamp(loss, min=0.0)
            
        return loss

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, save_dir='models'):
    """
    Train the quality detector model
    
    Args:
        model: The model to train
        dataloaders: Dictionary containing train and val dataloaders
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda or cpu)
        num_epochs: Number of epochs to train for
        save_dir: Directory to save model checkpoints
        
    Returns:
        model: The trained model
        history: Dictionary containing training and validation metrics
    """
    create_dir(save_dir)
    
    # Initialize history dictionary to store metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device).float() # BCEWithLogitsLoss expects float targets
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass - track history only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # Shape: [batch_size, 1]
                    outputs = outputs.squeeze() # Shape: [batch_size]
                    
                    # Calculate loss
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        # Clip gradients to prevent exploding gradients
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                running_corrects += torch.sum(preds == labels).item()
                
            # Calculate epoch metrics
            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects / dataset_size
            
            # Print metrics
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Store metrics in history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            
            # Save best validation model
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(save_dir, 'quality_detector_best.pth'))
                print(f'Saved new best model with validation accuracy: {epoch_acc:.4f}')
        
        # Save model after each epoch
        torch.save(model.state_dict(), os.path.join(save_dir, 'quality_detector_last.pth'))
        print()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'quality_detector_final.pth'))
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    
    return model, history

def plot_training_history(history, save_dir):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary containing training and validation metrics
        save_dir: Directory to save plots
    """
    create_dir(save_dir)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'quality_detector_training_history.png'))
    plt.close()

def evaluate_model(model, dataloader, criterion, device, save_dir=None, num_visualizations=5):
    """
    Evaluate the model on the test set and generate visualizations
    
    Args:
        model: The trained model
        dataloader: Test dataloader
        criterion: Loss function
        device: Device to evaluate on (cuda or cpu)
        save_dir: Directory to save visualizations
        num_visualizations: Number of visualizations to save
        
    Returns:
        test_loss: Average test loss
        test_acc: Test accuracy
    """
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    # For visualizations
    viz_samples = []
    viz_preds = []
    viz_labels = []
    
    # Evaluate model
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Testing'):
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            
            # Forward pass
            outputs = model(inputs)
            outputs = outputs.squeeze()
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            running_corrects += torch.sum(preds == labels).item()
            
            # Store predictions and labels for visualization
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Store samples for visualization (only store a few)
            if len(viz_samples) < num_visualizations:
                # Find both correct and incorrect predictions
                correct_indices = (preds == labels).nonzero(as_tuple=True)[0]
                incorrect_indices = (preds != labels).nonzero(as_tuple=True)[0]
                
                indices_to_vis = []
                # Prioritize some incorrect predictions if there are any
                if len(incorrect_indices) > 0:
                    indices_to_vis.extend(incorrect_indices[:min(2, len(incorrect_indices))])
                
                # Add correct predictions to reach the desired number
                if len(correct_indices) > 0:
                    indices_to_vis.extend(correct_indices[:min(num_visualizations - len(indices_to_vis), len(correct_indices))])
                
                for idx in indices_to_vis:
                    if len(viz_samples) < num_visualizations:
                        viz_samples.append(inputs[idx].cpu())
                        viz_preds.append(preds[idx].item())
                        viz_labels.append(labels[idx].item())
    
    # Calculate metrics
    dataset_size = len(dataloader.dataset)
    test_loss = running_loss / dataset_size
    test_acc = running_corrects / dataset_size
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    # Generate and save visualizations
    if save_dir and viz_samples:
        create_dir(os.path.join(save_dir, 'quality_visualizations'))
        
        for i, (img, pred, label) in enumerate(zip(viz_samples, viz_preds, viz_labels)):
            # Convert tensor to PIL image for visualization - with proper error handling
            try:
                # Make sure we have the correct shape and type
                img_np = img.numpy()
                
                # If img_np has shape like (1, 3, H, W) or some unexpected shape, fix it
                if img_np.ndim == 4 and img_np.shape[0] == 1:
                    img_np = img_np[0]  # Remove batch dimension
                
                # Ensure we have a proper 3-channel image in CHW format
                if img_np.shape[0] == 3:
                    # Convert from CHW to HWC format
                    img_np = np.transpose(img_np, (1, 2, 0))
                
                # Scale from [-1, 1] to [0, 255]
                img_np = (img_np * 0.5 + 0.5) * 255
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                
                # Create the PIL image
                original_image = Image.fromarray(img_np, 'RGB')
            except Exception as e:
                print(f"Error during image conversion: {e}")
                # Create a fallback image if conversion fails
                original_image = Image.new('RGB', (224, 224), color='gray')
            
            # Generate visualization
            _, visualization = model.predict_with_visualization(img, original_image)
            
            # Save visualization
            pred_str = "Good" if pred == 1 else "Bad"
            label_str = "Good" if label == 1 else "Bad" 
            result_str = "Correct" if pred == label else "Incorrect"
            
            save_path = os.path.join(save_dir, 'quality_visualizations', 
                                    f'quality_{i+1}_{pred_str}_{label_str}_{result_str}.png')
            visualization.save(save_path)
    
    return test_loss, test_acc

def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get device (cuda or cpu)
    device = get_device()
    print(f"Using device: {device}")
    
    # Get data paths
    base_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    raw_data_dir = os.path.join(base_data_dir, 'classification_sets', 'quality_detection')
    processed_data_dir = os.path.join(base_data_dir, 'processed', 'quality_detector')
    
    # Check if the processed data directory exists, if not create it
    if not os.path.exists(processed_data_dir):
        print(f"Processed data directory not found: {processed_data_dir}")
        print("Creating directory structure...")
        
        # Create the directory structure
        os.makedirs(os.path.join(processed_data_dir, 'train', 'good'), exist_ok=True)
        os.makedirs(os.path.join(processed_data_dir, 'train', 'bad'), exist_ok=True)
        os.makedirs(os.path.join(processed_data_dir, 'val', 'good'), exist_ok=True)
        os.makedirs(os.path.join(processed_data_dir, 'val', 'bad'), exist_ok=True)
        os.makedirs(os.path.join(processed_data_dir, 'test', 'good'), exist_ok=True)
        os.makedirs(os.path.join(processed_data_dir, 'test', 'bad'), exist_ok=True)
        
        print("Please copy the quality detection images into the appropriate directories before training.")
        print(f"Source: {raw_data_dir}")
        print(f"Destination: {processed_data_dir}")
        return
    
    # Get data loaders with augmentation for quality detection
    # Note: The get_data_loaders function should be modified to support quality detection specific augmentations
    dataloaders = get_data_loaders(
        processed_data_dir, 
        batch_size=args.batch_size,
        img_size=args.img_size, 
        num_workers=args.num_workers,
        augment_quality=True  # Enable quality-specific augmentations
    )
    
    # Create model
    model = QualityDetectorModel(in_channels=3, img_size=args.img_size)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = DiagnosticBCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Define paths for saving results
    model_save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results')
    create_dir(results_dir)
    
    # Train model
    model, history = train_model(
        model=model,
        dataloaders={'train': dataloaders['train'], 'val': dataloaders['val']},
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        save_dir=model_save_dir
    )
    
    # Plot training history
    plot_training_history(history, results_dir)
    
    # Evaluate model on test set and generate visualizations
    test_loss, test_acc = evaluate_model(
        model=model,
        dataloader=dataloaders['test'],
        criterion=criterion,
        device=device,
        save_dir=results_dir,
        num_visualizations=args.num_visualizations
    )
    
    print(f"Final test accuracy: {test_acc:.4f}")
    
    # Save test results
    with open(os.path.join(results_dir, 'quality_detector_test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Quality Detector Model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for resizing (224 or 256 as per PRD)')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_visualizations', type=int, default=10, help='Number of quality visualizations to generate')
    
    args = parser.parse_args()
    main(args)
