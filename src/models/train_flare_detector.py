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

from src.models.flare_detector import FlareDetectorModel
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
                
            # Print some sample outputs and targets for inspection
            if outputs.size(0) > 0:
                sample_idx = 0  # Just look at the first sample in the batch
                # print(f"Sample output: {outputs[sample_idx].item():.4f}, target: {targets[sample_idx].item():.1f}")
                # print(f"Sigmoid of output: {torch.sigmoid(outputs[sample_idx]).item():.6f}")
        
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
    Train the flare detector model
    
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
                # Labels: flare=1, no_flare=0 (as per PRD requirements)
                labels = labels.to(device).float().view(-1, 1)  # Convert to float and reshape for BCE loss
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs) > 0.5
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # Add gradient clipping to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                    # Monitor the raw logits to detect any issues
                    with torch.no_grad():
                        max_logit = torch.max(torch.abs(outputs)).item()
                        if max_logit > 20:  # Alert if logits are getting too large
                            print(f"WARNING: Large logit detected: {max_logit:.2f}. This may cause numerical instability.")
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.bool())
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase} Dataset Size: {len(dataloaders[phase].dataset)} images, {len(dataloaders[phase])} batches')
            
            # Print class distribution for better understanding
            class_counts = {}
            for _, labels_batch in dataloaders[phase]:
                for label in labels_batch:
                    label_val = 'flare' if label.item() == 1 else 'no_flare'
                    if label_val in class_counts:
                        class_counts[label_val] += 1
                    else:
                        class_counts[label_val] = 1
            print(f'{phase} Class Distribution: {class_counts}')
            
            # Store metrics
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Save best model
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(save_dir, 'flare_detector_best.pth'))
        
        print()
    
    # Load best model weights
    model.load_state_dict(torch.load(os.path.join(save_dir, 'flare_detector_best.pth')))
    
    return model, history

def plot_training_history(history, save_dir):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary containing training and validation metrics
        save_dir: Directory to save plots
    """
    create_dir(save_dir)
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'flare_detector_loss.png'))
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'flare_detector_accuracy.png'))
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
        num_visualizations: Number of visualizations to generate
        
    Returns:
        test_loss: Average test loss
        test_acc: Test accuracy
    """
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    
    # For visualization
    vis_samples = []
    vis_count = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Testing'):
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)
            
            # Forward pass
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.bool())
            
            # Generate visualizations for flare-positive samples
            if save_dir and vis_count < num_visualizations:
                for i, (pred, label) in enumerate(zip(preds, labels)):
                    # Only visualize true positives and false positives
                    if pred.item() == True:  # Flare detected
                        # Get the input image
                        img_tensor = inputs[i].cpu()
                        
                        # Denormalize the tensor for visualization
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_tensor = img_tensor * std + mean
                        img_tensor = torch.clamp(img_tensor, 0, 1)
                        
                        # Convert to PIL for visualization
                        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        img_pil = Image.fromarray(img_np)
                        
                        # Generate visualization
                        _, visualization = model.predict_with_visualization(inputs[i].unsqueeze(0), img_pil)
                        
                        # Save visualization
                        vis_path = os.path.join(save_dir, f'flare_vis_{vis_count}_pred{int(pred.item())}_true{int(label.item())}.png')
                        visualization.save(vis_path)
                        print(f"Saved flare visualization to {vis_path}")
                        
                        # Add to samples for grid
                        vis_np = np.array(visualization)
                        vis_tensor = torch.from_numpy(vis_np.transpose((2, 0, 1))).float() / 255.0
                        vis_samples.append(vis_tensor)
                        
                        vis_count += 1
                        if vis_count >= num_visualizations:
                            break
    
    # Create a grid of visualizations
    if save_dir and vis_samples:
        grid = make_grid(vis_samples, nrow=min(5, len(vis_samples)))
        grid_path = os.path.join(save_dir, 'flare_visualizations_grid.png')
        save_image(grid, grid_path)
        print(f"Saved visualization grid to {grid_path}")
    
    test_loss = running_loss / len(dataloader.dataset)
    test_acc = running_corrects.double() / len(dataloader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    return test_loss, test_acc.item()

def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get data loaders
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                           'data', 'processed', 'flare_detector')
    
    # Check if the processed data directory exists, if not create it
    if not os.path.exists(data_dir):
        print(f"Processed data directory not found: {data_dir}")
        print("Creating directory structure...")
        
        # Create the directory structure
        os.makedirs(os.path.join(data_dir, 'train', 'flare'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'train', 'no_flare'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'val', 'flare'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'val', 'no_flare'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'test', 'flare'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'test', 'no_flare'), exist_ok=True)
        
        print("Please copy the flare detection images into the appropriate directories before training.")
        print(f"Source: {os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'classification_sets', 'flare_detection')}")
        print(f"Destination: {data_dir}")
        return
    
    dataloaders = get_data_loaders(data_dir, batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers)
    
    # Create model
    model = FlareDetectorModel(in_channels=3, img_size=args.img_size)  # 3 channels for RGB images
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = DiagnosticBCEWithLogitsLoss()  # Use our diagnostic wrapper
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay * 10)  # Increase weight decay for regularization
    
    # Train model
    model_save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results')
    create_dir(results_dir)
    
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
    with open(os.path.join(results_dir, 'flare_detector_test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Flare Detector Model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for resizing (224 or 256 as per PRD)')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_visualizations', type=int, default=10, help='Number of flare visualizations to generate')
    
    args = parser.parse_args()
    main(args)
