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

from src.models.horizon_detector import HorizonDetectorModel
from src.data.dataset import get_data_loaders
from src.utils.common import set_seed, get_device, create_dir

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, save_dir='models'):
    """
    Train the horizon detector model
    
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
                # Labels: horizon=1, no_horizon=0 (as per PRD requirements)
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
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.bool())
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
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
                    torch.save(model.state_dict(), os.path.join(save_dir, 'horizon_detector_best.pth'))
                    print(f"Saved best model with validation accuracy: {best_val_acc:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, os.path.join(save_dir, f'horizon_detector_epoch_{epoch+1}.pth'))
        
        print()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'horizon_detector_final.pth'))
    
    # Plot training history
    plot_training_history(history, save_dir)
    
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
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'horizon_detector_training_history.png'))
    plt.close()
    
    # Display a message about where to find the saved plots
    print(f"\nTraining plots saved to {os.path.join(save_dir, 'horizon_detector_training_history.png')}")

def evaluate_model(model, dataloader, criterion, device, save_dir=None, num_visualizations=5):
    """
    Evaluate the model on the test set and generate visualizations
    
    Args:
        model: The trained model
        dataloader: Test dataloader
        criterion: Loss function
        device: Device to evaluate on (cuda or cpu)
        save_dir: Directory to save visualizations
        num_visualizations: Number of example visualizations to save
        
    Returns:
        test_loss: Average test loss
        test_acc: Test accuracy
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    # For visualization
    if save_dir:
        create_dir(os.path.join(save_dir, 'visualizations'))
        vis_count = 0
        vis_samples = []
    
    # Iterate over data
    for inputs, labels in tqdm(dataloader, desc='Testing'):
        inputs = inputs.to(device)
        labels = labels.to(device).float().view(-1, 1)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            loss = criterion(outputs, labels)
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.bool())
        
        # Save visualizations for correctly predicted horizons
        if save_dir and vis_count < num_visualizations:
            for i in range(inputs.size(0)):
                # Only visualize correctly predicted horizons
                if preds[i] == 1 and labels[i] == 1 and vis_count < num_visualizations:
                    # Generate visualization
                    img_tensor = inputs[i].cpu()
                    prediction, visualization = model.predict_with_visualization(img_tensor)
                    
                    if visualization:
                        # Save the visualization
                        vis_path = os.path.join(save_dir, 'visualizations', f'horizon_vis_{vis_count}.png')
                        visualization.save(vis_path)
                        print(f"Saved horizon visualization to {vis_path}")
                        
                        # Add to samples for grid
                        vis_np = np.array(visualization)
                        vis_tensor = torch.from_numpy(vis_np.transpose((2, 0, 1))).float() / 255.0
                        vis_samples.append(vis_tensor)
                        
                        vis_count += 1
    
    # Create a grid of visualizations
    if save_dir and vis_samples:
        grid = make_grid(vis_samples, nrow=min(5, len(vis_samples)))
        grid_path = os.path.join(save_dir, 'horizon_visualizations_grid.png')
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
                           'data', 'processed', 'horizon_detector')
    dataloaders = get_data_loaders(data_dir, batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers)
    
    # Create model
    model = HorizonDetectorModel(in_channels=3, img_size=args.img_size)  # 3 channels for RGB images
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
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
    with open(os.path.join(results_dir, 'horizon_detector_test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Horizon Detector Model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for resizing (224 or 256 as per PRD)')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for regularization')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_visualizations', type=int, default=10, help='Number of horizon visualizations to generate')
    
    args = parser.parse_args()
    main(args)

