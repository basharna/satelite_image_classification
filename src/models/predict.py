import os
import sys
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.horizon_detector import HorizonDetectorModel
from src.models.flare_detector import FlareDetectorModel
from src.utils.common import get_device

def load_model(model_path, model_type='horizon'):
    """
    Load a trained model from a checkpoint file
    
    Args:
        model_path: Path to the model checkpoint
        model_type: Type of model to load ('horizon' or 'flare')
        
    Returns:
        tuple: (model, img_size) - The loaded model and the image size it expects
    """
    device = get_device()
    
    # Load state dict first to check dimensions
    state_dict = torch.load(model_path, map_location=device)
    
    # Calculate the correct image size based on the fc1.weight shape in the saved model
    if 'fc1.weight' in state_dict:
        fc1_shape = state_dict['fc1.weight'].shape
        # fc1_shape[1] is the flattened feature size (128 * feature_size * feature_size)
        # We need to solve for feature_size
        # feature_size = sqrt(fc1_shape[1] / 128)
        flat_features = fc1_shape[1]
        feature_size = int((flat_features / 128) ** 0.5)
        img_size = feature_size * 16  # Because feature_size = img_size // 16
    else:
        # Default to 224 if we can't determine from the state dict
        img_size = 224
    
    print(f"Using image size {img_size} for {model_type} model")
    
    if model_type == 'horizon':
        model = HorizonDetectorModel(in_channels=3, img_size=img_size)
    elif model_type == 'flare':
        model = FlareDetectorModel(in_channels=3, img_size=img_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, img_size

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess an image for model prediction
    
    Args:
        image_path: Path to the image file
        target_size: Size to resize the image to
        
    Returns:
        tensor: Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, image

def predict_horizon(model, image_tensor, device):
    """
    Make a horizon prediction using the model
    
    Args:
        model: The trained horizon detector model
        image_tensor: Preprocessed image tensor
        device: Device to run prediction on
        
    Returns:
        is_horizon: Boolean indicating if horizon is detected
        visualization: Visualization image if horizon is detected, None otherwise
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        try:
            prediction, visualization = model.predict_with_visualization(image_tensor)
        except Exception as e:
            print(f"Error during horizon prediction: {e}")
            return False, None
        
    return prediction, visualization

def predict_flare(model, image_tensor, device):
    """
    Make a flare prediction using the model
    
    Args:
        model: The trained flare detector model
        image_tensor: Preprocessed image tensor
        device: Device to run prediction on
        
    Returns:
        is_flare: Boolean indicating if flare is detected
        visualization: Visualization image if flare is detected, None otherwise
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        try:
            prediction, visualization = model.predict_with_visualization(image_tensor)
        except Exception as e:
            print(f"Error during flare prediction: {e}")
            return False, None
        
    return prediction, visualization

def visualize_predictions(image, horizon_result, flare_result, quality_result=None, save_path=None, model_type=None):
    """
    Visualize predictions based on the selected model type
    
    Args:
        image: Original image
        horizon_result: Tuple of (is_horizon, visualization)
        flare_result: Tuple of (is_flare, visualization)
        quality_result: Quality assessment result (optional)
        save_path: Path to save the visualization (optional)
        model_type: Type of model to visualize ('horizon', 'flare', or None for both)
    """
    is_horizon, horizon_vis = horizon_result
    is_flare, flare_vis = flare_result
    
    # Determine how many subplots to create based on model_type
    if model_type == 'horizon':
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis('off')
        
        # Horizon detection
        if is_horizon and horizon_vis is not None:
            axes[1].imshow(horizon_vis)
            axes[1].set_title("Horizon Detected", fontsize=12)
        else:
            axes[1].imshow(image)
            axes[1].set_title("No Horizon Detected", fontsize=12)
        axes[1].axis('off')
        
    elif model_type == 'flare':
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis('off')
        
        # Flare detection
        if is_flare and flare_vis is not None:
            axes[1].imshow(flare_vis)
            axes[1].set_title("Flare Detected", fontsize=12)
        else:
            axes[1].imshow(image)
            axes[1].set_title("No Flare Detected", fontsize=12)
        axes[1].axis('off')
        
    else:  # Show both models
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis('off')
        
        # Horizon detection
        if is_horizon and horizon_vis is not None:
            axes[1].imshow(horizon_vis)
            axes[1].set_title("Horizon Detected", fontsize=12)
        else:
            axes[1].imshow(image)
            axes[1].set_title("No Horizon Detected", fontsize=12)
        axes[1].axis('off')
        
        # Flare detection
        if is_flare and flare_vis is not None:
            axes[2].imshow(flare_vis)
            axes[2].set_title("Flare Detected", fontsize=12)
        else:
            axes[2].imshow(image)
            axes[2].set_title("No Flare Detected", fontsize=12)
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def compress_image(image, target_size_kb=100, quality_start=95):
    """
    Compress an image to a target file size
    
    Args:
        image: PIL Image to compress
        target_size_kb: Target file size in KB
        quality_start: Starting quality for compression
        
    Returns:
        compressed_image: Compressed image as bytes
        compressed_size_kb: Size of the compressed image in KB
    """
    quality = quality_start
    min_quality = 20  # Don't go below this quality
    
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    
    # Try different quality settings until we get below target size
    while quality >= min_quality:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=quality)
        img_byte_arr.seek(0)
        size_kb = len(img_byte_arr.getvalue()) / 1024
        
        if size_kb <= target_size_kb:
            break
            
        # Reduce quality for next iteration
        quality -= 5
    
    return img_byte_arr.getvalue(), size_kb

def classify_image(image_path, models_dir=None, visualize=False, output_path=None, model_type=None):
    """
    Classify an image using all available models
    
    Args:
        image_path: Path to the image file
        models_dir: Directory containing model checkpoints
        visualize: Whether to visualize the predictions
        output_path: Path to save the visualization
        
    Returns:
        result: Dictionary containing classification results
    """
    device = get_device()
    
    # Set models directory if not provided
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
    
    # Define paths to model checkpoints
    horizon_model_path = os.path.join(models_dir, 'horizon_detector_best.pth')
    flare_model_path = os.path.join(models_dir, 'flare_detector_best.pth')
    
    # Load models if available
    horizon_model = None
    flare_model = None
    horizon_img_size = 224  # Default
    flare_img_size = 224    # Default
    
    # If model_type is specified, only load that model
    if model_type == 'horizon' or model_type is None:
        if os.path.exists(horizon_model_path):
            print(f"Loading horizon model from {horizon_model_path}")
            horizon_model, horizon_img_size = load_model(horizon_model_path, model_type='horizon')
        elif model_type == 'horizon':  # Only warn if specifically requested
            print(f"Warning: Horizon model not found at {horizon_model_path}")
    
    if model_type == 'flare' or model_type is None:
        if os.path.exists(flare_model_path):
            print(f"Loading flare model from {flare_model_path}")
            flare_model, flare_img_size = load_model(flare_model_path, model_type='flare')
        elif model_type == 'flare':  # Only warn if specifically requested
            print(f"Warning: Flare model not found at {flare_model_path}")
    
    # Initialize results
    result = {
        "horizon": False,
        "flare": False,
        "quality": "bad",  # Default to bad until we implement quality detection
        "compressed": None,
        "compressed_size_kb": None
    }
    
    # Make predictions
    horizon_result = (False, None)
    flare_result = (False, None)
    
    # Preprocess and predict for horizon model
    if horizon_model is not None:
        # Preprocess image with the correct size for horizon model
        print(f"Processing image for horizon detection with size {horizon_img_size}x{horizon_img_size}")
        horizon_tensor, original_image = preprocess_image(image_path, target_size=(horizon_img_size, horizon_img_size))
        is_horizon, horizon_vis = predict_horizon(horizon_model, horizon_tensor, device)
        result["horizon"] = bool(is_horizon)
        horizon_result = (is_horizon, horizon_vis)
        print(f"Horizon detection: {'Detected' if is_horizon else 'Not detected'}")
    
    # Preprocess and predict for flare model
    if flare_model is not None:
        # Preprocess image with the correct size for flare model
        print(f"Processing image for flare detection with size {flare_img_size}x{flare_img_size}")
        flare_tensor, original_image = preprocess_image(image_path, target_size=(flare_img_size, flare_img_size))
        is_flare, flare_vis = predict_flare(flare_model, flare_tensor, device)
        result["flare"] = bool(is_flare)
        flare_result = (is_flare, flare_vis)
        print(f"Flare detection: {'Detected' if is_flare else 'Not detected'}")
    
    # For now, set quality to "good" if no flare is detected
    # This is temporary until we implement the quality detection model
    result["quality"] = "good" if not result["flare"] else "bad"
    
    # Compress image if quality is good
    if result["quality"] == "good":
        try:
            from io import BytesIO
            import io
            
            # Compress the image
            compressed_data, size_kb = compress_image(original_image)
            result["compressed_size_kb"] = round(size_kb, 2)
            print(f"Image compressed to {size_kb:.2f} KB")
            
            # Save compressed image if output path is provided
            if output_path:
                compressed_path = os.path.splitext(output_path)[0] + "_compressed.jpg"
                with open(compressed_path, 'wb') as f:
                    f.write(compressed_data)
                print(f"Compressed image saved to {compressed_path}")
        except Exception as e:
            print(f"Error compressing image: {e}")
    
    # Visualize predictions
    if visualize:
        save_path = output_path if output_path else None
        visualize_predictions(original_image, horizon_result, flare_result, save_path=save_path, model_type=model_type)
    
    return result

def main(args):
    # Classify the image
    result = classify_image(
        image_path=args.image_path,
        models_dir=args.models_dir,
        visualize=args.visualize,
        output_path=args.output_path,
        model_type=args.model_type
    )
    
    # Print results as JSON
    print("\nClassification results:")
    print(json.dumps(result, indent=2))
    
    # Save results to file if output path is provided
    if args.output_path:
        results_path = os.path.splitext(args.output_path)[0] + "_results.json"
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify Satellite Image')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
    parser.add_argument('--models_dir', type=str, default=None, help='Directory containing model checkpoints')
    parser.add_argument('--visualize', action='store_true', help='Visualize the predictions')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the visualization and results')
    parser.add_argument('--model_type', type=str, choices=['horizon', 'flare'], default=None, 
                        help='Specify a single model type to use (horizon or flare). If not specified, both models will be used.')
    
    args = parser.parse_args()
    main(args)
