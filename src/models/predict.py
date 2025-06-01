import os
import sys
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.horizon_detector import HorizonDetectorModel
from src.utils.common import get_device

def load_model(model_path):
    """
    Load a trained model from a checkpoint file
    
    Args:
        model_path: Path to the model checkpoint
        
    Returns:
        model: The loaded model
    """
    device = get_device()
    model = HorizonDetectorModel(in_channels=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

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

def predict(model, image_tensor, device):
    """
    Make a prediction using the model
    
    Args:
        model: The trained model
        image_tensor: Preprocessed image tensor
        device: Device to run prediction on
        
    Returns:
        class_name: Predicted class name ('horizon' or 'no_horizon')
        confidence: Confidence score (0 to 1)
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        confidence = torch.sigmoid(output).item()
        
        # In our dataset, class 0 is 'horizon' and class 1 is 'no_horizon'
        # The model outputs a high value for class 1, so we need to invert the confidence
        # to get the confidence for the 'horizon' class
        horizon_confidence = 1.0 - confidence
        class_name = 'horizon' if horizon_confidence >= 0.5 else 'no_horizon'
        
    return class_name, horizon_confidence if class_name == 'horizon' else confidence

def visualize_prediction(image, class_name, confidence, save_path=None):
    """
    Visualize the prediction
    
    Args:
        image: Original image
        class_name: Predicted class name ('horizon' or 'no_horizon')
        confidence: Confidence score (0 to 1)
        save_path: Path to save the visualization (optional)
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    
    result = "Horizon Detected" if class_name == 'horizon' else "No Horizon"
    title = f"{result} (Confidence: {confidence:.4f})"
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def main(args):
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    if args.model_path is None or not os.path.exists(args.model_path):
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                 'models', 'horizon_detector_best.pth')
    else:
        model_path = args.model_path
    
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    # Preprocess image
    print(f"Processing image: {args.image_path}")
    image_tensor, original_image = preprocess_image(args.image_path)
    
    # Make prediction
    class_name, confidence = predict(model, image_tensor, device)
    
    # Print results
    result = "Horizon Detected" if class_name == 'horizon' else "No Horizon"
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.4f}")
    
    # Visualize prediction
    if args.visualize:
        save_path = args.output_path if args.output_path else None
        visualize_prediction(original_image, class_name, confidence, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Horizon in Satellite Image')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model checkpoint')
    parser.add_argument('--visualize', action='store_true', help='Visualize the prediction')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the visualization')
    
    args = parser.parse_args()
    main(args)
