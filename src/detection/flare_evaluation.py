import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, Dict, Optional, Union

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the FlareDetectorModel class
from src.models.flare_detector import FlareDetectorModel

class FlareEvaluator:
    """
    Class for evaluating flare/sunburn detection in satellite images
    """
    def __init__(self, model_path: str = None, img_size: int = 224, device: str = None):
        """
        Initialize the flare evaluator with a pre-trained model
        
        Args:
            model_path: Path to the pre-trained flare detector model
            img_size: Size of input images
            device: Device to run inference on ('cpu', 'cuda', or 'mps')
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                      ('mps' if torch.backends.mps.is_available() else 'cpu'))
        else:
            self.device = torch.device(device)
        
        # Set image size
        self.img_size = img_size
        
        # Create model
        self.model = FlareDetectorModel(in_channels=3, img_size=img_size)
        
        # Load pre-trained weights if provided
        if model_path:
            if os.path.exists(model_path):
                try:
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    print(f"Loaded flare detector model from {model_path}")
                except Exception as e:
                    print(f"Error loading flare detector model: {e}")
                    print("Using untrained model")
            else:
                print(f"Model path {model_path} does not exist. Using untrained model.")
        else:
            # Try to find the model in default location
            default_model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'models', 'flare_detector_best.pth'
            )
            if os.path.exists(default_model_path):
                try:
                    self.model.load_state_dict(torch.load(default_model_path, map_location=self.device))
                    print(f"Loaded flare detector model from default path: {default_model_path}")
                except Exception as e:
                    print(f"Error loading flare detector model from default path: {e}")
                    print("Using untrained model")
            else:
                print("No model specified and no default model found. Using untrained model.")
        
        # Set model to evaluation mode
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Define transform for inference
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def has_flare(self, image_path: str) -> bool:
        """
        Determine if an image contains a sun flare or sunburn
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if image has flare, False otherwise
        """
        # Load and preprocess the image
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return False  # Assume no flare if image can't be loaded
        
        # Make prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            confidence = torch.sigmoid(output).item()
            prediction = confidence > 0.5  # Flare present is label 1
        
        return prediction
    
    def evaluate_with_visualization(self, 
                                   image_path: str) -> Tuple[bool, float, Optional[Image.Image]]:
        """
        Evaluate flare presence and return prediction, confidence, and visualization
        
        Args:
            image_path: Path to the image file
            
        Returns:
            tuple: (has_flare, confidence, visualization_image)
        """
        # Load and preprocess the image
        try:
            original_image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return False, 0.0, None
        
        # Make prediction with visualization
        with torch.no_grad():
            # Get prediction
            output = self.model(image_tensor)
            sigmoid_value = torch.sigmoid(output).item()
            prediction = sigmoid_value > 0.5  # Flare present is label 1
            
            # Calculate confidence (as % certainty, regardless of class)
            # For 'has flare' images: confidence = sigmoid_value
            # For 'no flare' images: confidence = 1 - sigmoid_value
            confidence = sigmoid_value if prediction else (1.0 - sigmoid_value)
            
            # Create a side-by-side visualization
            try:
                # Convert original image to numpy array
                np_img = np.array(original_image)
                h, w = np_img.shape[:2]
                
                # Create a copy of the original image for the prediction side
                pred_img = np_img.copy()
                
                # Try to use the model's visualization method if available
                try:
                    # This should work if the model has a visualize_flare method
                    _, vis_img = self.model.visualize_flare(
                        image_tensor.cpu(), original_image, confidence
                    )
                    if vis_img is not None:
                        pred_img = np.array(vis_img)
                except Exception:
                    # Fallback to simple overlay if model visualization fails
                    # Create an overlay with consistent dimensions
                    overlay = np.zeros((h, w, 3), dtype=np.uint8)
                    
                    if prediction:  # Has flare
                        # Yellow overlay for flare
                        overlay[:,:,1] = int(confidence * 100)  # G channel
                        overlay[:,:,2] = int(confidence * 100)  # R channel
                        border_color = (0, 255, 255)  # Yellow border
                    else:  # No flare
                        # Blue overlay for no flare
                        overlay[:,:,0] = int(confidence * 100)  # B channel
                        border_color = (255, 0, 0)  # Blue border
                    
                    # Create the prediction overlay image
                    pred_img = cv2.addWeighted(pred_img, 0.7, overlay, 0.3, 0)
                
                # Add border to prediction image
                border_thickness = 5
                pred_img = cv2.copyMakeBorder(
                    pred_img, 
                    border_thickness, border_thickness, border_thickness, border_thickness,
                    cv2.BORDER_CONSTANT, 
                    value=border_color if 'border_color' in locals() else (0, 255, 255)
                )
                
                # Add text labels to both images
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                text_color = (255, 255, 255)  # White text
                
                # Add "Original" label to original image
                cv2.putText(np_img, "Original", (10, 30), font, font_scale, text_color, font_thickness)
                
                # Add prediction label to prediction image
                flare_text = f"Prediction: {'FLARE' if prediction else 'NO FLARE'} ({confidence:.4f})"
                cv2.putText(pred_img, flare_text, (10, 30), font, font_scale, text_color, font_thickness)
                
                # Create side-by-side image
                # Ensure both images have same height (adding border might change dimensions)
                h1, w1 = np_img.shape[:2]
                h2, w2 = pred_img.shape[:2]
                
                # Create a blank canvas for the side-by-side display
                side_by_side = np.ones((max(h1, h2), w1 + w2, 3), dtype=np.uint8) * 255
                
                # Place the images side by side
                side_by_side[:h1, :w1] = np_img
                side_by_side[:h2, w1:w1+w2] = pred_img
                
                # Convert to PIL Image
                visualization = Image.fromarray(side_by_side)
                
            except Exception as e:
                print(f"Error generating visualization: {e}")
                visualization = None
        
        return prediction, confidence, visualization
    
    def evaluate_image_dict(self, image_path: str) -> Dict[str, Union[bool, float]]:
        """
        Evaluate flare presence and return a dictionary with results
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Dictionary with flare evaluation results
        """
        prediction, confidence, _ = self.evaluate_with_visualization(image_path)
        
        return {
            "flare": prediction,
            "flare_confidence": float(confidence)
        }


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate flare presence")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image file")
    parser.add_argument("--model_path", type=str, 
                       default="models/flare_detector_best.pth", 
                       help="Path to flare detector model weights")
    parser.add_argument("--show", action="store_true", help="Show visualization")
    
    args = parser.parse_args()
    
    # Create flare evaluator
    evaluator = FlareEvaluator(model_path=args.model_path)
    
    # Evaluate flare presence
    has_flare, confidence, visualization = evaluator.evaluate_with_visualization(args.image_path)
    
    # Print result
    result = "FLARE" if has_flare else "NO FLARE"
    print(f"Flare detection: {result} (confidence: {confidence:.4f})")
    
    # Show visualization if requested
    if args.show and visualization:
        visualization.show()
