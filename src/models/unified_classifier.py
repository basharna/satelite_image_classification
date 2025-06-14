import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, Any, Union, Optional
import json
import cv2

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the detector models
from src.models.horizon_detector import HorizonDetectorModel
from src.models.flare_detector import FlareDetectorModel
from src.quality.quality_evaluation import QualityEvaluator
from src.compression.compress import compress_image  # Assuming this function exists

class SatelliteImageClassifier:
    """
    Unified satellite image classification system that:
    1. Detects horizon
    2. Detects flares (sunburn)
    3. Evaluates image quality
    4. Compresses good quality images
    """
    
    def __init__(
        self,
        horizon_model_path: Optional[str] = None,
        flare_model_path: Optional[str] = None,
        quality_model_path: Optional[str] = None,
        img_size: int = 224,
        device: Optional[str] = None
    ):
        """
        Initialize the classifier with pre-trained models
        
        Args:
            horizon_model_path: Path to the pre-trained horizon detector model
            flare_model_path: Path to the pre-trained flare detector model
            quality_model_path: Path to the pre-trained quality detector model
            img_size: Size of input images
            device: Device to run inference on ('cpu', 'cuda', or 'mps')
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                      ('mps' if torch.backends.mps.is_available() else 'cpu'))
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Set image size
        self.img_size = img_size
        
        # Define transform for inference
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Get models directory if paths not provided
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'models'
        )
        
        # Initialize horizon detector
        self.horizon_detector = self._init_horizon_detector(
            horizon_model_path or os.path.join(models_dir, 'horizon_detector_best.pth')
        )
        
        # Initialize flare detector
        self.flare_detector = self._init_flare_detector(
            flare_model_path or os.path.join(models_dir, 'flare_detector_best.pth')
        )
        
        # Initialize quality evaluator
        self.quality_evaluator = QualityEvaluator(
            model_path=quality_model_path or os.path.join(models_dir, 'quality_detector_best.pth'),
            img_size=img_size,
            device=self.device
        )
    
    def _init_horizon_detector(self, model_path: str) -> Optional[HorizonDetectorModel]:
        """Initialize and load the horizon detector model"""
        try:
            model = HorizonDetectorModel(in_channels=3, img_size=self.img_size)
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded horizon detector model from {model_path}")
            else:
                print(f"Horizon model path {model_path} does not exist. Using untrained model.")
            
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error initializing horizon detector: {e}")
            return None
    
    def _init_flare_detector(self, model_path: str) -> Optional[FlareDetectorModel]:
        """Initialize and load the flare detector model"""
        try:
            model = FlareDetectorModel(in_channels=3, img_size=self.img_size)
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded flare detector model from {model_path}")
            else:
                print(f"Flare model path {model_path} does not exist. Using untrained model.")
            
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error initializing flare detector: {e}")
            return None
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """
        Classify a satellite image and return the results
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Dictionary with classification results
        """
        # Load the image
        try:
            original_image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
        except Exception as e:
            return {
                "error": f"Error loading image: {str(e)}",
                "horizon": False,
                "flare": False,
                "quality": "bad",
                "compressed": None
            }
        
        result = {}
        
        # Detect horizon
        if self.horizon_detector:
            with torch.no_grad():
                horizon_output = self.horizon_detector(image_tensor)
                horizon_confidence = torch.sigmoid(horizon_output).item()
                is_horizon = horizon_confidence > 0.5
                # Calculate confidence as certainty percentage regardless of class
                confidence = horizon_confidence if is_horizon else (1.0 - horizon_confidence)
                result["horizon"] = bool(is_horizon)
                result["horizon_confidence"] = float(confidence)
        else:
            result["horizon"] = False
            result["horizon_confidence"] = 0.0
            
        # Detect flare
        if self.flare_detector:
            with torch.no_grad():
                flare_output = self.flare_detector(image_tensor)
                flare_confidence = torch.sigmoid(flare_output).item()
                is_flare = flare_confidence > 0.5
                # Calculate confidence as certainty percentage regardless of class
                confidence = flare_confidence if is_flare else (1.0 - flare_confidence)
                result["flare"] = bool(is_flare)
                result["flare_confidence"] = float(confidence)
        else:
            result["flare"] = False
            result["flare_confidence"] = 0.0
        
        # Evaluate quality
        quality_result = self.quality_evaluator.evaluate_image_dict(image_path)
        result["quality"] = quality_result["quality"]
        result["quality_confidence"] = quality_result["quality_confidence"]
        
        # Compress if good quality
        if result["quality"] == "good":
            try:
                compressed_path = image_path.replace(".jpg", "_compressed.jpg").replace(".png", "_compressed.jpg")
                compressed_size = compress_image(image_path, compressed_path)
                result["compressed"] = {
                    "path": compressed_path,
                    "compressed_size_kb": compressed_size
                }
            except Exception as e:
                result["compressed"] = None
                result["compression_error"] = str(e)
        else:
            result["compressed"] = None
        
        return result
    
    def visualize_results(self, image_path: str, save_path: Optional[str] = None) -> Optional[Image.Image]:
        """
        Create a visualization of the classification results
        
        Args:
            image_path: Path to the image file
            save_path: Path to save the visualization (if None, doesn't save)
            
        Returns:
            PIL.Image.Image or None: Visualization image
        """
        # Get classification results
        results = self.classify_image(image_path)
        
        # Load the original image
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image for visualization: {e}")
            return None
        
        # Resize for display if needed
        display_height = 500
        h, w = image.shape[:2]
        display_width = int(w * (display_height / h))
        image_display = cv2.resize(image, (display_width, display_height))
        
        # Create a blank area for text
        text_area_height = 150
        text_area = np.ones((text_area_height, display_width, 3), dtype=np.uint8) * 255
        
        # Combine image and text area
        combined = np.vstack([image_display, text_area])
        
        # Add classification results text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        line_height = 30
        
        # Format lines of text
        lines = [
            f"Horizon: {'YES' if results.get('horizon', False) else 'NO'} (Confidence: {results.get('horizon_confidence', 0):.2f})",
            f"Flare: {'YES' if results.get('flare', False) else 'NO'} (Confidence: {results.get('flare_confidence', 0):.2f})",
            f"Quality: {results.get('quality', 'bad').upper()} (Confidence: {results.get('quality_confidence', 0):.2f})"
        ]
        
        if results.get('compressed'):
            lines.append(f"Compressed Size: {results['compressed']['compressed_size_kb']:.2f} KB")
        else:
            lines.append("Not compressed (bad quality)")
        
        # Add text to the combined image
        for i, line in enumerate(lines):
            y = display_height + (i+1) * line_height
            color = (0, 0, 0)  # Black text
            cv2.putText(combined, line, (10, y), font, font_scale, color, 2)
        
        # Convert back to PIL Image
        visualization = Image.fromarray(combined)
        
        # Save if requested
        if save_path:
            try:
                # Create directory if it doesn't exist
                output_dir = os.path.dirname(save_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"Created directory: {output_dir}")
                
                # Add default extension if none provided
                if not os.path.splitext(save_path)[1]:
                    save_path = f"{save_path}.jpg"
                
                # Save the visualization
                visualization.save(save_path)
                print(f"Saved visualization to {save_path}")
            except Exception as e:
                print(f"Error saving visualization: {e}")
        
        return visualization


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Satellite Image Classifier")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image file")
    parser.add_argument("--horizon_model", type=str, default=None, help="Path to horizon detector model")
    parser.add_argument("--flare_model", type=str, default=None, help="Path to flare detector model")
    parser.add_argument("--quality_model", type=str, default=None, help="Path to quality detector model")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    parser.add_argument("--save_viz", type=str, default=None, help="Path to save visualization")
    
    args = parser.parse_args()
    
    # Create classifier
    classifier = SatelliteImageClassifier(
        horizon_model_path=args.horizon_model,
        flare_model_path=args.flare_model,
        quality_model_path=args.quality_model
    )
    
    # Classify image
    result = classifier.classify_image(args.image_path)
    
    # Print result
    print(json.dumps(result, indent=2))
    
    # Visualize if requested
    if args.visualize or args.save_viz:
        visualization = classifier.visualize_results(args.image_path, save_path=args.save_viz)
        if args.visualize and visualization:
            visualization.show()
