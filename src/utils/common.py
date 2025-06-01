import os
import random
import numpy as np
import torch
from PIL import Image

def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_device():
    """Get device for PyTorch."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path, target_size=(256, 256)):
    """Load and preprocess an image."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    return img
