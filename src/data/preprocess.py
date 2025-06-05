import os
import shutil
import random
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.common import create_dir, set_seed

def resize_and_convert(img_path, output_path, target_size=(224, 224)):
    """
    Resize an image to the target size
    
    Args:
        img_path: Path to the input image
        output_path: Path to save the processed image
        target_size: Size to resize the image to (224x224 or 256x256 as per PRD)
    """
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def apply_augmentation(img_path, output_path_base, target_size=(224, 224)):
    """
    Apply data augmentation to an image and save multiple augmented versions
    """
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)
        base_filename = os.path.splitext(os.path.basename(output_path_base))[0]
        ext = os.path.splitext(output_path_base)[1]
        output_dir = os.path.dirname(output_path_base)
        
        # Save original resized image
        img.save(output_path_base)
        
        # Apply rotation (30 degrees)
        angle = random.randint(-30, 30)  # Increased rotation range
        rotated_img = img.rotate(angle)
        rotated_path = os.path.join(output_dir, f"{base_filename}_rotated{ext}")
        rotated_img.save(rotated_path)
        
        # Apply horizontal flip
        h_flipped_img = ImageOps.mirror(img)
        h_flipped_path = os.path.join(output_dir, f"{base_filename}_hflip{ext}")
        h_flipped_img.save(h_flipped_path)
        
        # Apply vertical flip
        v_flipped_img = ImageOps.flip(img)
        v_flipped_path = os.path.join(output_dir, f"{base_filename}_vflip{ext}")
        v_flipped_img.save(v_flipped_path)
        
        # Apply both flips (horizontal + vertical)
        hv_flipped_img = ImageOps.flip(ImageOps.mirror(img))
        hv_flipped_path = os.path.join(output_dir, f"{base_filename}_hvflip{ext}")
        hv_flipped_img.save(hv_flipped_path)
        
        # Apply brightness and contrast adjustments
        bright_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(bright_factor)
        enhancer = ImageEnhance.Contrast(bright_img)
        adjusted_img = enhancer.enhance(contrast_factor)
        
        adjusted_path = os.path.join(output_dir, f"{base_filename}_adjusted{ext}")
        adjusted_img.save(adjusted_path)
        
        return True
    except Exception as e:
        print(f"Error augmenting {img_path}: {e}")
        return False

def process_horizon_dataset(classification_sets_dir, processed_data_dir, target_size=(224, 224), 
                            train_split=0.7, val_split=0.15, test_split=0.15, 
                            apply_augmentations=True):
    """
    Process the horizon detection dataset from the classification_sets directory
    
    Args:
        classification_sets_dir: Directory containing the classification sets
        processed_data_dir: Directory to save processed data
        target_size: Size to resize images to (224x224 or 256x256 as per PRD)
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        apply_augmentations: Whether to apply data augmentation
    """
    # Ensure splits sum to 1
    assert abs(train_split + val_split + test_split - 1.0) < 1e-10, "Splits must sum to 1"
    
    # Create directories
    create_dir(processed_data_dir)
    train_dir = os.path.join(processed_data_dir, 'train')
    val_dir = os.path.join(processed_data_dir, 'val')
    test_dir = os.path.join(processed_data_dir, 'test')
    
    for dir_path in [train_dir, val_dir, test_dir]:
        create_dir(os.path.join(dir_path, 'horizon'))
        create_dir(os.path.join(dir_path, 'no_horizon'))
    
    # Get all image files from the classification_sets directory
    horizon_dir = os.path.join(classification_sets_dir, 'horizon_detection', 'horizon')
    no_horizon_dir = os.path.join(classification_sets_dir, 'horizon_detection', 'no_horizon')
    
    horizon_files = [os.path.join(horizon_dir, f) for f in os.listdir(horizon_dir) 
                    if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]
    no_horizon_files = [os.path.join(no_horizon_dir, f) for f in os.listdir(no_horizon_dir) 
                       if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]
    
    # Shuffle files
    random.shuffle(horizon_files)
    random.shuffle(no_horizon_files)
    
    # Split into train, validation, and test sets
    n_horizon = len(horizon_files)
    train_horizon = horizon_files[:int(n_horizon * train_split)]
    val_horizon = horizon_files[int(n_horizon * train_split):int(n_horizon * (train_split + val_split))]
    test_horizon = horizon_files[int(n_horizon * (train_split + val_split)):]
    
    n_no_horizon = len(no_horizon_files)
    train_no_horizon = no_horizon_files[:int(n_no_horizon * train_split)]
    val_no_horizon = no_horizon_files[int(n_no_horizon * train_split):int(n_no_horizon * (train_split + val_split))]
    test_no_horizon = no_horizon_files[int(n_no_horizon * (train_split + val_split)):]
    
    # Process horizon images
    print("Processing horizon images...")
    for img_path in tqdm(train_horizon):
        filename = os.path.basename(img_path)
        output_path = os.path.join(train_dir, 'horizon', filename)
        resize_and_convert(img_path, output_path, target_size)
        
        # Apply augmentations for training set
        if apply_augmentations:
            aug_output_path = os.path.join(train_dir, 'horizon', filename)
            apply_augmentation(img_path, aug_output_path, target_size)
    
    for img_path in val_horizon:
        filename = os.path.basename(img_path)
        output_path = os.path.join(val_dir, 'horizon', filename)
        resize_and_convert(img_path, output_path, target_size)
    
    for img_path in test_horizon:
        filename = os.path.basename(img_path)
        output_path = os.path.join(test_dir, 'horizon', filename)
        resize_and_convert(img_path, output_path, target_size)
    
    # Process no_horizon images
    print("Processing no_horizon images...")
    for img_path in tqdm(train_no_horizon):
        filename = os.path.basename(img_path)
        output_path = os.path.join(train_dir, 'no_horizon', filename)
        resize_and_convert(img_path, output_path, target_size)
        
        # Apply augmentations for training set
        if apply_augmentations:
            aug_output_path = os.path.join(train_dir, 'no_horizon', filename)
            apply_augmentation(img_path, aug_output_path, target_size)
    
    for img_path in val_no_horizon:
        filename = os.path.basename(img_path)
        output_path = os.path.join(val_dir, 'no_horizon', filename)
        resize_and_convert(img_path, output_path, target_size)
    
    for img_path in test_no_horizon:
        filename = os.path.basename(img_path)
        output_path = os.path.join(test_dir, 'no_horizon', filename)
        resize_and_convert(img_path, output_path, target_size)
    
    # Print dataset statistics
    print("\nDataset statistics:")
    print(f"Train: {len(train_horizon)} horizon, {len(train_no_horizon)} no_horizon")
    print(f"Validation: {len(val_horizon)} horizon, {len(val_no_horizon)} no_horizon")
    print(f"Test: {len(test_horizon)} horizon, {len(test_no_horizon)} no_horizon")
    
    if apply_augmentations:
        print(f"Applied augmentations to training set, creating 5 additional versions of each image")
        print(f"Total training images after augmentation: {len(train_horizon) * 6} horizon, {len(train_no_horizon) * 6} no_horizon")
    
    return {
        'train': {
            'horizon': len(train_horizon) * (6 if apply_augmentations else 1),
            'no_horizon': len(train_no_horizon) * (6 if apply_augmentations else 1)
        },
        'val': {
            'horizon': len(val_horizon),
            'no_horizon': len(val_no_horizon)
        },
        'test': {
            'horizon': len(test_horizon),
            'no_horizon': len(test_no_horizon)
        }
    }

if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)
    
    # Define paths
    classification_sets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'classification_sets')
    processed_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'processed', 'horizon_detector')
    
    # Process the horizon detection dataset
    stats = process_horizon_dataset(
        classification_sets_dir=classification_sets_dir,
        processed_data_dir=processed_data_dir,
        target_size=(224, 224),  # As per PRD, use 224x224 or 256x256
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        apply_augmentations=True
    )
    
    print("Horizon detection dataset preprocessing completed successfully!")
    print(f"Dataset is ready for training at: {processed_data_dir}")
