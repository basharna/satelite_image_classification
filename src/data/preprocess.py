import os
import shutil
import random
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.common import create_dir, set_seed

def resize_and_convert(img_path, output_path, target_size=(256, 256), convert_grayscale=False):
    """
    Resize an image and optionally convert to grayscale
    """
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)
        
        if convert_grayscale:
            img = ImageOps.grayscale(img)
            
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def apply_augmentation(img_path, output_path_base, target_size=(256, 256)):
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

def process_horizon_dataset(raw_data_dir, processed_data_dir, target_size=(256, 256), 
                            train_split=0.8, val_split=0.1, test_split=0.1, 
                            apply_augmentations=True, convert_grayscale=False):
    """
    Process the horizon detection dataset
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
    
    # Get all image files
    horizon_files = [os.path.join(raw_data_dir, 'horizon', f) for f in os.listdir(os.path.join(raw_data_dir, 'horizon')) if f.endswith(('.jpg', '.jpeg', '.png'))]
    space_files = [os.path.join(raw_data_dir, 'space', f) for f in os.listdir(os.path.join(raw_data_dir, 'space')) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Shuffle files
    random.shuffle(horizon_files)
    random.shuffle(space_files)
    
    # Split horizon files
    n_horizon = len(horizon_files)
    train_horizon = horizon_files[:int(n_horizon * train_split)]
    val_horizon = horizon_files[int(n_horizon * train_split):int(n_horizon * (train_split + val_split))]
    test_horizon = horizon_files[int(n_horizon * (train_split + val_split)):]
    
    # Split space files (no horizon)
    n_space = len(space_files)
    train_space = space_files[:int(n_space * train_split)]
    val_space = space_files[int(n_space * train_split):int(n_space * (train_split + val_split))]
    test_space = space_files[int(n_space * (train_split + val_split)):]
    
    # Process horizon images
    print("Processing horizon images...")
    for img_path in tqdm(train_horizon):
        filename = os.path.basename(img_path)
        output_path = os.path.join(train_dir, 'horizon', filename)
        resize_and_convert(img_path, output_path, target_size, convert_grayscale)
        
        # Apply augmentations for training set
        if apply_augmentations:
            aug_output_path = os.path.join(train_dir, 'horizon', filename)
            apply_augmentation(img_path, aug_output_path, target_size)
    
    for img_path, out_dir in zip(val_horizon + test_horizon, 
                                [val_dir, test_dir] * len(val_horizon + test_horizon)):
        if img_path in val_horizon:
            out_dir = val_dir
        else:
            out_dir = test_dir
        
        filename = os.path.basename(img_path)
        output_path = os.path.join(out_dir, 'horizon', filename)
        resize_and_convert(img_path, output_path, target_size, convert_grayscale)
    
    # Process space images (no horizon)
    print("Processing space images (no horizon)...")
    for img_path in tqdm(train_space):
        filename = os.path.basename(img_path)
        output_path = os.path.join(train_dir, 'no_horizon', filename)
        resize_and_convert(img_path, output_path, target_size, convert_grayscale)
        
        # Apply augmentations for training set
        if apply_augmentations:
            aug_output_path = os.path.join(train_dir, 'no_horizon', filename)
            apply_augmentation(img_path, aug_output_path, target_size)
    
    for img_path in val_space:
        filename = os.path.basename(img_path)
        output_path = os.path.join(val_dir, 'no_horizon', filename)
        resize_and_convert(img_path, output_path, target_size, convert_grayscale)
    
    for img_path in test_space:
        filename = os.path.basename(img_path)
        output_path = os.path.join(test_dir, 'no_horizon', filename)
        resize_and_convert(img_path, output_path, target_size, convert_grayscale)
    
    # Print dataset statistics
    print("\nDataset statistics:")
    print(f"Train: {len(train_horizon)} horizon, {len(train_space)} no_horizon")
    print(f"Validation: {len(val_horizon)} horizon, {len(val_space)} no_horizon")
    print(f"Test: {len(test_horizon)} horizon, {len(test_space)} no_horizon")
    
    if apply_augmentations:
        print(f"Applied augmentations to training set, creating 5 additional versions of each image (original, rotated, h-flipped, v-flipped, hv-flipped, and brightness/contrast adjusted)")
    
    return {
        'train': {
            'horizon': len(train_horizon) * (6 if apply_augmentations else 1),
            'no_horizon': len(train_space) * (6 if apply_augmentations else 1)
        },
        'val': {
            'horizon': len(val_horizon),
            'no_horizon': len(val_space)
        },
        'test': {
            'horizon': len(test_horizon),
            'no_horizon': len(test_space)
        }
    }

if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)
    
    # Define paths
    raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'raw')
    processed_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'processed', 'horizon_detector')
    
    # Process the dataset
    stats = process_horizon_dataset(
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        target_size=(256, 256),
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        apply_augmentations=True,
        convert_grayscale=False  # Keep RGB for now
    )
    
    print("Data preprocessing completed successfully!")
