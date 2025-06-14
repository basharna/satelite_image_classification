import os
import sys
import argparse
import shutil
import random
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps, ImageEnhance, ImageFile
from torchvision import transforms
from enum import Enum

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.common import create_dir, set_seed

class DatasetType(Enum):
    HORIZON = 'horizon'
    FLARE = 'flare'
    QUALITY = 'quality'

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
        angle = random.randint(-30, 30)
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
        
        return True
    except Exception as e:
        print(f"Error augmenting {img_path}: {e}")
        return False

def apply_validation_test_augmentation(img_path, output_path_base, target_size=(224, 224)):
    """
    Apply data augmentation specifically for validation and test sets
    with rotations every 20 degrees and flipping
    
    Args:
        img_path: Path to the input image
        output_path_base: Base path for output (will be modified for each augmentation)
        target_size: Size to resize the image to
    
    Returns:
        int: Number of augmented images created (including original)
    """
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)
        base_filename = os.path.splitext(os.path.basename(output_path_base))[0]
        ext = os.path.splitext(output_path_base)[1]
        output_dir = os.path.dirname(output_path_base)
        
        # Save original resized image
        img.save(output_path_base)
        count = 1
        
        # Apply rotations every 20 degrees from -60 to 60
        for angle in [-60, -40, -20, 20, 40, 60]:
            rotated_img = img.rotate(angle)
            rotated_path = os.path.join(output_dir, f"{base_filename}_rot{angle}{ext}")
            rotated_img.save(rotated_path)
            count += 1
        
        # Apply horizontal flip
        h_flipped_img = ImageOps.mirror(img)
        h_flipped_path = os.path.join(output_dir, f"{base_filename}_hflip{ext}")
        h_flipped_img.save(h_flipped_path)
        count += 1
        
        # Apply vertical flip
        v_flipped_img = ImageOps.flip(img)
        v_flipped_path = os.path.join(output_dir, f"{base_filename}_vflip{ext}")
        v_flipped_img.save(v_flipped_path)
        count += 1
        
        # Apply both flips (horizontal + vertical)
        hv_flipped_img = ImageOps.flip(ImageOps.mirror(img))
        hv_flipped_path = os.path.join(output_dir, f"{base_filename}_hvflip{ext}")
        hv_flipped_img.save(hv_flipped_path)
        count += 1
        
        # Apply rotations to the flipped images for more variety
        for angle in [-40, 40]:
            # Horizontal flip with rotation
            h_rot_img = h_flipped_img.rotate(angle)
            h_rot_path = os.path.join(output_dir, f"{base_filename}_hflip_rot{angle}{ext}")
            h_rot_img.save(h_rot_path)
            count += 1
            
            # Vertical flip with rotation
            v_rot_img = v_flipped_img.rotate(angle)
            v_rot_path = os.path.join(output_dir, f"{base_filename}_vflip_rot{angle}{ext}")
            v_rot_img.save(v_rot_path)
            count += 1
        
        return count
    except Exception as e:
        print(f"Error augmenting {img_path}: {e}")
        return 0

def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split a dataset into train, validation, and test sets
    
    Args:
        source_dir: Directory containing class folders
        dest_dir: Destination directory for the split dataset
        train_ratio: Ratio of images to use for training
        val_ratio: Ratio of images to use for validation
        test_ratio: Ratio of images to use for testing
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create destination directories
    for split in ['train', 'val', 'test']:
        for class_name in os.listdir(source_dir):
            if os.path.isdir(os.path.join(source_dir, class_name)):
                create_dir(os.path.join(dest_dir, split, class_name))
    
    # Process each class
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Get all image files
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(image_files)
        
        # Calculate split sizes
        total_images = len(image_files)
        train_size = int(train_ratio * total_images)
        val_size = int(val_ratio * total_images)
        
        # Split the dataset
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size + val_size]
        test_files = image_files[train_size + val_size:]
        
        print(f"Class: {class_name}")
        print(f"  Total images: {total_images}")
        print(f"  Training: {len(train_files)}")
        print(f"  Validation: {len(val_files)}")
        print(f"  Testing: {len(test_files)}")
        
        # Copy files to their respective directories
        for files, split in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
            for file in tqdm(files, desc=f"Copying {split}/{class_name}"):
                src_path = os.path.join(class_dir, file)
                dst_path = os.path.join(dest_dir, split, class_name, file)
                
                try:
                    # Check if the image can be opened
                    img = Image.open(src_path)
                    img.load()
                    
                    if img.width <= 0 or img.height <= 0:
                        raise ValueError(f"Invalid image dimensions: {img.width}x{img.height}")
                    
                    # Copy the file
                    shutil.copy2(src_path, dst_path)
                    
                except Exception as e:
                    logging.warning(f"Skipping corrupted image {src_path}: {e}")

def process_dataset_type(dataset_type, classification_sets_dir, processed_data_dir, target_size=(224, 224),
                      train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, apply_augmentations=True, seed=42):
    """
    Process a specific type of dataset (horizon, flare, or quality)
    
    Args:
        dataset_type: Type of dataset to process (DatasetType enum)
        classification_sets_dir: Base directory containing all classification sets
        processed_data_dir: Base directory for processed datasets
        target_size: Size to resize images to (224x224 or 256x256 as per PRD)
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        apply_augmentations: Whether to apply data augmentation
        seed: Random seed for reproducibility
    """
    # Set random seed
    set_seed(seed)
    
    # Define paths
    source_dir = os.path.join(classification_sets_dir, f"{dataset_type.value}_detection")
    dest_dir = os.path.join(processed_data_dir, f"{dataset_type.value}_detector")
    
    # Create destination directory
    create_dir(dest_dir)
    
    # Split the dataset
    print(f"Processing {dataset_type.value} dataset...")
    split_dataset(
        source_dir=source_dir,
        dest_dir=dest_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    
    # Apply augmentations to training set
    if apply_augmentations:
        print(f"Applying augmentations to {dataset_type.value} training set...")
        train_dir = os.path.join(dest_dir, 'train')
        
        for class_name in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            print(f"Augmenting training {class_name} images...")
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for file in tqdm(image_files):
                img_path = os.path.join(class_dir, file)
                output_path = os.path.join(class_dir, file)
                apply_augmentation(img_path, output_path, target_size)
    
    # Apply augmentations to validation and test sets
    print(f"Applying augmentations to {dataset_type.value} validation and test sets...")
    val_test_stats = {}
    
    for split in ['val', 'test']:
        split_dir = os.path.join(dest_dir, split)
        val_test_stats[split] = {}
        
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            print(f"Augmenting {split} {class_name} images...")
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                          and not ('_rot' in f or '_flip' in f)]  # Skip already augmented images
            
            total_augmented = 0
            for file in tqdm(image_files):
                img_path = os.path.join(class_dir, file)
                output_path = os.path.join(class_dir, file)
                num_created = apply_validation_test_augmentation(img_path, output_path, target_size)
                total_augmented += num_created
            
            val_test_stats[split][class_name] = {
                'original': len(image_files),
                'augmented': total_augmented,
                'ratio': total_augmented / len(image_files) if len(image_files) > 0 else 0
            }
    
    # Print augmentation statistics
    print("\nDataset augmentation statistics:")
    for split, class_stats in val_test_stats.items():
        print(f"\n{split.capitalize()} set:")
        for class_name, stats in class_stats.items():
            print(f"  {class_name}: {stats['original']} original images → {stats['augmented']} total images")
            print(f"    Augmentation ratio: {stats['ratio']:.1f}x")
    
    return dest_dir

def process_all_datasets(classification_sets_dir, processed_data_dir, target_size=(224, 224),
                       train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, apply_augmentations=True, seed=42):
    """
    Process all three types of datasets: horizon, flare, and quality detection
    """
    # Set random seed
    set_seed(seed)
    
    # Process each dataset type
    processed_dirs = {}
    for dataset_type in DatasetType:
        try:
            processed_dir = process_dataset_type(
                dataset_type=dataset_type,
                classification_sets_dir=classification_sets_dir,
                processed_data_dir=processed_data_dir,
                target_size=target_size,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                apply_augmentations=apply_augmentations,
                seed=seed
            )
            processed_dirs[dataset_type.value] = processed_dir
        except Exception as e:
            logging.error(f"Error processing {dataset_type.value} dataset: {e}")
    
    return processed_dirs

def main():
    parser = argparse.ArgumentParser(description='Process and Augment All Image Datasets')
    parser.add_argument('--classification_sets_dir', type=str, help='Base directory containing classification sets',
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                            'data', 'classification_sets'))
    parser.add_argument('--processed_data_dir', type=str, help='Base directory for processed datasets',
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                            'data', 'processed'))
    parser.add_argument('--target_size', type=int, default=224, help='Size to resize images to (224 or 256)')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of images for training')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of images for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Ratio of images for testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no_augment', action='store_true', help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Process all datasets
    processed_dirs = process_all_datasets(
        classification_sets_dir=args.classification_sets_dir,
        processed_data_dir=args.processed_data_dir,
        target_size=(args.target_size, args.target_size),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        apply_augmentations=not args.no_augment,
        seed=args.seed
    )
    
    print("\nAll datasets preprocessing completed successfully!")
    print("\nProcessed datasets are ready for training at:")
    for dataset_type, dir_path in processed_dirs.items():
        print(f"- {dataset_type}: {dir_path}")

if __name__ == "__main__":
    main()
