#!/usr/bin/env python3
"""
Script to run the flare detection dataset preprocessing with augmentation for validation and test sets
"""

import os
import sys
import argparse
from enum import Enum

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.preprocessing.preprocess_data import process_dataset_type, DatasetType
from src.utils.common import set_seed

def main(args):
    """
    Main function to run the flare detection dataset preprocessing
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    classification_sets_dir = os.path.join(project_root, 'data', 'classification_sets')
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    
    print(f"Starting flare detection dataset preprocessing with augmented validation and test sets...")
    print(f"Source directory: {classification_sets_dir}")
    print(f"Destination directory: {processed_data_dir}")
    print(f"Target image size: {args.target_size}x{args.target_size}")
    print(f"Train/Val/Test split: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    
    # Process the flare detection dataset
    flare_dir = process_dataset_type(
        dataset_type=DatasetType.FLARE,
        classification_sets_dir=classification_sets_dir,
        processed_data_dir=processed_data_dir,
        target_size=(args.target_size, args.target_size),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        apply_augmentations=True,
        seed=args.seed
    )
    
    print("\nFlare detection dataset preprocessing completed successfully!")
    print(f"Dataset is ready for training at: {flare_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare Flare Detection Dataset with Augmented Validation and Test Sets')
    parser.add_argument('--target_size', type=int, default=224, help='Size to resize images to (224 or 256 as per PRD)')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of images to use for training')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of images to use for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Ratio of images to use for testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)
