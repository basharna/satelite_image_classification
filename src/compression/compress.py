import os
import sys
from PIL import Image
import numpy as np
from typing import Optional, Union

def compress_image(
    input_path: str, 
    output_path: Optional[str] = None, 
    target_size_kb: int = 100, 
    quality_start: int = 90,
    format: str = "JPEG"
) -> float:
    """
    Compress an image to a target file size or less
    
    Args:
        input_path: Path to the input image
        output_path: Path to save the compressed image (if None, derives from input_path)
        target_size_kb: Target file size in kilobytes
        quality_start: Starting quality level (0-100, higher is better)
        format: Output format (JPEG, PNG, WebP, etc.)
        
    Returns:
        float: Size of compressed image in kilobytes
    """
    # Generate output path if not provided
    if output_path is None:
        filename = os.path.basename(input_path)
        directory = os.path.dirname(input_path)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(directory, f"{name}_compressed.jpg")
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Load the image
    try:
        image = Image.open(input_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image for compression: {e}")
        return 0.0
    
    # Start with the original size
    width, height = image.size
    
    # Initial compression with starting quality
    quality = quality_start
    
    # Save with initial quality
    image.save(output_path, format=format, quality=quality, optimize=True)
    current_size_kb = os.path.getsize(output_path) / 1024
    
    # Binary search for the right quality level to meet target size
    if current_size_kb > target_size_kb:
        min_quality = 5  # Don't go below this quality
        max_quality = quality_start
        
        while min_quality <= max_quality and current_size_kb > target_size_kb:
            quality = (min_quality + max_quality) // 2
            
            # Try this quality level
            image.save(output_path, format=format, quality=quality, optimize=True)
            current_size_kb = os.path.getsize(output_path) / 1024
            
            if current_size_kb > target_size_kb:
                # Still too big, try lower quality
                max_quality = quality - 1
            else:
                # Small enough, try higher quality to find optimal balance
                min_quality = quality + 1
    
    # If still too large, try resizing
    resize_factor = 1.0
    while current_size_kb > target_size_kb and resize_factor > 0.2:
        resize_factor -= 0.1
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
        
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        resized_image.save(output_path, format=format, quality=quality, optimize=True)
        
        current_size_kb = os.path.getsize(output_path) / 1024
    
    print(f"Compressed image saved to {output_path} ({current_size_kb:.2f} KB)")
    return current_size_kb


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compress an image to target size")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Path to output compressed image")
    parser.add_argument("--target_size", type=int, default=100, help="Target size in KB")
    
    args = parser.parse_args()
    
    compressed_size = compress_image(
        args.input, 
        args.output, 
        target_size_kb=args.target_size
    )
    
    print(f"Final compressed size: {compressed_size:.2f} KB")
