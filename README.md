# 🛰️ Satellite Image Classification System

A system that classifies satellite images into three categories and prepares them for transmission:
1. Horizon detection
2. Flare (sunburn) detection
3. Image quality evaluation
4. Compression (only if image is "good")

The system provides both individual detectors for each feature and a unified pipeline that combines all three detectors.

## 📁 Project Structure

```
sateliite_image_classification/
├── data/
│   ├── raw/                      # Original, untouched data (for archiving)
│   │   ├── earth/
│   │   ├── horizon/
│   │   ├── space/
│   │   └── sunburn/
│   │
│   ├── classification_sets/
│       ├── horizon_detection/
│       │   ├── horizon/         # Label = 1 (horizon visible)
│       │   └── no_horizon/      # Label = 0 (no horizon visible)
│       │
│       ├── flare_detection/
│       │   ├── flare/           # Label = 1 (visible sun flare or glare)
│       │   └── no_flare/        # Label = 0 (no flare)
│       │
│       └── quality_detection/
│           ├── good/            # Sharp, informative, usable image
│           └── bad/             # Overexposed, washed out, blurred, etc.
│
├── models/                       # Saved model weights
│   ├── horizon_detector_best.pth
│   ├── flare_detector_best.pth
│   └── quality_detector_best.pth
│
├── results/                      # Visualization output directory
│
├── src/
    ├── compression/             # Image compression module
    │   └── compress.py          # Utility for compressing images
    │
    ├── data/                    # Data handling modules
    │   ├── dataset.py           # Dataset classes and data loaders
    │   └── preprocess.py        # Data preparation scripts
    │
    ├── detection/               # Evaluation modules for detectors
    │   ├── horizon_evaluation.py # Horizon detector evaluation
    │   ├── flare_evaluation.py  # Flare detector evaluation
    │   └── quality_evaluation.py # Quality detector evaluation
    │
    ├── models/                  # Neural network models
    │   ├── horizon_detector.py  # Horizon detection model
    │   ├── flare_detector.py    # Flare detection model
    │   ├── quality_detector.py  # Quality detection model
    │   └── unified_classifier.py # Combined detection pipeline
    │
    └── utils/                   # Utility functions
        └── common.py            # Shared utility functions
```

## 🔧 Installation

1. Clone the repository
2. Download the data and models folders from [Google Drive](https://drive.google.com/drive/folders/1kvjjSlnMD-H9OYWsSSz7jtxZ4eFrjP-6?usp=sharing)
   - This link includes both the raw dataset (which needs to be preprocessed) and the trained models
   - Place the downloaded folders in the root directory as `data/` and `models/`
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🔄 Data Preprocessing

The preprocessing pipeline prepares the raw satellite images for training by:
1. Resizing images to a standard size (224x224 or 256x256)
2. Splitting data into train/validation/test sets
3. Applying data augmentation (for training set only)

### Preprocessing Commands

```bash
# Process the raw dataset
python src/data/preprocess.py
```

The preprocessing script:
- Resizes all images to 224x224 pixels
- Splits data into 70% training, 15% validation, and 15% test sets
- Applies augmentations to the training set (rotation, flips, brightness/contrast adjustments)
- Creates 5 additional versions of each training image through augmentation

## 🧠 Model Training

The system consists of three binary classification models:

### 1. Horizon Detection Model
- Classifies if a horizon is visible in the image
- Uses a CNN architecture with visualization capability

### 2. Flare Detection Model
- Classifies if sun flare/glare is present in the image
- Uses similar architecture as the horizon model

### 3. Image Quality Detection Model
- Classifies image quality as good or bad
- Combines features from horizon and flare detection

### Training Commands

```bash
# Train the horizon detector model
python src/models/train_horizon_detector.py --batch_size 32 --img_size 224 --num_epochs 20 --learning_rate 0.001

# Train the flare detector model
python src/models/train_flare_detector.py --batch_size 32 --img_size 224 --num_epochs 20 --learning_rate 0.001

# Train the quality detector model
python src/models/train_quality_detector.py --batch_size 32 --img_size 224 --num_epochs 20 --learning_rate 0.001

```

### Training Parameters

- `--batch_size`: Batch size for training (default: 32)
- `--img_size`: Image size for resizing (224 or 256) (default: 224)
- `--num_epochs`: Number of epochs to train for (default: 20)
- `--learning_rate`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay for regularization (default: 1e-4)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--seed`: Random seed for reproducibility (default: 42)
- `--num_visualizations`: Number of horizon visualizations to generate (default: 10)

## 📊 Evaluation and Visualization

The system provides evaluation modules for each detector:

### Individual Detector Evaluation

```bash
# Evaluate horizon detector on a single image
python -m src.detection.horizon_evaluation --image_path path/to/image.jpg --show

# Evaluate flare detector on a single image
python -m src.detection.flare_evaluation --image_path path/to/image.jpg --show

# Evaluate quality detector on a single image
python -m src.detection.quality_evaluation --image_path path/to/image.jpg --show
```

Each evaluation module:
- Loads the corresponding detector model (uses default model path if not specified)
- Classifies the input image
- Provides confidence scores as percentage certainty
- Generates side-by-side visualization with the original image and prediction
- Displays the visualization (with `--show` flag)

### Visualization Examples

Each detector produces a visualization with:
- Original image on the left
- Prediction overlay on the right
- Color-coded indicators (blue for horizon, yellow for flare, green/red for quality)
- Confidence scores displayed

## 🔄 Unified Pipeline

The unified classifier integrates all three detectors:

```bash
# Run unified pipeline with visualization
python -m src.models.unified_classifier --image_path path/to/image.jpg --visualize

# Save visualization to a file
python -m src.models.unified_classifier --image_path path/to/image.jpg --visualize --save_viz results/output.jpg

# Specify custom model paths
python -m src.models.unified_classifier \
  --image_path path/to/image.jpg \
  --horizon_model models/custom_horizon.pth \
  --flare_model models/custom_flare.pth \
  --quality_model models/custom_quality.pth \
  --visualize
```

The unified pipeline:
1. Processes input images through each detector model
2. Compresses "good" quality images to target size (≤100KB)
3. Generates a comprehensive visualization
4. Returns classification results with confidence scores as JSON

## 📉 Image Compression

Images classified as "good" quality are compressed to reduce transmission size:
- Uses standard image compression techniques
- Target size: ≤100KB per image

## 🔍 Example Output

```json
{
  "horizon": true,
  "horizon_confidence": 0.9984567165374756,
  "flare": false,
  "flare_confidence": 0.9999546021317656,
  "quality": "bad",
  "quality_confidence": 0.8538671880960464,
  "compressed": null
}
```

For good quality images, compression details are included:

```json
{
  "horizon": true,
  "horizon_confidence": 0.9568066096305847,
  "flare": false,
  "flare_confidence": 0.9999546021317656,
  "quality": "good",
  "quality_confidence": 0.9245392131805419,
  "compressed": {
    "path": "path/to/compressed_image.jpg",
    "compressed_size_kb": 83.45
  }
}
```

## 🧰 Compression Module

The system includes a standalone compression utility:

```bash
# Compress a single image
python -m src.compression.compress \
  --input path/to/image.jpg \
  --target_size 100
```

Compression features:
- Adaptive quality reduction to meet target size
- Falls back to resizing if quality reduction isn't enough
- Optimization for JPEG compression
- Target size: ≤100KB per image (configurable)
