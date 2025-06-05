# 🛰️ Satellite Image Classification System

A system that classifies satellite images into three categories and prepares them for transmission:
1. Horizon detection
2. Flare (sunburn) detection
3. Image quality evaluation
4. Compression (only if image is "good")

## 📁 Project Structure

```
dataset/
├── raw/                          # Original, untouched data (for archiving)
│   ├── earth/
│   ├── horizon/
│   ├── space/
│   └── sunburn/

├── classification_sets/
│   ├── horizon_detection/
│   │   ├── horizon/             # Label = 1 (horizon visible)
│   │   └── no_horizon/          # Label = 0 (no horizon visible)

│   ├── flare_detection/
│   │   ├── flare/               # Label = 1 (visible sun flare or glare)
│   │   └── no_flare/            # Label = 0 (no flare)

│   └── quality_detection/
       ├── good/                # Sharp, informative, usable image
       └── bad/                 # Overexposed, washed out, blurred, etc.
```

## 🔧 Installation

1. Clone the repository
2. Install dependencies:

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
# Process horizon detection dataset
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

# With custom parameters
python src/models/train_horizon_detector.py --batch_size 64 --img_size 256 --num_epochs 30 --learning_rate 0.0005 --weight_decay 1e-5 --num_workers 8 --seed 42 --num_visualizations 15
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

The training process includes:
- Saving model checkpoints every 5 epochs
- Saving the best model based on validation accuracy
- Generating training/validation loss and accuracy plots
- Creating visualizations of detected horizons with green dots

## 🔄 Model Integration

After training all three models, they are integrated into a unified pipeline that:
1. Processes input images through each model
2. Compresses "good" quality images
3. Returns classification results and compressed images

## 📉 Image Compression

Images classified as "good" quality are compressed to reduce transmission size:
- Uses standard image compression techniques
- Target size: ≤100KB per image

## 🔍 Example Output

```json
{
  "horizon": true,
  "flare": false,
  "quality": "good",
  "compressed_size_kb": 83
}
```
