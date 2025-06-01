# Satellite Image Classification System

A system for classifying satellite images for a nanosatellite, including:

- Horizon detection
- Star detection
- Flare/blink detection
- Image quality assessment
- Image compression for transmission

## Project Structure

```
├── data/
│   ├── raw/            # Raw satellite images
│   └── processed/      # Preprocessed images for model training
├── models/             # Saved trained models
├── src/
│   ├── data/           # Data processing scripts
│   ├── models/         # Model definition scripts
│   └── utils/          # Utility functions
└── requirements.txt    # Project dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Preprocess the data:

```bash
python src/data/preprocess.py
```

2. Train the horizon detector model:

```bash
python src/models/train_horizon_detector.py
```

3. Run predictions:

```bash
python src/models/predict.py --image path/to/image.jpg
```
