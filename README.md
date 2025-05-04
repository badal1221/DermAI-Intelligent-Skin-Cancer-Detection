# DermAI: Intelligent Skin Cancer Detector

## Overview
DermAI is an end-to-end deep learning framework for automated classification of dermoscopic skin lesion images into seven clinically significant categories. This project leverages state-of-the-art convolutional neural network architectures and provides a web-based interface for real-time inference.

## Table of Contents
1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Architectures](#model-architectures)
   - [Training Pipeline](#training-pipeline)
5. [Results](#results)
6. [Web Application](#web-application)
7. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Usage](#usage)
8. [Contributing](#contributing)
9. [License](#license)
10. [Acknowledgments](#acknowledgments)
11. [References](#references)

## Features
- Automated multi-class classification of skin lesions: BCC, DF, AKIEC, NV, VASC, MEL, BKL.
- Four deep learning architectures:
  - Custom Sequential CNN
  - MobileNetV2 (Transfer Learning)
  - EfficientNetB4 (Transfer Learning)
  - DenseNet121 (Transfer Learning)
- Data augmentation and class balancing to address dataset skew.
- Deployment-ready Flask web interface for image upload and real-time predictions.
- Modular, maintainable codebase following industry best practices.

## Project Structure

project-root/
│
├── data/ # Raw and preprocessed datasets
│ ├── HAM10000/ # HAM10000 image files
│ └── ISIC/ # ISIC image files
│
├── src/ # Core ML pipeline
│ ├── components/ # Data ingestion, preprocessing, model training modules
│ ├── pipeline/ # Orchestrators for training and inference
│ ├── utils.py # Utility functions
│ ├── exception.py # Custom exception handling
│ └── logger.py # Logging setup
│
├── webapp/ # Flask application
│ ├── static/ # CSS, JS, images
│ ├── templates/ # HTML templates
│ └── app.py # Flask server entrypoint
│
├── notebooks/ # Jupyter notebooks for EDA and experiments
├── outputs/ # Saved models and evaluation artifacts
├── requirements.txt # Python dependencies
└── README.md # Project overview and instructions



## Dataset
This project uses two publicly available dermoscopic image datasets:
- **HAM10000** (Human Against Machine with 10,000 training images)
- **ISIC** (International Skin Imaging Collaboration)

Images are organized into seven diagnostic categories:
- `BCC` (Basal Cell Carcinoma)
- `DF` (Dermatofibroma)
- `AKIEC` (Actinic Keratoses & Intraepithelial Carcinoma)
- `NV` (Melanocytic Nevi)
- `VASC` (Vascular Lesions)
- `MEL` (Melanoma)
- `BKL` (Benign Keratosis-like Lesions)

Before training, images are resized to 112×112 pixels, normalized, one-hot encoded, and augmented (rotation, flipping, scaling).

## Methodology
### Data Preprocessing
1. **Resizing**: Convert all dermoscopic images to 112×112×3 using bilinear interpolation.  
2. **Normalization**: Scale pixel values to the [0,1] range.  
3. **One-Hot Encoding**: Transform class labels (0–6) into seven-dimensional vectors.  
4. **Class Balancing**: Oversample minority classes and downsample majority classes to equalize representation.  
5. **Augmentation**: Apply real-time augmentations (flip, rotate, zoom) using `ImageDataGenerator`.

### Model Architectures
- **Custom Sequential CNN**: Five convolutional blocks (filters: 32→256), followed by dense layers and softmax output.  
- **MobileNetV2**: Pretrained on ImageNet; custom head with global average pooling and dropout.  
- **EfficientNetB4**: Compound-scaled CNN; fine-tuned layers 75+ with L2 regularization and dropout.  
- **DenseNet121**: Densely connected network; first 100 layers frozen, custom head with 512-unit dense layer.

### Training Pipeline
- **Optimizer**: Adam or AdamW  
- **Loss**: Categorical Cross-Entropy  
- **Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC  
- **Callbacks**:
  - `EarlyStopping` to prevent overfitting  
  - `ReduceLROnPlateau` to adjust learning rate  
  - `ModelCheckpoint` to save best weights  

## Results
| Model              | Accuracy | Weighted F1 | Comments                              |
|--------------------|----------|-------------|---------------------------------------|
| Sequential CNN     | 54.3%    | 0.53        | Baseline                              |
| MobileNetV2        | 62.9%    | 0.62        | Lightweight; efficient                |
| EfficientNetB4     | 75.0%    | 0.75        | High performance with moderate size   |
| **DenseNet121**    | **88.6%**| **0.885**   | Best overall; robust feature learning |

Confusion matrices indicate melanoma and benign keratosis as the most confused classes. DenseNet121 shows strong generalization across all categories.

## Web Application
The Flask-based web interface allows users to:
1. Upload a dermoscopic image (JPEG/PNG).
2. Receive the predicted lesion category and class probabilities in real time.

**API Endpoints**:
- `GET /` : Render homepage.
- `POST /predict` : Accepts image file, returns JSON with `prediction`, `confidence`, and `probabilities`.

**Error Handling**:
- Invalid file types or missing uploads return HTTP 400 with descriptive error messages.
- Server errors return HTTP 500 with fallback responses.

## Getting Started
### Prerequisites
- Python 3.8+  
- TensorFlow 2.x  
- Flask 2.x  
- Other dependencies in `requirements.txt`

```bash
pip install -r requirements.txt




