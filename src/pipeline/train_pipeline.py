import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from data_ingestion import load_metadata, load_images_from_dirs
from data_transformation import encode_labels, get_image_shape
from model_trainer import train_model

# Constants
IMAGE_DIRS = ['notebook\data\HAM10000_images_part_1','notebook\data\HAM10000_images_part_2']  
CSV_METADATA_PATH = 'notebook\data\HAM10000_metadata.csv' 
IMG_H = 112
IMG_W = 112

def main():
    # Step 1: Load metadata
    metadata = load_metadata(CSV_METADATA_PATH)

    # Step 2: Load images and labels
    images, labels = load_images_from_dirs(IMAGE_DIRS, metadata)

    # Step 3: Get image shape info
    get_image_shape(images)

    # Step 4: Encode labels
    labels_encoded = encode_labels(labels)

    # Step 5: Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
    )

    # Step 6: Preprocessing
    X_train = X_train / 255.0
    X_val = X_val / 255.0

    # Step 7: One-hot encode labels
    num_classes = len(np.unique(labels_encoded))
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)

    # Step 8: Train model
    input_shape = (IMG_H, IMG_W, 3)
    model, history = train_model(X_train, y_train, X_val, y_val, input_shape, num_classes)

    # Step 9: Save final model
    model.save('artifacts/DenseNet121/final_densenet_model.h5')
    print("[INFO] Model training complete and saved as 'artifacts/DenseNet121/final_densenet_model.h5'.")

if __name__ == "__main__":
    main()
