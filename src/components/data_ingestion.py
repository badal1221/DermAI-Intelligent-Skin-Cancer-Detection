import os
import cv2
import pandas as pd
import numpy as np

def load_metadata(csv_path: str) -> pd.DataFrame:
    """
    Load the metadata CSV file.
    """
    df = pd.read_csv(csv_path)
    print(f"[INFO] Metadata loaded. Shape: {df.shape}")
    return df

def load_images_from_dirs(image_dirs: list, metadata: pd.DataFrame) -> tuple:
    """
    Load images from directories and match them with metadata.
    
    Args:
        image_dirs (list): List of paths to image directories.
        metadata (pd.DataFrame): Metadata dataframe containing 'image_id' and 'dx'.
    
    Returns:
        images (np.array): Array of loaded images.
        labels (np.array): Array of corresponding labels.
    """
    images = []
    labels = []
    
    for image_dir in image_dirs:
        for filename in os.listdir(image_dir):
            if filename.endswith('.jpg'):
                image_id = filename[:-4]
                if image_id in metadata['image_id'].values:
                    row = metadata[metadata['image_id'] == image_id].iloc[0]
                    image_path = os.path.join(image_dir, filename)
                    image = cv2.imread(image_path)
                    images.append(image)
                    labels.append(row['dx'])
                else:
                    print(f"[WARNING] image_id '{image_id}' not found in metadata.")
    
    return np.array(images), np.array(labels)
