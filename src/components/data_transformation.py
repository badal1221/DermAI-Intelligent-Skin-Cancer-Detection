import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def encode_labels(labels: np.ndarray) -> np.ndarray:
    """
    Encode diagnosis labels to numeric classes.
    
    Args:
        labels (np.ndarray): Original string labels.
    
    Returns:
        np.ndarray: Encoded integer labels.
    """
    label_mapping = {
        'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3,
        'nv': 4, 'vasc': 5, 'mel': 6
    }
    encoded = np.vectorize(label_mapping.get)(labels)
    print("[INFO] Labels encoded.")
    return encoded

def visualize_label_distribution(labels: np.ndarray):
    """
    Plot the label distribution.
    """
    unique_labels = np.unique(labels)
    plt.figure(figsize=(10, 5))
    sns.countplot(x=labels, order=unique_labels)
    plt.title('Count Plot of Unique Labels')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

def get_image_shape(images: np.ndarray):
    """
    Print and return image shape.
    """
    shape = images[0].shape
    print(f"[INFO] Image shape: {shape}")
    return shape
