import numpy as np
import cv2
import tensorflow as tf

# Constants
IMG_H = 112
IMG_W = 112

def load_model(model_path: str):
    """
    Load a trained model.
    """
    model = tf.keras.models.load_model(model_path)
    print(f"[INFO] Model loaded from {model_path}")
    return model

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single image.
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict(model, image_array: np.ndarray, label_mapping: dict):
    """
    Predict the class of a preprocessed image.
    """
    preds = model.predict(image_array)
    class_idx = np.argmax(preds, axis=1)[0]
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    predicted_label = inv_label_mapping[class_idx]
    return predicted_label

def main():
    model_path = 'artifacts/DenseNet121/final_densenet_model.keras'  # or wherever your model is saved
    image_path = 'path_to_image.jpg'  # Replace with the path to your image
    
    label_mapping = {
        'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3,
        'nv': 4, 'vasc': 5, 'mel': 6
    }

    # Load model
    model = load_model(model_path)

    # Preprocess image
    image_array = preprocess_image(image_path)

    # Predict
    predicted_label = predict(model, image_array, label_mapping)
    print(f"[INFO] Predicted Label: {predicted_label}")

if __name__ == "__main__":
    main()
