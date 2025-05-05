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
    # DRIVE_FILE_ID = "14i3LvHBOZJGNceQ-FJ7V-QK_aQiATO2q"  # 👈 Replace this with your file ID

    # def download_model():
    #     if not os.path.exists("models"):
    #         os.makedirs("models")
    #     if not os.path.exists(MODEL_PATH):
    #         print("Downloading model from Google Drive...")
    #         url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    #         gdown.download(url, MODEL_PATH, quiet=False)

    # Download and load the trained Keras model for skin cancer classification
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(img):

    print(img)
    """Ensures image is in RGB format and resizes to (112, 112, 3) while maintaining aspect ratio."""
    if isinstance(img, Image.Image):
        img = img.convert("RGB")  # Convert to RGB to ensure 3 channels
        img = np.array(img)  # Convert PIL Image to NumPy array

    if img.shape[-1] == 4:  # If RGBA, convert to RGB
        img = img[:, :, :3]

    old_size = img.shape[:2]  # (height, width)
    ratio = min(float(target_size[0]) / old_size[0], float(target_size[1]) / old_size[1])
    new_size = tuple([int(x * ratio) for x in old_size])

    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]  # Black padding
    new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # Convert to float32 and normalize
    new_im = new_im.astype("float32") / 255.0

    return new_im

def detect_skin(image_np):
    print('here')
    """Detects if skin is present in the image using OpenCV."""
    image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)  # Convert to HSV

    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create mask for skin regions
    skin_mask = cv2.inRange(image_hsv, lower_skin, upper_skin)

    # Count nonzero pixels in the mask (i.e., potential skin regions)
    skin_ratio = np.count_nonzero(skin_mask) / skin_mask.size

    # If skin pixels make up more than a threshold (e.g., 15% of image), assume skin is present
    return skin_ratio > 0.15

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
