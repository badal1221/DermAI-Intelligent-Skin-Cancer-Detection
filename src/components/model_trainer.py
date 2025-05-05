import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Constants
IMG_H = 112
IMG_W = 112
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

def load_and_preprocess_data(csv_file_path):
    """Load image paths and labels from CSV, preprocess images."""
    df = pd.read_csv(csv_file_path)

    X = []
    y = []

    for idx, row in df.iterrows():
        img_path = row['image']
        label = row['label']

        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_W, IMG_H))
        img = img / 255.0  # Normalize

        X.append(img)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y


def build_model(input_shape, num_classes):
    """Build a transfer learning model using DenseNet121."""
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze first 75 layers
    for layer in base_model.layers[:75]:
        layer.trainable = False
    for layer in base_model.layers[75:]:
        layer.trainable = True

    # Custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.4)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model

def train_model(X_train, y_train, X_val, y_val, input_shape, num_classes):
    """Compile, train and return the model."""
    model = build_model(input_shape, num_classes)

    # Prepare callbacks
    callbacks = [
        ModelCheckpoint('artifacts/DenseNet121/best_densenet_model.keras', save_best_only=True, monitor='val_accuracy', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, mode='min', min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    ]

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    return model, history


def main():
    csv_file_path = 'your_data.csv'  # TODO: replace with your actual CSV path

    X, y = load_and_preprocess_data(csv_file_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = (IMG_H, IMG_W, 3)
    num_classes = len(np.unique(y))

    model, history = train_model(X_train, y_train, X_val, y_val, input_shape, num_classes)

    # Save the trained model
    model.save('trained_model.keras')
    print("Model training complete and saved as 'trained_model.h5'.")

if __name__ == "__main__":
    main()
