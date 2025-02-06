import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Load dataset
def load_dataset(dataset_path):
    images = []
    labels = []
    label_names = os.listdir(dataset_path)
    for label_id, label_name in enumerate(label_names):
        student_folder = os.path.join(dataset_path, label_name)
        for image_name in os.listdir(student_folder):
            image_path = os.path.join(student_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128, 128))  # Resize to 128x128
            images.append(image)
            labels.append(label_id)
    return np.array(images), np.array(labels), label_names

# Build the model
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model():
    dataset_path = "face_dataset"
    images, labels, label_names = load_dataset(dataset_path)
    images = images / 255.0  # Normalize pixel values
    images = np.expand_dims(images, axis=-1)  # Add channel dimension

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = build_model(input_shape=(128, 128, 1), num_classes=len(label_names))
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    model.save("models/face_recognition.h5")
    print("Model trained and saved.")

if __name__ == "__main__":
    train_model()
