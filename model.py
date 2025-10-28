import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(num_classes, dropout_rate, img_size=28):
    """Builds a simple CNN for character classification."""
    model = models.Sequential([
        layers.Input(shape=(img_size, img_size, 1)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate), # Use dropout rate from config

        layers.Dense(num_classes, activation='softmax') # Output layer
    ])
    return model