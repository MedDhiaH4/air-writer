# model.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(num_classes, dropout_rate, img_size=28):
    """
    Builds a simple Convolutional Neural Network (CNN).
    
    Args:
        num_classes (int): The number of output classes (36).
        dropout_rate (float): The dropout rate to use (from W&B).
        img_size (int): The height/width of the input images (28).
        
    Returns:
        A compiled Keras model.
    """
    model = models.Sequential()
    
    # Input Layer
    model.add(layers.Input(shape=(img_size, img_size, 1)))
    
    # Convolutional Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convolutional Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten and Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    # Use the dropout_rate from the sweep config
    model.add(layers.Dropout(dropout_rate))
    
    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model