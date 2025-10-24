# data.py

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# --- 1. Constants ---
# We are keeping 36 classes: 0-9 and A-Z
NUM_CLASSES = 36
CLASS_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

IMG_SIZE = 28
BATCH_SIZE = 64

# --- 2. Preprocessing Functions ---

def filter_classes(image, label):
    """Filters the dataset to keep only labels 0-35."""
    return label < NUM_CLASSES

def preprocess(image, label):
    """
    Normalizes and correctly orients the EMNIST images.
    """
    # Cast to float and normalize
    image = tf.cast(image, tf.float32) / 255.0
    
    # EMNIST images are loaded rotated and flipped.
    # Transposing the (x, y) axes fixes this.
    image = tf.transpose(image, perm=[1, 0, 2])
    
    return image, label

# --- 3. Main Data Loading Function ---

def load_and_preprocess_data():
    """
    Loads the EMNIST/byclass dataset, filters for our 36 classes,
    and sets up optimized training and test pipelines.
    
    Returns:
        (ds_train, ds_test): The optimized tf.data.Dataset objects.
    """
    print("Loading EMNIST/byclass dataset...")
    
    # Load the dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'emnist/byclass',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,  # Returns (image, label) tuples
        with_info=True
    )
    
    print("Filtering and preprocessing data...")
    
    # --- Training Pipeline ---
    ds_train = ds_train.filter(filter_classes)
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache() # Cache after preprocessing
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(BATCH_SIZE)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    
    # --- Test Pipeline ---
    ds_test = ds_test.filter(filter_classes)
    ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(BATCH_SIZE)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    
    return ds_train, ds_test

# --- 4. Sanity Check ---
# This block runs ONLY when you execute `python data.py` directly
if __name__ == "__main__":
    print("Running data.py sanity check...")
    ds_train, ds_test = load_and_preprocess_data()
    
    print("\n--- Data Sanity Check ---")
    
    # Take one batch from the training set and show the first 2 labels
    for images, labels in ds_train.take(1):
        print(f"Image 1 Label: {labels[0]}, Class: '{CLASS_NAMES[labels[0]]}'")
        print(f"Image 2 Label: {labels[1]}, Class: '{CLASS_NAMES[labels[1]]}'")
            
    print("------------------------")