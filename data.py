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
# This default will be overridden by the sweep
DEFAULT_BATCH_SIZE = 64

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

def load_and_preprocess_data(batch_size=DEFAULT_BATCH_SIZE):
    """
    Loads the EMNIST/byclass dataset, filters for our 36 classes,
    and sets up optimized training and test pipelines.
    
    Args:
        batch_size (int): The batch size to use for the dataset.
        
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
    # Use the batch_size passed from the sweep
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    
    # --- Test Pipeline ---
    ds_test = ds_test.filter(filter_classes)
    ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    # Use the batch_size passed from the sweep
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    
    return ds_train, ds_test