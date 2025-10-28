import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# --- Constants ---
NUM_CLASSES = 36 # 0-9, A-Z
CLASS_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]
IMG_SIZE = 28
DEFAULT_BATCH_SIZE = 64

# --- Preprocessing Functions ---

def filter_classes(image, label):
    """Keep only labels 0-35."""
    return label < NUM_CLASSES

def preprocess(image, label):
    """Normalize and orient EMNIST images."""
    image = tf.cast(image, tf.float32) / 255.0
    # Correct EMNIST rotation/flip by transposing
    image = tf.transpose(image, perm=[1, 0, 2])
    return image, label

# --- Data Loading Function ---

def load_and_preprocess_data(batch_size=DEFAULT_BATCH_SIZE):
    """Loads, filters, preprocesses, and batches EMNIST/byclass data."""
    print("Loading EMNIST/byclass dataset...")
    (ds_train, ds_test), ds_info = tfds.load(
        'emnist/byclass',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    print("Filtering and preprocessing data...")

    # Configure training pipeline
    ds_train = ds_train.filter(filter_classes)
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Configure test pipeline
    ds_test = ds_test.filter(filter_classes)
    ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test