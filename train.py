# train.py

import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger

# Import our custom modules
from data import load_and_preprocess_data, NUM_CLASSES, BATCH_SIZE
from model import build_model

# --- 1. Training Configuration ---
# This config dict is for W&B
config = {
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": BATCH_SIZE,
    "model_architecture": "SimpleCNN"
}

# --- 2. Weights & Biases Setup ---
# Initialize a new W&B run
wandb.init(
    project="air-writer",  # The name of your project in W&B
    config=config          # Pass the config dictionary
)

# --- 3. Main Training Logic ---

print("Loading data...")
ds_train, ds_test = load_and_preprocess_data()

print("Building model...")
model = build_model(num_classes=NUM_CLASSES)
model.summary()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
    loss='sparse_categorical_crossentropy', # Use sparse (integer labels)
    metrics=['accuracy']
)

print("Starting training...")
# Train the model
# WandbCallback automatically logs metrics, losses, and system stats
model.fit(
    ds_train,
    epochs=config["epochs"],
    validation_data=ds_test,
    callbacks=[WandbMetricsLogger()] # This is the magic W&B line
)

print("Training finished. Saving model...")

# Save the trained model
model.save("air_writer_model.h5")

wandb.finish()
print("Model saved as air_writer_model.h5")