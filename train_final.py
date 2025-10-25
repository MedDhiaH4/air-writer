# train_final.py

import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger

# Import our custom modules
from data import load_and_preprocess_data, NUM_CLASSES
from model import build_model

# --- 1. Final Model Configuration ---
# These are the best parameters you found from the W&B Sweep
config = {
    "learning_rate": 0.001,
    "epochs": 8, 
    "batch_size": 64,
    "dropout_rate": 0.3,
    "model_architecture": "SimpleCNN_final"
}

# --- 2. Weights & Biases Setup ---
# Initialize a new W&B run to track this *single* training
wandb.init(
    project="air-writer",
    config=config,
    name=f"final_train_epochs-{config['epochs']}" # Give it a custom name
)

# --- 3. Main Training Logic ---

print(f"\n--- Starting Final Training Run ---")
print(f"Parameters: {config}")

# Load data using the best batch_size
print("Loading and preprocessing data...")
ds_train, ds_test = load_and_preprocess_data(batch_size=config["batch_size"])

print("Building model...")
# Build the model with the best dropout_rate
model = build_model(num_classes=NUM_CLASSES, dropout_rate=config["dropout_rate"])
model.summary()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Starting training for {config['epochs']} epochs...")
# Train the model for 8 epochs
model.fit(
    ds_train,
    epochs=config["epochs"],
    validation_data=ds_test,
    callbacks=[WandbMetricsLogger()] # Log metrics to W&B
)

print("Training finished. Saving final model...")

# Save the final, trained model
model.save("air_writer_model.h5")

wandb.finish()
print("Model saved as air_writer_model.h5")