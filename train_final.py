import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger

from data import load_and_preprocess_data, NUM_CLASSES
from model import build_model

# --- Final Model Configuration (from best sweep run) ---
config = {
    "learning_rate": 0.001,
    "epochs": 8,
    "batch_size": 64,
    "dropout_rate": 0.3,
    "model_architecture": "SimpleCNN_final"
}

# --- W&B Setup for Final Run ---
wandb.init(
    project="air-writer",
    config=config,
    name=f"final_train_epochs-{config['epochs']}" # Custom run name
)

# --- Main Training Logic ---
print(f"\n--- Starting Final Training Run ---")
print(f"Parameters: {config}")

print("Loading data...")
ds_train, ds_test = load_and_preprocess_data(batch_size=config["batch_size"])

print("Building model...")
model = build_model(num_classes=NUM_CLASSES, dropout_rate=config["dropout_rate"])
model.summary() # Print model summary

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Starting training for {config['epochs']} epochs...")
model.fit(
    ds_train,
    epochs=config["epochs"],
    validation_data=ds_test,
    callbacks=[WandbMetricsLogger()]
)

print("Training finished. Saving final model...")
# Save the model with a clear final name
FINAL_MODEL_NAME = "air_writer_model_final.h5"
model.save(FINAL_MODEL_NAME)

wandb.finish()
print(f"Model saved as {FINAL_MODEL_NAME}")