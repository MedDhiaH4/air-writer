# train.py

import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger

# Import our custom modules
from data import load_and_preprocess_data, NUM_CLASSES
from model import build_model

# --- 1. W&B Sweep Configuration ---
# This dictionary tells W&B what to test.
sweep_config = {
    'method': 'random',  # Try 10 random combinations
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'   
    },
    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.0005, 0.0001]
        },
        'epochs': {
            'value': 3  # Keep epochs constant at 10 for this sweep
        },
        'batch_size': {
            'values': [64, 128]
        },
        'dropout_rate': {
            'values': [0.3, 0.4, 0.5]
        }
    }
}

print("--- W&B Sweep Configuration ---")
print(sweep_config)
print("---------------------------------")


# --- 2. Define the Training Function ---
# This function will be called by W&B for each new run
def train_main():
    # Initialize a new W&B run (W&B will provide the config)
    run = wandb.init()
    
    # Get the (hyper)parameters for this run from wandb.config
    config = wandb.config

    print(f"\n--- Starting Run: LR={config.learning_rate}, BS={config.batch_size}, DO={config.dropout_rate} ---")

    # Load data *using the batch_size from the sweep*
    print("Loading and preprocessing data...")
    ds_train, ds_test = load_and_preprocess_data(batch_size=config.batch_size)

    print("Building model...")
    # Pass the dropout_rate from the sweep config
    model = build_model(num_classes=NUM_CLASSES, dropout_rate=config.dropout_rate)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Starting training...")
    model.fit(
        ds_train,
        epochs=config.epochs,
        validation_data=ds_test,
        callbacks=[WandbMetricsLogger()] # Log metrics to W&B
    )

    print("Training finished.")
    run.finish()

# --- 3. Start the Sweep ---
print("Initializing W&B Sweep...")
# Initialize the sweep and get a Sweep ID
sweep_id = wandb.sweep(sweep_config, project="air-writer")

print(f"Sweep initialized. ID: {sweep_id}")
print("Starting W&B agent to run 10 trials...")

# Start the agent. This will run 'train_main' 10 times.
wandb.agent(sweep_id, function=train_main, count=10)

print("--- Sweep finished ---")
# After this, go to your W&B dashboard to see the best model!