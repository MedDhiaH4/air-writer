import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger

from data import load_and_preprocess_data, NUM_CLASSES
from model import build_model

# --- W&B Sweep Configuration ---
sweep_config = {
    'method': 'random',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'values': [0.001, 0.0005, 0.0001]},
        'epochs': {'value': 5}, # Shortened epochs for sweep
        'batch_size': {'values': [64, 128]},
        'dropout_rate': {'values': [0.3, 0.4, 0.5]}
    }
}
print("--- W&B Sweep Configuration ---")
print(sweep_config)

# --- Training Function (Called by W&B Agent) ---
def train_main():
    run = wandb.init() # Initialize run, automatically uses sweep config
    config = wandb.config
    print(f"\n--- Starting Run: LR={config.learning_rate}, BS={config.batch_size}, DO={config.dropout_rate} ---")

    print("Loading data...")
    ds_train, ds_test = load_and_preprocess_data(batch_size=config.batch_size)

    print("Building model...")
    model = build_model(num_classes=NUM_CLASSES, dropout_rate=config.dropout_rate)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Starting training for {config.epochs} epochs...")
    model.fit(
        ds_train,
        epochs=config.epochs,
        validation_data=ds_test,
        callbacks=[WandbMetricsLogger()] # Log metrics automatically
    )
    print("Training finished.")
    run.finish()

# --- Start the Sweep ---
if __name__ == "__main__":
    print("Initializing W&B Sweep...")
    sweep_id = wandb.sweep(sweep_config, project="air-writer")
    print(f"Sweep initialized. ID: {sweep_id}")
    print("Starting W&B agent to run 10 trials...")
    # Run 10 trials using the train_main function
    wandb.agent(sweep_id, function=train_main, count=10)
    print("--- Sweep finished ---")