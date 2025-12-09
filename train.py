"""
Training script for Pong Behavioral Cloning CNN
Supports smoke testing with small data subsets
"""
import os
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import create_pong_cnn


def load_data(data_path, smoke_test=False, subset_size=1000):
    """
    Load the processed Pong dataset.
    
    Args:
        data_path: Path to the .npz file
        smoke_test: If True, use only a small subset
        subset_size: Number of samples for smoke test
    
    Returns:
        (X_train, y_train, X_test, y_test)
    """
    print("\nLoading datasets...")
    data = np.load(data_path)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"Loaded train set: {X_train.shape[0]} samples")
    print(f"  Image shape: {X_train.shape[1:]}")
    print(f"  Label shape: {y_train.shape}")
    print(f"  Unique actions: {np.unique(y_train)}")
    
    print(f"Loaded test set: {X_test.shape[0]} samples")
    
    # Smoke test: use subset
    if smoke_test:
        print(f"\nSMOKE TEST MODE: Using {subset_size} samples")
        train_indices = np.random.choice(len(X_train), 
                                        min(subset_size, len(X_train)), 
                                        replace=False)
        test_indices = np.random.choice(len(X_test), 
                                       min(subset_size // 4, len(X_test)), 
                                       replace=False)
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]
        print(f"  Train subset: {len(X_train)} samples")
        print(f"  Test subset: {len(X_test)} samples")
    
    return X_train, y_train, X_test, y_test


def main():
    parser = argparse.ArgumentParser(description='Train Pong Behavioral Cloning CNN')
    
    # Data parameters
    parser.add_argument('--data-path', type=str, default='data/processed/pong_dataset.npz',
                        help='Path to processed dataset')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run smoke test with small subset of data')
    parser.add_argument('--subset-size', type=int, default=1000,
                        help='Number of samples for smoke test (default: 1000)')
    
    # Model parameters
    parser.add_argument('--num-actions', type=int, default=6,
                        help='Number of actions (default: 6)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    
    args = parser.parse_args()
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Using GPU: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("Using CPU")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data(
        args.data_path, 
        smoke_test=args.smoke_test, 
        subset_size=args.subset_size
    )
    
    # Create model
    print("\nCreating model...")
    model = create_pong_cnn(input_shape=(84, 84, 1), num_actions=args.num_actions)
    
    # Print model summary
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        # Save checkpoints
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.checkpoint_dir, 'checkpoint_epoch_{epoch:02d}.weights.h5'),
            save_weights_only=True,
            save_freq='epoch',
            verbose=1
        ),
        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.checkpoint_dir, 'best_model.weights.h5'),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        # CSV logger
        keras.callbacks.CSVLogger(
            os.path.join(args.checkpoint_dir, 'training_log.csv'),
            separator=',',
            append=False
        ),
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.checkpoint_dir, 'logs'),
            histogram_freq=0,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    # Training
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    total_time = time.time() - start_time
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Total training time: {total_time:.2f}s")
    
    # Evaluate on test set
    print("\nFinal evaluation on test set:")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc * 100:.2f}%")
    
    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, 'final_model.weights.h5')
    model.save_weights(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Print best results
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    print(f"\nBest validation accuracy: {best_val_acc * 100:.2f}% (Epoch {best_epoch})")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
