"""
CNN Model for Pong Behavioral Cloning
Architecture inspired by DQN/Atari networks
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_pong_cnn(input_shape=(84, 84, 1), num_actions=6):
    """
    Create a Convolutional Neural Network for Pong action prediction.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_actions: Number of possible actions (default: 6)
    
    Returns:
        Keras Model
    """
    
    # Input layer
    inputs = keras.Input(shape=input_shape, name='input_image')
    
    # Convolutional layers
    # Conv1: 32 filters, 8x8 kernel, stride 4
    x = layers.Conv2D(32, kernel_size=8, strides=4, activation='relu', 
                      name='conv1')(inputs)
    # Output: (20, 20, 32)
    
    # Conv2: 64 filters, 4x4 kernel, stride 2
    x = layers.Conv2D(64, kernel_size=4, strides=2, activation='relu',
                      name='conv2')(x)
    # Output: (9, 9, 64)
    
    # Conv3: 64 filters, 3x3 kernel, stride 1
    x = layers.Conv2D(64, kernel_size=3, strides=1, activation='relu',
                      name='conv3')(x)
    # Output: (7, 7, 64)
    
    # Flatten
    x = layers.Flatten(name='flatten')(x)
    # Output: (3136,)
    
    # Fully connected layers
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    
    # Output layer (logits, no activation)
    outputs = layers.Dense(num_actions, activation=None, name='output')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='PongCNN')
    
    return model


class PongCNN:
    """
    Wrapper class for the Pong CNN model.
    Provides consistent interface similar to PyTorch.
    """
    
    def __init__(self, input_shape=(84, 84, 1), num_actions=6):
        self.model = create_pong_cnn(input_shape, num_actions)
        self.input_shape = input_shape
        self.num_actions = num_actions
    
    def summary(self):
        """Print model summary."""
        return self.model.summary()
    
    def predict(self, x):
        """
        Predict actions for given input.
        
        Args:
            x: Input array of shape (batch_size, 84, 84, 1)
            
        Returns:
            Predicted action indices
        """
        logits = self.model.predict(x, verbose=0)
        actions = np.argmax(logits, axis=1)
        return actions
    
    def save(self, filepath):
        """Save model weights."""
        self.model.save_weights(filepath)
    
    def load(self, filepath):
        """Load model weights."""
        self.model.load_weights(filepath)


if __name__ == "__main__":
    # Test the model
    print("Testing PongCNN model...")
    model = PongCNN(input_shape=(84, 84, 1), num_actions=6)
    
    print("\nModel Architecture:")
    model.summary()
    
    # Create dummy input
    batch_size = 4
    dummy_input = np.random.randn(batch_size, 84, 84, 1).astype(np.float32)
    
    # Forward pass
    output = model.model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, 6)")
    
    # Test prediction
    predictions = model.predict(dummy_input)
    print(f"Predictions: {predictions}")
    print(f"Predictions shape: {predictions.shape}")
    
    # Count parameters
    total_params = model.model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n✓ Model test passed!")
