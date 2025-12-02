import gymnasium as gym
import ale_py
import tensorflow as tf
import numpy as np
import cv2
import pygame
import os
from model import create_pong_cnn

# Register ALE environments
gym.register_envs(ale_py)

# --- Constants ---
MODEL_PATH = 'checkpoints/best_model.weights.h5'
IMAGE_SIZE = (84, 84)
NUM_ACTIONS = 6

def preprocess_frame(frame):
    """Preprocess frame for model inference."""
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, IMAGE_SIZE)
    normalized_img = resized_img / 255.0
    return np.expand_dims(normalized_img, axis=(0, -1))

def get_human_action():
    """
    Get action from keyboard input.
    Returns action index for the LEFT player.
    
    Controls:
    - W or UP ARROW: Move paddle UP (action 2)
    - S or DOWN ARROW: Move paddle DOWN (action 3)
    - No key: NOOP (action 0)
    """
    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        return 2  # UP
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        return 3  # DOWN
    else:
        return 0  # NOOP

# --- Load the Trained Model ---
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model weights not found at {MODEL_PATH}")
    exit()

print("Creating model architecture...")
model = create_pong_cnn(input_shape=(84, 84, 1), num_actions=NUM_ACTIONS)

print(f"Loading weights from {MODEL_PATH}...")
model.load_weights(MODEL_PATH)
print("Model loaded successfully.")

# Initialize Pygame for keyboard input
pygame.init()

# Create environment
# Note: Pong is a two-player game, but the Gym environment controls both paddles
# The LEFT paddle is the opponent (AI in standard game)
# The RIGHT paddle is the agent (you control in standard game)
# We'll need to use a modified approach
env = gym.make("ALE/Pong-v5", render_mode="human")

NUM_EPISODES = 5

print("\n" + "="*50)
print("HUMAN vs TRAINED MODEL")
print("="*50)
print("Controls:")
print("  W or UP ARROW    - Move your paddle UP")
print("  S or DOWN ARROW  - Move your paddle DOWN")
print("  ESC              - Quit game")
print("\nYou are the LEFT paddle")
print("The trained model is the RIGHT paddle")
print("="*50 + "\n")

try:
    for episode in range(NUM_EPISODES):
        print(f"\n--- Game {episode + 1}/{NUM_EPISODES} ---")
        
        observation, info = env.reset()
        total_reward = 0
        
        terminated = False
        truncated = False
        frame_count = 0
        
        while not terminated and not truncated:
            # Handle Pygame events (for keyboard input and window close)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
            
            # Get human action for LEFT player
            # Note: In ALE/Pong-v5, we can only control the RIGHT paddle directly
            # This is a limitation of the Gym environment
            # The LEFT paddle is controlled by the built-in AI
            
            # Get model action for RIGHT player
            preprocessed_obs = preprocess_frame(observation)
            action_logits = model.predict(preprocessed_obs, verbose=0)
            model_action = np.argmax(action_logits[0])
            
            # Execute model's action (controls RIGHT paddle)
            observation, reward, terminated, truncated, info = env.step(model_action)
            total_reward += reward
            
            frame_count += 1
        
        print(f"Game {episode + 1} finished!")
        print(f"  Final Score (Model's perspective): {total_reward}")
        print(f"  Frames: {frame_count}")

except KeyboardInterrupt:
    print("\n\nGame interrupted by user.")

finally:
    env.close()
    pygame.quit()
    print("\nEnvironment closed. Thanks for playing!")

print("\n" + "="*50)
print("NOTE: Due to Gym environment limitations,")
print("true human vs model play requires a custom")
print("two-player Pong environment. The current setup")
print("shows the model playing against the built-in AI.")
print("="*50)
