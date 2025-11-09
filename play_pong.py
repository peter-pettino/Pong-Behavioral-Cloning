import gymnasium as gym
import ale_py
import tensorflow as tf
import numpy as np
import cv2
import time
import os
from model import create_pong_cnn  # Import model architecture

# Register ALE environments
gym.register_envs(ale_py)

# --- Constants ---
MODEL_PATH = 'checkpoints/best_model.weights.h5'  # Path to trained weights
IMAGE_SIZE = (84, 84)
NUM_ACTIONS = 6  # Full Atari action space

def preprocess_frame(frame):
    """
    Preprocesses a single game frame.
    
    1. Grayscale
    2. Resize
    3. Normalize
    4. Add channel and batch dimensions
    """
    # 1. Grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize
    resized_img = cv2.resize(gray_img, IMAGE_SIZE)
    
    # 3. Normalize
    normalized_img = resized_img / 255.0
    
    # 4. Add channel and batch dimensions
    # Shape becomes (1, 84, 84, 1)
    return np.expand_dims(normalized_img, axis=(0, -1))

# --- Action Mapping ---
# The model predicts one of 6 Atari actions
# We'll use the model's prediction directly
# Atari Pong actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=RIGHTFIRE, 5=LEFTFIRE
# In practice, 2 (UP) and 3 (DOWN) are the main actions used

# --- Load the Trained Model ---
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model weights not found at {MODEL_PATH}")
    print("Available files in checkpoints/:")
    if os.path.exists('checkpoints'):
        print(os.listdir('checkpoints'))
    exit()

print("Creating model architecture...")
model = create_pong_cnn(input_shape=(84, 84, 1), num_actions=NUM_ACTIONS)

print(f"Loading weights from {MODEL_PATH}...")
model.load_weights(MODEL_PATH)
print("Model loaded successfully.")

# --- Create the Environment ---
# Using render_mode="human" will open a window to watch the game
env = gym.make("ALE/Pong-v5", render_mode="human")

NUM_EPISODES = 10  # How many games to play

try:
    for episode in range(NUM_EPISODES):
        print(f"--- Starting Episode: {episode + 1} ---")
        total_reward = 0
        
        # Reset the environment for a new game
        observation, info = env.reset()
        
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            
            # 1. PREPROCESS the current game frame
            preprocessed_obs = preprocess_frame(observation)
            
            # 2. PREDICT the action
            # Model outputs logits for 6 actions
            action_logits = model.predict(preprocessed_obs, verbose=0)
            predicted_action = np.argmax(action_logits[0])
            
            # 3. TAKE STEP: Use the predicted action directly
            observation, reward, terminated, truncated, info = env.step(predicted_action)
            
            total_reward += reward
            
            # Optional: Add a small delay so you can watch
            # time.sleep(0.01) 

        print(f"Episode {episode + 1} finished. Total Reward: {total_reward}")

finally:
    # Always close the environment
    env.close()
    print("--------------------")
    print("All episodes complete. Environment closed.")