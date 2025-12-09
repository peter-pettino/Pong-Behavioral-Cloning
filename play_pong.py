import gymnasium as gym
import ale_py
import tensorflow as tf
import numpy as np
import cv2
import time
import os
import pandas as pd
from model import create_pong_cnn

# Register ALE environments
gym.register_envs(ale_py)

# --- Constants ---
MODEL_PATH = 'model/best_model.weights.h5'  # Path to trained weights
IMAGE_SIZE = (84, 84)
NUM_ACTIONS = 6

# Map action index to name for reporting
ACTION_MAP = {0: 'NOOP', 1: 'FIRE', 2: 'UP', 3: 'DOWN', 4: 'RIGHTFIRE', 5: 'LEFTFIRE'}

# --- BENCHMARK SETTINGS ---
NUM_EPISODES = 100
MAX_STEPS = 1000

def preprocess_frame(frame):
    """
    Preprocesses a single game frame.
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

def generate_report(results):
    """Calculates and saves the final aggregate metrics for the report."""
    df = pd.DataFrame(results)
    
    # 1. Score Metrics
    avg_score = df['cumulative_reward'].mean()
    std_dev_score = df['cumulative_reward'].std()
    
    # 2. Latency Metric
    avg_latency_ms = df['avg_latency_ms'].mean()
    
    # 3. Action Usage Metrics
    total_actions = df['total_actions'].sum()
    action_metrics = {}
    for action_name in ACTION_MAP.values():
        total_count = df[f'{action_name}_count'].sum()
        action_metrics[f'{action_name} (%)'] = (total_count / total_actions) * 100
        
    print("\n\n--- FINAL BENCHMARK REPORT ---")
    print(f"Total Episodes Run: {NUM_EPISODES}")
    print(f"Max Steps Per Episode: {MAX_STEPS}")
    print(f"Average Cumulative Reward: {avg_score:.2f}")
    print(f"Standard Deviation: {std_dev_score:.2f}")
    print(f"Average Action Latency: {avg_latency_ms:.2f} ms")
    print("Action Usage (%)")
    for name, pct in action_metrics.items():
        print(f"  {name}: {pct:.2f}%")
        
    # Save the raw episode data and the final aggregate report
    df.to_csv('pong_agent_raw_episode_data.csv', index=False)
    print("\nRaw episode data saved to: pong_agent_raw_episode_data.csv")
    
    # Create the comparison DataFrame 
    comparison_data = {
        'Metric': ['Average Cumulative Reward', 'Standard Deviation', 'Average Action Latency (ms)'] + list(action_metrics.keys()),
        'BC Model Result': [f'{avg_score:.2f}', f'{std_dev_score:.2f}', f'{avg_latency_ms:.2f}'] + [f'{v:.2f}' for v in action_metrics.values()]
    }
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv('pong_agent_comparison_report.csv', index=False)
    print("Comparison report saved to: pong_agent_comparison_report.csv")

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
env = gym.make("ALE/Pong-v5", render_mode="human")

episode_results = []  # To store results of each episode

try:
    for episode in range(NUM_EPISODES):
        print(f"--- Starting Episode: {episode + 1}/{NUM_EPISODES} ---")
        
        # Reset episode metrics
        total_reward = 0
        total_latency = 0
        action_counts = {f"{name}_count": 0 for name in ACTION_MAP.values()}
        
        observation, info = env.reset()
        
        # Reset step-based flags
        terminated = False
        truncated = False
        
        # The game loop now runs for a fixed number of steps (1000)
        step = 0
        for step in range(MAX_STEPS):
            if terminated or truncated:
                break
                
            # 1. START LATENCY TIMER
            start_time = time.time()
            
            # 2. PREPROCESS & PREDICT
            preprocessed_obs = preprocess_frame(observation)
            action_logits = model.predict(preprocessed_obs, verbose=0)
            predicted_action_idx = np.argmax(action_logits[0])
            
            # 3. STOP LATENCY TIMER & AGGREGATE
            latency_ms = (time.time() - start_time) * 1000
            total_latency += latency_ms
            
            # 4. RECORD ACTION
            predicted_action_name = ACTION_MAP.get(predicted_action_idx, 'UNKNOWN')
            action_counts[f"{predicted_action_name}_count"] += 1
            
            # 5. TAKE STEP
            observation, reward, terminated, truncated, info = env.step(predicted_action_idx)
            total_reward += reward

        # --- END OF EPISODE LOOP ---
        
        # Save results for this episode
        episode_data = {
            'episode': episode + 1,
            'cumulative_reward': total_reward,
            'avg_latency_ms': total_latency / (step + 1) if step >= 0 else 0,
            'total_actions': step + 1
        }
        episode_data.update(action_counts)
        episode_results.append(episode_data)

        print(f"Episode {episode + 1} finished. Total Reward: {total_reward}. Steps: {step + 1}")

finally:
    # Always close the environment
    env.close()
    print("--------------------")
    print(f"All {NUM_EPISODES} episodes complete. Environment closed.")
    
    # Generate the final report
    if episode_results:
        generate_report(episode_results)