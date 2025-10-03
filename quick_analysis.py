"""
Quick Dataset Analysis Example - No Plots

This script demonstrates basic analysis of the Pong behavioral cloning dataset
without generating plots (useful for headless environments or quick inspection).
"""

import numpy as np
from pathlib import Path

def quick_dataset_analysis():
    """Perform a quick analysis of the dataset without visualizations."""
    
    dataset_path = Path("PongNoFrameskip-v4")
    files = list(dataset_path.glob("*.npz"))
    files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    
    print("=" * 60)
    print("QUICK PONG DATASET ANALYSIS")
    print("=" * 60)
    print(f"Found {len(files)} episodes")
    
    # Analyze first few episodes for quick overview
    sample_episodes = min(10, len(files))
    print(f"\nAnalyzing first {sample_episodes} episodes...")
    
    episode_stats = []
    for i in range(sample_episodes):
        data = np.load(files[i])
        
        stats = {
            'episode': i,
            'length': len(data['rewards']),
            'return': data['episode_returns'][0],
            'positive_rewards': np.sum(data['rewards'] > 0),
            'negative_rewards': np.sum(data['rewards'] < 0),
            'action_agreement': np.mean(data['model selected actions'].flatten() == data['taken actions'].flatten()),
            'unique_actions': len(np.unique(data['taken actions'])),
            'repeated_frames': np.sum(data['repeated'])
        }
        episode_stats.append(stats)
    
    # Print results
    print(f"\n{'Episode':<8} {'Length':<8} {'Return':<8} {'Pos/Neg':<10} {'Agreement':<10} {'Actions':<8}")
    print("-" * 70)
    
    for stats in episode_stats:
        agreement_pct = stats['action_agreement'] * 100
        reward_ratio = f"{stats['positive_rewards']}/{stats['negative_rewards']}"
        print(f"{stats['episode']:<8} {stats['length']:<8} {stats['return']:<8.1f} {reward_ratio:<10} {agreement_pct:<10.1f}% {stats['unique_actions']:<8}")
    
    # Summary statistics
    lengths = [s['length'] for s in episode_stats]
    returns = [s['return'] for s in episode_stats]
    agreements = [s['action_agreement'] for s in episode_stats]
    
    print(f"\nSUMMARY (first {sample_episodes} episodes):")
    print(f"Average length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"Average return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Average agreement: {np.mean(agreements)*100:.1f}%")
    
    # Quick action analysis
    print(f"\nACTION ANALYSIS (Episode 0):")
    data = np.load(files[0])
    actions = data['taken actions'].flatten()
    unique_actions, counts = np.unique(actions, return_counts=True)
    
    action_names = {0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN"}
    
    for action, count in zip(unique_actions, counts):
        action_name = action_names.get(action, f"Action_{action}")
        percentage = count / len(actions) * 100
        print(f"  {action_name}: {count} times ({percentage:.1f}%)")

if __name__ == "__main__":
    quick_dataset_analysis()