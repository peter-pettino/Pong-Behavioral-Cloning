"""
Pong Behavioral Cloning Dataset Analysis Script

This script provides comprehensive analysis tools for the Pong behavioral cloning dataset.
The dataset contains expert demonstrations for training imitation learning models.

Dataset Structure:
- Each .npz file contains one episode of Pong gameplay
- Files contain observations, actions, rewards, and other metadata
- Total of 200 episodes in the dataset

Usage:
    python analyze_dataset.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class PongDatasetAnalyzer:
    def __init__(self, dataset_path="PongNoFrameskip-v4"):
        """
        Initialize the analyzer with the dataset path.
        
        Args:
            dataset_path (str): Path to the directory containing .npz files
        """
        self.dataset_path = Path(dataset_path)
        self.files = list(self.dataset_path.glob("*.npz"))
        self.files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        print(f"Found {len(self.files)} episodes in the dataset")
        
        # Pong action mapping (Atari environment)
        self.action_names = {
            0: "NOOP",
            1: "FIRE", 
            2: "UP",
            3: "RIGHT",
            4: "LEFT",
            5: "DOWN"
        }
        
    def load_episode(self, episode_idx):
        """Load a single episode from the dataset."""
        if episode_idx >= len(self.files):
            raise IndexError(f"Episode {episode_idx} not found. Dataset has {len(self.files)} episodes.")
        
        data = np.load(self.files[episode_idx])
        return {
            'model_actions': data['model selected actions'].flatten(),
            'taken_actions': data['taken actions'].flatten(),
            'observations': data['obs'],
            'rewards': data['rewards'],
            'episode_returns': data['episode_returns'][0],
            'episode_starts': data['episode_starts'],
            'repeated': data['repeated']
        }
    
    def get_dataset_overview(self):
        """Get comprehensive overview of the entire dataset."""
        print("=" * 60)
        print("PONG BEHAVIORAL CLONING DATASET OVERVIEW")
        print("=" * 60)
        
        total_steps = 0
        total_rewards = 0
        episode_lengths = []
        episode_returns = []
        all_actions = []
        all_model_actions = []
        action_agreement = []
        
        print("Loading all episodes...")
        for i, file_path in enumerate(self.files):
            if i % 50 == 0:
                print(f"Processing episode {i}/{len(self.files)}")
                
            data = np.load(file_path)
            
            # Episode statistics
            episode_length = len(data['rewards'])
            episode_return = data['episode_returns'][0]
            
            episode_lengths.append(episode_length)
            episode_returns.append(episode_return)
            total_steps += episode_length
            total_rewards += episode_return
            
            # Action analysis
            model_actions = data['model selected actions'].flatten()
            taken_actions = data['taken actions'].flatten()
            
            all_actions.extend(taken_actions)
            all_model_actions.extend(model_actions)
            action_agreement.extend(model_actions == taken_actions)
        
        # Calculate statistics
        print("\n" + "=" * 40)
        print("DATASET STATISTICS")
        print("=" * 40)
        print(f"Total Episodes: {len(self.files)}")
        print(f"Total Steps: {total_steps:,}")
        print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"Shortest Episode: {min(episode_lengths)} steps")
        print(f"Longest Episode: {max(episode_lengths)} steps")
        
        print(f"\nTotal Reward: {total_rewards:.1f}")
        print(f"Average Episode Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
        print(f"Best Episode Return: {max(episode_returns):.1f}")
        print(f"Worst Episode Return: {min(episode_returns):.1f}")
        
        print(f"\nAction Agreement Rate: {np.mean(action_agreement)*100:.1f}%")
        print(f"Total Disagreements: {np.sum(~np.array(action_agreement)):,}")
        
        return {
            'episode_lengths': episode_lengths,
            'episode_returns': episode_returns,
            'total_steps': total_steps,
            'all_actions': all_actions,
            'all_model_actions': all_model_actions,
            'action_agreement': action_agreement
        }
    
    def analyze_actions(self, stats=None):
        """Analyze action distributions and patterns."""
        if stats is None:
            stats = self.get_dataset_overview()
        
        print("\n" + "=" * 40)
        print("ACTION ANALYSIS")
        print("=" * 40)
        
        # Action distribution
        action_counts = Counter(stats['all_actions'])
        model_action_counts = Counter(stats['all_model_actions'])
        
        print("Taken Actions Distribution:")
        for action_id in sorted(action_counts.keys()):
            action_name = self.action_names.get(action_id, f"Unknown_{action_id}")
            count = action_counts[action_id]
            percentage = count / len(stats['all_actions']) * 100
            print(f"  {action_name:8}: {count:8,} ({percentage:5.1f}%)")
        
        print("\nModel Selected Actions Distribution:")
        for action_id in sorted(model_action_counts.keys()):
            action_name = self.action_names.get(action_id, f"Unknown_{action_id}")
            count = model_action_counts[action_id]
            percentage = count / len(stats['all_model_actions']) * 100
            print(f"  {action_name:8}: {count:8,} ({percentage:5.1f}%)")
        
        return action_counts, model_action_counts
    
    def analyze_episode_structure(self):
        """Analyze the structure and patterns within episodes."""
        print("\n" + "=" * 40)
        print("EPISODE STRUCTURE ANALYSIS")
        print("=" * 40)
        
        # Sample a few episodes for detailed analysis
        sample_episodes = [0, len(self.files)//4, len(self.files)//2, 3*len(self.files)//4, -1]
        
        for i, ep_idx in enumerate(sample_episodes):
            if ep_idx == -1:
                ep_idx = len(self.files) - 1
                
            episode = self.load_episode(ep_idx)
            
            print(f"\nEpisode {ep_idx}:")
            print(f"  Length: {len(episode['rewards'])} steps")
            print(f"  Return: {episode['episode_returns']:.1f}")
            print(f"  Positive rewards: {np.sum(episode['rewards'] > 0)}")
            print(f"  Negative rewards: {np.sum(episode['rewards'] < 0)}")
            print(f"  Action agreement: {np.mean(episode['model_actions'] == episode['taken_actions'])*100:.1f}%")
            print(f"  Repeated frames: {np.sum(episode['repeated'])}")
    
    def create_visualizations(self, stats=None):
        """Create comprehensive visualizations of the dataset."""
        if stats is None:
            stats = self.get_dataset_overview()
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Episode Length Distribution
        plt.subplot(3, 3, 1)
        plt.hist(stats['episode_lengths'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Episode Length (steps)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Episode Lengths')
        plt.grid(True, alpha=0.3)
        
        # 2. Episode Returns Distribution
        plt.subplot(3, 3, 2)
        plt.hist(stats['episode_returns'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Episode Return')
        plt.ylabel('Frequency')
        plt.title('Distribution of Episode Returns')
        plt.grid(True, alpha=0.3)
        
        # 3. Action Distribution (Taken Actions)
        plt.subplot(3, 3, 3)
        action_counts = Counter(stats['all_actions'])
        actions = [self.action_names.get(a, f"Action_{a}") for a in sorted(action_counts.keys())]
        counts = [action_counts[a] for a in sorted(action_counts.keys())]
        
        bars = plt.bar(actions, counts, alpha=0.7, edgecolor='black')
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.title('Taken Actions Distribution')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add percentages on bars
        total = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                    f'{count/total*100:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 4. Model vs Taken Actions Comparison
        plt.subplot(3, 3, 4)
        model_action_counts = Counter(stats['all_model_actions'])
        model_counts = [model_action_counts[a] for a in sorted(action_counts.keys())]
        
        x = np.arange(len(actions))
        width = 0.35
        
        plt.bar(x - width/2, counts, width, label='Taken Actions', alpha=0.7)
        plt.bar(x + width/2, model_counts, width, label='Model Selected', alpha=0.7)
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.title('Model vs Taken Actions')
        plt.xticks(x, actions, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Episode Returns Over Time
        plt.subplot(3, 3, 5)
        episode_numbers = range(len(stats['episode_returns']))
        plt.plot(episode_numbers, stats['episode_returns'], alpha=0.7, linewidth=1)
        plt.scatter(episode_numbers[::10], [stats['episode_returns'][i] for i in episode_numbers[::10]], 
                   alpha=0.5, s=20)
        plt.xlabel('Episode Number')
        plt.ylabel('Episode Return')
        plt.title('Episode Returns Over Time')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(episode_numbers, stats['episode_returns'], 1)
        p = np.poly1d(z)
        plt.plot(episode_numbers, p(episode_numbers), "r--", alpha=0.8, linewidth=2, label=f'Trend')
        plt.legend()
        
        # 6. Action Agreement Rate
        plt.subplot(3, 3, 6)
        agreement_rate = np.mean(stats['action_agreement']) * 100
        disagreement_rate = 100 - agreement_rate
        
        plt.pie([agreement_rate, disagreement_rate], 
               labels=[f'Agreement\n{agreement_rate:.1f}%', f'Disagreement\n{disagreement_rate:.1f}%'],
               autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
        plt.title('Model-Human Action Agreement')
        
        # 7. Cumulative Returns
        plt.subplot(3, 3, 7)
        cumulative_returns = np.cumsum(stats['episode_returns'])
        plt.plot(cumulative_returns, linewidth=2)
        plt.xlabel('Episode Number')
        plt.ylabel('Cumulative Return')
        plt.title('Cumulative Returns Across Episodes')
        plt.grid(True, alpha=0.3)
        
        # 8. Episode Length vs Return Scatter
        plt.subplot(3, 3, 8)
        plt.scatter(stats['episode_lengths'], stats['episode_returns'], alpha=0.6, s=30)
        plt.xlabel('Episode Length (steps)')
        plt.ylabel('Episode Return')
        plt.title('Episode Length vs Return')
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(stats['episode_lengths'], stats['episode_returns'])[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        # 9. Action Sequence Analysis (sample episode)
        plt.subplot(3, 3, 9)
        sample_episode = self.load_episode(0)
        episode_length = min(200, len(sample_episode['taken_actions']))  # Show first 200 steps
        steps = range(episode_length)
        
        plt.plot(steps, sample_episode['taken_actions'][:episode_length], 'o-', 
                alpha=0.7, markersize=4, linewidth=1, label='Taken')
        plt.plot(steps, sample_episode['model_actions'][:episode_length], 's-', 
                alpha=0.7, markersize=3, linewidth=1, label='Model Selected')
        plt.xlabel('Time Step')
        plt.ylabel('Action ID')
        plt.title(f'Action Sequence (Episode 0, First {episode_length} steps)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pong_dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nVisualization saved as 'pong_dataset_analysis.png'")
    
    def analyze_behavioral_patterns(self):
        """Analyze behavioral patterns and expertise level."""
        print("\n" + "=" * 40)
        print("BEHAVIORAL PATTERN ANALYSIS")
        print("=" * 40)
        
        # Analyze action sequences and patterns
        action_transitions = defaultdict(lambda: defaultdict(int))
        reaction_patterns = []
        
        # Sample episodes for pattern analysis
        sample_size = min(50, len(self.files))
        sampled_indices = np.linspace(0, len(self.files)-1, sample_size, dtype=int)
        
        print(f"Analyzing behavioral patterns from {sample_size} episodes...")
        
        for ep_idx in sampled_indices:
            episode = self.load_episode(ep_idx)
            actions = episode['taken_actions']
            rewards = episode['rewards']
            
            # Analyze action transitions
            for i in range(len(actions) - 1):
                current_action = actions[i]
                next_action = actions[i + 1]
                action_transitions[current_action][next_action] += 1
            
            # Analyze reaction to rewards
            for i in range(len(rewards) - 1):
                if rewards[i] != 0:  # Non-zero reward
                    reaction_patterns.append({
                        'reward': rewards[i],
                        'action_before': actions[i],
                        'action_after': actions[i + 1] if i + 1 < len(actions) else None
                    })
        
        # Print action transition patterns
        print("\nMost Common Action Transitions:")
        all_transitions = []
        for from_action in action_transitions:
            for to_action in action_transitions[from_action]:
                count = action_transitions[from_action][to_action]
                all_transitions.append((from_action, to_action, count))
        
        all_transitions.sort(key=lambda x: x[2], reverse=True)
        
        for i, (from_action, to_action, count) in enumerate(all_transitions[:10]):
            from_name = self.action_names.get(from_action, f"Action_{from_action}")
            to_name = self.action_names.get(to_action, f"Action_{to_action}")
            print(f"  {i+1:2d}. {from_name:8} -> {to_name:8}: {count:5,} times")
        
        # Analyze reward reactions
        if reaction_patterns:
            positive_rewards = [p for p in reaction_patterns if p['reward'] > 0]
            negative_rewards = [p for p in reaction_patterns if p['reward'] < 0]
            
            print(f"\nReward Reaction Analysis:")
            print(f"  Positive reward events: {len(positive_rewards)}")
            print(f"  Negative reward events: {len(negative_rewards)}")
            
            if positive_rewards:
                pos_actions_after = [p['action_after'] for p in positive_rewards if p['action_after'] is not None]
                pos_action_dist = Counter(pos_actions_after)
                print("  Most common actions after positive reward:")
                for action_id, count in pos_action_dist.most_common(3):
                    action_name = self.action_names.get(action_id, f"Action_{action_id}")
                    print(f"    {action_name}: {count} times")
            
            if negative_rewards:
                neg_actions_after = [p['action_after'] for p in negative_rewards if p['action_after'] is not None]
                neg_action_dist = Counter(neg_actions_after)
                print("  Most common actions after negative reward:")
                for action_id, count in neg_action_dist.most_common(3):
                    action_name = self.action_names.get(action_id, f"Action_{action_id}")
                    print(f"    {action_name}: {count} times")
    
    def export_summary_statistics(self, output_file="dataset_summary.csv"):
        """Export summary statistics to CSV for further analysis."""
        print(f"\nExporting summary statistics to {output_file}...")
        
        episode_data = []
        for i, file_path in enumerate(self.files):
            data = np.load(file_path)
            
            episode_info = {
                'episode_id': i,
                'episode_length': len(data['rewards']),
                'episode_return': data['episode_returns'][0],
                'positive_rewards': np.sum(data['rewards'] > 0),
                'negative_rewards': np.sum(data['rewards'] < 0),
                'total_reward_magnitude': np.sum(np.abs(data['rewards'])),
                'action_agreement_rate': np.mean(data['model selected actions'].flatten() == data['taken actions'].flatten()),
                'repeated_frames': np.sum(data['repeated']),
                'unique_actions': len(np.unique(data['taken actions'])),
                'most_common_action': np.bincount(data['taken actions'].flatten()).argmax()
            }
            episode_data.append(episode_info)
        
        df = pd.DataFrame(episode_data)
        df.to_csv(output_file, index=False)
        print(f"Summary statistics exported to {output_file}")
        
        return df

def main():
    """Main function to run the complete analysis."""
    print("Pong Behavioral Cloning Dataset Analyzer")
    print("=========================================")
    
    # Initialize analyzer
    analyzer = PongDatasetAnalyzer()
    
    # Run comprehensive analysis
    stats = analyzer.get_dataset_overview()
    analyzer.analyze_actions(stats)
    analyzer.analyze_episode_structure()
    analyzer.analyze_behavioral_patterns()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.create_visualizations(stats)
    
    # Export summary statistics
    df = analyzer.export_summary_statistics()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("Files generated:")
    print("  - pong_dataset_analysis.png (comprehensive visualizations)")
    print("  - dataset_summary.csv (episode-level statistics)")
    print("\nUse these files for further analysis, reporting, or model development.")

if __name__ == "__main__":
    main()