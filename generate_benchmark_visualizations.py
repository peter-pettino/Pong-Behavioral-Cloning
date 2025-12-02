import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Extraction and Preparation ---

# Load the two CSV files
df_comp = pd.read_csv('pong_agent_comparison_report.csv')
df_raw = pd.read_csv('pong_agent_raw_episode_data.csv')

# Extract BC Model metrics from the comparison report
bc_data = {}
for index, row in df_comp.iterrows():
    metric = row['Metric']
    # Clean and convert the result
    result = str(row['BC Model Result']).replace(' ', '')
    result = result.replace('ms', '').replace('%', '')
    
    try:
        if result.replace('.', '', 1).isdigit() or (result.startswith('-') and result[1:].replace('.', '', 1).isdigit()):
            bc_data[metric] = float(result)
        else:
            bc_data[metric] = result 
    except ValueError:
        bc_data[metric] = result 

# Define LLM and Human Benchmarks (from Atari-GPT paper, Table 1 )
LLM_BENCHMARK = {
    'GPT-4V Turbo': -25.25,
    'GPT-4o': -22.5,
    'Gemini 1.5 Flash': -26.0,
    'Claude 3 Haiku': -26.0,
    'Human': 2.0,
    'BC Model (Ours)': bc_data['Average Cumulative Reward']
}

# Define Latency Benchmarks (from paper discussion, 2-7 seconds )
LLM_LATENCY = {
    'GPT-4o': 2500,
    'Gemini 1.5 Flash': 2000,
    'Claude 3 Haiku': 2500,
    'GPT-4 Turbo': 6000,
    'BC Model (Ours)': bc_data['Average Action Latency (ms)']
}

# Define Action Usage data (from BC Model results)
ACTION_USAGE = {
    'NOOP': bc_data['NOOP (%)'],
    'FIRE': bc_data['FIRE (%)'],
    'UP': bc_data['UP (%)'],
    'DOWN': bc_data['DOWN (%)'],
    'RIGHTFIRE': bc_data['RIGHTFIRE (%)'],
    'LEFTFIRE': bc_data['LEFTFIRE (%)']
}

# --- 2. Visualization Generation ---

plt.style.use('ggplot')

# V1: Performance Comparison (Average Cumulative Reward)
fig1, ax1 = plt.subplots(figsize=(8, 5))
agents = list(LLM_BENCHMARK.keys())
scores = list(LLM_BENCHMARK.values())
colors = ['#A9D0F5', '#3A8FB7', '#64B5A6', '#1E804F', '#808080', '#E5734E'] 
bars = ax1.bar(agents, scores, color=colors, edgecolor='black')
ax1.axhline(2.0, color='g', linestyle='--', linewidth=1, label='Human Benchmark')
ax1.set_title('Pong Performance: BC Model vs. LLM Agents (1000 Timesteps)')
ax1.set_ylabel('Average Cumulative Reward')
ax1.set_xlabel('Agent Policy')
ax1.tick_params(axis='x', rotation=45)
y_min = min(scores) - 4
y_max = max(scores) + 4
ax1.set_ylim(y_min, y_max)
for bar in bars:
    yval = bar.get_height()
    # Adjust label position for negative bars
    if yval < 0:
        label_y = yval - 1.5
        va = 'top'
    else:
        label_y = yval + 1
        va = 'bottom'
        
    ax1.text(bar.get_x() + bar.get_width()/2, label_y, f'{yval:.2f}', ha='center', va=va, fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('V1_Performance_Comparison.png')
plt.close(fig1)

# V2: Efficiency Comparison (Latency)
fig2, ax2 = plt.subplots(figsize=(7, 5))
v2_agents = list(LLM_LATENCY.keys())
v2_latency = list(LLM_LATENCY.values())
colors_latency = ['#4CAF50', '#81C784', '#E53935'] 
bars = ax2.bar(v2_agents, v2_latency, color=colors_latency, edgecolor='black', log=True) 
ax2.set_title('Action Latency: Behavioral Cloning vs. LLM Agents')
ax2.set_ylabel('Average Latency (ms, Log Scale)')
ax2.set_xlabel('Agent Policy')
ax2.set_ylim(0.1, 20000)
for i, bar in enumerate(bars):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval * 1.5, f'{yval:.2f} ms', ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')
plt.tight_layout()
plt.savefig('V2_Latency_Comparison.png')
plt.close(fig2)

# V3: Behavioral Analysis (Action Distribution)
fig3, ax3 = plt.subplots(figsize=(6, 6))
relevant_actions = {k: v for k, v in ACTION_USAGE.items() if v > 0.5}
ax3.pie(
    relevant_actions.values(), 
    labels=[f'{k}\n({v:.1f}%)' for k, v in relevant_actions.items()], 
    autopct='',
    startangle=90, 
    colors=['#FFC107', '#4CAF50', '#2196F3', '#00BCD4', '#FF5722', '#9C27B0']
)
ax3.set_title('BC Model Action Usage Distribution (100 Episodes)')
plt.tight_layout()
plt.savefig('V3_Action_Distribution.png')
plt.close(fig3)

# V4: Score Consistency (Episode Reward Histogram)
fig4, ax4 = plt.subplots(figsize=(7, 5))
min_score = df_raw['cumulative_reward'].min()
max_score = df_raw['cumulative_reward'].max()
bins = np.arange(min_score - 0.5, max_score + 1.5, 1)
ax4.hist(df_raw['cumulative_reward'], bins=bins, edgecolor='black', color='#4E89B7', alpha=0.8)
ax4.axvline(LLM_BENCHMARK['Human'], color='g', linestyle='--', linewidth=2, label=f'Human Benchmark ({LLM_BENCHMARK["Human"]:.1f})')
ax4.axvline(LLM_BENCHMARK['BC Model (Ours)'], color='#E5734E', linestyle='-', linewidth=2, label=f'BC Model Average ({LLM_BENCHMARK["BC Model (Ours)"]:.2f})')
ax4.set_title('Consistency Check: BC Model Episode Score Distribution')
ax4.set_xlabel('Cumulative Reward per 1000-Step Episode')
ax4.set_ylabel('Frequency (Number of Episodes)')
ax4.legend()
plt.tight_layout()
plt.savefig('V4_Score_Distribution_Histogram.png')
plt.close(fig4)