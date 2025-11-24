import matplotlib.pyplot as plt
import numpy as np
import time

def plot_training_comparison(comparison_results):
    """Plot training runtime and evaluation rewards"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training runtime
    ax1.boxplot([comparison_results['actionspace_training_runtimes'], 
                 comparison_results['actionset_training_runtimes']],
                labels=['ActionSpace', 'ActionSet'])
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Runtime')
    
    # Evaluation rewards
    ax2.boxplot([comparison_results['actionspace_eval_rewards'], 
                 comparison_results['actionset_eval_rewards']],
                labels=['ActionSpace', 'ActionSet'])
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Policy Performance')
    
    plt.tight_layout()
    plt.show()

def plot_runtime_vs_reward(comparison_results):
    """Trade-off between training time and performance"""
    plt.figure(figsize=(10, 6))
    
    plt.scatter(comparison_results['actionspace_training_runtimes'], 
                comparison_results['actionspace_eval_rewards'],
                alpha=0.7, label='ActionSpace', s=80)
    plt.scatter(comparison_results['actionset_training_runtimes'], 
                comparison_results['actionset_eval_rewards'],
                alpha=0.7, label='ActionSet', s=80)
    
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Average Reward')
    plt.title('Training Time vs Performance')
    plt.legend()
    plt.show()

def plot_statistical_summary(comparison_results):
    """Bar plot with error bars for key metrics"""
    metrics = ['Training Time', 'Average Reward']
    actionspace_means = [
        np.mean(comparison_results['actionspace_training_runtimes']),
        np.mean(comparison_results['actionspace_eval_rewards'])
    ]
    actionset_means = [
        np.mean(comparison_results['actionset_training_runtimes']),
        np.mean(comparison_results['actionset_eval_rewards'])
    ]
    
    actionspace_stds = [
        np.std(comparison_results['actionspace_training_runtimes']),
        np.std(comparison_results['actionspace_eval_rewards'])
    ]
    actionset_stds = [
        np.std(comparison_results['actionset_training_runtimes']),
        np.std(comparison_results['actionset_eval_rewards'])
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, actionspace_means, width, yerr=actionspace_stds, 
           label='ActionSpace', capsize=5, alpha=0.8)
    ax.bar(x + width/2, actionset_means, width, yerr=actionset_stds, 
           label='ActionSet', capsize=5, alpha=0.8)
    
    ax.set_ylabel('Performance')
    ax.set_title('ActionSpace vs ActionSet Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.show()

def plot_runs_vs_runtime(tradeoff_data):
    """Plot number of training runs vs total runtime"""
    plt.figure(figsize=(10, 6))
    
    num_runs = tradeoff_data['num_runs']
    total_runtimes = tradeoff_data['total_runtimes']
    runtime_stds = tradeoff_data['runtime_stds']
    
    plt.errorbar(num_runs, total_runtimes, yerr=runtime_stds, 
                 marker='o', capsize=5, linewidth=2, markersize=8)
    
    plt.xlabel('Number of Training Runs')
    plt.ylabel('Total Training Runtime (seconds)')
    plt.title('Training Cost vs Number of Runs')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(num_runs, total_runtimes, 1)
    p = np.poly1d(z)
    plt.plot(num_runs, p(num_runs), 'r--', alpha=0.8, label=f'Linear trend')
    plt.legend()
    
    plt.show()

def plot_runs_vs_reward(tradeoff_data):
    """Plot number of training runs vs evaluation reward"""
    plt.figure(figsize=(10, 6))
    
    num_runs = tradeoff_data['num_runs']
    average_rewards = tradeoff_data['average_rewards']
    reward_stds = tradeoff_data['reward_stds']
    
    plt.errorbar(num_runs, average_rewards, yerr=reward_stds,
                 marker='s', capsize=5, linewidth=2, markersize=8, color='orange')
    
    plt.xlabel('Number of Training Runs')
    plt.ylabel('Average Evaluation Reward')
    plt.title('Performance vs Number of Training Runs')
    plt.grid(True, alpha=0.3)
    
    plt.show()

def plot_size_vs_performance(tradeoff_data):
    """Plot final action set size vs performance metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sizes = tradeoff_data['final_sizes']
    rewards = tradeoff_data['average_rewards']
    runtimes = tradeoff_data['total_runtimes']
    
    # Size vs Reward
    ax1.errorbar(sizes, rewards, yerr=tradeoff_data['reward_stds'],
                 marker='o', capsize=5, linewidth=2, markersize=6)
    ax1.set_xlabel('Final Action Set Size')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Action Set Size vs Performance')
    ax1.grid(True, alpha=0.3)
    
    # Size vs Runtime
    ax2.errorbar(sizes, runtimes, yerr=tradeoff_data['runtime_stds'],
                 marker='s', capsize=5, linewidth=2, markersize=6, color='red')
    ax2.set_xlabel('Final Action Set Size')
    ax2.set_ylabel('Total Runtime (seconds)')
    ax2.set_title('Action Set Size vs Training Cost')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_comprehensive_tradeoff(tradeoff_data):
    """Comprehensive plot showing all trade-offs"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    num_runs = tradeoff_data['num_runs']
    sizes = tradeoff_data['final_sizes']
    rewards = tradeoff_data['average_rewards']
    runtimes = tradeoff_data['total_runtimes']
    
    # Plot 1: Runs vs Reward
    ax1.errorbar(num_runs, rewards, yerr=tradeoff_data['reward_stds'],
                 marker='o', capsize=5, linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Training Runs')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Performance vs Training Runs')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Runs vs Runtime
    ax2.errorbar(num_runs, runtimes, yerr=tradeoff_data['runtime_stds'],
                 marker='s', capsize=5, linewidth=2, markersize=6, color='red')
    ax2.set_xlabel('Number of Training Runs')
    ax2.set_ylabel('Total Runtime (seconds)')
    ax2.set_title('Training Cost vs Training Runs')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Size vs Reward
    ax3.scatter(sizes, rewards, s=100, alpha=0.7)
    ax3.set_xlabel('Action Set Size')
    ax3.set_ylabel('Average Reward')
    ax3.set_title('Action Set Size vs Performance')
    ax3.grid(True, alpha=0.3)
    
    # Add size labels
    for i, (size, reward) in enumerate(zip(sizes, rewards)):
        ax3.annotate(f'{num_runs[i]} runs', (size, reward), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 4: Size vs Runtime
    ax4.scatter(sizes, runtimes, s=100, alpha=0.7, color='red')
    ax4.set_xlabel('Action Set Size')
    ax4.set_ylabel('Total Runtime (seconds)')
    ax4.set_title('Action Set Size vs Training Cost')
    ax4.grid(True, alpha=0.3)
    
    # Add size labels
    for i, (size, runtime) in enumerate(zip(sizes, runtimes)):
        ax4.annotate(f'{num_runs[i]} runs', (size, runtime), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def analyze_runs_tradeoff(runner, max_runs=8, episodes_per_run=1000):
    """Complete analysis of trade-offs between number of runs and performance"""
    print("="*60)
    print("ANALYZING TRADE-OFF: Number of Runs vs Performance")
    print("="*60)
    
    # Collect data
    tradeoff_data = collect_runs_tradeoff_data(
        runner, 
        max_runs=max_runs, 
        episodes_per_run=episodes_per_run
    )
    
    # Generate all plots
    plot_runs_vs_runtime(tradeoff_data)
    plot_runs_vs_reward(tradeoff_data)
    plot_size_vs_performance(tradeoff_data)
    plot_comprehensive_tradeoff(tradeoff_data)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for i, num_runs in enumerate(tradeoff_data['num_runs']):
        print(f"{num_runs} runs: Size={tradeoff_data['final_sizes'][i]}, "
              f"Reward={tradeoff_data['average_rewards'][i]:.2f} ± {tradeoff_data['reward_stds'][i]:.2f}, "
              f"Runtime={tradeoff_data['total_runtimes'][i]:.2f}s")
    
    # Find optimal point (highest reward per runtime)
    efficiencies = [reward / runtime for reward, runtime in 
                   zip(tradeoff_data['average_rewards'], tradeoff_data['total_runtimes'])]
    optimal_idx = np.argmax(efficiencies)
    
    print(f"\n🎯 Most efficient: {tradeoff_data['num_runs'][optimal_idx]} runs "
          f"(Reward/Runtime: {efficiencies[optimal_idx]:.4f})")
    
    return tradeoff_data

def plot_num_runs_vs_performance(tradeoff_data):
    """Plot how num_runs affects performance metrics"""
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    
    num_runs = tradeoff_data['num_runs']
    
    # Calculate statistics using the simplified key names
    asp_reward_means = [np.mean(rewards) for rewards in tradeoff_data['actionspace_rewards']]
    ast_reward_means = [np.mean(rewards) for rewards in tradeoff_data['actionset_rewards']]
    asp_reward_stds = [np.std(rewards) for rewards in tradeoff_data['actionspace_rewards']]
    ast_reward_stds = [np.std(rewards) for rewards in tradeoff_data['actionset_rewards']]
    
    asp_time_means = [np.mean(times) for times in tradeoff_data['actionspace_runtimes']]
    ast_time_means = [np.mean(times) for times in tradeoff_data['actionset_runtimes']]
    asp_time_stds = [np.std(times) for times in tradeoff_data['actionspace_runtimes']]
    ast_time_stds = [np.std(times) for times in tradeoff_data['actionset_runtimes']]
    
    # Plot 1: num_runs vs Reward
    ax1.errorbar(num_runs, asp_reward_means, yerr=asp_reward_stds, 
                 marker='o', label='ActionSpace', capsize=5, linewidth=2, markersize=8)
    ax1.errorbar(num_runs, ast_reward_means, yerr=ast_reward_stds,
                 marker='s', label='ActionSet', capsize=5, linewidth=2, markersize=8)
    ax1.set_xlabel('K')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Reward at Evaluation Phase')
    ax1.set_xticks(num_runs)  # Set integer ticks directly
    ax1.set_xticklabels(num_runs)  # Use original K values as labels
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: num_runs vs Training Runtime (with error bars)
    ax2.errorbar(num_runs, asp_time_means, yerr=asp_time_stds,
                 marker='o', label='ActionSpace', capsize=5, linewidth=2, markersize=8)
    ax2.errorbar(num_runs, ast_time_means, yerr=ast_time_stds,
                 marker='s', label='ActionSet', capsize=5, linewidth=2, markersize=8)
    ax2.set_xlabel('K')
    ax2.set_ylabel('Average Training Runtime (seconds)')
    ax2.set_title('Runtime at Training Phase')
    ax2.set_xticks(num_runs)  # Set integer ticks directly
    ax2.set_xticklabels(num_runs)  # Use original K values as labels
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def collect_num_runs_tradeoff_data(runner, num_runs_list, episodes_per_run=10000, num_comparisons=30):
    """
    Collect trade-off data for different num_runs values
    
    Args:
        runner: MetaQLearningRunner instance
        num_runs_list: List of num_runs values to test [1, 3, 5, 10, 15, 20]
        episodes_per_run: Episodes per training run
        num_comparisons: Number of evaluation comparisons per trial
    """
    tradeoff_data = {
        'num_runs': [],
        'actionspace_rewards': [],
        'actionset_rewards': [],
        'actionspace_runtimes': [],
        'actionset_runtimes': []
    }
    
    # # Store initial state
    # initial_actionset = runner.actionset_dict.copy()
    
    last_num_runs = 0
    for num_runs in num_runs_list:
        print(f"\n{'='*50}")
        print(f"Testing with num_runs = {num_runs}")
        print(f"{'='*50}")
        
        # # Reset actionset_dict for each trial
        # runner.actionset_dict = initial_actionset.copy()
        
        # Phase 1: Training with current num_runs
        runner.run_multiple_experiments(
            num_runs=int(num_runs-last_num_runs), 
            episodes_per_run=episodes_per_run,
            update_actionset_dict=True,
            use_actionset_as_actionspace=False,
            verbose=True
        )
        
        # Remove repetitions
        runner.remove_action_repetitions()
        
        # Phase 2: Compare performance
        comparison_results = runner.compare_policy_performance(
            training_episodes=episodes_per_run,
            evaluation_episodes=1,
            num_comparisons=num_comparisons
        )

        # Extract the detailed results (this is a list of 30 comparison runs)
        detailed_results = comparison_results['detailed_results']
        
        # Store data for this num_runs value
        tradeoff_data['num_runs'].append(num_runs)
        tradeoff_data['actionspace_runtimes'].append(detailed_results['actionspace_training_runtimes'])
        tradeoff_data['actionset_runtimes'].append(detailed_results['actionset_training_runtimes'])
        tradeoff_data['actionspace_rewards'].append(detailed_results['actionspace_eval_rewards'])
        tradeoff_data['actionset_rewards'].append(detailed_results['actionset_eval_rewards'])
        
        print(f"Completed: num_runs={num_runs}")

        last_num_runs = num_runs
    
    return tradeoff_data

import pickle
import os
from datetime import datetime

def analyze_num_runs_tradeoff(runner, num_runs_list=[1, 3, 5, 8, 10], 
                              episodes_per_run=10000, num_comparisons=50, save_dir="results"):
    """Complete analysis of num_runs trade-off with auto-save"""
    print("="*60)
    print("ANALYZING TRADE-OFF: num_runs vs Performance")
    print("="*60)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"num_runs_tradeoff_{timestamp}"
    
    try:
        # Collect data
        tradeoff_data = collect_num_runs_tradeoff_data(
            runner, 
            num_runs_list=num_runs_list,
            episodes_per_run=episodes_per_run,
            num_comparisons=num_comparisons
        )
        
        # Save data immediately after collection
        data_file = os.path.join(save_dir, f"{base_filename}_data.pkl")
        with open(data_file, 'wb') as f:
            pickle.dump({
                'tradeoff_data': tradeoff_data,
                'num_runs_list': num_runs_list,
                'timestamp': timestamp,
                'runner_config': {
                    'num_actions': runner.num_actions,
                    'lr': runner.lr,
                    'gamma': runner.gamma,
                    'epsilon': runner.epsilon
                }
            }, f)
        print(f"✓ Data saved to: {data_file}")
        
        # Generate plots
        plot_num_runs_vs_performance(tradeoff_data)
        
        # Save the plot
        plot_file = os.path.join(save_dir, f"{base_filename}_plot.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {plot_file}")
        
        # Print detailed summary
        print("\n" + "="*60)
        print("DETAILED RESULTS SUMMARY")
        print("="*60)
        
        for i, num_runs in enumerate(tradeoff_data['num_runs']):
            asp_reward = np.mean(tradeoff_data['actionspace_rewards'][i])
            ast_reward = np.mean(tradeoff_data['actionset_rewards'][i])
            improvement = ast_reward - asp_reward
            
            print(f"num_runs={num_runs:2d}: "
                  f"ActionSpace={asp_reward:6.2f}, "
                  f"ActionSet={ast_reward:6.2f}, "
                  f"Improvement={improvement:+.2f}")
        
        return tradeoff_data
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Attempting to save partial data...")
        
        # Try to save whatever data we have
        if 'tradeoff_data' in locals():
            emergency_file = os.path.join(save_dir, f"{base_filename}_EMERGENCY_SAVE.pkl")
            with open(emergency_file, 'wb') as f:
                pickle.dump({
                    'tradeoff_data': tradeoff_data,
                    'error': str(e),
                    'timestamp': timestamp
                }, f)
            print(f"⚠️  Emergency save created: {emergency_file}")
        raise

def load_tradeoff_data(filename):
    """Load previously saved tradeoff data"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['tradeoff_data']

def resume_analysis(runner, previous_data_file, additional_runs=[]):
    """Resume analysis from a previous save file"""
    print("🔄 Resuming analysis from previous save...")
    
    # Load previous data
    with open(previous_data_file, 'rb') as f:
        saved_data = pickle.load(f)
    
    previous_tradeoff_data = saved_data['tradeoff_data']
    previous_runs = previous_tradeoff_data['num_runs']
    
    # Find which runs we still need to do
    remaining_runs = [r for r in additional_runs if r not in previous_runs]
    
    if not remaining_runs:
        print("✓ No additional runs to process")
        return previous_tradeoff_data
    
    print(f"Processing additional runs: {remaining_runs}")
    
    # Collect data for remaining runs
    new_tradeoff_data = collect_num_runs_tradeoff_data(
        runner, 
        num_runs_list=remaining_runs,
        episodes_per_run=1000,
        num_comparisons=30
    )
    
    # Merge with previous data
    merged_data = {}
    for key in previous_tradeoff_data.keys():
        merged_data[key] = previous_tradeoff_data[key] + new_tradeoff_data[key]
    
    # Sort by num_runs to keep order
    sorted_indices = sorted(range(len(merged_data['num_runs'])), 
                           key=lambda i: merged_data['num_runs'][i])
    
    for key in merged_data.keys():
        merged_data[key] = [merged_data[key][i] for i in sorted_indices]
    
    return merged_data