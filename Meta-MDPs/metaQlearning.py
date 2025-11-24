import time
import random
from typing import Dict, List, Tuple, Any
import numpy as np
from Qtabularfunctions import*
import pickle

class MetaQLearningRunner:
    def __init__(self, gen, low: float, high: float, num_actions: int, 
                 actionset_dict: Dict, actionspace_dict: Dict, lr: float = 0.1, gamma: float = 0.99, 
                 epsilon: float = 1.0, force_mag: float = 10.0,
                 min_td_error: float = 0.001, consecutive_small_errors: int = 10):
        """
        Initialize the Q-learning experiment runner.
        
        Args:
            gen: Environment generator
            low: Lower bound for state space discretization
            high: Upper bound for state space discretization  
            num_actions: Number of discrete actions
            actionset_dict: Dictionary to store learned actions
            lr: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            force_mag: Force magnitude for agent
            min_td_error: Minimum TD error threshold for early stopping
            consecutive_small_errors: Consecutive small errors needed for early stop
        """
        self.gen = gen
        self.low = low
        self.high = high
        self.num_actions = num_actions # the number of action in actionspace. 
        self.actionset_dict = actionset_dict
        self.actionspace_dict = actionspace_dict
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.force_mag = force_mag
        self.min_td_error = min_td_error
        self.consecutive_small_errors = consecutive_small_errors
        
        # Store run results
        self.run_results = []

    def run_multiple_experiments(self, num_runs: int, episodes_per_run: int, 
                            use_actionset_as_actionspace: bool = False,
                            update_actionset_dict: bool = True,
                            verbose: bool = True) -> List[Dict]:
        """
        Run multiple Q-learning experiments.
        
        Args:
            num_runs: Number of experiments to run
            episodes_per_run: Number of episodes per experiment
            categories: List of categories to use (cycles through if provided)
            use_actionset_as_actionspace: If True, use actionset_dict as action space
            update_actionset_dict: Whether to update actionset_dict with learned policies
            verbose: Whether to print progress information
            
        Returns:
            List of results from each run
        """
        self.run_results = []
        
        if verbose:
            print(f"\nStarting {num_runs} experiments with:")
            print(f"  Episodes per run: {episodes_per_run}")
            print(f"  Using actionset as action space: {use_actionset_as_actionspace}")
            print(f"  Updating actionset_dict: {update_actionset_dict}")
        
        for run_idx in range(num_runs):
            if verbose:
                print(f"\n--- Starting Run {run_idx + 1}/{num_runs} ---")
            
            result = self.run_single_experiment(
                episodes=episodes_per_run,
                env=None,
                use_actionset_as_actionspace=use_actionset_as_actionspace,
                update_actionset_dict=update_actionset_dict,
                verbose=verbose
            )
            
            result['run_id'] = run_idx
            self.run_results.append(result)
            
            if verbose:
                print(f"Run {run_idx + 1} completed in {result['runtime_seconds']:.2f}s")
        
        # Print summary if multiple runs
        if verbose and num_runs > 1:
            self._print_multiple_runs_summary()
        
        return self.run_results

    def _print_multiple_runs_summary(self):
        """Print a summary of multiple runs."""
        runtimes = [r['runtime_seconds'] for r in self.run_results]
        rewards = [r['mean_reward'] for r in self.run_results]
        
        print(f"\n--- Multiple Runs Summary ---")
        print(f"Total runtime: {sum(runtimes):.2f}s")
        print(f"Average runtime per run: {np.mean(runtimes):.2f} ± {np.std(runtimes):.2f}s")
        print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

    def print_summary(self):
        """Print a summary of all runs."""
        if not self.run_results:
            print("No results to summarize. Run experiments first.")
            return
        
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        
        runtimes = [r['runtime_seconds'] for r in self.run_results]
        mean_rewards = [r['mean_reward'] for r in self.run_results]
        
        print(f"Total runs: {len(self.run_results)}")
        print(f"Average runtime: {np.mean(runtimes):.2f} ± {np.std(runtimes):.2f} seconds")
        print(f"Average mean reward: {np.mean(mean_rewards):.2f} ± {np.std(mean_rewards):.2f}")
        print(f"Total actions in actionset_dict: {len(self.actionset_dict)}")
        
        print("\nDetailed results:")
        for result in self.run_results:
            print(f"Run {result['run_id']+1}: {result['category']} - "
                  f"{result['runtime_seconds']:.2f}s, "
                  f"reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")

    def remove_action_repetitions(self) -> Dict:
        """
        Remove duplicate actions from actionset_dict for each state.
        
        Returns:
            Updated actionset_dict with unique actions per state
        """
        for state_tuple, actions in self.actionset_dict.items():
            self.actionset_dict[state_tuple] = list(set(actions))
        return self.actionset_dict

    def save_actionset_dict(self, filepath: str):
        """
        Save actionset_dict to file using pickle.
        
        Args:
            filepath: Path to save the file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.actionset_dict, f)
        print(f"actionset_dict saved to {filepath}")

    def load_actionset_dict(self, filepath: str) -> Dict:
        """
        Load actionset_dict from file using pickle.
        
        Args:
            filepath: Path to load the file from
            
        Returns:
            Loaded actionset_dict
        """
        with open(filepath, 'rb') as f:
            self.actionset_dict = pickle.load(f)
        print(f"actionset_dict loaded from {filepath}")
        return self.actionset_dict

    def evaluate_policy(self, episodes: int = 1000, category: str = None,
                    use_actionset_as_actionspace: bool = False,
                    max_steps_per_episode: int = 1000,
                    env=None,  # Accept existing environment
                    verbose: bool = False) -> Dict[str, Any]:
        """
        Evaluate a policy without training or early stopping.
        """
        start_time = time.time()
        
        # Generate environment if not provided
        if env is None:
            if category is None:
                category = random.choice(list(self.gen.categories.keys()))
            env = self.gen.generate_env(category)
            close_env = True
        else:
            close_env = False
            category = "provided_env"
        
        if verbose:
            print(f"Evaluating policy with {'provided environment' if env else f'category: {category}'}")
            print(f"Using {'actionset_dict' if use_actionset_as_actionspace else 'actionspace_dict'} as action space")
        
        # Choose which action space to use
        if use_actionset_as_actionspace:
            action_space = self.actionset_dict
        else:
            action_space = self.actionspace_dict
        
        # Create agent
        agent = TabularQLearningAgent(
            statespace=[self.low, self.high],
            num_actions=self.num_actions,
            actionspace=action_space, 
            lr=0.0,  # No learning during evaluation
            gamma=self.gamma,
            epsilon=0.0,  # No exploration during evaluation
            force_mag=self.force_mag
        )
        
        # Track evaluation metrics
        episode_rewards = []
        episode_steps = []
        episode_success = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < max_steps_per_episode:
                # Choose action greedily (no exploration)
                a = agent.choose_action(state)
                action = agent.discrete_actions[a]
                
                # Take action in environment
                result = env.step(action)
                if len(result) == 4:
                    next_state, reward, done, info = result
                else:
                    next_state, reward, done, truncated, info = result
                    done = done or truncated
                
                state = next_state
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            episode_success.append(1 if done and steps < max_steps_per_episode else 0)
            
            if verbose and episode % 100 == 0:
                print(f"Evaluation Episode {episode}: Reward = {total_reward:.2f}, Steps = {steps}")
        
        # Close environment only if we created it
        if close_env:
            env.close()
        
        end_time = time.time()
        runtime = end_time - start_time
        
        # Compile results
        result = {
            'category': category,
            'runtime_seconds': runtime,
            'total_episodes': episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_steps': np.mean(episode_steps),
            'std_steps': np.std(episode_steps),
            'success_rate': np.mean(episode_success),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
        
        if verbose:
            print(f"\nEvaluation completed in {runtime:.2f} seconds")
            print(f"Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
            print(f"Success rate: {result['success_rate']:.2%}")
            print(f"Steps per episode: {result['mean_steps']:.1f} ± {result['std_steps']:.1f}")
        
        return result

    def _calculate_comparison_summary(self, comparison_results: Dict) -> Dict[str, Any]:
        """Calculate summary statistics for comparison results."""
        return {
            'actionspace_train_time_mean': np.mean(comparison_results['actionspace_training_runtimes']),
            'actionspace_train_time_std': np.std(comparison_results['actionspace_training_runtimes']),
            'actionset_train_time_mean': np.mean(comparison_results['actionset_training_runtimes']),
            'actionset_train_time_std': np.std(comparison_results['actionset_training_runtimes']),
            'actionspace_eval_reward_mean': np.mean(comparison_results['actionspace_eval_rewards']),
            'actionspace_eval_reward_std': np.std(comparison_results['actionspace_eval_rewards']),
            'actionset_eval_reward_mean': np.mean(comparison_results['actionset_eval_rewards']),
            'actionset_eval_reward_std': np.std(comparison_results['actionset_eval_rewards']),
            'actionspace_eval_time_mean': np.mean(comparison_results['actionspace_eval_runtimes']),
            'actionspace_eval_time_std': np.std(comparison_results['actionspace_eval_runtimes']),
            'actionset_eval_time_mean': np.mean(comparison_results['actionset_eval_runtimes']),
            'actionset_eval_time_std': np.std(comparison_results['actionset_eval_runtimes']),
            'actionspace_success_mean': np.mean(comparison_results['actionspace_success_rates']),
            'actionset_success_mean': np.mean(comparison_results['actionset_success_rates']),
            'train_time_improvement': (np.mean(comparison_results['actionset_training_runtimes']) - 
                                    np.mean(comparison_results['actionspace_training_runtimes'])),
            'reward_improvement': (np.mean(comparison_results['actionset_eval_rewards']) - 
                                np.mean(comparison_results['actionspace_eval_rewards']))
        }
    
    def _print_comparison_summary(self, summary: Dict):
        """Print a formatted summary of the comparison results."""
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        print(f"\n📊 TRAINING PERFORMANCE:")
        print(f"ActionSpace Dict: {summary['actionspace_train_time_mean']:.2f} ± {summary['actionspace_train_time_std']:.2f} s")
        print(f"ActionSet Dict:   {summary['actionset_train_time_mean']:.2f} ± {summary['actionset_train_time_std']:.2f} s")
        print(f"Training Time Difference: {summary['train_time_improvement']:+.2f} s")
        
        print(f"\n🏆 EVALUATION PERFORMANCE (Average Reward):")
        print(f"ActionSpace Dict: {summary['actionspace_eval_reward_mean']:.2f} ± {summary['actionspace_eval_reward_std']:.2f}")
        print(f"ActionSet Dict:   {summary['actionset_eval_reward_mean']:.2f} ± {summary['actionset_eval_reward_std']:.2f}")
        print(f"Reward Difference: {summary['reward_improvement']:+.2f}")
        
        print(f"\n✅ SUCCESS RATES:")
        print(f"ActionSpace Dict: {summary['actionspace_success_mean']:.2%}")
        print(f"ActionSet Dict:   {summary['actionset_success_mean']:.2%}")
        
        # Performance interpretation
        if summary['reward_improvement'] > 0:
            print(f"\n✅ ActionSet policy performs BETTER by {summary['reward_improvement']:.2f} average reward")
        else:
            print(f"\n❌ ActionSet policy performs WORSE by {abs(summary['reward_improvement']):.2f} average reward")
        
        if summary['train_time_improvement'] < 0:
            print(f"✅ ActionSet policy trains FASTER by {abs(summary['train_time_improvement']):.2f} seconds")
        else:
            print(f"❌ ActionSet policy trains SLOWER by {summary['train_time_improvement']:.2f} seconds")

    def run_single_experiment(self, episodes: int, category: str = None, 
                            use_actionset_as_actionspace: bool = False,
                            update_actionset_dict: bool = True,
                            env=None,
                            verbose: bool = False) -> Dict[str, Any]:
        """
        Run a single Q-learning experiment.
        
        Returns:
            Dictionary containing run results and metrics AND the trained agent
        """
        start_time = time.time()
        
        # Choose category if not specified and no env provided
        if category is None and env is None:
            category = random.choice(list(self.gen.categories.keys())[:2])
        
        if verbose:
            print(f"Starting experiment with {'provided environment' if env else f'category: {category}'}")
            if use_actionset_as_actionspace:
                print("Using actionset_dict as action space")
            else:
                mode = "training" + (" (updating actionset_dict)" if update_actionset_dict else " (not updating actionset_dict)")
                print(f"Using actionspace_dict as action space ({mode})")
        
        # Generate environment if not provided
        if env is None:
            env = self.gen.generate_env(category)
            close_env = True
        else:
            close_env = False
            category = "provided_env"

        # Choose which action space to use
        if use_actionset_as_actionspace:
            action_space = self.actionset_dict
        else:
            action_space = self.actionspace_dict
        
        # Create agent
        agent = TabularQLearningAgent(
            statespace=[self.low, self.high],
            num_actions=self.num_actions,
            actionspace=action_space, 
            lr=self.lr,
            gamma=self.gamma,
            epsilon=self.epsilon,
            force_mag=self.force_mag
        )
        
        # Track episode rewards and steps
        episode_rewards = []
        episode_steps = []
        early_stops = 0
        
        # Training loop
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0
            small_error_count = 0
            
            while not done:
                # Choose action (returns index 0-9)
                a = agent.choose_action(state)
                action = agent.discrete_actions[a]
                
                # Take action in environment
                result = env.step(action)
                if len(result) == 4:
                    next_state, reward, done, info = result
                else:
                    next_state, reward, done, truncated, info = result
                    done = done or truncated
                
                # Update Q-table and get TD error
                td_error = agent.update(state, a, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # Check if TD error is small enough to stop episode
                if abs(td_error) < self.min_td_error:
                    small_error_count += 1
                else:
                    small_error_count = 0
                    
                # Stop episode if TD error has been small for consecutive steps
                if small_error_count >= self.consecutive_small_errors:
                    done = True
                    early_stops += 1
                    if verbose and episode % 1000 == 0:
                        print(f"Episode {episode} stopped early due to small TD error")
            
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            
            # Decrease exploration over time
            if episode % 100 == 0:
                agent.decrease_epsilon()
        
        # Close environment only if we created it
        if close_env:
            env.close()

        end_time = time.time()
        runtime = end_time - start_time
        
        # Update actionset_dict only if requested AND we're not using actionset_dict as action space
        if update_actionset_dict and not use_actionset_as_actionspace:
            for state_tuple in agent.disc.get_all_discrete_states():
                self.actionset_dict[state_tuple].append(np.argmax(agent.Q[state_tuple]))
        
        # Compile results
        result = {
            'category': category,
            'runtime_seconds': runtime,
            'total_episodes': episodes,
            'early_stops': early_stops,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_steps': np.mean(episode_steps),
            'final_epsilon': agent.epsilon,
            'mode': 'evaluation' if use_actionset_as_actionspace else 'training',
            'actionset_updated': update_actionset_dict and not use_actionset_as_actionspace,
            'agent': agent  # Return the trained agent!
        }
        
        if verbose:
            print(f"Experiment completed in {runtime:.2f} seconds")
            print(f"Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
            print(f"Early stops: {early_stops}/{episodes}")
            print(f"Mode: {result['mode']}")
            if result['actionset_updated']:
                print("Actionset dictionary was updated")
        
        return result

    def evaluate_trained_agent(self, agent, episodes: int = 1000, 
                            env=None, max_steps_per_episode: int = 1000,
                            verbose: bool = False) -> Dict[str, Any]:
        """
        Evaluate a trained agent without any learning or exploration.
        
        Args:
            agent: The trained TabularQLearningAgent to evaluate
            episodes: Number of evaluation episodes
            env: Environment to use (if None, creates new one)
            max_steps_per_episode: Maximum steps per evaluation episode
            verbose: Whether to print progress information
        """
        start_time = time.time()
        
        # Generate environment if not provided
        if env is None:
            category = random.choice(list(self.gen.categories.keys()))
            env = self.gen.generate_env(category)
            close_env = True
        else:
            close_env = False
            category = "provided_env"
        
        if verbose:
            print(f"Evaluating trained agent with {'provided environment' if env else f'category: {category}'}")
        
        # Track evaluation metrics
        episode_rewards = []
        episode_steps = []
        episode_success = []
        
        # Set agent to evaluation mode (no exploration, no learning)
        original_epsilon = agent.epsilon
        original_lr = agent.lr
        agent.epsilon = 0.0  # No exploration
        # Note: We don't change learning rate since agent.update won't be called
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < max_steps_per_episode:
                # Choose action greedily (no exploration)
                a = agent.choose_action(state)
                action = agent.discrete_actions[a]
                
                # Take action in environment
                result = env.step(action)
                if len(result) == 4:
                    next_state, reward, done, info = result
                else:
                    next_state, reward, done, truncated, info = result
                    done = done or truncated
                
                state = next_state
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            episode_success.append(1 if done and steps < max_steps_per_episode else 0)
            
            if verbose and episode % 100 == 0:
                print(f"Evaluation Episode {episode}: Reward = {total_reward:.2f}, Steps = {steps}")
        
        # Restore agent's original parameters
        agent.epsilon = original_epsilon
        
        if close_env:
            env.close()
        
        end_time = time.time()
        runtime = end_time - start_time
        
        # Compile results
        result = {
            'category': category,
            'runtime_seconds': runtime,
            'total_episodes': episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_steps': np.mean(episode_steps),
            'std_steps': np.std(episode_steps),
            'success_rate': np.mean(episode_success),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'median_reward': np.median(episode_rewards)
        }
        
        if verbose:
            print(f"\nEvaluation completed in {runtime:.2f} seconds")
            print(f"Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
            print(f"Success rate: {result['success_rate']:.2%}")
            print(f"Steps per episode: {result['mean_steps']:.1f} ± {result['std_steps']:.1f}")
        
        return result

    def compare_policy_performance(self, training_episodes: int = 1000, 
                                evaluation_episodes: int = 1000,
                                num_comparisons: int = 5,
                                max_eval_steps: int = 1000) -> Dict[str, Any]:
        """
        Compare performance between policies learned with actionspace_dict vs actionset_dict.
        """
        comparison_results = {
            'actionspace_training_runtimes': [],
            'actionset_training_runtimes': [],
            'actionspace_eval_rewards': [],
            'actionset_eval_rewards': [],
            'actionspace_eval_runtimes': [],
            'actionset_eval_runtimes': [],
            'actionspace_success_rates': [],
            'actionset_success_rates': []
        }
        
        print(f"\n{'='*60}")
        print("POLICY PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        
        for comp_run in range(num_comparisons):
            print(f"\n--- Comparison Run {comp_run + 1}/{num_comparisons} ---")
            
            # Create the SAME environment for this comparison run
            category = random.choice(list(self.gen.categories.keys())[:2])
            env = self.gen.generate_env(category)
            print(f"Using category: {category}")
            
            # Save original actionset_dict to restore later
            original_actionset_dict = self.actionset_dict.copy()
            
            # 1. Train with actionspace_dict using SAME environment
            print("Training with actionspace_dict...")
            actionspace_train_result = self.run_single_experiment(
                episodes=training_episodes,
                env=env,  # Use the same environment
                use_actionset_as_actionspace=False,
                update_actionset_dict=False,
                verbose=False
            )
            
            # Evaluate the trained actionspace agent using SAME environment
            print("Evaluating actionspace_dict policy...")
            actionspace_eval_result = self.evaluate_trained_agent(
                agent=actionspace_train_result['agent'],  # Use the trained agent
                episodes=evaluation_episodes,
                env=env,  # Use the same environment
                max_steps_per_episode=max_eval_steps,
                verbose=False
            )
            
            # 2. Restore actionset_dict and train with actionset_dict using SAME environment
            self.actionset_dict = original_actionset_dict.copy()
            
            print("Training with actionset_dict...")
            actionset_train_result = self.run_single_experiment(
                episodes=training_episodes,
                env=env,  # Use the same environment
                use_actionset_as_actionspace=True,
                update_actionset_dict=False,
                verbose=False
            )
            
            # Evaluate the trained actionset agent using SAME environment
            print("Evaluating actionset_dict policy...")
            actionset_eval_result = self.evaluate_trained_agent(
                agent=actionset_train_result['agent'],  # Use the trained agent
                episodes=evaluation_episodes,
                env=env,  # Use the same environment
                max_steps_per_episode=max_eval_steps,
                verbose=False
            )
            
            # Close the environment
            env.close()
            
            # Store results
            comparison_results['actionspace_training_runtimes'].append(actionspace_train_result['runtime_seconds'])
            comparison_results['actionset_training_runtimes'].append(actionset_train_result['runtime_seconds'])
            comparison_results['actionspace_eval_rewards'].append(actionspace_eval_result['mean_reward'])
            comparison_results['actionset_eval_rewards'].append(actionset_eval_result['mean_reward'])
            comparison_results['actionspace_eval_runtimes'].append(actionspace_eval_result['runtime_seconds'])
            comparison_results['actionset_eval_runtimes'].append(actionset_eval_result['runtime_seconds'])
            comparison_results['actionspace_success_rates'].append(actionspace_eval_result['success_rate'])
            comparison_results['actionset_success_rates'].append(actionset_eval_result['success_rate'])
            
            print(f"ActionSpace - Train: {actionspace_train_result['runtime_seconds']:.2f}s, "
                f"Eval Reward: {actionspace_eval_result['mean_reward']:.2f}, "
                f"Success: {actionspace_eval_result['success_rate']:.2%}")
            print(f"ActionSet   - Train: {actionset_train_result['runtime_seconds']:.2f}s, "
                f"Eval Reward: {actionset_eval_result['mean_reward']:.2f}, "
                f"Success: {actionset_eval_result['success_rate']:.2%}")
        
        # Calculate summary statistics
        summary = self._calculate_comparison_summary(comparison_results)
        
        # Print comprehensive comparison
        self._print_comparison_summary(summary)
        
        return {
            'detailed_results': comparison_results,
            'summary': summary
        }