import numpy as np

class Discretizer:
    def __init__(self, statespace):
        self.state_min,self.state_max  = statespace
        self.original_bins = 100
        self.new_bins = 5
        self.bins=[1]+[self.new_bins]*3
        
    def discretize(self, state):
        discrete_state = []
        
        for i, value in enumerate(state):
            # Convert continuous value to original bin [0, 19]
            normalized = (value - self.state_min[i]) / (self.state_max[i] - self.state_min[i])
            original_bin = int(normalized * self.original_bins)
            
            if i == 0:  # First dimension - always map to bin 0
                new_bin = 0
            else:
            # Map to 5 optimized bins for other dimensions
                if original_bin <= 42:
                    new_bin = 0  # Big group: 0-45
                elif original_bin <= 48:
                    new_bin = 1  # Single bin: 46-48
                elif original_bin <= 50:
                    new_bin = 2  # Single bin: 49-50
                elif original_bin <= 55:
                    new_bin = 3  # Single bin: 51-58
                else:
                    new_bin = 4  # Big group: 59-99
            
            discrete_state.append(new_bin)
        return tuple(discrete_state)

    def get_all_discrete_states(self):
        """Generator that yields all possible discrete states"""
        self.bins=[1]+[self.new_bins]*3
        ranges = [range(bin_size) for bin_size in self.bins]
        from itertools import product
        for state_tuple in product(*ranges):
            yield state_tuple\

class TabularQLearningAgent:
    def __init__(self, statespace ,num_actions=10, actionspace=[] ,lr=0.1, gamma=0.99, epsilon=0.1, force_mag=10.0):
        self.disc = Discretizer(statespace)
        self.num_actions = num_actions
        self.discrete_actions = np.linspace(-force_mag, force_mag, self.num_actions)
        if len(actionspace) < 1:
            self.actionspace={state_tuple: [*range(num_actions)] for state_tuple in self.disc.get_all_discrete_states()}
        else:
            self.actionspace=actionspace
        
        self.Q = np.zeros((*self.disc.bins, num_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        s = self.disc.discretize(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actionspace[s])
        maxloc=np.argmax(self.Q[s][self.actionspace[s]])
        return self.actionspace[s][maxloc]

    def update(self, state, action, reward, next_state, done):
        s = self.disc.discretize(state)
        ns = self.disc.discretize(next_state)

        current_q = self.Q[s][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[ns][self.actionspace[s]])
        
        # Calculate the update
        td_error = target - current_q
        update_amount = self.lr * td_error
        
        # Apply update
        self.Q[s][action] += update_amount
        
        # Debug information (print every 1000 updates or so)
        # if hasattr(self, 'debug_counter'):
        #     self.debug_counter += 1
        # else:
        #     self.debug_counter = 0
            
        # if self.debug_counter % 1000 == 0:
        #     print(f"\n--- Q-update Debug ---")
        #     print(f"State: {s}, Action: {action}, Next State: {ns}")
        #     print(f"Current Q: {current_q:.3f}")
        #     print(f"Target: {target:.3f}")
        #     print(f"TD Error: {td_error:.3f}")
        #     print(f"Update amount: {update_amount:.3f}")
        #     print(f"New Q: {self.Q[s][action]:.3f}")
        #     print(f"Max Q in next state: {np.max(self.Q[ns]):.3f}")
        
        return td_error  # Optional: return for monitoring

    def decrease_epsilon(self, factor=0.9, min_epsilon=0.01):
        """Gradually decrease exploration rate"""
        self.epsilon = max(min_epsilon, self.epsilon * factor)

    def get_policy_dict(self):
        """
        Returns the policy as a dictionary mapping discrete states to actions
        
        Returns:
            dict: {state_tuple: action_index} where action_index corresponds 
                  to self.discrete_actions[action_index]
        """
        policy = {}
        
        # Iterate through all possible discrete states
        for state_tuple in self.disc.get_all_discrete_states():
            # Get the best action for this state
            best_action = np.argmax(self.Q[state_tuple][self.actionspace[state_tuple]])
            policy[state_tuple] = best_action
        
        return policy

    def get_detailed_policy(self):
        """
        Returns a detailed policy with Q-values and action information
        
        Returns:
            dict: {state_tuple: {'action_index': int, 
                                'continuous_action': float,
                                'q_value': float,
                                'all_q_values': list}}
        """
        detailed_policy = {}
        
        for state_tuple in self.disc.get_all_discrete_states():
            best_action_idx = np.argmax(self.Q[state_tuple][self.actionspace[state_tuple]])
            best_q_value = self.Q[state_tuple][best_action_idx]
            continuous_action = self.discrete_actions[best_action_idx]
            all_q_values = self.Q[state_tuple].tolist()
            
            detailed_policy[state_tuple] = {
                'action_index': best_action_idx,
                'continuous_action': continuous_action,
                'q_value': best_q_value,
                'all_q_values': all_q_values
            }
        
        return detailed_policy