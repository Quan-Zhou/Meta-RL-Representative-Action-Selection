import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd
import time

class GaussianCTS:
    def __init__(self, n_arms, budget, prior_var=1.0, noise_var=1.0):
        """
        n_arms: number of base arms
        budget: number of arms to select per round (top-k)
        prior_var: prior variance of each arm
        noise_var: observation noise variance
        """
        self.n_arms = n_arms
        self.budget = budget
        self.prior_var = prior_var
        self.noise_var = noise_var

        # Posterior parameters
        self.mu = np.zeros(n_arms)
        self.var = np.ones(n_arms) * prior_var

    def step(self, get_rewards):
        """
        Perform one round of Gaussian CTS (top-k oracle)

        get_rewards: function(chosen) -> rewards for chosen arms
        """
        # 1. Sample theta from posterior
        theta = np.random.normal(self.mu, np.sqrt(self.var))

        # 2. Top-k oracle: choose arms with largest sampled values
        chosen = np.argsort(theta)[-self.budget:]

        # 3. Observe rewards from the environment
        rewards = get_rewards[chosen]

        # 4. Update posterior for chosen arms
        for i, r in zip(chosen, rewards):
            prior_prec = 1.0 / self.var[i]
            like_prec = 1.0 / self.noise_var
            post_prec = prior_prec + like_prec
            self.mu[i] = (self.mu[i]*prior_prec + r*like_prec) / post_prec
            self.var[i] = 1.0 / post_prec

        return chosen

class CombinatorialUCB:
    def __init__(self, n_arms, budget, noise_var=1.0):
        """
        n_arms: number of base arms
        budget: number of arms to select per round (top-k)
        noise_var: known variance of rewards
        """
        self.n_arms = n_arms
        self.budget = budget
        self.noise_var = noise_var

        self.means = np.zeros(n_arms)    # empirical means
        self.counts = np.zeros(n_arms)   # number of pulls
        self.t = 1                       # time step

    def step(self, get_rewards):
        """
        Perform one round of combinatorial UCB

        get_rewards: function(chosen) -> rewards for chosen arms
        """
        # 1. Compute UCB for each arm
        ucb = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                ucb[i] = float('inf')  # ensure each arm is pulled at least once
            else:
                ucb[i] = self.means[i] + np.sqrt(2 * self.noise_var * np.log(self.t) / self.counts[i])

        # 2. Top-k selection
        chosen = np.argsort(ucb)[-self.budget:]

        # 3. Observe rewards
        rewards = get_rewards[chosen]

        # 4. Update empirical means and counts
        for i, r in zip(chosen, rewards):
            self.counts[i] += 1
            self.means[i] += (r - self.means[i]) / self.counts[i]

        self.t += 1
        return chosen

# class GPfunctions:
#     def __init__(self, K, length_scale=None, IfStationary=True):
#         self.K=K
#         self.num_points= 500 #number of actions
#         actionspace =  np.linspace(-5,5,self.num_points) #.reshape(-1, 1) # grid points
#         self.actionspace = np.sort(actionspace,axis=0)
#         self.length_scale=length_scale
#         # Compute covariance matrix
#         if IfStationary == True:
#             self.kernel = self.rbf_kernel()
#         else:
#             self.kernel = self.gibbs_kernel()
   
#         self.subset = self.algorithm()

#     # Stationary Gaussian Kernel
#     def rbf_kernel(self):
#         """Computes the RBF kernel matrix."""
#         actionset=self.actionspace.reshape((-1,1))
#         sq_dist = cdist(actionset,actionset, 'sqeuclidean')
#         return np.exp(-sq_dist / (2 * self.length_scale ** 2))
    
#     # Non-stationary Gibbs Kernel
#     def gibbs_kernel(self):
#         """Computes the Gibbs kernel matrix."""
#         K = np.zeros((self.num_points,self.num_points))
#         # Compute the kernel matrix
#         for i in range(self.num_points):
#             for j in range(self.num_points):
#                 K[i,j] = self.gibbs_kernel_fun(self.actionspace[i],self.actionspace[j])
#         return K

#     # Define an input-dependent length scale function l(x)
#     def length_scale_fun(self, x):
#         return 0.5 + 0.5* np.exp(-(x/self.length_scale)**2)  # Short length scale near 0, longer away

#     # Define the 1D Gibbs kernel function
#     def gibbs_kernel_fun(self, x, x_prime):
#         l_x = self.length_scale_fun(x)
#         l_xp = self.length_scale_fun(x_prime)
#         numerator = 2 * l_x * l_xp
#         denominator = l_x**2 + l_xp**2
#         prefactor = np.sqrt(numerator / denominator)
#         exponent = - (x - x_prime)**2 / denominator
#         return prefactor * np.exp(exponent)

#     def samples(self,size):
#         # Sample multiple functions from the GP
#         return np.random.multivariate_normal(mean=np.zeros(self.num_points), cov=self.kernel,size=size)
    
#     def algorithm(self):
#         # Sample multiple functions from the GP
#         f_samples = self.samples(size=self.K)
#         # Find the max index for each batch
#         max_indices = np.argmax(f_samples, axis=1)  # Shape: (num_batches,)
#         # Get unique max indices
#         subset = np.unique(max_indices)

#         while len(subset) < self.K: # add more items until K distinct actions are found
#             f_samples = self.samples(size=self.K-len(subset))
#             max_indices = np.argmax(f_samples, axis=1)
#             subset = np.unique(np.append(subset,max_indices))
  
#         return subset
    
#     def test(self,subset):
#         num_batches = 10**5  # Number of function samples for testing
#         # Sample multiple functions from the GP
#         f_samples = self.samples(size=num_batches) # np.random.multivariate_normal(mean=np.zeros(self.num_points), cov=self.kernel, size=num_batches)
#         return np.average(np.max(f_samples, axis=1)-np.max(f_samples[:,subset], axis=1))

#     def ucb_action_selection(self, N):
#         """
#         Selects a subset of K actions using Upper Confidence Bound (UCB).
#         Returns: List of selected action indices.
#         """
#         # Sample multiple functions from the GP
#         f_samples = self.samples(size=N) 
#         ucb = CombinatorialUCB(self.num_points, self.K, noise_var=1.0)

#         for t in range(N):
#             selected_actions = ucb.step(f_samples[t,:])
#         return selected_actions
    
#     def ts_action_selection(self, N):
#         """
#         Selects a subset of K actions using Thompson Sampling.
#         Returns: List of selected action indices.
#         """
#         # Sample multiple functions from the GP
#         f_samples = self.samples(size=N) 
        
#         cts = GaussianCTS(self.num_points, self.K, prior_var=1.0, noise_var=1.0)

#         for t in range(N):
#             selected_actions = cts.step(f_samples[t,:])
#         return selected_actions  
    
class GPfunctions_noise:
    def __init__(self, K, length_scale=None, IfStationary=True):
        self.K=K
        self.num_points= 500 #number of actions
        actionspace =  np.linspace(-5,5,self.num_points) #.reshape(-1, 1) # grid points
        self.actionspace = np.sort(actionspace,axis=0)
        self.length_scale=length_scale
        # Compute covariance matrix
        if IfStationary == True:
            self.kernel = self.rbf_kernel()
        else:
            self.kernel = self.gibbs_kernel()
        self.noise_var=1e-2
        self.subset = self.algorithm()

    # Stationary Gaussian Kernel
    def rbf_kernel(self):
        """Computes the RBF kernel matrix."""
        actionset=self.actionspace.reshape((-1,1))
        sq_dist = cdist(actionset,actionset, 'sqeuclidean')
        return np.exp(-sq_dist / (2 * self.length_scale ** 2))
    
    def samples(self,size):
        # Sample multiple functions from the GP
        return np.random.multivariate_normal(mean=np.zeros(self.num_points), cov=self.kernel,size=size)
    
    # def argmax(self,f_samples,N):
    #     cts = GaussianCTS(self.num_points, budget=1, prior_var=1.0, noise_var=self.noise_var)
    #     for t in range(N):
    #         # Y_samples = f_samples + np.random.normal(loc=0.0, scale=self.noise_var, size=(1,self.num_points))
    #         selected_actions = cts.step(f_samples[0,:])
    #     return selected_actions  

    # def algorithm(self):
    #     subset=np.array([], dtype=int)
    #     while len(subset) < self.K: # add more items until K distinct actions are found
    #         f_samples = self.samples(size=1) #(size=min(1,self.K-len(subset)))
    #         max_indices = self.argmax(f_samples, N=500)  #np.argmax(f_samples, axis=1)
    #         subset = np.unique(np.append(subset,max_indices))
    #     return subset

    def argmax(self, f_samples, N):
        # cts = GaussianCTS(self.num_points, budget=1, prior_var=1.0, noise_var=self.noise_var)
        # for t in range(N-1):  # only update internal state
        #     _ = cts.step(f_samples[0,:])
        # selected_actions = cts.step(f_samples[0,:])  # final result
        # return selected_actions
    
        bandit = GaussianTS(self.num_points, known_variance=1)
        rewards = []
        for _ in range(N):
            arm = bandit.select_arm()
            # print(selected_actions)
            reward = f_samples[0,arm] 
            bandit.update(arm, reward)
            rewards.append(reward)
        # print("Estimated means:", bandit.prior_mean)
        return bandit.select_arm()

    def algorithm(self):
        subset = set()
        samplesize=0
        while len(subset) < self.K:
            f_samples = self.samples(size=self.K - len(subset))
            for f in f_samples:
                max_indices = self.argmax(f[np.newaxis, :], N=300)
                samplesize+=300
                # subset.update(max_indices)
                subset.add(max_indices)
        self.epsilon_samplesize = samplesize
        return np.array(list(subset))
    
    def test(self,subset):
        num_batches = 10**5  # Number of function samples for testing
        # Sample multiple functions from the GP
        f_samples = self.samples(size=num_batches) # np.random.multivariate_normal(mean=np.zeros(self.num_points), cov=self.kernel, size=num_batches)
        return np.average(np.max(f_samples, axis=1)-np.max(f_samples[:,subset], axis=1))

    def ucb_action_selection(self, N):
        """
        Selects a subset of K actions using Upper Confidence Bound (UCB).
        Returns: List of selected action indices.
        """
        # Sample multiple functions from the GP
        f_samples = self.samples(size=N) 
        ucb = CombinatorialUCB(self.num_points, self.K)

        for t in range(N):
            selected_actions = ucb.step(f_samples[t,:])
        return selected_actions
    
    def ts_action_selection(self, N):
        """
        Selects a subset of K actions using Thompson Sampling.
        Returns: List of selected action indices.
        """
        # Sample multiple functions from the GP
        f_samples = self.samples(size=N) 
        
        cts = GaussianCTS(self.num_points, self.K, prior_var=1.0, noise_var=self.noise_var)

        for t in range(N):
            selected_actions = cts.step(f_samples[t,:])
        return selected_actions  

class UCBPolicy:
    def __init__(self, K):
        self.K=K
        self.counts = np.zeros(K, dtype=int)     # Number of times each arm has been pulled
        self.values = np.zeros(K, dtype=float)   # Empirical means of each arm
        self.total_time = 0                      # Global time step

    def select_arm(self):
        self.total_time += 1
        t = self.total_time

        if t <= self.K:
            # Force one pull per arm in the first K rounds
            return t - 1

        # UCB1 formula assuming known variance = 1
        ucb_values = self.values + np.sqrt(2 * np.log(t) / (self.counts + 1e-8))
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        """Update empirical mean of selected arm with new reward."""
        self.counts[arm] += 1
        n = self.counts[arm]
        old_mean = self.values[arm]
        self.values[arm] = old_mean + (reward - old_mean) / n

    def get_estimates(self):
        return self.values

    def get_counts(self):
        return self.counts
    
class GaussianTS:
    def __init__(self, n_arms, known_variance=1.0):
        self.n_arms = n_arms
        self.prior_mean = np.zeros(n_arms)
        self.prior_precision = np.ones(n_arms) * 1e-6  # Start with vague prior
        self.known_variance = known_variance

    def select_arm(self):
        sampled_means = np.random.normal(
            self.prior_mean,
            np.sqrt(self.known_variance / self.prior_precision)
        )
        return np.argmax(sampled_means)

    def update(self, arm, reward):
        self.prior_precision[arm] += 1
        self.prior_mean[arm] += (reward - self.prior_mean[arm]) / self.prior_precision[arm]

class GPfunctions_Epsilon:
    def __init__(self, K, length_scale=None, actionspace=None, kernel=None):
        self.K=K
        self.length_scale=length_scale
        self.actionspace=actionspace
        self.kernel=kernel
        self.num_points=len(actionspace)
        if len(actionspace)==0:
            self.num_points= 1000 #number of actions
            actionspace =  np.linspace(0,2,self.num_points) #.reshape(-1, 1) # grid points
            self.actionspace = np.sort(actionspace,axis=0)        
            # Compute covariance matrix
            if IfStationary == True:
                self.kernel = self.rbf_kernel()
            else:
                self.kernel = self.gibbs_kernel()
        self.subset = self.algorithm()

    # Stationary Gaussian Kernel
    def rbf_kernel(self):
        """Computes the RBF kernel matrix."""
        actionset=self.actionspace.reshape((-1,1))
        sq_dist = cdist(actionset,actionset, 'sqeuclidean')
        return np.exp(-sq_dist / (2 * self.length_scale ** 2))

    def samples(self,size):
        # Sample multiple functions from the GP
        return np.random.multivariate_normal(mean=np.zeros(self.num_points), cov=self.kernel,size=size)

    def algorithm(self):
        # Sample multiple functions from the GP
        f_samples = self.samples(size=self.K)
        # Find the max index for each batch
        max_indices = np.argmax(f_samples, axis=1)  # Shape: (num_batches,)
        # Get unique max indices
        # subset = np.unique(max_indices)
        # while len(subset)< self.K: # add more items until K distinct actions are found
        #     f_samples = self.samples(size=self.K-len(subset))
        #     max_indices = np.argmax(f_samples, axis=1)
        #     subset = np.append(subset,np.unique(max_indices))
        # # print("Unique actions:", subset)
        return max_indices
    
class gibbsmatrix:
    def __init__(self, num_points=1000,length_scale=0.1):
        self.num_points= num_points#number of actions
        actionspace =  np.linspace(0,2,self.num_points) #.reshape(-1, 1) # grid points
        self.actionspace = np.sort(actionspace,axis=0)        
        # Compute covariance matrix
        self.length_scale=length_scale
        self.kernel = self.gibbs_kernel()

    # Non-stationary Gibbs Kernel
    def gibbs_kernel(self):
        """Computes the Gibbs kernel matrix."""
        K = np.zeros((self.num_points,self.num_points))
        # Compute the kernel matrix
        for i in range(self.num_points):
            for j in range(self.num_points):
                K[i,j] = self.gibbs_kernel_fun(self.actionspace[i],self.actionspace[j])
        return K

    # Define an input-dependent length scale function l(x)
    def length_scale_fun(self, x):
        return self.length_scale + (1-self.length_scale)*np.exp(-x**2)  # Short length scale near 0, longer away

    # Define the 1D Gibbs kernel function
    def gibbs_kernel_fun(self, x, x_prime):
        l_x = self.length_scale_fun(x)
        l_xp = self.length_scale_fun(x_prime)
        numerator = 2 * l_x * l_xp
        denominator = l_x**2 + l_xp**2
        prefactor = np.sqrt(numerator / denominator)
        exponent = - (x - x_prime)**2 / denominator
        return prefactor * np.exp(exponent)
    
class GPfunctions:
    def __init__(self, K, length_scale=None, IfStationary=True, actionspace=None, kernel=None):
        self.K=K
        self.length_scale=length_scale
        if actionspace is None:
            self.num_points= 15 #number of actions
            actionspace =  np.linspace(0,2,self.num_points) #.reshape(-1, 1) # grid points
        self.num_points = len(actionspace)
        self.actionspace = np.sort(actionspace,axis=0)
        # Compute covariance matrix
        if IfStationary and kernel is None:
            self.kernel = self.rbf_kernel()
        else:
            self.kernel = kernel # gibbs kernel is slow to compute 
        self.subset = self.algorithm()
        self.geneate_combinations()
    
    def geneate_combinations(self):
        # Generate all K-action combinations
        all_combinations = list(combinations(range(self.num_points), self.K))
        # Create both dictionaries
        self.combination_index = {comb: idx for idx, comb in enumerate(all_combinations)}
        self.index_combination = {idx: comb for idx, comb in enumerate(all_combinations)}
        self.num_superarm = len(self.combination_index) # number of super arm
        # # Example usage:
        # example_comb = (0, 1, 2, 3, 4)
        # example_idx = 1234
        
    # Stationary Gaussian Kernel
    def rbf_kernel(self):
        """Computes the RBF kernel matrix."""
        actionset=self.actionspace.reshape((-1,1))
        sq_dist = cdist(actionset,actionset, 'sqeuclidean')
        return np.exp(-sq_dist / (2 * self.length_scale ** 2))

    def samples(self,size):
        # Sample multiple functions from the GP
        return np.random.multivariate_normal(mean=np.zeros(self.num_points), cov=self.kernel, size=size)

    def algorithm(self):
        # Sample multiple functions from the GP
        f_samples = self.samples(size=self.K)
        # Find the max index for each batch
        max_indices = np.argmax(f_samples, axis=1)  # Shape: (num_batches,)
        # Get unique max indices
        subset = np.unique(max_indices)

        total_steps=len(subset)
        while len(subset) < self.K: # add more items until K distinct actions are found
            total_steps+=self.K-len(subset)
            f_samples = self.samples(size=self.K-len(subset))
            max_indices = np.argmax(f_samples, axis=1)
            subset = np.unique(np.append(subset,max_indices))
        # print("Unique actions:", subset)

        # print("total samples of bandit instances of ep:",total_steps)
        self.ep_steps=total_steps
        return list(subset)
    
    def test(self,subset):
        num_batches = 10**5  # Number of function samples for testing
        # Sample multiple functions from the GP
        f_samples = self.samples(size=num_batches) # np.random.multivariate_normal(mean=np.zeros(self.num_points), cov=self.kernel, size=num_batches)
        return np.average(np.max(f_samples, axis=1)-np.max(f_samples[:,subset], axis=1))

    def run_ucb(self, time_steps=3000):
        bandit = UCBPolicy(self.num_superarm)
        rewards = []
        for t in range(time_steps+self.num_superarm): #the initialization doesn't count into time steps
            arm = bandit.select_arm()
            selected_actions = list(self.index_combination[arm])
            # if t>len(self.combination_index):
            #     print(selected_actions)
            f_samples = self.samples(size=1).reshape(-1)
            reward = np.max(f_samples[selected_actions])
            bandit.update(arm, reward)
            rewards.append(reward)
        return selected_actions
    
    def run_ts(self, time_steps=3000):
        bandit = GaussianTS(self.num_superarm, known_variance=1)
        rewards = []
        for _ in range(time_steps):
            arm = bandit.select_arm()
            selected_actions = list(self.index_combination[arm])
            # print(selected_actions)
            f_samples = self.samples(size=1).reshape(-1)
            reward = np.max(f_samples[selected_actions]) 
            bandit.update(arm, reward)
            rewards.append(reward)
        # print("Estimated means:", bandit.prior_mean)
        return selected_actions
    
    def run_sh(self, budget=37000):
        num_rounds = int(np.ceil(np.log2(self.num_superarm)))
        arms = [*range(self.num_superarm)]
        current_arms = [*range(self.num_superarm)]
        pulls = {arm: 0 for arm in arms}
        rewards = {arm: 0.0 for arm in arms}
        
        for r in range(num_rounds):
            n_arms = len(current_arms)
            
            pulls_per_arm = budget // (n_arms * num_rounds)
            if pulls_per_arm==0:
                break;

            for _ in range(pulls_per_arm):
                f_samples = self.samples(size=1).reshape(-1)
                for arm in current_arms:
                    selected_actions = list(self.index_combination[arm])
                    reward = np.max(f_samples[selected_actions]) #np.max(f_samples)-
                    rewards[arm] += reward
                    pulls[arm] += 1

            means = {arm: rewards[arm] / pulls[arm] for arm in current_arms if pulls[arm] > 0}
            sorted_arms = sorted(current_arms, key=lambda a: means[a], reverse=True)
            current_arms = sorted_arms[:max(1, n_arms // 2)]  # always keep at least one arm
  
        return list(self.index_combination[current_arms[0]])
    
    def run_sh_sequence(self, budget=37000):
        num_rounds = int(np.ceil(np.log2(self.num_superarm)))
        arms = [*range(self.num_superarm)]
        current_arms = [*range(self.num_superarm)]
        pulls = {arm: 0 for arm in arms}
        rewards = {arm: 0.0 for arm in arms}
        
        total_steps=0
        total_pulls=0
        arm_seq = []
        pull_seq = []
        for r in range(num_rounds):
            n_arms = len(current_arms)
            
            pulls_per_arm = budget // (n_arms * num_rounds)
            if pulls_per_arm==0:
                break;
            total_steps+=pulls_per_arm
            total_pulls+=pulls_per_arm*n_arms
            for _ in range(pulls_per_arm):
                f_samples = self.samples(size=1).reshape(-1)
                for arm in current_arms:
                    selected_actions = list(self.index_combination[arm])
                    reward = np.max(f_samples[selected_actions]) #np.max(f_samples)-
                    rewards[arm] += reward
                    pulls[arm] += 1

            means = {arm: rewards[arm] / pulls[arm] for arm in current_arms if pulls[arm] > 0}
            sorted_arms = sorted(current_arms, key=lambda a: means[a], reverse=True)
            current_arms = sorted_arms[:max(1, n_arms // 2)]  # always keep at least one arm

            arm_seq = arm_seq+[list(self.index_combination[current_arms[0]])]
            pull_seq = pull_seq+[total_pulls]
        
        # print("total samples of bandit instances of sh:",total_steps)
        # self.sh_steps=total_pulls
        # print("total pulls of super arms:",total_pulls)
        return arm_seq, pull_seq