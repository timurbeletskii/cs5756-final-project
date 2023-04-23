import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm

# Setting the seed to ensure reproducability
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        """Policy network for the REINFORCE algorithm.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(PolicyNet, self).__init__()
        ## TODO: Implement the policy network for the REINFORCE algorithm here
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.activation = torch.nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, action_dim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, state: torch.Tensor):
        """Forward pass of the policy network.

        Args:
            state (torch.Tensor): State of the environment.

        Returns:
            x (torch.Tensor): Probabilities of the actions.
        """
        ## TODO: Implement the forward pass of the policy network here
        x = self.linear1(state)
        x = self.activation(x)
        x = self.linear2(x)
        return self.softmax(x)
    

class PolicyGradient:
    def __init__(self, env, policy_net: PolicyNet, reward_to_go: bool = False):
        """Policy gradient algorithm based on the REINFORCE algorithm.

        Args:
            env (gym.Env): Environment
            policy_net (PolicyNet): Policy network
            reward_to_go (bool): Whether to use returns or reward-to-go
        """
        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = policy_net.to(self.device)
        self.reward_to_go = reward_to_go
        
    def select_action(self, state):
        """Select an action based on the policy network

        Args:
            state (np.ndarray): State of the environment

        Returns:
            action (int): Action to be taken
        """
        ## TODO: Implement the action selection here based on the policy network output probabilities
        ## Hint: Use torch.distributions.Categorical
        state_tensor = torch.from_numpy(state).to(self.device)
        action_probs = self.policy_net(state_tensor)
        m = torch.distributions.Categorical(action_probs)
        return m.sample().item()

    def compute_loss(self, episode, gamma = 0.99):
        """Compute the loss function for the REINFORCE algorithm

        Args:
            episode (list): List of tuples (state, action, reward). 
            gamma (float): Discount factor

        Returns:
            loss (torch.Tensor): Loss function
        """
        # Extract states, actions and rewards from the episode
        states, actions, rewards = [], [], []
        for state, action, reward in episode:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
        
        # Compute the discounted returns
        discounted_returns = []
        if not self.reward_to_go:
          ## TODO: Part 1: Compute the discounted returns here
            total_reward = 0.0
            for idx, reward in enumerate(rewards):
                total_reward += gamma ** idx * reward
            discounted_returns = [total_reward for _ in range(len(episode))]
        else:
          ## TODO: Part 2: Compute the discounted reward to go here
            for step in range(len(episode)):
                total_reward = 0.0
                for idx, reward in enumerate(rewards[step:]):
                    total_reward += gamma ** (idx + step) * reward
                discounted_returns.append(total_reward)

        discounted_returns = torch.tensor(discounted_returns)

        ## TODO: Implement the loss function for the REINFORCE algorithm here based on the discounted returns and the log probabilities of the actions
        losses = []
        for step in range(len(episode)):
            state_tensor = torch.from_numpy(states[step]).to(self.device)
            losses.append(-discounted_returns[step] * torch.log(self.policy_net(state_tensor)[actions[step]]))

        return torch.stack(losses).sum()
    
    def update_policy(self, episodes, optimizer, gamma):
        """Update the policy network using the batch of episodes

        Args:
            episodes (list): List of episodes
            optimizer (torch.optim): Optimizer
            gamma (float): Discount factor
        """
        # Compute the loss function for each episode
        losses = 0.0
        for episode in episodes:
            losses += self.compute_loss(episode, gamma)
        losses /= len(episodes)

        # Compute the gradients and update the policy network
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    def run_episode(self, render = False):
        """
        Run an episode of the environment and return the episode
        
        Returns:
            episode (list): List of tuples (state, action, reward)
        """
        state = self.env.reset()
        episode = []
        done = False
        while not done:
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        if render:
          self.env.play()
        return episode

    def train(self, num_outer_loop, num_episodes, gamma, lr, plot_steps):
        """Train the policy network using the REINFORCE algorithm

        Args:
            num_outer_loop (int): Number of outerloops, i.e., calls to update_policy
            num_episodes (int): Number of episodes to collect in each iteration
            gamma (float): Discount factor
            lr (float): Learning rate
        """
        ## TODO: Implement the training loop for the REINFORCE algorithm here 
        ## using the update_policy function to update the policy network
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        rewards = []
        for epoch in tqdm(range(num_outer_loop)):
            episodes = []
            for ep in range(num_episodes):
                episodes.append(self.run_episode())
            self.update_policy(episodes, optimizer, gamma)
            if epoch % plot_steps == 0:
                rewards.append(self.evaluate(num_episodes))
        return rewards

    def evaluate(self, num_episodes = 100, max_steps = 1000):
        """Evaluate the policy network by running multiple episodes.

        Args:
            num_episodes (int): Number of episodes to run
            max_steps (int): Maximum number of steps in the episode
        Returns:
            average_reward (float): Average reward over the episodes
        """
        ## TODO: Implement the evaluation loop for the REINFORCE algorithm here by running multiple episodes and averaging the returns
        total_rewards = 0.0
        for ep in range(num_episodes):
            state = self.env.reset()
            done, step = False, 0
            while not done and step < max_steps:
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                total_rewards += reward
                state = next_state
                step += 1
        return total_rewards/num_episodes