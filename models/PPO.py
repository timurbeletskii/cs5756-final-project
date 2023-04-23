import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm

class Poke_PPO:
    def __init__(self, env):
        self.poke_ppo = PPO("MlpPolicy", env, verbose=1)
    
    def train(self):
        self.poke_ppo.learn(total_timesteps=10, progress_bar=True)
        
    def evaluate(self, env, num_episodes = 100, max_steps = 1000):
        """Evaluate the policy network by running multiple episodes.

        Args:
            num_episodes (int): Number of episodes to run
            max_steps (int): Maximum number of steps in the episodes
        Returns:
            average_reward (float): Average reward over the episodes
        """
        total_rewards = 0.0
        for ep in range(num_episodes):
            state = env.reset()
            done, step = False, 0
            while not done and step < max_steps:
                action = self.poke_ppo.predict(state)[0]
                next_state, reward, done, info = env.step(action)
                total_rewards += reward
                state = next_state
                step += 1
        return total_rewards/num_episodes
        