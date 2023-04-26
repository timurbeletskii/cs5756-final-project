from abc import ABC, abstractmethod
from stable_baselines3 import PPO, DQN
from stable_baselines3.dqn.policies import MlpPolicy

class Stablebaseline_Base(ABC):
    def __init__(self, rl_model):
        self.rl_model = rl_model
    
    @abstractmethod
    def load_model(self, file_path: str):
        pass

    def save_model(self, file_path: str):
        self.rl_model.save(file_path)

    def train(self, total_timesteps:int, progress_bar:bool=True):
        self.rl_model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
        
    def evaluate(self, env, num_episodes:int=100, max_steps:int=1000):
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
                action, _states = self.rl_model.predict(state)
                next_state, reward, done, _info = env.step(action)
                total_rewards += reward
                state = next_state
                step += 1
        return total_rewards/num_episodes
    
        
class PPO_Stablebaseline(Stablebaseline_Base):
    def load_model(self, file_path: str):
        self.rl_model = PPO.load(file_path)


class DQN_Stablebaseline(Stablebaseline_Base):
    def load_model(self, file_path: str):
        self.rl_model = DQN.load(file_path)