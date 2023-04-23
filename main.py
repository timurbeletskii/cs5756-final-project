from types import MethodType
import numpy as np

from gym import Space
from gym.spaces import Box
from gym.utils.env_checker import check_env

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import (
    ObservationType,
    RandomPlayer,
    Gen8EnvSinglePlayer,
)
from poke_env import LocalhostServerConfiguration
from poke_env.data import GenData

from models.REINFORCE import PolicyGradient, PolicyNet
from reward_def import reward_computing_helper_poke_rl


class RL_Agent(Gen8EnvSinglePlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GenData.from_gen(8).type_chart
                )

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )

def train_player():
    opponent = RandomPlayer(
        battle_format="gen8randombattle",
        server_configuration=LocalhostServerConfiguration,
    )
    rl_agent_env = RL_Agent(
        battle_format="gen8randombattle",
        server_configuration=LocalhostServerConfiguration,
        start_challenging=True,
        opponent=opponent,
        use_old_gym_api=True,
    )
    rl_agent_env.reward_computing_helper = MethodType(reward_computing_helper_poke_rl, rl_agent_env)
    # check_env(rl_agent_env)
    print(rl_agent_env.observation_space)
    print(rl_agent_env.action_space)

    policy_net = PolicyNet(rl_agent_env.observation_space.shape[0], rl_agent_env.action_space.n, 128)
    policy_gradient = PolicyGradient(rl_agent_env, policy_net, True)
    rewards_1 = policy_gradient.train(num_outer_loop=10, num_episodes=5, gamma=0.99, lr=0.01)
    avg_reward = policy_gradient.evaluate(num_episodes=5)
    print(f"Average reward {avg_reward}")
    rl_agent_env.close()

    # Evaluating the model
    opponent = RandomPlayer(battle_format="gen8randombattle")
    eval_env = RL_Agent(
        battle_format="gen8randombattle", 
        start_challenging=True, 
        opponent=opponent, 
        use_old_gym_api=True,
    )
    print("Results against random player:")
    policy_gradient.env = eval_env
    avg_reward = policy_gradient.evaluate(num_episodes=2)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    eval_env.reset_env(restart=False)

if __name__ == "__main__":
    train_player()