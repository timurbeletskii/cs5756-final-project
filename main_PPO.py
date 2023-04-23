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
from models.PPO import Poke_PPO
from reward_def import reward_computing_helper_poke_rl
import argparse
import teams
from utils import plot_training

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

class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

    
def train_player(num_outer_loop: int, num_episodes: int, 
                 gamma: float, plot_steps: int, 
                 reward_def: str, num_eval_episodes: int):
    opponent = MaxDamagePlayer(
        battle_format="gen8ou",
        team=teams.OP_TEAM,
        server_configuration=LocalhostServerConfiguration,
    )
    rl_agent_env = RL_Agent(
        battle_format="gen8ou",
        team=teams.OUR_TEAM,
        server_configuration=LocalhostServerConfiguration,
        start_challenging=True,
        opponent=opponent,
        use_old_gym_api=True,
    )
    if (reward_def == "poke_rl"):
        rl_agent_env.reward_computing_helper = MethodType(reward_computing_helper_poke_rl, rl_agent_env)
    # check_env(rl_agent_env)

    model = Poke_PPO(rl_agent_env)
    model.train()
    rl_agent_env.close()

    # Evaluating the model
    opponent = RandomPlayer(battle_format="gen8ou", team=teams.OP_TEAM)
    eval_random_opp_env = RL_Agent(
        battle_format="gen8ou", 
        team=teams.OUR_TEAM,
        start_challenging=True, 
        opponent=opponent, 
        use_old_gym_api=True,
    )
    print("Results against random player:")
    rwd = model.evaluate(env=eval_random_opp_env,num_episodes=num_eval_episodes)
    print(
        f"PPO Evaluation: {eval_random_opp_env.n_won_battles} victories out of {eval_random_opp_env.n_finished_battles} battles"
    )
    eval_random_opp_env.reset_env(restart=False)
    opponent = MaxDamagePlayer(battle_format="gen8ou", team=teams.OP_TEAM)
    eval_maxdamage_opp_env = RL_Agent(
        battle_format="gen8ou", 
        team=teams.OUR_TEAM,
        start_challenging=True, 
        opponent=opponent, 
        use_old_gym_api=True,
    )
    print("Results against max damage player:")
    rwd = model.evaluate(env=eval_maxdamage_opp_env,num_episodes=num_eval_episodes)
    print(
        f"PPO Evaluation: {eval_maxdamage_opp_env.n_won_battles} victories out of {eval_maxdamage_opp_env.n_finished_battles} battles"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reward Definition Experiment')
    parser.add_argument('--num_outer_loop', type=int, nargs='?',
                        default=100, help="Number of outer loops")
    parser.add_argument('--num_episodes', type=int, nargs='?',
                        default=10, help="Number of episodes to rollout at each loop")
    parser.add_argument('--gamma', type=float, nargs='?',
                        default=0.99, help="Gamma")
    parser.add_argument('--plot_steps', type=int, nargs='?',
                        default=5, help="Number of episodes to rollout at each loop")
    parser.add_argument('--reward_def', type=str, nargs='?',
                        default="default", help="Choose which reward definition to use")
    parser.add_argument('--num_eval_episodes', type=int, nargs='?',
                        default=10, help="Number of episodes to evaluate on")
    args = parser.parse_args()
    train_player(args.num_outer_loop, args.num_episodes, 
                 args.gamma, args.plot_steps, 
                 args.reward_def, args.num_eval_episodes)