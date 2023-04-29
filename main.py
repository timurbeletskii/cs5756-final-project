import numpy as np

from gym import Space
from gym.spaces import Box
from gym.utils.env_checker import check_env

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import (
    ObservationType,
    RandomPlayer,
    Gen8EnvSinglePlayer,
    SimpleHeuristicsPlayer,
)
from poke_env import LocalhostServerConfiguration
from poke_env.data import GenData
from stable_baselines3 import DQN, PPO, A2C

from models.REINFORCE import PolicyGradient, PolicyNet
from models.stablebaseline_models import A2C_Stablebaseline, DQN_Stablebaseline, PPO_Stablebaseline
from reward_def import reward_computing_helper_custom 
import argparse
import teams
from utils import plot_training


class RL_Agent(Gen8EnvSinglePlayer):
    def __init__(self, reward_type="default", fainted_value=2.0, 
        hp_value=1.0, victory_value=15.0, status_value=0.15, opponent_weight = 1.0, active_weight = 0.0, hp_shift = 0.0,
        *args, **kwargs):
        super(RL_Agent, self).__init__(*args, **kwargs)
        self.opponent_weight = opponent_weight
        self.reward_type = reward_type
        self.fainted_value=fainted_value
        self.hp_value=hp_value
        self.victory_value = victory_value
        self.status_value = status_value
        self.active_weight = active_weight
        self.hp_shift = hp_shift

    def calc_reward(self, last_battle, current_battle) -> float:
        if self.reward_type == "default":
            return self.reward_computing_helper(
                current_battle, fainted_value=self.fainted_value, 
                hp_value=self.hp_value, victory_value=self.victory_value, status_value=self.status_value 
            )
        elif self.reward_type == "custom":
            return reward_computing_helper_custom(
                self, current_battle, fainted_value = self.fainted_value, hp_value=self.hp_value,
                victory_value=self.victory_value, status_value=self.status_value,
                opponent_value=self.opponent_weight, active_weight=self.active_weight,
                hp_shift=self.hp_shift
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

    
def train_reinforce(num_outer_loop: int, num_episodes: int, 
                 gamma: float, lr: float, plot_steps: int, 
                 reward_type: str, num_eval_episodes: int, model_path: str, log_num: int):
    opponent = MaxDamagePlayer(
        battle_format="gen8ou",
        team=teams.OP_TEAM,
        server_configuration=LocalhostServerConfiguration,
    )
    rl_agent_env = RL_Agent(
        reward_type=reward_type,
        battle_format="gen8ou",
        team=teams.OUR_TEAM,
        server_configuration=LocalhostServerConfiguration,
        # start_challenging=True,
        opponent=opponent,
        # use_old_gym_api=True,
    )
    # check_env(rl_agent_env)

    policy_net = PolicyNet(rl_agent_env.observation_space.shape[0], rl_agent_env.action_space.n, 128)
    policy_gradient = PolicyGradient(rl_agent_env, policy_net, True)
    rewards = policy_gradient.train(num_outer_loop, num_episodes, gamma, lr, plot_steps)
    plot_training(list(range(0, num_outer_loop, plot_steps)), rewards, "training_plot")
    avg_reward = policy_gradient.evaluate(num_episodes=num_episodes)
    print(f"Average reward {avg_reward}")
    rl_agent_env.close()
    policy_gradient.save_model(model_path)

    # Evaluating the model
    evaluate_trained_model(policy_gradient, "REINFORCE", num_eval_episodes, log_num)


def train_ppo(total_timestep: int, n_steps: int, n_epochs: int,
                 gamma: float, lr: float, 
                 reward_type: str, num_eval_episodes: int, model_path: str, reward_def_dict: dict, log_num: int):
    opponent = SimpleHeuristicsPlayer(
        battle_format="gen8ou",
        team=teams.OP_TEAM,
        server_configuration=LocalhostServerConfiguration,
    )
    rl_agent_env = RL_Agent(
        reward_type=reward_type,
        battle_format="gen8ou",
        team=teams.OUR_TEAM,
        server_configuration=LocalhostServerConfiguration,
        start_challenging=True,
        opponent=opponent,
        use_old_gym_api=True,
        fainted_value=reward_def_dict["fainted_value"],
        hp_value=reward_def_dict["hp_value"],
        victory_value=reward_def_dict["victory_value"],
        active_weight=reward_def_dict["active_weight"],
        opponent_weight=reward_def_dict["opponent_weight"],
        hp_shift=reward_def_dict["hp_shift"],
        status_value=reward_def_dict["status_value"],
    )
    # check_env(rl_agent_env)

    # n_epochs is the number of epoch when optimizing the surrogate loss (default 10)
    # n_steps is the number of steps to run for each environment per update (default 2048)
    # total_timesteps used in learn method is the total number of samples (env steps) to train on
    # So total_timesteps = n_epochs * n_steps?

    ppo = PPO("MlpPolicy", rl_agent_env, 
              n_steps=n_steps, n_epochs=n_epochs, batch_size=n_steps,
              gamma=gamma, learning_rate=lr, verbose=1, 
              tensorboard_log="tensorboard_logs/ppo/")
    model = PPO_Stablebaseline(ppo)
    model.train(total_timesteps=total_timestep)
    rl_agent_env.close()
    model.save_model(model_path)

    # Evaluating the mode
    evaluate_trained_model(model, "PPO", num_eval_episodes, log_num)


def train_dqn(total_timesteps: int, num_episodes: int, 
                 gamma: float, lr: float,
                 reward_type: str, num_eval_episodes: int, model_path: str, reward_def_dict: dict, log_num: int):
    opponent = MaxDamagePlayer(
        battle_format="gen8ou",
        team=teams.OP_TEAM,
        server_configuration=LocalhostServerConfiguration,
    )
    rl_agent_env = RL_Agent(
        reward_type=reward_type,
        battle_format="gen8ou",
        team=teams.OUR_TEAM,
        server_configuration=LocalhostServerConfiguration,
        start_challenging=True,
        opponent=opponent,
        use_old_gym_api=True,
    )
    # check_env(rl_agent_env)

    dqn = DQN("MlpPolicy", rl_agent_env, gamma=gamma, learning_rate=lr, verbose=1, 
              tensorboard_log="tensorboard_logs/dqn/")
    model = DQN_Stablebaseline(dqn)
    model.train(total_timesteps=total_timesteps)
    rl_agent_env.close()
    model.save_model(model_path)

    # Evaluating the model
    evaluate_trained_model(model, "DQN", num_eval_episodes, log_num)


def evaluate_trained_model(model, model_name: str, num_eval_episodes: int, log_num: int):
    with open(f'./results/{model_name}_{log_num}.log', 'a') as file:
        # Evaluating the model against random player
        opponent1 = RandomPlayer(battle_format="gen8ou", team=teams.OP_TEAM)
        eval_env = RL_Agent(
            battle_format="gen8ou", 
            team=teams.OUR_TEAM,
            opponent=opponent1, 
        )    
        file.write("Results against random player:\n")
        if model_name == "REINFORCE":
            rwd = model.evaluate(num_episodes=num_eval_episodes)
        else:
            rwd = model.evaluate(env=eval_env,num_episodes=num_eval_episodes)
        file.write(
            f"{model_name} Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} battles\n"
        )

        # Evaluating the model against max damage player
        opponent2 = MaxDamagePlayer(battle_format="gen8ou", team=teams.OP_TEAM)
        eval_env.reset_env(restart=True, opponent=opponent2)
        file.write("Results against max damage player:\n")
        rwd = model.evaluate(env=eval_env,num_episodes=num_eval_episodes)
        file.write(
            f"{model_name} Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} battles\n"
        )
        
        opponent3 = SimpleHeuristicsPlayer(battle_format="gen8ou", team=teams.OP_TEAM)
        eval_env.reset_env(restart=True, opponent=opponent3)
        rwd = model.evaluate(env=eval_env, num_episodes=num_eval_episodes)
        file.write("Results against simple heuristic player:\n")
        file.write(
            f"{model_name} Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} battles\n"
        )
        eval_env.reset_env(restart=False)


def train_a2c(total_timesteps: int, num_episodes: int, 
                 gamma: float, lr: float,
                 reward_type: str, num_eval_episodes: int, model_path: str, reward_def_dict: dict, log_num: int):
    opponent = MaxDamagePlayer(
        battle_format="gen8ou",
        team=teams.OP_TEAM,
        server_configuration=LocalhostServerConfiguration,
    )
    rl_agent_env = RL_Agent(
        reward_type=reward_type,
        battle_format="gen8ou",
        team=teams.OUR_TEAM,
        server_configuration=LocalhostServerConfiguration,
        start_challenging=True,
        opponent=opponent,
        use_old_gym_api=True,
        fainted_value=reward_def_dict["fainted_value"],
        hp_value=reward_def_dict["hp_value"],
        victory_value=reward_def_dict["victory_value"],
        active_weight=reward_def_dict["active_weight"],
        opponent_weight=reward_def_dict["opponent_weight"],
        hp_shift=reward_def_dict["hp_shift"],
        status_value=reward_def_dict["status_value"],
    )

    dqn = A2C("MlpPolicy", rl_agent_env, verbose=1, 
              tensorboard_log="tensorboard_logs/a2c/")
    model = A2C_Stablebaseline(dqn)
    model.train(total_timesteps=total_timesteps)
    rl_agent_env.close()
    model.save_model(model_path)

    # Evaluating the model
    evaluate_trained_model(model, "A2C", num_eval_episodes, log_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reward Definition Experiment')
    parser.add_argument('--num_outer_loop', type=int, nargs='?',
                        default=100, help="Number of outer loops")
    parser.add_argument('--num_episodes', type=int, nargs='?',
                        default=10, help="Number of episodes to rollout at each loop")
    parser.add_argument('--total_timesteps', type=int, nargs='?',
                        default=10000, help="Number of timesteps (stablebaseline)")
    parser.add_argument('--n_steps', type=int, nargs='?',
                        default=64, help="Number of steps to rollout per update (stablebaseline)")
    parser.add_argument('--gamma', type=float, nargs='?',
                        default=0.75, help="Gamma")
    parser.add_argument('--lr', type=float, nargs='?',
                        default=0.00025, help="Learning rate")
    parser.add_argument('--plot_steps', type=int, nargs='?',
                        default=5, help="Number of episodes to rollout at each loop")
    parser.add_argument('--reward_type', type=str, nargs='?',
                        default="default", help="Choose which reward definition to use")
    parser.add_argument('--num_eval_episodes', type=int, nargs='?',
                        default=100, help="Number of episodes to evaluate on")
    parser.add_argument("--algo", type=str, nargs='?',default="ppo", help="Choose which algorithm to use")
    parser.add_argument("--model_path", type=str, nargs='?',default="model.zip", help="Path to save model")
    parser.add_argument("--evaluate_only", action="store_true", help="Evaluate only")
    parser.add_argument('--fainted_val', type=float, nargs='?', default=2.0)
    parser.add_argument('--hp_val', type=float, nargs='?', default=1.0)
    parser.add_argument('--victory_val', type=float, nargs='?', default=15.0)
    parser.add_argument('--status_val', type=float, nargs='?', default=0.15)
    parser.add_argument('--op_wgt', type=float, nargs='?', default=1.0)
    parser.add_argument('--act_wgt', type=float, nargs='?', default=0.0)
    parser.add_argument('--hp_shift', type=float, nargs='?', default=0.0)
    parser.add_argument('--log_num', type=int, nargs='?', default=0)
    args = parser.parse_args()

    if args.evaluate_only:
        match args.algo:
            case "reinforce":
                pass
            case "ppo":
                model = PPO_Stablebaseline(None)
                model.load_model(args.model_path)
                evaluate_trained_model(model, "PPO", args.num_eval_episodes, args.log_num)
            case "dqn":
                model = DQN_Stablebaseline(None)
                model.load_model(args.model_path)
                evaluate_trained_model(model, "DQN", args.num_eval_episodes, args.log_num)
                
    else:
        reward_def_dict = {"fainted_value": args.fainted_val,
                "hp_value": args.hp_val, "victory_value": args.victory_val, 'status_value': args.status_val, 
                'opponent_weight'  : args.op_wgt, 'active_weight'  : args.act_wgt, 'hp_shift'  : args.hp_shift}
        match args.algo:
            case "reinforce":
                train_reinforce(args.num_outer_loop, args.num_episodes,
                            args.gamma, args.lr, args.plot_steps,
                            args.reward_type, args.num_eval_episodes, args.model_path, args.log_num)
            case "ppo":
                epochs = args.total_timesteps // args.n_steps
                train_ppo(args.total_timesteps, args.n_steps, epochs,
                    args.gamma, args.lr, 
                    args.reward_type, args.num_eval_episodes, args.model_path, reward_def_dict, args.log_num)
            case "dqn":
                train_dqn(args.total_timesteps, args.num_episodes, 
                        args.gamma, args.lr, 
                        args.reward_type, args.num_eval_episodes, args.model_path, reward_def_dict, args.log_num)
            case "a2c":
                train_a2c(args.total_timesteps, args.num_episodes, 
                        args.gamma, args.lr, 
                        args.reward_type, args.num_eval_episodes, args.model_path, reward_def_dict, args.log_num)