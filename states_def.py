from poke_env.environment.abstract_battle import AbstractBattle

from poke_env.player import (
    ObservationType,
)
import numpy as np

from poke_env.data import GenData
from teams import NAME_TO_ID_DICT

def embed_battle_poke_rl(self, battle: AbstractBattle) -> ObservationType:
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
            )

    # We count how many pokemons have not fainted in each team
    n_fainted_mon_team = (
        len([mon for mon in battle.team.values() if mon.fainted])
    )
    n_fainted_mon_opponent = (
        len([mon for mon in battle.opponent_team.values() if mon.fainted])
    )

    state= np.concatenate([
        [NAME_TO_ID_DICT[str(battle.active_pokemon).split(' ')[0]]],
        [NAME_TO_ID_DICT[str(battle.opponent_active_pokemon).split(' ')[0]]],
        [move_base_power for move_base_power in moves_base_power],
        [move_dmg_multiplier for move_dmg_multiplier in moves_dmg_multiplier],
        [n_fainted_mon_team,
        n_fainted_mon_opponent]])
    
    return state

def embed_battle_new(self, battle: AbstractBattle) -> ObservationType:
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
            )
    for available_switch in battle.available_switches:
        pass
    # We count how many pokemons have not fainted in each team
    n_fainted_mon_team = (
        len([mon for mon in battle.team.values() if mon.fainted])
    )
    n_fainted_mon_opponent = (
        len([mon for mon in battle.opponent_team.values() if mon.fainted])
    )

    state= np.concatenate([
        [NAME_TO_ID_DICT[str(battle.active_pokemon).split(' ')[0]]],
        [NAME_TO_ID_DICT[str(battle.opponent_active_pokemon).split(' ')[0]]],
        [move_base_power for move_base_power in moves_base_power],
        [move_dmg_multiplier for move_dmg_multiplier in moves_dmg_multiplier],
        [n_fainted_mon_team,
        n_fainted_mon_opponent]])
    
    return state