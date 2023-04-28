from poke_env.environment.abstract_battle import AbstractBattle

def reward_computing_helper_custom(
    self,
    battle: AbstractBattle,
    *,
    fainted_value: float = 0.15,
    hp_value: float = 0.15,
    number_of_pokemons: int = 6,
    starting_value: float = 0.0,
    status_value: float = 0.15,
    victory_value: float = 1.0,
    opponent_value: float = 1.0,
    active_weight: float = 0.0,
    hp_shift: float = 0.0
) -> float: 
    # 1st compute
    if battle not in self._reward_buffer:
        self._reward_buffer[battle] = starting_value
    current_value = 0

    # Verify if pokemon have fainted or have status
    for mon in battle.team.values():
        current_value += mon.current_hp_fraction * hp_value
        if mon.fainted:
            current_value -= fainted_value
        elif mon.status is not None:
            current_value -= status_value

    if battle.active_pokemon is not None:
        active = battle.active_pokemon
        current_value -= active.current_hp_fraction * hp_value # Reset hp contribution of active pokemon to 0
        current_value += (active.current_hp_fraction - hp_shift) * hp_value * (1 + active_weight)
        # Add shifted hp reward value for active pokemon
        if active.fainted:
            current_value -= fainted_value * active_weight
        elif active.status is not None:
            current_value -= status_value * active_weight

    current_value += (number_of_pokemons- len(battle.team)) * hp_value

    # Verify if opponent pokemon have fainted or have status
    for mon in battle.opponent_team.values():
        current_value -= mon.current_hp_fraction * hp_value * opponent_value
        if mon.fainted:
            current_value += fainted_value * opponent_value
        elif mon.status is not None:
            current_value += status_value * opponent_value

    current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value * opponent_value

    if battle.opponent_active_pokemon is not None:
        active = battle.opponent_active_pokemon
        current_value -= active.current_hp_fraction * hp_value * opponent_value # Reset hp contribution of active pokemon to 0
        current_value += (active.current_hp_fraction - hp_shift) * hp_value * opponent_value * (1 + active_weight)
        # Add shifted hp reward value for active pokemon
        if active.fainted:
            current_value -= fainted_value * active_weight * opponent_value
        elif active.status is not None:
            current_value -= status_value * active_weight * opponent_value

    # Verify if we won or lost
    if battle.won:
        current_value += victory_value
    elif battle.lost:
        current_value -= victory_value

    # Value to return
    to_return = current_value - self._reward_buffer[battle]
    self._reward_buffer[battle] = current_value
    return to_return