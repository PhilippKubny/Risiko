from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .actions import Action
from .map import ADJACENCY
from .state import PHASES, GameState, calculate_reinforcements


def is_action_legal(state: GameState, action: Action, num_players: int) -> bool:
    if action.kind == "reinforce":
        return state.phase == "reinforce" and state.reinforcements > 0
    if action.kind == "attack":
        if state.phase != "attack":
            return False
        from_id = action.params["from"]
        to_id = action.params["to"]
        return (
            state.owners[from_id] == state.current_player
            and state.owners[to_id] != state.current_player
            and state.troops[from_id] > 1
            and to_id in ADJACENCY[from_id]
        )
    if action.kind == "fortify":
        if state.phase != "fortify":
            return False
        from_id = action.params["from"]
        to_id = action.params["to"]
        return (
            state.owners[from_id] == state.current_player
            and state.owners[to_id] == state.current_player
            and state.troops[from_id] > 1
            and to_id in ADJACENCY[from_id]
        )
    if action.kind == "end_phase":
        return state.phase != "reinforce" or state.reinforcements == 0
    return False


def step_state(
    state: GameState,
    action: Action,
    num_players: int,
    max_turns: int,
    rng: np.random.Generator | None = None,
) -> Tuple[GameState, float, bool, Dict[str, object]]:
    rng = rng or np.random.default_rng()
    reward = 0.0
    done = False
    info: Dict[str, object] = {}

    if not is_action_legal(state, action, num_players):
        raise ValueError(f"Illegal action {action}")

    if action.kind == "reinforce":
        territory = action.params["territory"]
        state.troops[territory] += 1
        state.reinforcements -= 1
        if state.reinforcements == 0:
            state.phase = "attack"

    elif action.kind == "attack":
        from_id = action.params["from"]
        to_id = action.params["to"]
        reward += _resolve_attack(state, from_id, to_id, rng)

    elif action.kind == "fortify":
        from_id = action.params["from"]
        to_id = action.params["to"]
        state.troops[from_id] -= 1
        state.troops[to_id] += 1
        state.phase = "reinforce"
        state.current_player = (state.current_player + 1) % num_players
        state.reinforcements = calculate_reinforcements(state.owners, state.current_player)
        state.turn += 1

    elif action.kind == "end_phase":
        _advance_phase(state, num_players)

    winner = _check_winner(state, num_players)
    if winner is not None:
        done = True
        reward = 1.0 if winner == state.current_player else -1.0
        info["winner"] = winner

    if state.turn >= max_turns:
        done = True
        info["timeout"] = True

    return state, reward, done, info


def _advance_phase(state: GameState, num_players: int) -> None:
    if state.phase == "attack":
        state.phase = "fortify"
    elif state.phase == "fortify":
        state.phase = "reinforce"
        state.current_player = (state.current_player + 1) % num_players
        state.reinforcements = calculate_reinforcements(state.owners, state.current_player)
        state.turn += 1


def _resolve_attack(
    state: GameState, from_id: int, to_id: int, rng: np.random.Generator
) -> float:
    attack_dice = min(3, state.troops[from_id] - 1)
    defend_dice = min(2, state.troops[to_id])
    attack_rolls = rng.integers(1, 7, size=attack_dice)
    defend_rolls = rng.integers(1, 7, size=defend_dice)
    attack_rolls.sort()
    defend_rolls.sort()

    attacker_losses = 0
    defender_losses = 0
    for attack_die, defend_die in zip(attack_rolls[::-1], defend_rolls[::-1]):
        if attack_die > defend_die:
            state.troops[to_id] -= 1
            defender_losses += 1
        else:
            state.troops[from_id] -= 1
            attacker_losses += 1

    if state.troops[to_id] <= 0:
        state.owners[to_id] = state.current_player
        state.troops[from_id] -= 1
        state.troops[to_id] = 1
        return 0.25

    if attacker_losses > 0 and defender_losses == 0:
        return -0.1
    return 0.0


def _check_winner(state: GameState, num_players: int) -> int | None:
    for player in range(num_players):
        if np.all(state.owners == player):
            return player
    return None
