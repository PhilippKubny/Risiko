from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .actions import Action, ActionSpace, build_action_space
from .map import TERRITORIES
from .rules import is_action_legal, step_state
from .state import PHASES, GameState, calculate_reinforcements, initial_state


@dataclass
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: Dict[str, object]


class RiskEnv:
    def __init__(self, num_players: int = 2, max_turns: int = 200, seed: int | None = None):
        self.num_players = num_players
        self.max_turns = max_turns
        self.seed = seed
        self.state: GameState | None = None
        self.action_space = build_action_space()

    def reset(self) -> GameState:
        self.state = initial_state(self.num_players, self.seed)
        self.state.reinforcements = calculate_reinforcements(
            self.state.owners, self.state.current_player
        )
        return self.state.clone()

    def legal_action_indices(self) -> List[int]:
        if self.state is None:
            raise RuntimeError("Reset the environment before requesting actions.")
        indices: List[int] = []
        action_list = self.action_space.all_actions()
        for idx, action in enumerate(action_list):
            if is_action_legal(self.state, action, self.num_players):
                indices.append(idx)
        return indices

    def step(self, action: Action) -> StepResult:
        if self.state is None:
            raise RuntimeError("Reset the environment before stepping.")
        state = self.state
        reward = 0.0
        done = False
        info: Dict[str, object] = {}

        state, reward, done, info = step_state(
            state, action, self.num_players, self.max_turns
        )

        return StepResult(state.clone(), reward, done, info)

    def encode_state(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Reset the environment before encoding.")
        owners = self.state.owners
        troops = self.state.troops.astype(np.float32)
        num_territories = len(TERRITORIES)
        owner_plane = np.zeros((self.num_players, num_territories), dtype=np.float32)
        owner_plane[owners, np.arange(num_territories)] = 1.0
        troops_plane = troops / (troops.max(initial=1.0))
        phase_plane = np.zeros((len(PHASES),), dtype=np.float32)
        phase_plane[PHASES.index(self.state.phase)] = 1.0
        player_plane = np.zeros((self.num_players,), dtype=np.float32)
        player_plane[self.state.current_player] = 1.0
        return np.concatenate(
            [
                owner_plane.reshape(-1),
                troops_plane,
                phase_plane,
                player_plane,
            ]
        )

