from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .map import TERRITORIES


PHASES = ("reinforce", "attack", "fortify")


@dataclass
class GameState:
    owners: np.ndarray
    troops: np.ndarray
    current_player: int
    phase: str
    reinforcements: int
    turn: int

    def clone(self) -> "GameState":
        return GameState(
            owners=self.owners.copy(),
            troops=self.troops.copy(),
            current_player=self.current_player,
            phase=self.phase,
            reinforcements=self.reinforcements,
            turn=self.turn,
        )


def initial_state(num_players: int, seed: int | None = None) -> GameState:
    rng = np.random.default_rng(seed)
    num_territories = len(TERRITORIES)
    owners = rng.integers(0, num_players, size=num_territories)
    troops = np.ones(num_territories, dtype=np.int64)
    return GameState(
        owners=owners,
        troops=troops,
        current_player=0,
        phase=PHASES[0],
        reinforcements=0,
        turn=0,
    )


def count_territories(owners: np.ndarray, player: int) -> int:
    return int(np.sum(owners == player))


def calculate_reinforcements(owners: np.ndarray, player: int) -> int:
    owned = count_territories(owners, player)
    return max(3, owned // 3)
