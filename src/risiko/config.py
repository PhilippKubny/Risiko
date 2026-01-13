from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    num_players: int = 2
    max_turns: int = 200
    seed: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    games_per_iteration: int = 8
    mcts_simulations: int = 96
    learning_rate: float = 2e-3
    batch_size: int = 64
    epochs: int = 4
    discount: float = 0.99
