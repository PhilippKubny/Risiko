from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from risiko.game.actions import ActionSpace, build_action_space
from risiko.game.env import RiskEnv
from risiko.game.state import GameState
from .mcts import run_mcts, state_encode


@dataclass
class SelfPlaySample:
    state: List[float]
    policy: List[float]
    value: float


@dataclass
class SelfPlayConfig:
    num_simulations: int = 96
    temperature: float = 1.0
    max_moves: int = 400


class SelfPlayRunner:
    def __init__(
        self,
        network: torch.nn.Module,
        env: RiskEnv,
        config: SelfPlayConfig | None = None,
    ) -> None:
        self.network = network
        self.env = env
        self.config = config or SelfPlayConfig()
        self.action_space: ActionSpace = env.action_space or build_action_space()

    def play_game(self) -> List[SelfPlaySample]:
        state = self.env.reset()
        game_history: List[Dict[str, object]] = []
        done = False
        moves = 0

        while not done and moves < self.config.max_moves:
            policy, _ = run_mcts(
                state,
                self.network,
                action_space=self.action_space,
                num_simulations=self.config.num_simulations,
                num_players=self.env.num_players,
                max_turns=self.env.max_turns,
            )
            action_index = self._sample_action(policy)
            action = self.action_space.all_actions()[action_index]
            game_history.append(
                {
                    "state": state_encode(state, self.env.num_players).tolist(),
                    "policy": policy.tolist(),
                    "player": state.current_player,
                }
            )
            step = self.env.step(action)
            state = step.state
            done = step.done
            moves += 1

        winner = None
        if done and "winner" in step.info:
            winner = step.info["winner"]

        samples: List[SelfPlaySample] = []
        for entry in game_history:
            player = int(entry["player"])
            value = 0.0
            if winner is not None:
                value = 1.0 if winner == player else -1.0
            samples.append(
                SelfPlaySample(
                    state=entry["state"],
                    policy=entry["policy"],
                    value=value,
                )
            )
        return samples

    def _sample_action(self, policy: np.ndarray) -> int:
        if self.config.temperature <= 0.0:
            return int(np.argmax(policy))
        scaled = np.power(policy, 1.0 / self.config.temperature)
        if scaled.sum() == 0:
            scaled = np.ones_like(scaled) / len(scaled)
        else:
            scaled = scaled / scaled.sum()
        return int(np.random.choice(len(policy), p=scaled))
