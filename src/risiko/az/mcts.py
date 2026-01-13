from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch

from risiko.game.actions import ActionSpace, build_action_space
from risiko.game.rules import is_action_legal, step_state
from risiko.game.state import GameState


@dataclass
class Node:
    state: GameState
    prior: np.ndarray
    visit_count: np.ndarray
    value_sum: np.ndarray
    children: Dict[int, "Node"] = field(default_factory=dict)

    def q_values(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(self.visit_count > 0, self.value_sum / self.visit_count, 0.0)
        return q


def run_mcts(
    root_state: GameState,
    network: torch.nn.Module,
    action_space: ActionSpace | None = None,
    num_simulations: int = 96,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_frac: float = 0.25,
    num_players: int = 2,
    max_turns: int = 200,
) -> Tuple[np.ndarray, Node]:
    action_space = action_space or build_action_space()
    root, _ = _expand(root_state, network, action_space, num_players)
    root = _add_dirichlet_noise(root, dirichlet_alpha, dirichlet_frac)

    for _ in range(num_simulations):
        node = root
        search_path: List[Tuple[Node, int]] = []

        while True:
            action_index = _select_action(node, c_puct)
            search_path.append((node, action_index))
            if action_index in node.children:
                node = node.children[action_index]
                continue
            break

        next_state = node.state.clone()
        next_state, reward, done, _ = step_state(
            next_state,
            action_space.all_actions()[action_index],
            num_players,
            max_turns,
        )
        if done:
            child = Node(
                state=next_state.clone(),
                prior=np.ones(action_space.size, dtype=np.float32) / action_space.size,
                visit_count=np.zeros(action_space.size, dtype=np.float32),
                value_sum=np.zeros(action_space.size, dtype=np.float32),
            )
            node.children[action_index] = child
            _backpropagate(search_path, reward)
            continue

        child, value = _expand(next_state, network, action_space, num_players)
        node.children[action_index] = child

        _backpropagate(search_path, value)

    visit_counts = root.visit_count.astype(np.float32)
    if visit_counts.sum() == 0:
        visit_counts = np.ones_like(visit_counts)
    policy = visit_counts / visit_counts.sum()
    return policy, root


def _expand(
    state: GameState,
    network: torch.nn.Module,
    action_space: ActionSpace,
    num_players: int,
) -> Tuple[Node, float]:
    legal_mask = np.zeros(action_space.size, dtype=np.float32)
    for idx, action in enumerate(action_space.all_actions()):
        if is_action_legal(state, action, num_players):
            legal_mask[idx] = 1.0
    if legal_mask.sum() == 0:
        legal_mask[:] = 1.0
    encoded = torch.tensor(state_encode(state, num_players), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = network(encoded)
    logits = output.policy_logits.squeeze(0).cpu().numpy()
    logits = logits - logits.max()
    policy = np.exp(logits) * legal_mask
    if policy.sum() == 0:
        policy = legal_mask / legal_mask.sum()
    else:
        policy = policy / policy.sum()
    value = output.value.squeeze().item()
    node = Node(
        state=state.clone(),
        prior=policy,
        visit_count=np.zeros(action_space.size, dtype=np.float32),
        value_sum=np.zeros(action_space.size, dtype=np.float32),
    )
    return node, value


def _select_action(node: Node, c_puct: float) -> int:
    total_visits = node.visit_count.sum() + 1.0
    q = node.q_values()
    u = c_puct * node.prior * np.sqrt(total_visits) / (1.0 + node.visit_count)
    scores = q + u
    scores = np.where(node.prior > 0, scores, -np.inf)
    return int(np.argmax(scores))


def _backpropagate(search_path: List[Tuple[Node, int]], value: float) -> None:
    current_value = value
    for node, action_index in reversed(search_path):
        node.visit_count[action_index] += 1
        node.value_sum[action_index] += current_value
        current_value = -current_value


def _add_dirichlet_noise(root: Node, alpha: float, frac: float) -> Node:
    noise = np.random.dirichlet([alpha] * len(root.prior))
    root.prior = (1 - frac) * root.prior + frac * noise
    return root


def state_encode(state: GameState, num_players: int) -> np.ndarray:
    owners = state.owners
    troops = state.troops.astype(np.float32)
    num_territories = len(owners)
    owner_plane = np.zeros((num_players, num_territories), dtype=np.float32)
    owner_plane[owners, np.arange(num_territories)] = 1.0
    troops_plane = troops / (troops.max(initial=1.0))
    phase_plane = np.zeros(3, dtype=np.float32)
    phase_index = ["reinforce", "attack", "fortify"].index(state.phase)
    phase_plane[phase_index] = 1.0
    player_plane = np.zeros(num_players, dtype=np.float32)
    player_plane[state.current_player] = 1.0
    return np.concatenate(
        [owner_plane.reshape(-1), troops_plane, phase_plane, player_plane]
    )
