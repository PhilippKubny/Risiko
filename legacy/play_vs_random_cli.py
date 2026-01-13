# file: play_vs_random_cli.py
MODEL_PATH = "risk_az_latest.pth"
from __future__ import annotations
from typing import Any, Dict, Tuple, List

import random
import numpy as np
import torch

from risk_env import RiskEnv
from risk_az_net import RiskAZNet
from mcts_risk import MCTS
from action_mapping import N_ACTIONS, action_to_index, index_to_action

Action = Tuple[str, Dict[str, Any]]

MODEL_PATH = "risk_az_latest.pth"  # ggf. anpassen


def load_net(device: str = "cpu") -> RiskAZNet:
    # Dummy-Env für Input-Dim
    env = RiskEnv()
    obs, _ = env.reset()
    input_dim = obs["flat_features"].shape[0]

    net = RiskAZNet(input_dim=input_dim, n_actions=N_ACTIONS).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def choose_action_nn(env: RiskEnv, net: RiskAZNet, n_simulations: int, device: str) -> Action:
    mcts = MCTS(env, net, c_puct=1.5, device=device)
    pi = mcts.run_search(n_simulations=n_simulations)

    legal_actions = env.get_legal_actions()
    legal_indices: List[Tuple[int, Action]] = []
    for a in legal_actions:
        try:
            idx = action_to_index(a)
            legal_indices.append((idx, a))
        except ValueError:
            continue

    if not legal_indices:
        return ("end_phase", {})

    # Greedy auf MCTS-Policy
    logits = np.array([pi[idx] for idx, _ in legal_indices], dtype=np.float32)
    best_idx = int(np.argmax(logits))
    _, action = legal_indices[best_idx]
    return action


def choose_action_random(env: RiskEnv) -> Action:
    legal = env.get_legal_actions()
    return random.choice(legal)


def print_state_info(env: RiskEnv) -> None:
    s = env.state
    assert s is not None
    print(f"\n=== Step {s.step_count} | Player: {env.players[s.current_player_idx]} | Phase: {env.phase_str} ===")


def main():
    device = "cpu"
    net = load_net(device=device)

    env = RiskEnv(max_steps=5000)
    obs, info = env.reset()
    done = False

    print("Starte Spiel: Lila (NN+MCTS) vs. Rot (Random).")

    while not done:
        print_state_info(env)
        current_player = env.players[env.state.current_player_idx]  # type: ignore

        if current_player == env.lila_name:
            action = choose_action_nn(env, net, n_simulations=4, device=device)
            print("Lila spielt:", action)
        else:
            action = choose_action_random(env)
            print("Rot (Random) spielt:", action)

        obs, reward, done, info = env.step(action)

        if done:
            print("\nSpiel beendet:", info, "Reward aus Sicht von Lila:", reward)
            break

        # Optional: kleine Pause, damit man mitlesen kann
        # import time; time.sleep(0.1)


if __name__ == "__main__":
    main()
