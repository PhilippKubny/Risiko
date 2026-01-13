from __future__ import annotations
from typing import List, Tuple, Dict, Any
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import multiprocessing as mp

os.makedirs("checkpoints", exist_ok=True)

from risk_env import RiskEnv
from risk_az_net import RiskAZNet
from mcts_risk import MCTS
from action_mapping import N_ACTIONS, action_to_index
from risk_hahn_heuristic import compute_hahn_score

StateVec = np.ndarray
PolicyVec = np.ndarray
ValueScalar = float

MAX_STEPS_PER_EPISODE = 1000


# ---------- VS-RANDOM mit Winner + Draw-Heuristik ----------

def run_vs_random_episode(
    net: RiskAZNet,
    n_simulations: int = 3,
    temperature: float = 1.0,
    device: str = "cpu",
) -> Tuple[List[StateVec], List[PolicyVec], ValueScalar, str | None]:
    env = RiskEnv()
    obs, info = env.reset()

    states: List[StateVec] = []
    policies: List[PolicyVec] = []

    done = False
    step_idx = 0
    aborted = False
    winner: str | None = None

    while not done:
        step_idx += 1
        if step_idx >= MAX_STEPS_PER_EPISODE:
            print(
                f"[VS-RANDOM] reached MAX_STEPS_PER_EPISODE={MAX_STEPS_PER_EPISODE}, aborting.",
                flush=True,
            )
            aborted = True
            break

        s = env.state
        assert s is not None
        current_player = env.players[s.current_player_idx]

        if current_player == env.lila_name:
            # NN + MCTS
            mcts = MCTS(env, net, c_puct=1.5, device=device)
            pi = mcts.run_search(n_simulations=n_simulations)

            legal_actions = env.get_legal_actions()
            legal_indices: List[Tuple[int, Any]] = []
            for a in legal_actions:
                try:
                    idx = action_to_index(a)
                    legal_indices.append((idx, a))
                except ValueError:
                    continue

            masked_pi = np.zeros_like(pi, dtype=np.float32)

            if not legal_indices:
                action = ("end_phase", {})
                try:
                    end_idx = action_to_index(action)
                    masked_pi[end_idx] = 1.0
                except ValueError:
                    pass
            else:
                for idx, _ in legal_indices:
                    masked_pi[idx] = pi[idx]
                s_sum = masked_pi.sum()
                if s_sum > 0:
                    masked_pi /= s_sum
                else:
                    for idx, _ in legal_indices:
                        masked_pi[idx] = 1.0 / len(legal_indices)

                logits = np.array(
                    [masked_pi[idx] for idx, _ in legal_indices],
                    dtype=np.float32,
                )
                if temperature > 0:
                    probs = logits ** (1.0 / temperature)
                    s_sum = probs.sum()
                    if s_sum <= 0:
                        probs = np.ones_like(probs) / len(probs)
                    else:
                        probs /= s_sum
                    choice = np.random.choice(len(legal_indices), p=probs)
                else:
                    choice = int(np.argmax(logits))

                chosen_idx, action = legal_indices[choice]

            states.append(obs["flat_features"].astype(np.float32))
            policies.append(masked_pi.astype(np.float32))

        else:
            # Rot = Random
            legal_actions = env.get_legal_actions()
            action = random.choice(legal_actions)

        obs, reward, done, info = env.step(action)

        if done:
            winner = info.get("winner", None)
            break

    # ---------- Value-Target + Draw-Heuristik ----------

    s = env.state
    assert s is not None

    if aborted:
        winner = None

    owners = s.owners
    troops = s.troops

    lila_idx = env.players.index(env.lila_name)

    lila_territories = int(np.sum(owners == lila_idx))
    rot_territories = int(np.sum((owners != lila_idx) & (owners >= 0)))

    lila_troops = float(troops[owners == lila_idx].sum())
    rot_troops = float(troops[(owners != lila_idx) & (owners >= 0)].sum())

    score_lila = 2 * lila_territories + lila_troops
    score_rot = 2 * rot_territories + rot_troops

    delta = score_lila - score_rot
    K = 1000.0
    z_cap = float(max(-1.0, min(1.0, delta / K)))

    draw_eps = 0.01
    if winner is None and abs(z_cap) < draw_eps:
        z = 0.0
        winner = None
        return states, policies, z, winner

    if winner == env.lila_name:
        z = 1.0
    elif winner is not None:
        z = -1.0
    else:
        z = z_cap

    return states, policies, z, winner


# ---------- Self-Play mit Step-Limit ----------

def run_self_play_episode(
    net: RiskAZNet,
    n_simulations: int = 100,
    temperature: float = 1.0,
    device: str = "cpu",
) -> Tuple[List[StateVec], List[PolicyVec], ValueScalar]:
    env = RiskEnv()
    obs, info = env.reset()
    mcts = MCTS(env, net, c_puct=1.5, device=device)

    states: List[StateVec] = []
    policies: List[PolicyVec] = []

    done = False
    final_reward = 0.0
    step_idx = 0
    aborted = False

    while not done:
        step_idx += 1
        if step_idx >= MAX_STEPS_PER_EPISODE:
            print(
                f"[Self-Play] reached MAX_STEPS_PER_EPISODE={MAX_STEPS_PER_EPISODE}, aborting.",
                flush=True,
            )
            aborted = True
            break

        pi = mcts.run_search(n_simulations=n_simulations)

        legal_actions = env.get_legal_actions()
        legal_indices: List[Tuple[int, Any]] = []
        for a in legal_actions:
            try:
                idx = action_to_index(a)
                legal_indices.append((idx, a))
            except ValueError:
                continue

        masked_pi = np.zeros_like(pi, dtype=np.float32)
        if not legal_indices:
            action = ("end_phase", {})
            try:
                end_idx = action_to_index(action)
                masked_pi[end_idx] = 1.0
            except ValueError:
                pass
        else:
            for idx, _ in legal_indices:
                masked_pi[idx] = pi[idx]
            s_sum = masked_pi.sum()
            if s_sum > 0:
                masked_pi /= s_sum
            else:
                for idx, _ in legal_indices:
                    masked_pi[idx] = 1.0 / len(legal_indices)

            logits = np.array(
                [masked_pi[idx] for idx, _ in legal_indices],
                dtype=np.float32,
            )
            if temperature > 0:
                probs = logits ** (1.0 / temperature)
                s_sum = probs.sum()
                if s_sum <= 0:
                    probs = np.ones_like(probs) / len(probs)
                else:
                    probs /= s_sum
                choice = np.random.choice(len(legal_indices), p=probs)
            else:
                choice = int(np.argmax(logits))
            chosen_idx, action = legal_indices[choice]

        states.append(obs["flat_features"].astype(np.float32))
        policies.append(masked_pi.astype(np.float32))

        obs, reward, done, info = env.step(action)

        if done:
            final_reward = reward
            break

        mcts = MCTS(env, net, c_puct=1.5, device=device)

    if aborted:
        z = 0.0
    else:
        if final_reward > 0:
            z = 1.0
        elif final_reward < 0:
            z = -1.0
        else:
            z = 0.0

    return states, policies, z


# ---------- Hahn-Off-Policy-Pretraining ----------

def pretrain_with_hahn(
    net: RiskAZNet,
    device: str = "cpu",
    n_states: int = 20000,
    batch_size: int = 512,
    lr: float = 5e-4,
):
    env = RiskEnv()
    obs, _ = env.reset()

    input_dim = obs["flat_features"].shape[0]
    if hasattr(net, "input_dim"):
        assert net.input_dim == input_dim

    optimizer = optim.Adam(net.parameters(), lr=lr)

    all_states = []
    all_values = []

    # Dataset sammeln
    while len(all_states) < n_states:
        obs, _ = env.reset()
        done = False
        while not done and len(all_states) < n_states:
            s = env.state
            assert s is not None

            lila_idx = env.players.index(env.lila_name)
            h_score = compute_hahn_score(s, lila_idx)

            all_states.append(obs["flat_features"].astype(np.float32))
            all_values.append(h_score)

            action = random.choice(env.get_legal_actions())
            obs, reward, done, info = env.step(action)

    states_arr = np.stack(all_states, axis=0)
    values_arr = np.array(all_values, dtype=np.float32)

    # Normierung
    std = max(values_arr.std(), 1e-6)
    values_arr = values_arr / (3.0 * std)

    states_t = torch.from_numpy(states_arr).to(device)
    values_t = torch.from_numpy(values_arr).unsqueeze(1).to(device)

    net.train()
    optimizer.zero_grad()

    # Annahme: net(x) -> (policy_logits, value)
    _, value_pred = net(states_t)
    loss = F.mse_loss(value_pred, values_t)
    loss.backward()
    optimizer.step()

    print(f"[Hahn-Pretrain] n_states={len(all_states)}, loss={loss.item():.4f}")


# ---------- Loss & Training-Loop ----------

def alpha_zero_loss(
    net: RiskAZNet,
    batch_states: torch.Tensor,
    batch_policy_targets: torch.Tensor,
    batch_value_targets: torch.Tensor,
) -> torch.Tensor:
    policy_logits, values = net(batch_states)
    value_loss = F.mse_loss(values, batch_value_targets)
    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -torch.mean(torch.sum(batch_policy_targets * log_probs, dim=-1))
    return value_loss + policy_loss


def self_play_worker(
    worker_id: int,
    net_state_dict,
    input_dim: int,
    n_simulations: int,
    device: str,
    episodes_per_worker: int,
    out_queue: mp.Queue,
) -> None:
    net = RiskAZNet(input_dim=input_dim, n_actions=N_ACTIONS).to(device)
    net.load_state_dict(net_state_dict)
    net.eval()

    for ep in range(episodes_per_worker):
        states, policies, z = run_self_play_episode(
            net=net,
            n_simulations=n_simulations,
            temperature=1.0,
            device=device,
        )
        print(
            f"[Worker {worker_id}] finished episode {ep+1}/{episodes_per_worker} "
            f"with z={z}",
            flush=True,
        )
        out_queue.put((states, policies, z))


def generate_self_play_parallel(
    net: RiskAZNet,
    input_dim: int,
    n_simulations: int,
    device: str,
    total_episodes: int,
    num_workers: int,
):
    mp.set_start_method("spawn", force=True)

    net_state_dict = net.state_dict()
    out_queue: mp.Queue = mp.Queue()

    episodes_per_worker = total_episodes // num_workers
    extra = total_episodes % num_workers

    procs = []
    for wid in range(num_workers):
        n_eps = episodes_per_worker + (1 if wid < extra else 0)
        if n_eps == 0:
            continue
        p = mp.Process(
            target=self_play_worker,
            args=(
                wid,
                net_state_dict,
                input_dim,
                n_simulations,
                device,
                n_eps,
                out_queue,
            ),
        )
        p.start()
        procs.append(p)

    all_states: List[StateVec] = []
    all_policies: List[PolicyVec] = []
    all_values: List[ValueScalar] = []

    for _ in range(total_episodes):
        states, policies, z = out_queue.get()
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend([z] * len(states))

    for p in procs:
        p.join()

    return all_states, all_policies, all_values


def simple_training_loop(
    num_episodes: int = 40,
    n_simulations: int = 100,
    device: str = "cpu",
    lr: float = 5e-4,
    net: RiskAZNet | None = None,
) -> RiskAZNet:
    env = RiskEnv()
    obs, _ = env.reset()
    input_dim = obs["flat_features"].shape[0]

    if net is None:
        net = RiskAZNet(input_dim=input_dim, n_actions=N_ACTIONS).to(device)
        try:
            state_dict = torch.load("risk_az_latest.pth", map_location=device)
            net.load_state_dict(state_dict)
            print("[Train] Loaded existing model from risk_az_latest.pth")
        except FileNotFoundError:
            print("[Train] No existing model found, starting from scratch.")
    else:
        print("[Train] Using provided (pretrained) network.")

    optimizer = optim.Adam(net.parameters(), lr=lr)

    num_workers = 8

    all_states, all_policies, all_values = generate_self_play_parallel(
        net=net,
        input_dim=input_dim,
        n_simulations=n_simulations,
        device=device,
        total_episodes=num_episodes,
        num_workers=num_workers,
    )

    if not all_states:
        print("[Train] Warning: no states collected, aborting training.")
        return net

    states_arr = np.stack(all_states, axis=0)
    policies_arr = np.stack(all_policies, axis=0)
    values_arr = np.array(all_values, dtype=np.float32)

    states_t = torch.from_numpy(states_arr).to(device)
    policies_t = torch.from_numpy(policies_arr).to(device)
    values_t = torch.from_numpy(values_arr).unsqueeze(1).to(device)

    net.train()
    optimizer.zero_grad()
    loss = alpha_zero_loss(net, states_t, policies_t, values_t)
    loss.backward()
    optimizer.step()

    print(f"[Train] Loss: {loss.item():.4f}")

    MODEL_PATH = "risk_az_latest.pth"
    torch.save(net.state_dict(), MODEL_PATH)
    print(f"[Save] Model saved to {MODEL_PATH}")

    return net


if __name__ == "__main__":
    device = "cpu"

    # Netz initialisieren
    env = RiskEnv()
    obs, _ = env.reset()
    input_dim = obs["flat_features"].shape[0]
    net = RiskAZNet(input_dim=input_dim, n_actions=N_ACTIONS).to(device)

    # Hahn-Pretraining
    pretrain_with_hahn(net, device=device, n_states=100000)

    # Danach Self-Play-Training mit dem vortrainierten Netz
    trained_net = simple_training_loop(
        num_episodes=40,
        n_simulations=100,
        device=device,
        lr=5e-4,
        net=net,
    )
