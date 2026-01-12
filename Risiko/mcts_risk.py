# file: mcts_risk.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any

import math
import numpy as np
import torch

from risk_env import RiskEnv
from risk_az_net import RiskAZNet
from action_mapping import action_to_index, index_to_action, N_ACTIONS

Action = Tuple[str, Dict[str, Any]]  # wie in risk_state / risk_env


@dataclass
class MCTSNode:
    parent: Optional["MCTSNode"]
    action_from_parent: Optional[Action]

    # Statistiken
    N: Dict[int, int] = field(default_factory=dict)      # action_idx -> Visit count
    W: Dict[int, float] = field(default_factory=dict)    # action_idx -> Total value
    Q: Dict[int, float] = field(default_factory=dict)    # action_idx -> Mean value
    P: Dict[int, float] = field(default_factory=dict)    # action_idx -> Prior prob

    is_expanded: bool = False

    # Legal Actions an diesem Knoten
    legal_actions: List[Action] = field(default_factory=list)
    # Mapping von local_idx (Index in legal_actions) auf globalen Action-Index
    action_indices: Dict[int, int] = field(default_factory=dict)

    # Kinder
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)  # local_idx -> child


class MCTS:
    """
    AlphaZero-artiger MCTS für RiskEnv + RiskAZNet.

    Annahmen:
      - Value ist aus Sicht von Lila (env.lila_name) in [-1, 1].
      - Reward des Envs am Ende der Partie spiegelt Sieg/Niederlage von Lila wider.
    """

    def __init__(
        self,
        env: RiskEnv,
        net: RiskAZNet,
        c_puct: float = 1.5,
        device: str = "cpu",
    ):
        self.env = env
        self.net = net.to(device)
        self.net.eval()
        self.c_puct = c_puct
        self.device = device

        self.root: Optional[MCTSNode] = None

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def run_search(self, n_simulations: int) -> np.ndarray:
        """
        Führt MCTS vom aktuellen Zustand der Env aus und liefert
        eine Policy π(a|s) als Verteilung über den globalen Action-Space (N_ACTIONS).
        """
        # Root-Env klonen
        root_env = self._clone_env(self.env)
        self.root = self._make_root(root_env)

        for _ in range(n_simulations):
            env_sim = self._clone_env(root_env)
            node = self.root
            path: List[Tuple[MCTSNode, int]] = []  # (node, a_idx)

            # 1) Selection
            while node.is_expanded and node.legal_actions:
                a_idx, local_idx = self._select_action(node)
                action = node.legal_actions[local_idx]
                path.append((node, a_idx))

                obs, value, done = self._step_env(env_sim, action)
                if done:
                    v = value  # Value aus Sicht von Lila am Terminal
                    break

                # Child holen oder erstellen
                if local_idx not in node.children:
                    node.children[local_idx] = MCTSNode(
                        parent=node,
                        action_from_parent=action,
                    )
                node = node.children[local_idx]
            else:
                # 2) Expansion + Evaluation (falls nicht bereits done)
                obs, value, done = self._get_obs(env_sim)
                if done:
                    v = value
                else:
                    v = self._expand_and_evaluate(node, obs, env_sim)

            # 3) Backpropagation
            self._backprop(path, v)

        # 4) Policy π aus Root-Visit-Counts
        pi = np.zeros(N_ACTIONS, dtype=np.float32)
        if self.root is None:
            return pi

        for a_idx, N_sa in self.root.N.items():
            pi[a_idx] = N_sa

        s = pi.sum()
        if s > 0:
            pi /= s

        return pi

    # --------------------------------------------------
    # Node-Initialisierung / Expansion
    # --------------------------------------------------
    def _make_root(self, env: RiskEnv) -> MCTSNode:
        node = MCTSNode(parent=None, action_from_parent=None)
        obs, value, done = self._get_obs(env)
        if not done:
            self._expand_and_evaluate(node, obs, env)
        else:
            node.is_expanded = True
        return node

    def _expand_and_evaluate(
        self,
        node: MCTSNode,
        obs: Dict[str, np.ndarray],
        env: RiskEnv,
    ) -> float:
        """
        - Ruft Netz auf, um Policy-Priors und Value zu bekommen.
        - Legt Legal-Actions + P(a|s) im Node ab.
        - Gibt Value aus Sicht von Lila zurück.
        """
        x = torch.from_numpy(obs["flat_features"]).unsqueeze(0).to(self.device)  # (1, D)
        with torch.no_grad():
            policy_logits, value = self.net(x)
        policy_logits = policy_logits[0].cpu().numpy()
        v = float(value[0, 0].cpu().item())

        # Softmax für Priors
        max_logit = np.max(policy_logits)
        exp_logits = np.exp(policy_logits - max_logit)
        priors = exp_logits / (exp_logits.sum() + 1e-8)

        # Legal Actions holen
        legal_actions = env.get_legal_actions()
        node.legal_actions = legal_actions
        node.action_indices = {}

        for local_idx, action in enumerate(legal_actions):
            try:
                a_idx = action_to_index(action)
            except ValueError:
                # Sollte mit deinem Mapping selten vorkommen
                continue

            node.action_indices[local_idx] = a_idx
            node.N[a_idx] = 0
            node.W[a_idx] = 0.0
            node.Q[a_idx] = 0.0
            node.P[a_idx] = priors[a_idx] if a_idx < len(priors) else 0.0

        node.is_expanded = True
        return v

    # --------------------------------------------------
    # Selection / Backprop
    # --------------------------------------------------
    def _select_action(self, node: MCTSNode) -> Tuple[int, int]:
        """
        Wählt Action nach PUCT-Regel:
        a* = argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a)) ]
        Returns:
            (a_idx_global, local_idx_in_node.legal_actions)
        """
        N_sum = sum(node.N.values()) + 1e-8
        best_score = -1e9
        best_a_idx = -1
        best_local_idx = -1

        for local_idx, action in enumerate(node.legal_actions):
            if local_idx not in node.action_indices:
                continue
            a_idx = node.action_indices[local_idx]

            Q_sa = node.Q.get(a_idx, 0.0)
            P_sa = node.P.get(a_idx, 0.0)
            N_sa = node.N.get(a_idx, 0)

            U_sa = self.c_puct * P_sa * math.sqrt(N_sum) / (1 + N_sa)
            score = Q_sa + U_sa

            if score > best_score:
                best_score = score
                best_a_idx = a_idx
                best_local_idx = local_idx

        if best_a_idx < 0 or best_local_idx < 0:
            # Fallback: irgendeine Action wählen
            best_local_idx = 0
            a_idx = node.action_indices.get(best_local_idx, 0)
            best_a_idx = a_idx

        return best_a_idx, best_local_idx

    def _backprop(self, path: List[Tuple[MCTSNode, int]], v: float) -> None:
        """
        Backpropagiert den Value v entlang des Pfades.
        Annahme: v ist bereits aus Sicht von Lila,
        und der Reward/Value ist symmetrisch (Lila gewinnt => +1, sonst -1).
        """
        for node, a_idx in reversed(path):
            N_sa = node.N.get(a_idx, 0)
            W_sa = node.W.get(a_idx, 0.0)

            N_sa += 1
            W_sa += v
            node.N[a_idx] = N_sa
            node.W[a_idx] = W_sa
            node.Q[a_idx] = W_sa / N_sa

    # --------------------------------------------------
    # Env-Helfer
    # --------------------------------------------------
    def _clone_env(self, env: RiskEnv) -> RiskEnv:
        """
        Klont RiskEnv inkl. State-Arrays (owners, troops, etc.).
        """
        # Nur gamma und players übernehmen; max_steps / noprogress_limit
        # kommen aus der RiskEnv-Default-Initialisierung.
        new_env = RiskEnv(
            gamma=env.gamma,
            players=list(env.players),
        )
        s = env.state
        assert s is not None
        new_state = type(s)(
            owners=s.owners.copy(),
            troops=s.troops.copy(),
            current_player_idx=s.current_player_idx,
            phase=s.phase,
            reinforcements_left=s.reinforcements_left,
            step_count=s.step_count,
            max_steps=s.max_steps,
            noprogress_steps=s.noprogress_steps,
            noprogress_limit=s.noprogress_limit,
        )
        new_env.state = new_state
        new_env.lila_last_terr_count = env.lila_last_terr_count
        return new_env

    def _get_obs(self, env: RiskEnv) -> Tuple[Dict[str, np.ndarray], float, bool]:
        """
        Gibt (obs, value, done) zurück.
        Aktuell:
          - value = 0, solange nicht done.
          - done  = False, bis Env terminal wird.
        """
        obs = env.encode_state()
        s = env.state
        assert s is not None

        done = False
        value = 0.0
        return obs, value, done

    def _step_env(self, env: RiskEnv, action: Action) -> Tuple[Dict[str, np.ndarray], float, bool]:
        """
        Führt Action in der Sim-Env aus und gibt (obs, value, done) zurück.
        value ist das Terminal-Reward aus Sicht von Lila, sonst 0.
        """
        obs, reward, done, info = env.step(action)
        value = reward if done else 0.0
        return obs, value, done


# Mini-Demo (nur Smoke-Test)
if __name__ == "__main__":
    env = RiskEnv()
    obs, info = env.reset()

    net = RiskAZNet(input_dim=obs["flat_features"].shape[0], n_actions=N_ACTIONS)
    mcts = MCTS(env, net, c_puct=1.5, device="cpu")

    pi = mcts.run_search(n_simulations=5)
    print("pi shape:", pi.shape)
    print("sum(pi):", float(pi.sum()))
    print("non-zero entries:", int((pi > 0).sum()))
