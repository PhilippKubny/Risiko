from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import random

from risk_state import (
    PLAYERS,
    NUM_TERR,
    PHASE_REINFORCE,
    PHASE_ATTACK,
    PHASE_FORTIFY,
    RiskState,
    build_random_initial_state,
    get_legal_actions as rs_get_legal_actions,
    apply_place,
    apply_attack,
    apply_fortify,
    calculate_reinforcements,
    check_winner,
    CONTINENTS,
    CONTINENT_BONUS,
)

Action = Tuple[str, Dict[str, Any]]

class RiskEnv:
    """
    Schlanke Env für Risk, basierend auf RiskState (nur Arrays).

    Bietet:
    - reset()
    - step(action)
    - get_legal_actions()
    - encode_state() / flatten_state() für NN
    """

    def __init__(
        self,
        gamma: float = 0.99,
        players: Optional[List[str]] = None,
        max_steps: int = 20_000,
        no_progress_limit: int = 50_000,
    ):
        self.players: List[str] = players if players is not None else list(PLAYERS)
        self.gamma = gamma
        self.max_steps = max_steps
        self.no_progress_limit = no_progress_limit

        self.state: Optional[RiskState] = None
        self.lila_name = "Lila"

        # Tracking für No-Progress (optional)
        self.lila_last_terr_count: int = 0

        # Fortify nur einmal pro Runde erlauben
        self._fortify_used_this_turn: bool = False

    # ----------------------------------------------------------
    # Reset / Step / Legal Actions
    # ----------------------------------------------------------

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Startet ein neues Spiel mit random Setup nach Risk-Regeln.
        """
        self.state = build_random_initial_state(
            players=self.players,
            max_steps=self.max_steps,
        )
        s = self.state
        assert s is not None

        # Erste Verstärkungen für Startspieler berechnen
        s.reinforcements_left = calculate_reinforcements(
            s, self.players, s.current_player_idx
        )

        # RiskState-Feld heißt noprogress_limit
        s.noprogress_limit = self.no_progress_limit  # type: ignore[assignment]

        # Neue Runde -> Fortify-Flag zurücksetzen
        self._fortify_used_this_turn = False

        obs = self.encode_state()
        info = {"winner": None, "phase": self.phase_str}
        self._update_lila_progress(obs)
        return obs, info

    def step(self, action: Action) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Führt eine Action aus und gibt (obs, reward, done, info) zurück.
        """
        assert self.state is not None, "Environment not reset. Call reset() first."
        kind, params = action
        s = self.state

        s.step_count += 1
        done = False
        reward = 0.0
        winner_idx: Optional[int] = None
        info: Dict[str, Any] = {}

        # ----------------- Aktion ausführen -----------------
        if kind == "place":
            apply_place(s, params["territory_id"])

        elif kind == "attack":
            apply_attack(
                s,
                from_id=params["from_id"],
                to_id=params["to_id"],
                move_count=params.get("move_count", 1),
            )

        elif kind == "fortify":
            # Nur ein Fortify pro Runde erlaubt
            if self._fortify_used_this_turn:
                # Weitere Fortify-Versuche beenden einfach die Runde
                self._advance_phase_or_player()
            else:
                apply_fortify(
                    s,
                    from_id=params["from_id"],
                    to_id=params["to_id"],
                    count=params["count"],
                )
                self._fortify_used_this_turn = True
                # Nach dem (einzigen) Fortify ist die Runde fertig
                self._advance_phase_or_player()

        elif kind == "end_phase":
            self._advance_phase_or_player()

        else:
            raise ValueError(f"Unknown action kind: {kind}")

        # ----------------- Gewinner-Check -------------------
        winner_idx = check_winner(s, self.players)
        if winner_idx is not None:
            done = True
            info["winner"] = self.players[winner_idx]
        else:
            unique_owners = np.unique(s.owners)
            if len(unique_owners) == 1 and unique_owners[0] >= 0:
                winner_idx = int(unique_owners[0])
                done = True
                info["winner"] = self.players[winner_idx]

        # Optional: Max-Steps
        if not done and s.step_count >= s.max_steps:
            done = True
            info["time_limit"] = True

        obs = self.encode_state()

        # No-Progress nur auf Lila
        if not done and self.lila_name in self.players:
            lila_idx = self.players.index(self.lila_name)
            lila_terr_count = int(np.sum(s.owners == lila_idx))
            if lila_terr_count != self.lila_last_terr_count:
                self.lila_last_terr_count = lila_terr_count
                s.noprogress_steps = 0
            else:
                s.noprogress_steps += 1

            if s.noprogress_steps >= s.noprogress_limit:
                done = True
                info["no_progress_termination"] = True

        # Reward nur am Ende
        if done:
            if winner_idx is not None and self.players[winner_idx] == self.lila_name:
                reward = 1.0
            elif winner_idx is not None:
                reward = -1.0

        return obs, reward, done, info

    def get_legal_actions(self) -> List[Action]:
        assert self.state is not None
        return rs_get_legal_actions(self.state, self.players)

    # ----------------------------------------------------------
    # Phase / Spielerwechsel
    # ----------------------------------------------------------

    @property
    def phase_str(self) -> str:
        assert self.state is not None
        if self.state.phase == PHASE_REINFORCE:
            return "reinforce"
        if self.state.phase == PHASE_ATTACK:
            return "attack"
        if self.state.phase == PHASE_FORTIFY:
            return "fortify"
        return "unknown"

    def _advance_phase_or_player(self) -> None:
        """
        Phase-Logik: reinforce -> attack -> fortify -> nächster Spieler (reinforce).
        Reinforcements werden beim Eintritt in reinforce berechnet.
        """
        s = self.state
        assert s is not None

        if s.phase == PHASE_REINFORCE:
            s.phase = PHASE_ATTACK

        elif s.phase == PHASE_ATTACK:
            s.phase = PHASE_FORTIFY

        elif s.phase == PHASE_FORTIFY:
            # Runde des Spielers endet, nächster Spieler beginnt
            s.current_player_idx = (s.current_player_idx + 1) % len(self.players)
            s.phase = PHASE_REINFORCE
            s.reinforcements_left = calculate_reinforcements(
                s, self.players, s.current_player_idx
            )
            # Neue Runde -> Fortify-Flag zurücksetzen
            self._fortify_used_this_turn = False

    # ----------------------------------------------------------
    # State-Encoding für NN / MCTS
    # ----------------------------------------------------------

    def encode_state(self) -> Dict[str, np.ndarray]:
        """
        Roh-orientiertes Encoding mit vielen Infos, aber ohne Heuristiken:
        - per_territory: (NUM_TERR, F_terr)
          [owner_one_hot..., troops, troops_log]
        - extra: (F_extra,)
          [current_player_one_hot..., phase_one_hot..., reinf_left,
           step_count, continents_owned_per_player..., total_troops_per_player..., total_terr_per_player...]
        - flat_features: alles geflattet für Feedforward-Netze
        """
        s = self.state
        assert s is not None
        num_players = len(self.players)

        owners = s.owners.astype(np.int32, copy=True)   # (NUM_TERR,)
        troops = s.troops.astype(np.int32, copy=True)   # (NUM_TERR,)

        # --------- Per-Territory Features ---------
        owner_one_hot = np.zeros((NUM_TERR, num_players), dtype=np.float32)
        for tid in range(NUM_TERR):
            pid = owners[tid]
            if 0 <= pid < num_players:
                owner_one_hot[tid, pid] = 1.0

        troops_f = troops.astype(np.float32)
        troops_log = np.log1p(troops_f)

        # Shape: (NUM_TERR, num_players + 2)
        per_territory = np.concatenate(
            [
                owner_one_hot,
                troops_f.reshape(NUM_TERR, 1),
                troops_log.reshape(NUM_TERR, 1),
            ],
            axis=1,
        ).astype(np.float32)

        # --------- Globale Features ---------
        current_player_oh = np.zeros(num_players, dtype=np.float32)
        current_player_oh[s.current_player_idx] = 1.0

        # Phase-One-Hot über Konstanten
        phase_oh = np.zeros(3, dtype=np.float32)
        if s.phase == PHASE_REINFORCE:
            phase_oh[0] = 1.0
        elif s.phase == PHASE_ATTACK:
            phase_oh[1] = 1.0
        elif s.phase == PHASE_FORTIFY:
            phase_oh[2] = 1.0

        reinf_left = float(s.reinforcements_left)
        step_count = float(s.step_count)

        continents_owned_per_player = np.zeros(num_players, dtype=np.float32)
        for cont_name, terr_ids in CONTINENTS.items():
            owner0 = s.owners[terr_ids[0]]
            if owner0 < 0:
                continue
            if all(s.owners[tid] == owner0 for tid in terr_ids):
                continents_owned_per_player[owner0] += 1.0

        total_troops_per_player = np.zeros(num_players, dtype=np.float32)
        total_terr_per_player = np.zeros(num_players, dtype=np.float32)
        for pid in range(num_players):
            mask = (owners == pid)
            total_terr_per_player[pid] = float(mask.sum())
            total_troops_per_player[pid] = float(troops[mask].sum())

        extra = np.concatenate(
            [
                current_player_oh,
                phase_oh,
                np.array([reinf_left, step_count], dtype=np.float32),
                continents_owned_per_player,
                total_troops_per_player,
                total_terr_per_player,
            ],
            axis=0,
        ).astype(np.float32)

        flat_features = np.concatenate(
            [
                per_territory.flatten(),
                extra,
            ],
            axis=0,
        ).astype(np.float32)

        return {
            "owners": owners,
            "troops": troops,
            "per_territory": per_territory,  # (NUM_TERR, F_terr)
            "extra": extra,                  # (F_extra,)
            "flat_features": flat_features,  # (D,)
        }

    def flatten_state(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return obs["flat_features"]

    # ----------------------------------------------------------
    # Hilfsfunktionen
    # ----------------------------------------------------------

    def _update_lila_progress(self, obs: Dict[str, np.ndarray]) -> None:
        if self.lila_name in self.players:
            lila_idx = self.players.index(self.lila_name)
            owners = obs["owners"]
            self.lila_last_terr_count = int(np.sum(owners == lila_idx))


# Mini-Demo zum Testen
if __name__ == "__main__":
    env = RiskEnv()
    obs, info = env.reset()
    done = False
    while not done:
        legal = env.get_legal_actions()
        a = random.choice(legal)
        obs, r, done, info = env.step(a)
    print("Spiel beendet:", info)
