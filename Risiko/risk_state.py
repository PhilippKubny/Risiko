from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import random
import numpy as np

# ---------------------------------------------------------
# Phasen-Konstanten (für risk_env.py)
# ---------------------------------------------------------

PHASE_REINFORCE = "reinforce"
PHASE_ATTACK    = "attack"
PHASE_FORTIFY   = "fortify"

PHASE_NAMES = [PHASE_REINFORCE, PHASE_ATTACK, PHASE_FORTIFY]

# ---------------------------------------------------------
# Konstante Spiel-Definition (klassische Risk-Weltkarte)
# ---------------------------------------------------------

PLAYERS: List[str] = ["Lila", "Rot"]

TERRITORIES: List[str] = [
    # Nordamerika
    "Alaska",              # 0
    "Nordwest-Territorium",# 1
    "Grönland",            # 2
    "Alberta",             # 3
    "Ontario",             # 4
    "Quebec",              # 5
    "Weststaaten",         # 6
    "Oststaaten",          # 7
    "Mittelamerika",       # 8
    # Südamerika
    "Venezuela",           # 9
    "Peru",                # 10
    "Brasilien",           # 11
    "Argentinien",         # 12
    # Europa
    "Island",              # 13
    "Großbritannien",      # 14
    "Skandinavien",        # 15
    "Westeuropa",          # 16
    "Mitteleuropa",        # 17
    "Südeuropa",           # 18
    "Ukraine",             # 19
    # Afrika
    "Nordafrika",          # 20
    "Ägypten",             # 21
    "Ostafrika",           # 22
    "Zentralafrika",       # 23
    "Südafrika",           # 24
    "Madagaskar",          # 25
    # Asien
    "Ural",                # 26
    "Sibirien",            # 27
    "Jakutien",            # 28
    "Irkutsk",             # 29
    "Kamtschatka",         # 30
    "Mongolei",            # 31
    "Japan",               # 32
    "China",               # 33
    "Mittlerer Osten",     # 34
    "Indien",              # 35
    "Siam",                # 36
    "Afghanistan",         # 37
    # Australien
    "Indonesien",          # 38
    "Neu-Guinea",          # 39
    "West-Australien",     # 40
    "Ost-Australien",      # 41
]

NUM_TERR: int = len(TERRITORIES)

CONTINENTS: Dict[str, List[int]] = {
    "Nordamerika": list(range(0, 9)),
    "Südamerika": list(range(9, 13)),
    "Europa": list(range(13, 20)),
    "Afrika": list(range(20, 26)),
    "Asien": list(range(26, 38)),
    "Australien": list(range(38, 42)),
}

CONTINENT_BONUS: Dict[str, int] = {
    "Nordamerika": 5,
    "Südamerika": 2,
    "Europa": 5,
    "Afrika": 3,
    "Asien": 7,
    "Australien": 2,
}

# klassische Risk-Weltkarte – Adjazenzliste
ADJACENCY: Dict[int, List[int]] = {
    # Nordamerika
    0: [1, 3, 30],            # Alaska: Nordwest-Territorium, Alberta, Kamtschatka
    1: [0, 2, 3, 4],          # Nordwest-Territorium: Alaska, Grönland, Alberta, Ontario
    2: [1, 4, 5, 13],         # Grönland: Nordwest-Territorium, Ontario, Quebec, Island
    3: [0, 1, 4, 6],          # Alberta: Alaska, Nordwest-Territorium, Ontario, Weststaaten
    4: [1, 2, 3, 5, 6, 7],    # Ontario: NW-Territorium, Grönland, Alberta, Quebec, West-, Oststaaten
    5: [2, 4, 7],             # Quebec: Grönland, Ontario, Oststaaten
    6: [3, 4, 7, 8],          # Weststaaten: Alberta, Ontario, Oststaaten, Mittelamerika
    7: [4, 5, 6, 8],          # Oststaaten: Ontario, Quebec, Weststaaten, Mittelamerika
    8: [6, 7, 9],             # Mittelamerika: Weststaaten, Oststaaten, Venezuela

    # Südamerika
    9:  [8, 10, 11],          # Venezuela: Mittelamerika, Peru, Brasilien
    10: [9, 11, 12],          # Peru: Venezuela, Brasilien, Argentinien
    11: [9, 10, 12, 20],      # Brasilien: Venezuela, Peru, Argentinien, Nordafrika
    12: [10, 11],             # Argentinien: Peru, Brasilien

    # Europa
    13: [2, 14, 15],          # Island: Grönland, Großbritannien, Skandinavien
    14: [13, 15, 17, 16],     # Großbritannien: Island, Skandinavien, Mitteleuropa, Westeuropa
    15: [13, 14, 19, 17],     # Skandinavien: Island, GB, Ukraine, Mitteleuropa
    16: [14, 17, 20, 18],     # Westeuropa: GB, Mitteleuropa, Nordafrika, Südeuropa
    17: [14, 15, 16, 18, 19], # Mitteleuropa (Northern Europe): GB, Skandinavien, West-, Süd-Europa, Ukraine
    18: [16, 17, 19, 20, 21, 34],  # Südeuropa: West-, Mittel-, Ukraine, Nordafrika, Ägypten, Mittlerer Osten
    19: [15, 17, 18, 26, 37, 34],  # Ukraine: Skandinavien, Mitteleuropa, Südeuropa, Ural, Afghanistan, Mittlerer Osten

    # Afrika
    20: [16, 18, 21, 22, 23, 11],  # Nordafrika: West-, Süd-Europa, Ägypten, Ost-, Zentralafrika, Brasilien
    21: [18, 20, 22, 34],          # Ägypten: Südeuropa, Nordafrika, Ostafrika, Mittlerer Osten
    22: [20, 21, 23, 24, 25, 34],  # Ostafrika: Nordafrika, Ägypten, Zentral-, Südafrika, Madagaskar, Mittlerer Osten
    23: [20, 22, 24],              # Zentralafrika (Congo): Nordafrika, Ostafrika, Südafrika
    24: [22, 23, 25],              # Südafrika: Ost-, Zentralafrika, Madagaskar
    25: [22, 24],                  # Madagaskar: Ostafrika, Südafrika

    # Asien
    26: [19, 27, 33, 37],          # Ural: Ukraine, Sibirien, China, Afghanistan
    27: [26, 28, 29, 31, 33],      # Sibirien: Ural, Jakutien, Irkutsk, Mongolei, China
    28: [27, 29, 30],              # Jakutien: Sibirien, Irkutsk, Kamtschatka
    29: [27, 28, 30, 31],          # Irkutsk: Sibirien, Jakutien, Kamtschatka, Mongolei
    30: [28, 29, 31, 32, 0],       # Kamtschatka: Jakutien, Irkutsk, Mongolei, Japan, Alaska
    31: [27, 29, 30, 32, 33],      # Mongolei: Sibirien, Irkutsk, Kamtschatka, Japan, China
    32: [30, 31],                  # Japan: Kamtschatka, Mongolei
    33: [26, 27, 31, 35, 37, 36],  # China: Ural, Sibirien, Mongolei, Indien, Afghanistan, Siam
    34: [18, 21, 22, 19, 37, 35],  # Mittlerer Osten: Südeuropa, Ägypten, Ostafrika, Ukraine, Afghanistan, Indien
    35: [34, 33, 36, 37],          # Indien: Mittlerer Osten, China, Siam, Afghanistan
    36: [35, 33, 38],              # Siam: Indien, China, Indonesien
    37: [26, 19, 34, 33, 35],      # Afghanistan: Ural, Ukraine, Mittlerer Osten, China, Indien

    # Australien
    38: [36, 39, 40],              # Indonesien: Siam, Neu-Guinea, West-Australien
    39: [38, 40, 41],              # Neu-Guinea: Indonesien, West-, Ost-Australien
    40: [38, 39, 41],              # West-Australien: Indonesien, Neu-Guinea, Ost-Australien
    41: [39, 40],                  # Ost-Australien: Neu-Guinea, West-Australien
}
# ---------------------------------------------------------
# State-Definition
# ---------------------------------------------------------

@dataclass
class RiskState:
    owners: np.ndarray
    troops: np.ndarray
    current_player_idx: int
    phase: str
    reinforcements_left: int

    # ab hier Defaults, damit MCTS RiskState(...) mit weniger Feldern bauen kann
    winner: Optional[int] = None
    move_count: int = 0
    step_count: int = 0
    max_steps: int = 20_000
    noprogress_steps: int = 0
    noprogress_limit: int = 50_000

    @staticmethod
    def new() -> RiskState:
        owners = np.full(NUM_TERR, fill_value=-1, dtype=np.int8)
        troops = np.zeros(NUM_TERR, dtype=np.int16)
        return RiskState(
            owners=owners,
            troops=troops,
            current_player_idx=0,
            phase=PHASE_REINFORCE,
            reinforcements_left=0,
            winner=None,
            move_count=0,
            step_count=0,
            max_steps=20_000,
            noprogress_steps=0,
            noprogress_limit=50_000,
        )

# ---------------------------------------------------------
# Hilfsfunktionen auf State
# ---------------------------------------------------------

def build_random_initial_state(
    players: Optional[List[str]] = None,
    max_steps: int = 20_000,
) -> RiskState:
    if players is None:
        players = list(PLAYERS)
    num_players = len(players)

    state = RiskState.new()
    state.max_steps = max_steps

    owners = np.full(NUM_TERR, fill_value=-1, dtype=np.int8)
    terr_ids = list(range(NUM_TERR))
    random.shuffle(terr_ids)

    # Schritt 1: Länder zufällig verteilen, je 1 Armee
    p = 0
    for tid in terr_ids:
        owners[tid] = p
        state.troops[tid] = 1
        p = (p + 1) % num_players

    state.owners = owners

    # Schritt 2: zusätzliche Startarmeen wie im Regelbuch-Idee (hier Hausregel für 2 Spieler)
    # Beispiel: jeder Spieler hat insgesamt 40 Armeen.
    # 42 Länder sind schon mit je 1 Armee belegt, also hat jeder Spieler bereits
    # etwa 21 Armeen gesetzt. Wir geben jedem Spieler noch 19 zusätzliche.
    extra_per_player = 19  # kannst du anpassen

    owned_lists = [
        [tid for tid in range(NUM_TERR) if owners[tid] == pid]
        for pid in range(num_players)
    ]

    # reihum zusätzliche Armeen auf eigene Länder setzen
    remaining = [extra_per_player] * num_players
    current_pid = 0
    while any(r > 0 for r in remaining):
        if remaining[current_pid] > 0 and owned_lists[current_pid]:
            tid = random.choice(owned_lists[current_pid])
            state.troops[tid] += 1
            remaining[current_pid] -= 1
        current_pid = (current_pid + 1) % num_players

    state.current_player_idx = 0
    state.phase = PHASE_REINFORCE
    state.reinforcements_left = 0
    state.winner = None
    state.move_count = 0
    state.step_count = 0
    state.noprogress_steps = 0
    return state

def calc_reinforcements(state: RiskState, player_idx: int) -> int:
    owned = np.sum(state.owners == player_idx)
    base = max(3, owned // 3)
    bonus = 0
    for cont_name, terr_ids in CONTINENTS.items():
        if all(state.owners[tid] == player_idx for tid in terr_ids):
            bonus += CONTINENT_BONUS.get(cont_name, 0)
    return base + bonus

def calculate_reinforcements(state: RiskState, players: List[str], player_idx: int) -> int:
    return calc_reinforcements(state, player_idx)

def clone_state(state: RiskState) -> RiskState:
    return RiskState(
        owners=state.owners.copy(),
        troops=state.troops.copy(),
        current_player_idx=state.current_player_idx,
        phase=state.phase,
        reinforcements_left=state.reinforcements_left,
        winner=state.winner,
        move_count=state.move_count,
        step_count=state.step_count,
        max_steps=state.max_steps,
        noprogress_steps=state.noprogress_steps,
        noprogress_limit=state.noprogress_limit,
    )

def check_winner(state: RiskState, players: List[str]) -> Optional[int]:
    """
    Gewinner: ein Spieler mit mind. einem Gebiet,
    alle anderen Spieler besitzen kein Gebiet mehr.
    Neutrale (-1) sind egal.
    """
    num_players = len(players)
    for p_idx in range(num_players):
        has_any = np.any(state.owners == p_idx)
        if not has_any:
            continue
        others_have = any(
            np.any(state.owners == q)
            for q in range(num_players)
            if q != p_idx
        )
        if not others_have:
            return p_idx
    return None
# ---------------------------------------------------------
# Primitive Regeln
# ---------------------------------------------------------

def apply_reinforce(state: RiskState, terr_id: int, count: int = 1) -> None:
    if state.phase != PHASE_REINFORCE:
        return
    if state.reinforcements_left <= 0:
        return
    if state.owners[terr_id] != state.current_player_idx:
        return
    c = min(count, state.reinforcements_left)
    state.troops[terr_id] += c
    state.reinforcements_left -= c

def apply_place(state: RiskState, territory_id: int, count: int = 1) -> None:
    apply_reinforce(state, territory_id, count)

def apply_attack(state: RiskState, from_id: int, to_id: int, move_count: int) -> None:
    """
    Risiko-Würfelkampf ohne Karten/Missionen:

    - Angreifer: bis zu 3 Würfel, max = Truppen_im_Angriffsland - 1
    - Verteidiger: bis zu 2 Würfel, max = Truppen_im_Zielland
    - Würfel absteigend sortieren, paarweise vergleichen:
      * höherer Wurf gewinnt, bei Gleichstand gewinnt Verteidiger.
    - Für jeden verlorenen Vergleich verliert die entsprechende Seite 1 Truppe.
    - Wird das Zielland leer, erobert Angreifer es und zieht Truppen hinein.
    """
    if state.phase != PHASE_ATTACK:
        return
    if state.owners[from_id] != state.current_player_idx:
        return
    if state.owners[to_id] == state.current_player_idx:
        return
    if to_id not in ADJACENCY[from_id]:
        return

    atk_troops = int(state.troops[from_id])
    def_troops = int(state.troops[to_id])

    # Angreifer braucht mindestens 2 Truppen, Verteidiger mindestens 1
    if atk_troops <= 1 or def_troops <= 0:
        return

    # Anzahl Würfel wie im Regelbuch
    atk_dice = min(3, atk_troops - 1)
    def_dice = min(2, def_troops)
    if atk_dice <= 0 or def_dice <= 0:
        return

    # Würfel würfeln und absteigend sortieren
    atk_rolls = sorted([random.randint(1, 6) for _ in range(atk_dice)], reverse=True)
    def_rolls = sorted([random.randint(1, 6) for _ in range(def_dice)], reverse=True)

    # Paarweise vergleichen
    pairs = min(len(atk_rolls), len(def_rolls))
    atk_losses = 0
    def_losses = 0
    for i in range(pairs):
        if atk_rolls[i] > def_rolls[i]:
            def_losses += 1
        else:
            # Verteidiger gewinnt Gleichstand
            atk_losses += 1

    # Verluste anwenden
    atk_troops_after = max(0, atk_troops - atk_losses)
    def_troops_after = max(0, def_troops - def_losses)

    state.troops[from_id] = atk_troops_after
    state.troops[to_id] = def_troops_after

    # Zielland erobert?
    if def_troops_after <= 0 and atk_troops_after > 1:
        state.owners[to_id] = state.current_player_idx

        # Im Regelbuch: mindestens so viele Truppen hineinziehen,
        # wie Angreifer-Würfel verwendet wurden.
        max_move = atk_troops_after - 1
        min_move = min(atk_dice, max_move)
        # move_count als Wunsch nutzen, aber clampen
        m = max(min_move, min(move_count, max_move))
        state.troops[from_id] -= m
        state.troops[to_id] = m

    # Winner-Check
    w = check_winner(state, PLAYERS)
    if w is not None:
        state.winner = w

def apply_fortify(state: RiskState, from_id: int, to_id: int, count: int) -> None:
    if state.phase != PHASE_FORTIFY:
        return
    if state.owners[from_id] != state.current_player_idx:
        return
    if state.owners[to_id] != state.current_player_idx:
        return
    if to_id not in ADJACENCY[from_id]:
        return

    troops_from = int(state.troops[from_id])
    if troops_from <= 1:
        return
    max_move = troops_from - 1
    m = max(1, min(count, max_move))
    state.troops[from_id] -= m
    state.troops[to_id] += m

    # Neue Regel: Fortify wurde in diesem Zug genutzt
    state.has_fortified_this_turn = True

# ---------------------------------------------------------
# Actions & Legal Actions
# ---------------------------------------------------------

Action = Tuple[str, Dict[str, Any]]

def encode_action(kind: str, **kwargs: Any) -> Action:
    return kind, kwargs

def owned_territories(state: RiskState, player_idx: int) -> np.ndarray:
    return np.nonzero(state.owners == player_idx)[0]

def border_pairs(state: RiskState, player_idx: int) -> List[Tuple[int, int]]:
    res: List[Tuple[int, int]] = []
    own = owned_territories(state, player_idx)
    for tid in own:
        for nb in ADJACENCY[tid]:
            res.append((tid, nb))
    return res

def get_legal_actions(state: RiskState, players: List[str]) -> List[Action]:
    legal: List[Action] = []
    p = state.current_player_idx

    if state.winner is not None:
        legal.append(encode_action("end_phase"))
        return legal

    if state.phase == PHASE_REINFORCE:
        if state.reinforcements_left > 0:
            for tid in owned_territories(state, p):
                legal.append(encode_action("place", territory_id=tid, count=1))
        legal.append(encode_action("end_phase"))

    elif state.phase == PHASE_ATTACK:
        for from_id, to_id in border_pairs(state, p):
            if state.owners[to_id] == p:
                continue
            troops_from = int(state.troops[from_id])
            if troops_from <= 1:
                continue
            max_move = troops_from - 1
            for move in range(1, max_move + 1):
                legal.append(
                    encode_action(
                        "attack",
                        from_id=from_id,
                        to_id=to_id,
                        move_count=move,
                    )
                )
        legal.append(encode_action("end_phase"))

    elif state.phase == PHASE_FORTIFY:
        for from_id, to_id in border_pairs(state, p):
            if state.owners[from_id] != p or state.owners[to_id] != p:
                continue
            troops_from = int(state.troops[from_id])
            if troops_from <= 1:
                continue
            max_move = troops_from - 1
            for c in range(1, max_move + 1):
                legal.append(
                    encode_action(
                        "fortify",
                        from_id=from_id,
                        to_id=to_id,
                        count=c,
                    )
                )
        legal.append(encode_action("end_phase"))

    if not legal:
        legal.append(encode_action("end_phase"))

    return legal
