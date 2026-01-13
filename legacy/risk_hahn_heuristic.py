import numpy as np

# Kontinente und Boni (kopiert aus risk_env.py)
CONTINENTS = {
    "Nordamerika": list(range(0, 9)),
    "Südamerika": list(range(9, 13)),
    "Europa": list(range(13, 20)),
    "Afrika": list(range(20, 26)),
    "Asien": list(range(26, 38)),
    "Australien": list(range(38, 42)),
}

CONTINENT_BONUS = {
    "Nordamerika": 5,
    "Südamerika": 2,
    "Europa": 5,
    "Afrika": 3,
    "Asien": 7,
    "Australien": 2,
}

# Adjazenz (auch kopiert aus risk_env.py)
ADJACENCY = {
    0: [1, 3, 30],
    1: [0, 2, 3, 4],
    2: [1, 4, 5, 13],
    3: [0, 1, 4, 6],
    4: [1, 2, 3, 5, 6, 7],
    5: [2, 4, 7],
    6: [3, 4, 7, 8],
    7: [4, 5, 6, 8],
    8: [6, 7, 9],
    9: [8, 10, 11],
    10: [9, 11, 12],
    11: [9, 10, 12, 20],
    12: [10, 11],
    13: [2, 14, 15],
    14: [13, 15, 17, 16],
    15: [13, 14, 19, 17],
    16: [14, 17, 20, 18],
    17: [14, 15, 16, 18, 19],
    18: [16, 17, 19, 20, 21, 34],
    19: [15, 17, 18, 26, 37, 34],
    20: [16, 18, 21, 22, 23, 11],
    21: [18, 20, 22, 34],
    22: [20, 21, 23, 24, 25, 34],
    23: [20, 22, 24],
    24: [22, 23, 25],
    25: [22, 24],
    26: [19, 27, 33, 37],
    27: [26, 28, 29, 31, 33],
    28: [27, 29, 30],
    29: [27, 28, 30, 31],
    30: [28, 29, 31, 32, 0],
    31: [27, 29, 30, 32, 33],
    32: [30, 31],
    33: [26, 27, 31, 35, 37, 36],
    34: [18, 21, 22, 19, 37, 35],
    35: [34, 33, 36, 37],
    36: [35, 33, 38],
    37: [26, 19, 34, 33, 35],
    38: [36, 39, 40],
    39: [38, 40, 41],
    40: [38, 39, 41],
    41: [39, 40],
}


def compute_hahn_score(state, player_idx: int) -> float:
    """
    Heuristik nach Hahn:
    1) Overall Strength (Länder, Armeen, Kontinente)
    2) Expected Reinforcements
    3) Defensive Fronts
    """
    owners = state.owners
    troops = state.troops

    # ---------- 1) Overall Strength ----------
    my_countries = (owners == player_idx)
    my_territories = int(np.sum(my_countries))
    my_troops = float(troops[my_countries].sum())

    continent_bonus = 0.0
    for cont_name, terr_ids in CONTINENTS.items():
        if all(owners[tid] == player_idx for tid in terr_ids):
            continent_bonus += CONTINENT_BONUS.get(cont_name, 0)

    overall_strength = (
        2.0 * my_territories +
        1.0 * my_troops +
        5.0 * continent_bonus
    )

    # ---------- 2) Expected Reinforcements ----------
    base_reinf = max(3, my_territories // 3) if my_territories > 0 else 0
    expected_reinf = float(base_reinf + continent_bonus)

    # ---------- 3) Defensive Fronts ----------
    front_count = 0
    num_terr = len(owners)
    for terr in range(num_terr):
        if owners[terr] != player_idx:
            continue
        for nb in ADJACENCY[terr]:
            if owners[nb] != player_idx and owners[nb] >= 0:
                front_count += 1
                break

    defensive_fronts = -float(front_count)

    score = (
        1.0 * overall_strength +
        3.0 * expected_reinf +
        5.0 * defensive_fronts
    )
    return score
