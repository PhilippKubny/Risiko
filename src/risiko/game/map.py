from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Territory:
    id: int
    name: str


TERRITORIES = [
    Territory(0, "Alaska"),
    Territory(1, "Nordwest-Territorium"),
    Territory(2, "Grönland"),
    Territory(3, "Alberta"),
    Territory(4, "Ontario"),
    Territory(5, "Quebec"),
    Territory(6, "Weststaaten"),
    Territory(7, "Oststaaten"),
    Territory(8, "Mittelamerika"),
    Territory(9, "Venezuela"),
    Territory(10, "Peru"),
    Territory(11, "Brasilien"),
    Territory(12, "Argentinien"),
    Territory(13, "Island"),
    Territory(14, "Großbritannien"),
    Territory(15, "Skandinavien"),
    Territory(16, "Westeuropa"),
    Territory(17, "Mitteleuropa"),
    Territory(18, "Südeuropa"),
    Territory(19, "Ukraine"),
    Territory(20, "Nordafrika"),
    Territory(21, "Ägypten"),
    Territory(22, "Ostafrika"),
    Territory(23, "Zentralafrika"),
    Territory(24, "Südafrika"),
    Territory(25, "Madagaskar"),
    Territory(26, "Ural"),
    Territory(27, "Sibirien"),
    Territory(28, "Jakutien"),
    Territory(29, "Irkutsk"),
    Territory(30, "Kamtschatka"),
    Territory(31, "Mongolei"),
    Territory(32, "Japan"),
    Territory(33, "China"),
    Territory(34, "Mittlerer Osten"),
    Territory(35, "Indien"),
    Territory(36, "Siam"),
    Territory(37, "Afghanistan"),
    Territory(38, "Indonesien"),
    Territory(39, "Neu-Guinea"),
    Territory(40, "West-Australien"),
    Territory(41, "Ost-Australien"),
]

ADJACENCY = {
    0: (1, 3, 30),
    1: (0, 2, 3, 4),
    2: (1, 4, 5, 13),
    3: (0, 1, 4, 6),
    4: (1, 2, 3, 5, 6, 7),
    5: (2, 4, 7),
    6: (3, 4, 7, 8),
    7: (4, 5, 6, 8),
    8: (6, 7, 9),
    9: (8, 10, 11),
    10: (9, 11, 12),
    11: (9, 10, 12, 20),
    12: (10, 11),
    13: (2, 14, 15),
    14: (13, 15, 17, 16),
    15: (13, 14, 19, 17),
    16: (14, 17, 20, 18),
    17: (14, 15, 16, 18, 19),
    18: (16, 17, 19, 20, 21, 34),
    19: (15, 17, 18, 26, 37, 34),
    20: (16, 18, 21, 22, 23, 11),
    21: (18, 20, 22, 34),
    22: (20, 21, 23, 24, 25, 34),
    23: (20, 22, 24),
    24: (22, 23, 25),
    25: (22, 24),
    26: (19, 27, 33, 37),
    27: (26, 28, 29, 31, 33),
    28: (27, 29, 30),
    29: (27, 28, 30, 31),
    30: (28, 29, 31, 32, 0),
    31: (27, 29, 30, 32, 33),
    32: (30, 31),
    33: (26, 27, 31, 35, 37, 36),
    34: (18, 21, 22, 19, 37, 35),
    35: (34, 33, 36, 37),
    36: (35, 33, 38),
    37: (26, 19, 34, 33, 35),
    38: (36, 39, 40),
    39: (38, 40, 41),
    40: (38, 39, 41),
    41: (39, 40),
}

EDGE_LIST = [
    (src, dst)
    for src, neighbors in ADJACENCY.items()
    for dst in neighbors
]
