from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Territory:
    id: int
    name: str


TERRITORIES = [
    Territory(0, "Nord"),
    Territory(1, "Ost"),
    Territory(2, "Sued"),
    Territory(3, "West"),
    Territory(4, "Delta"),
    Territory(5, "Echo"),
    Territory(6, "Fjord"),
    Territory(7, "Gulf"),
]

ADJACENCY = {
    0: (1, 3),
    1: (0, 2, 4),
    2: (1, 3, 5),
    3: (0, 2, 6),
    4: (1, 5, 7),
    5: (2, 4, 7),
    6: (3, 7),
    7: (4, 5, 6),
}

EDGE_LIST = [
    (src, dst)
    for src, neighbors in ADJACENCY.items()
    for dst in neighbors
]
