# file: action_mapping.py
from __future__ import annotations
from typing import Dict, Tuple, Any, List, Hashable

from risk_state import NUM_TERR, ADJACENCY

Action = Tuple[str, Dict[str, Any]]

# Intern verwenden wir eine hashbare Repräsentation:
# ("place", tid), ("attack", from_id, to_id), ("fortify", from_id, to_id, count), ("end_phase",)
KeyType = Tuple[Hashable, ...]


_INDEX_TO_ACTION: List[Action] = []
_ACTION_TO_INDEX: Dict[KeyType, int] = {}


def _make_key(action: Action) -> KeyType:
    kind, params = action
    if kind == "place":
        return ("place", int(params["territory_id"]))
    elif kind == "attack":
        return ("attack", int(params["from_id"]), int(params["to_id"]))
    elif kind == "fortify":
        return (
            "fortify",
            int(params["from_id"]),
            int(params["to_id"]),
            int(params["count"]),
        )
    elif kind == "end_phase":
        return ("end_phase",)
    else:
        raise ValueError(f"Unknown action kind: {kind}")


def _register(action: Action) -> None:
    key = _make_key(action)
    if key in _ACTION_TO_INDEX:
        return
    idx = len(_INDEX_TO_ACTION)
    _INDEX_TO_ACTION.append(action)
    _ACTION_TO_INDEX[key] = idx


def build_action_space() -> int:
    from_id: int
    to_id: int

    # place
    for tid in range(NUM_TERR):
        _register(("place", {"territory_id": tid}))

    # attack
    for from_id in range(NUM_TERR):
        for to_id in ADJACENCY[from_id]:
            _register(("attack", {"from_id": from_id, "to_id": to_id}))

    # fortify
    MAX_FORTIFY_COUNT = 100
    for from_id in range(NUM_TERR):
        for to_id in ADJACENCY[from_id]:
            if from_id == to_id:
                continue
            for count in range(1, MAX_FORTIFY_COUNT + 1):
                _register(
                    (
                        "fortify",
                        {"from_id": from_id, "to_id": to_id, "count": count},
                    )
                )

    # end_phase
    _register(("end_phase", {}))

    return len(_INDEX_TO_ACTION)


N_ACTIONS = build_action_space()


def action_to_index(action: Action) -> int:
    kind, params = action

    if kind == "place":
        key = ("place", int(params["territory_id"]))
    elif kind == "attack":
        key = ("attack", int(params["from_id"]), int(params["to_id"]))
    elif kind == "fortify":
        raw_count = int(params["count"])
        MAX_FORTIFY_COUNT = 100
        count = max(1, min(raw_count, MAX_FORTIFY_COUNT))
        key = (
            "fortify",
            int(params["from_id"]),
            int(params["to_id"]),
            count,
        )
    elif kind == "end_phase":
        key = ("end_phase",)
    else:
        raise ValueError(f"Unknown action kind: {kind}")

    if key not in _ACTION_TO_INDEX:
        raise ValueError(f"Action not in mapping: {key}")

    return _ACTION_TO_INDEX[key]


def index_to_action(idx: int) -> Action:
    if idx < 0 or idx >= len(_INDEX_TO_ACTION):
        raise IndexError(f"Action index out of range: {idx}")
    kind, params = _INDEX_TO_ACTION[idx]
    return kind, dict(params)


def legal_actions_to_indices(legal_actions: List[Action]) -> List[int]:
    indices: List[int] = []
    for a in legal_actions:
        try:
            idx = action_to_index(a)
            indices.append(idx)
        except ValueError:
            continue
    return indices
