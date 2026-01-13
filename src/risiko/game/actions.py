from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .map import EDGE_LIST, TERRITORIES


@dataclass(frozen=True)
class Action:
    kind: str
    params: Dict[str, int]


@dataclass(frozen=True)
class ActionSpace:
    reinforce_actions: Tuple[Action, ...]
    attack_actions: Tuple[Action, ...]
    fortify_actions: Tuple[Action, ...]
    end_phase_action: Action

    @property
    def size(self) -> int:
        return (
            len(self.reinforce_actions)
            + len(self.attack_actions)
            + len(self.fortify_actions)
            + 1
        )

    def all_actions(self) -> Tuple[Action, ...]:
        return (
            *self.reinforce_actions,
            *self.attack_actions,
            *self.fortify_actions,
            self.end_phase_action,
        )


def build_action_space() -> ActionSpace:
    reinforce_actions = tuple(
        Action("reinforce", {"territory": territory.id}) for territory in TERRITORIES
    )
    attack_actions = tuple(
        Action("attack", {"from": src, "to": dst}) for src, dst in EDGE_LIST
    )
    fortify_actions = tuple(
        Action("fortify", {"from": src, "to": dst}) for src, dst in EDGE_LIST
    )
    end_phase_action = Action("end_phase", {})
    return ActionSpace(reinforce_actions, attack_actions, fortify_actions, end_phase_action)


def index_action_map(space: ActionSpace) -> Tuple[Dict[int, Action], Dict[Action, int]]:
    actions = list(space.all_actions())
    to_action = {idx: action for idx, action in enumerate(actions)}
    to_index = {action: idx for idx, action in enumerate(actions)}
    return to_action, to_index


def mask_from_indices(space: ActionSpace, indices: Iterable[int]) -> List[float]:
    mask = [0.0] * space.size
    for idx in indices:
        mask[idx] = 1.0
    return mask
