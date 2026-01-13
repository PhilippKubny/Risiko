from __future__ import annotations

import argparse
import math
import random
import tkinter as tk
from dataclasses import dataclass
from typing import Dict, Tuple

from risiko.game.env import RiskEnv
from risiko.game.map import ADJACENCY, TERRITORIES


WINDOW_SIZE = (900, 600)
BACKGROUND_COLOR = "#1c1e24"
EDGE_COLOR = "#505a6e"
TEXT_COLOR = "#f0f0f0"
PANEL_COLOR = "#2d323c"
PLAYER_COLORS = [
    "#b33f3f",
    "#4682b4",
    "#3cb371",
    "#daa520",
]


@dataclass
class GuiState:
    last_action: str = ""
    paused: bool = False
    step_requested: bool = False
    elapsed_ms: int = 0


def build_positions(center: Tuple[int, int], radius: int) -> Dict[int, Tuple[int, int]]:
    positions: Dict[int, Tuple[int, int]] = {}
    count = len(TERRITORIES)
    for idx in range(count):
        angle = (2 * math.pi / count) * idx - math.pi / 2
        x = center[0] + int(math.cos(angle) * radius)
        y = center[1] + int(math.sin(angle) * radius)
        positions[idx] = (x, y)
    return positions


def format_action(action) -> str:
    if action.kind == "reinforce":
        return f"reinforce {TERRITORIES[action.params['territory']].name}"
    if action.kind == "attack":
        return (
            "attack "
            f"{TERRITORIES[action.params['from']].name} -> "
            f"{TERRITORIES[action.params['to']].name}"
        )
    if action.kind == "fortify":
        return (
            "fortify "
            f"{TERRITORIES[action.params['from']].name} -> "
            f"{TERRITORIES[action.params['to']].name}"
        )
    return "end phase"


def draw_state(
    canvas: tk.Canvas,
    env: RiskEnv,
    positions: Dict[int, Tuple[int, int]],
    gui_state: GuiState,
) -> None:
    canvas.delete("all")
    canvas.configure(bg=BACKGROUND_COLOR)
    canvas.create_rectangle(0, 0, WINDOW_SIZE[0], 60, fill=PANEL_COLOR, outline="")
    state = env.state
    if state is None:
        return

    for src, neighbors in ADJACENCY.items():
        for dst in neighbors:
            start = positions[src]
            end = positions[dst]
            canvas.create_line(*start, *end, fill=EDGE_COLOR, width=3)

    for territory in TERRITORIES:
        pos = positions[territory.id]
        owner = int(state.owners[territory.id])
        color = PLAYER_COLORS[owner % len(PLAYER_COLORS)]
        canvas.create_oval(
            pos[0] - 34,
            pos[1] - 34,
            pos[0] + 34,
            pos[1] + 34,
            fill=color,
            outline="#0f0f0f",
            width=2,
        )
        troops = int(state.troops[territory.id])
        canvas.create_text(pos[0], pos[1], text=str(troops), fill=TEXT_COLOR, font=("Arial", 16, "bold"))
        canvas.create_text(
            pos[0],
            pos[1] + 44,
            text=territory.name,
            fill=TEXT_COLOR,
            font=("Arial", 10),
        )

    header_text = (
        f"Turn {state.turn} | Player {state.current_player + 1} | "
        f"Phase: {state.phase} | Reinforcements: {state.reinforcements}"
    )
    canvas.create_text(20, 20, text=header_text, fill=TEXT_COLOR, anchor="w", font=("Arial", 10))
    action_text = f"Last action: {gui_state.last_action}" if gui_state.last_action else "Last action: -"
    canvas.create_text(20, 40, text=action_text, fill=TEXT_COLOR, anchor="w", font=("Arial", 10))
    if gui_state.paused:
        canvas.create_text(
            WINDOW_SIZE[0] - 80,
            30,
            text="PAUSED",
            fill="#ffc850",
            font=("Arial", 12, "bold"),
        )


def step_env(env: RiskEnv, rng: random.Random, gui_state: GuiState, auto_reset: bool) -> None:
    legal_indices = env.legal_action_indices()
    if not legal_indices:
        return
    action = env.action_space.all_actions()[rng.choice(legal_indices)]
    gui_state.last_action = format_action(action)
    result = env.step(action)
    if result.done and auto_reset:
        env.reset()
        gui_state.last_action = "auto reset"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a live GUI for the Risiko env.")
    parser.add_argument("--delay", type=float, default=0.8, help="Seconds between moves.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--auto-reset", action="store_true", help="Restart after a win.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = RiskEnv(seed=args.seed)
    env.reset()

    root = tk.Tk()
    root.title("Risiko Live Viewer")
    root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1] + 80}")
    root.configure(bg=BACKGROUND_COLOR)

    canvas = tk.Canvas(root, width=WINDOW_SIZE[0], height=WINDOW_SIZE[1], bg=BACKGROUND_COLOR, highlightthickness=0)
    canvas.pack(side=tk.TOP)

    control_frame = tk.Frame(root, bg=BACKGROUND_COLOR)
    control_frame.pack(fill=tk.X, pady=6)

    gui_state = GuiState()
    positions = build_positions((WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2 + 30), 200)

    def toggle_pause() -> None:
        gui_state.paused = not gui_state.paused
        draw_state(canvas, env, positions, gui_state)

    def step_once() -> None:
        gui_state.step_requested = True

    def reset_game() -> None:
        env.reset()
        gui_state.last_action = ""
        gui_state.paused = False
        gui_state.step_requested = False
        gui_state.elapsed_ms = 0
        draw_state(canvas, env, positions, gui_state)

    tk.Button(control_frame, text="Pause/Resume (Space)", command=toggle_pause).pack(side=tk.LEFT, padx=6)
    tk.Button(control_frame, text="Step (N)", command=step_once).pack(side=tk.LEFT, padx=6)
    tk.Button(control_frame, text="Reset (R)", command=reset_game).pack(side=tk.LEFT, padx=6)

    root.bind("<space>", lambda _event: toggle_pause())
    root.bind("n", lambda _event: step_once())
    root.bind("r", lambda _event: reset_game())

    delay_ms = max(0, int(args.delay * 1000))

    def tick() -> None:
        gui_state.elapsed_ms += 16
        should_step = (not gui_state.paused and gui_state.elapsed_ms >= delay_ms) or gui_state.step_requested
        if should_step:
            gui_state.elapsed_ms = 0
            gui_state.step_requested = False
            step_env(env, rng, gui_state, args.auto_reset)
        draw_state(canvas, env, positions, gui_state)
        root.after(16, tick)

    tick()
    root.mainloop()


if __name__ == "__main__":
    main()
