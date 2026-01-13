from __future__ import annotations

import argparse
import math
import random
from typing import Dict, Tuple

import pygame

from risiko.game.env import RiskEnv
from risiko.game.map import ADJACENCY, TERRITORIES


WINDOW_SIZE = (900, 600)
BACKGROUND_COLOR = (28, 30, 36)
EDGE_COLOR = (80, 90, 110)
TEXT_COLOR = (240, 240, 240)
PANEL_COLOR = (45, 50, 60)
PLAYER_COLORS = [
    (179, 63, 63),
    (70, 130, 180),
    (60, 179, 113),
    (218, 165, 32),
]


def build_positions(center: Tuple[int, int], radius: int) -> Dict[int, Tuple[int, int]]:
    positions: Dict[int, Tuple[int, int]] = {}
    count = len(TERRITORIES)
    for idx in range(count):
        angle = (2 * math.pi / count) * idx - math.pi / 2
        x = center[0] + int(math.cos(angle) * radius)
        y = center[1] + int(math.sin(angle) * radius)
        positions[idx] = (x, y)
    return positions


def draw_state(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    env: RiskEnv,
    positions: Dict[int, Tuple[int, int]],
    last_action: str,
    paused: bool,
) -> None:
    screen.fill(BACKGROUND_COLOR)
    pygame.draw.rect(screen, PANEL_COLOR, (0, 0, WINDOW_SIZE[0], 60))
    state = env.state
    if state is None:
        return

    for src, neighbors in ADJACENCY.items():
        for dst in neighbors:
            start = positions[src]
            end = positions[dst]
            pygame.draw.line(screen, EDGE_COLOR, start, end, 3)

    for territory in TERRITORIES:
        pos = positions[territory.id]
        owner = int(state.owners[territory.id])
        color = PLAYER_COLORS[owner % len(PLAYER_COLORS)]
        pygame.draw.circle(screen, color, pos, 34)
        pygame.draw.circle(screen, (15, 15, 15), pos, 34, 2)

        troops = int(state.troops[territory.id])
        troop_surf = font.render(str(troops), True, TEXT_COLOR)
        troop_rect = troop_surf.get_rect(center=pos)
        screen.blit(troop_surf, troop_rect)

        name_surf = small_font.render(territory.name, True, TEXT_COLOR)
        name_rect = name_surf.get_rect(midtop=(pos[0], pos[1] + 40))
        screen.blit(name_surf, name_rect)

    header_text = (
        f"Turn {state.turn} | Player {state.current_player + 1} | "
        f"Phase: {state.phase} | Reinforcements: {state.reinforcements}"
    )
    header = small_font.render(header_text, True, TEXT_COLOR)
    screen.blit(header, (20, 18))

    action_text = f"Last action: {last_action}" if last_action else "Last action: -"
    action = small_font.render(action_text, True, TEXT_COLOR)
    screen.blit(action, (20, 36))

    if paused:
        paused_surf = font.render("PAUSED", True, (255, 200, 80))
        paused_rect = paused_surf.get_rect(center=(WINDOW_SIZE[0] - 100, 30))
        screen.blit(paused_surf, paused_rect)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a live GUI for the Risiko env.")
    parser.add_argument("--delay", type=float, default=0.8, help="Seconds between moves.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--auto-reset", action="store_true", help="Restart after a win.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = RiskEnv(seed=args.seed)
    env.reset()

    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Risiko Live Viewer")
    font = pygame.font.SysFont("arial", 26)
    small_font = pygame.font.SysFont("arial", 18)
    clock = pygame.time.Clock()

    positions = build_positions((WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2 + 30), 200)
    last_action = ""
    paused = False
    step_requested = False
    elapsed = 0.0
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        elapsed += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_n:
                    step_requested = True
                elif event.key == pygame.K_r:
                    env.reset()
                    last_action = ""
                    paused = False
                    step_requested = False
                    elapsed = 0.0

        should_step = (not paused and elapsed >= args.delay) or step_requested
        if should_step and env.state is not None:
            elapsed = 0.0
            step_requested = False
            legal_indices = env.legal_action_indices()
            if legal_indices:
                action = env.action_space.all_actions()[rng.choice(legal_indices)]
                last_action = format_action(action)
                result = env.step(action)
                if result.done and args.auto_reset:
                    env.reset()
                    last_action = "auto reset"

        draw_state(screen, font, small_font, env, positions, last_action, paused)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
