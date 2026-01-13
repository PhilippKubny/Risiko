from __future__ import annotations
from typing import Dict, Tuple, Any

import sys
import math
import random
import numpy as np
import pygame
import torch

from risk_env import RiskEnv
from risk_az_net import RiskAZNet
from mcts_risk import MCTS
from action_mapping import N_ACTIONS, action_to_index
from risk_state import NUM_TERR, TERRITORIES, ADJACENCY  # type: ignore

# ---------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------

MODEL_PATH = "risk_az_latest.pth"
WINDOW_WIDTH, WINDOW_HEIGHT = 1600, 900
FPS = 30

TOP_BAR_HEIGHT = 60
BOTTOM_BAR_HEIGHT = 60

BG_COLOR = (20, 20, 30)
TOP_BAR_COLOR = (40, 40, 60)
BOTTOM_BAR_COLOR = (40, 40, 60)
TEXT_COLOR = (230, 230, 230)

LILA_COLOR = (180, 80, 255)
ROT_COLOR = (220, 60, 60)
NEUTRAL_COLOR = (150, 150, 150)

LAND_EDGE_COLOR = (120, 120, 120)
SEA_EDGE_COLOR = (120, 180, 220)

TERR_RADIUS = 26

# ---------------------------------------------------------
# Territoriums-Positionen – an Vorlage angelehnt
# ---------------------------------------------------------

TERR_POS: Dict[int, Tuple[int, int]] = {
    # Nordamerika (0–8)
    0: (125, 210),  # Alaska
    1: (255, 200),  # Nordwest-Territorium
    2: (430, 145),  # Grönland
    3: (230, 285),  # Alberta
    4: (330, 280),  # Ontario
    5: (450, 275),  # Quebec
    6: (230, 360),  # Weststaaten
    7: (335, 355),  # Oststaaten
    8: (295, 440),  # Mittelamerika

    # Südamerika (9–12)
    9:  (340, 520),  # Venezuela
    10: (335, 600),  # Peru
    11: (440, 585),  # Brasilien
    12: (365, 685),  # Argentinien

    # Europa (13–19)
    13: (560, 180),  # Island
    14: (610, 275),  # Großbritannien
    15: (710, 215),  # Skandinavien
    16: (620, 345),  # Westeuropa
    17: (700, 290),  # Mitteleuropa
    18: (745, 370),  # Südeuropa
    19: (840, 300),  # Ukraine

    # Afrika (20–25)
    20: (660, 485),  # Nordafrika
    21: (760, 450),  # Ägypten
    22: (850, 530),  # Ostafrika
    23: (700, 575),  # Zentralafrika
    24: (730, 665),  # Südafrika
    25: (900, 640),  # Madagaskar

    # Asien (26–37)
    26: (900, 255),  # Ural
    27: (1010, 215), # Sibirien
    28: (1135, 160), # Jakutien
    29: (1225, 225), # Irkutsk
    30: (1355, 235), # Kamtschatka

    31: (1145, 305), # Mongolei
    32: (1350, 320), # Japan
    33: (1045, 345), # China
    34: (900, 400),  # Mittlerer Osten
    35: (1010, 410), # Indien
    36: (1125, 435), # Siam
    37: (930, 330),  # Afghanistan

    # Australien (38–41)
    38: (1035, 565), # Indonesien
    39: (1160, 555), # Neu-Guinea
    40: (1075, 670), # West-Australien
    41: (1205, 680), # Ost-Australien
}

# ---------------------------------------------------------
# Seewege – klassische Risk-Karte
# ---------------------------------------------------------

SEA_EDGES = {
    # Südamerika ↔ Afrika
    (11, 20), (20, 11),   # Brasilien – Nordafrika

    # Nordamerika ↔ Asien
    (0, 30), (30, 0),     # Alaska – Kamtschatka

    # Grönland ↔ Nordamerika (Landbrücken über See)
    (2, 1), (1, 2),       # Grönland – Nordwest-Territorium
    (2, 4), (4, 2),       # Grönland – Ontario
    (2, 5), (5, 2),       # Grönland – Quebec

    # Island-Dreieck
    (13, 2),  (2, 13),    # Island – Grönland
    (13, 14), (14, 13),   # Island – Großbritannien
    (13, 15), (15, 13),   # Island – Skandinavien

    # Europa ↔ Afrika
    (16, 20), (20, 16),   # Westeuropa – Nordafrika
    (18, 21), (21, 18),   # Südeuropa – Ägypten

    # Australien-Bereich
    (38, 40), (40, 38),   # Indonesien – West-Australien
    (38, 39), (39, 38),   # Indonesien – Neu-Guinea
    (39, 41), (41, 39),   # Neu-Guinea – Ost-Australien
}

# ---------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------

def load_net(device: str = "cpu") -> RiskAZNet:
    env = RiskEnv()
    obs, _ = env.reset()
    input_dim = obs["flat_features"].shape[0]
    net = RiskAZNet(input_dim=input_dim, n_actions=N_ACTIONS).to(device)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        print(f"[GUI] Model loaded from {MODEL_PATH} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    except FileNotFoundError:
        print(f"[GUI] WARNING: {MODEL_PATH} not found, using random weights.")
    net.eval()
    return net

def choose_action_nn(env: RiskEnv, net: RiskAZNet, n_simulations: int, device: str) -> Any:
    mcts = MCTS(env, net, c_puct=1.5, device=device)
    pi = mcts.run_search(n_simulations=n_simulations)

    legal_actions = env.get_legal_actions()
    legal_indices = []
    for a in legal_actions:
        try:
            idx = action_to_index(a)
            legal_indices.append((idx, a))
        except ValueError:
            continue

    if not legal_indices:
        return ("end_phase", {})

    logits = np.array([pi[idx] for idx, _ in legal_indices], dtype=np.float32)
    best_idx = int(np.argmax(logits))
    _, action = legal_indices[best_idx]
    return action

def choose_action_random(env: RiskEnv) -> Any:
    legal = env.get_legal_actions()
    return random.choice(legal)

def compute_stats(env: RiskEnv) -> Dict[str, int]:
    s = env.state
    assert s is not None
    owners = s.owners
    troops = s.troops
    players = env.players

    stats: Dict[str, int] = {}
    for name in players:
        pid = players.index(name)
        terr_mask = (owners == pid)
        terr_cnt = int(terr_mask.sum())
        troops_cnt = int(troops[terr_mask].sum())
        stats[f"{name}_terr"] = terr_cnt
        stats[f"{name}_troops"] = troops_cnt
    return stats

# ---------------------------------------------------------
# Drawing
# ---------------------------------------------------------

def draw_top_bar(
    screen,
    font,
    small_font,
    env: RiskEnv,
    episode: int,
    lila_wins: int,
    games_played: int,
    model_player: str,
):
    pygame.draw.rect(screen, TOP_BAR_COLOR, (0, 0, WINDOW_WIDTH, TOP_BAR_HEIGHT))

    s = env.state
    assert s is not None
    step = s.step_count
    current_player = env.players[s.current_player_idx]
    phase = env.phase_str

    text_left = f"Step: {step}   Player: {current_player}   Phase: {phase}"
    surf = font.render(text_left, True, TEXT_COLOR)
    screen.blit(surf, (10, 10))

    random_player = next(name for name in env.players if name != model_player)
    role_text = f"AI: {model_player}   Random: {random_player}"
    role_surf = small_font.render(role_text, True, TEXT_COLOR)
    screen.blit(role_surf, (10, 34))

    winrate = 0.0 if games_played == 0 else lila_wins / games_played
    text_right = f"Episode: {episode}   Lila Winrate: {winrate:.2f}"
    surf_r = font.render(text_right, True, TEXT_COLOR)
    rect = surf_r.get_rect()
    rect.top = 10
    rect.right = WINDOW_WIDTH - 10
    screen.blit(surf_r, rect)

def draw_bottom_bar(screen, font, env: RiskEnv, model_player: str):
    pygame.draw.rect(
        screen,
        BOTTOM_BAR_COLOR,
        (0, WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT, WINDOW_WIDTH, BOTTOM_BAR_HEIGHT),
    )

    stats = compute_stats(env)
    random_player = next(name for name in env.players if name != model_player)
    text = (
        f"Model ({model_player}): {stats.get(f'{model_player}_troops', 0)} troops, "
        f"{stats.get(f'{model_player}_terr', 0)} terrs   |   "
        f"Random ({random_player}): {stats.get(f'{random_player}_troops', 0)} troops, "
        f"{stats.get(f'{random_player}_terr', 0)} terrs"
    )
    surf = font.render(text, True, TEXT_COLOR)
    rect = surf.get_rect()
    rect.centerx = WINDOW_WIDTH // 2
    rect.centery = WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT // 2
    screen.blit(surf, rect)

def draw_dashed_line(surface, color, start_pos, end_pos, width=1, dash_length=10, gap_length=5):
    x1, y1 = start_pos
    x2, y2 = end_pos
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    drawn = 0.0
    draw = True
    while drawn < length:
        start = (x1 + dx * drawn, y1 + dy * drawn)
        end = (x1 + dx * min(drawn + dash_length, length),
               y1 + dy * min(drawn + dash_length, length))
        if draw:
            pygame.draw.line(surface, color, start, end, width)
        drawn += dash_length + gap_length
        draw = not draw

def draw_wrap_edge(surface, color, alaska_pos, kam_pos, width=2):
    ax, ay = alaska_pos
    kx, ky = kam_pos
    mid_y = (ay + ky) // 2
    margin = 50
    pygame.draw.line(surface, color, (ax, ay), (-margin, mid_y), width)
    pygame.draw.line(surface, color, (WINDOW_WIDTH + margin, mid_y), (kx, ky), width)

def draw_world(screen, small_font, env: RiskEnv):
    s = env.state
    assert s is not None
    owners = s.owners
    troops = s.troops
    players = env.players

    # Kanten nach ADJACENCY (Land + See)
    for from_id in range(NUM_TERR):
        if from_id not in TERR_POS:
            continue
        x1, y1 = TERR_POS[from_id]
        for to_id in ADJACENCY[from_id]:
            if to_id <= from_id or to_id not in TERR_POS:
                continue
            x2, y2 = TERR_POS[to_id]

            # Alaska <-> Kamtschatka Wrap-Around
            if {from_id, to_id} == {0, 30}:
                draw_wrap_edge(screen, SEA_EDGE_COLOR, TERR_POS[0], TERR_POS[30], width=2)
                continue

            if (from_id, to_id) in SEA_EDGES or (to_id, from_id) in SEA_EDGES:
                draw_dashed_line(
                    screen,
                    SEA_EDGE_COLOR,
                    (x1, y1),
                    (x2, y2),
                    width=2,
                    dash_length=10,
                    gap_length=6,
                )
            else:
                pygame.draw.line(screen, LAND_EDGE_COLOR, (x1, y1), (x2, y2), 2)

    # Territories
    for tid in range(NUM_TERR):
        if tid not in TERR_POS:
            continue
        x, y = TERR_POS[tid]
        owner_idx = owners[tid]
        troop_cnt = int(troops[tid])

        if 0 <= owner_idx < len(players):
            name = players[owner_idx]
            if name == "Lila":
                color = LILA_COLOR
            elif name == "Rot":
                color = ROT_COLOR
            else:
                color = NEUTRAL_COLOR
        else:
            color = NEUTRAL_COLOR

        pygame.draw.circle(screen, color, (x, y), TERR_RADIUS)
        pygame.draw.circle(screen, (0, 0, 0), (x, y), TERR_RADIUS, 2)

        troop_text = str(troop_cnt)
        surf_t = small_font.render(troop_text, True, (0, 0, 0))
        rect_t = surf_t.get_rect(center=(x, y))
        screen.blit(surf_t, rect_t)

        name_text = TERRITORIES[tid]
        surf_n = small_font.render(name_text, True, TEXT_COLOR)
        rect_n = surf_n.get_rect(center=(x, y - TERR_RADIUS - 14))
        screen.blit(surf_n, rect_n)

# ---------------------------------------------------------
# Haupt-Loop
# ---------------------------------------------------------

def check_conquest_winner(env: RiskEnv) -> str | None:
    s = env.state
    if s is None:
        return None
    for name in env.players:
        pid = env.players.index(name)
        if np.all(s.owners == pid):
            return name
    return None

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Risk GUI: Model vs Random")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 26)
    small_font = pygame.font.SysFont("arial", 18)

    device = "cpu"
    net = load_net(device=device)

    episode = 0
    lila_wins = 0
    games_played = 0

    running = True

    while running:
        episode += 1
        env = RiskEnv(max_steps=1000)
        obs, info = env.reset()
        done = False
        model_player = random.choice(env.players)
        pygame.display.set_caption(
            f"Risk GUI: Model ({model_player}) vs Random"
        )

        # Startaufstellung 3s
        start_show_ms = 3000
        start_time = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_time < start_show_ms:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)

            screen.fill(BG_COLOR)
            draw_world(screen, small_font, env)
            draw_top_bar(
                screen,
                font,
                small_font,
                env,
                episode,
                lila_wins,
                games_played,
                model_player,
            )
            draw_bottom_bar(screen, font, env, model_player)
            pygame.display.flip()
            clock.tick(FPS)

        # Spiel-Loop
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not running:
                break

            s = env.state
            assert s is not None
            current_player = env.players[s.current_player_idx]

            if current_player == model_player:
                action = choose_action_nn(env, net, n_simulations=4, device=device)
            else:
                action = choose_action_random(env)

            obs, reward, done, info = env.step(action)
            if not done:
                conquest_winner = check_conquest_winner(env)
                if conquest_winner is not None:
                    done = True
                    info["winner"] = conquest_winner

            screen.fill(BG_COLOR)
            draw_world(screen, small_font, env)
            draw_top_bar(
                screen,
                font,
                small_font,
                env,
                episode,
                lila_wins,
                games_played,
                model_player,
            )
            draw_bottom_bar(screen, font, env, model_player)
            pygame.display.flip()
            clock.tick(FPS)

        # Ende: Winner zeigen
        games_played += 1
        winner = info.get("winner")
        if winner == env.lila_name:
            lila_wins += 1

        end_show_ms = 3000
        end_start = pygame.time.get_ticks()
        while pygame.time.get_ticks() - end_start < end_show_ms and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill(BG_COLOR)
            draw_world(screen, small_font, env)
            draw_top_bar(
                screen,
                font,
                small_font,
                env,
                episode,
                lila_wins,
                games_played,
                model_player,
            )
            draw_bottom_bar(screen, font, env, model_player)

            winner_text = f"Winner: {winner}" if winner is not None else "Winner: None"
            surf_w = font.render(winner_text, True, TEXT_COLOR)
            rect_w = surf_w.get_rect(center=(WINDOW_WIDTH // 2, TOP_BAR_HEIGHT + 20))
            screen.blit(surf_w, rect_w)

            pygame.display.flip()
            clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
