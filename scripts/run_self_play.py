from __future__ import annotations

import argparse

from risiko.az.network import PolicyValueNet
from risiko.az.self_play import SelfPlayRunner
from risiko.game.env import RiskEnv
from risiko.game.map import TERRITORIES
from risiko.utils.serialization import save_samples


def parse_random_players(raw_value: str, num_players: int) -> set[int]:
    raw_value = raw_value.strip()
    if not raw_value:
        return set()
    if raw_value.lower() == "all":
        return set(range(num_players))
    indices: set[int] = set()
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        index = int(chunk)
        if index < 0 or index >= num_players:
            raise ValueError(f"Random player index {index} out of range 0..{num_players - 1}.")
        indices.add(index)
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Run self-play games.")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--output", type=str, default="data/self_play.jsonl")
    parser.add_argument(
        "--random-players",
        type=str,
        default="1",
        help="Comma-separated 0-based player indices that act randomly (or 'all').",
    )
    args = parser.parse_args()

    env = RiskEnv()
    action_dim = env.action_space.size
    input_dim = (
        env.num_players * len(TERRITORIES)
        + len(TERRITORIES)
        + 3
        + env.num_players
    )
    network = PolicyValueNet(input_dim=input_dim, action_dim=action_dim)
    runner = SelfPlayRunner(network=network, env=env)
    random_players = parse_random_players(args.random_players, env.num_players)

    all_samples = []
    for _ in range(args.games):
        all_samples.extend(runner.play_game(random_players=random_players))

    save_samples(args.output, all_samples)
    print(f"Saved {len(all_samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
