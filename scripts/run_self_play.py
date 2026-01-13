from __future__ import annotations

import argparse

from risiko.az.network import PolicyValueNet
from risiko.az.self_play import SelfPlayRunner
from risiko.game.env import RiskEnv
from risiko.game.map import TERRITORIES
from risiko.utils.serialization import save_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run self-play games.")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--output", type=str, default="data/self_play.jsonl")
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

    all_samples = []
    for _ in range(args.games):
        all_samples.extend(runner.play_game())

    save_samples(args.output, all_samples)
    print(f"Saved {len(all_samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
