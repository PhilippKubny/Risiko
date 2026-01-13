from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import torch

from risiko.az.network import PolicyValueNet
from risiko.az.self_play import SelfPlayConfig, SelfPlayRunner
from risiko.az.train import build_dataset, train_network
from risiko.game.env import RiskEnv
from risiko.game.map import TERRITORIES


def build_network(env: RiskEnv) -> PolicyValueNet:
    action_dim = env.action_space.size
    input_dim = (
        env.num_players * len(TERRITORIES)
        + len(TERRITORIES)
        + 3
        + env.num_players
    )
    return PolicyValueNet(input_dim=input_dim, action_dim=action_dim)


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


def load_checkpoint(path: Path, network: PolicyValueNet) -> None:
    if not path.exists():
        return
    checkpoint = torch.load(path, map_location="cpu")
    network.load_state_dict(checkpoint["state_dict"])


def save_checkpoint(path: Path, network: PolicyValueNet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    network.eval()
    network.cpu()
    checkpoint = {
        "state_dict": network.state_dict(),
    }
    torch.save(checkpoint, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run iterative self-play training.")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--games-per-iteration", type=int, default=8)
    parser.add_argument("--buffer-size", type=int, default=5000)
    parser.add_argument("--random-players", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="data/training")
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--mcts-sims", type=int, default=128)
    parser.add_argument("--mcts-cpuct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-frac", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-decay", type=float, default=0.97)
    parser.add_argument("--min-temperature", type=float, default=0.25)
    parser.add_argument("--max-moves", type=int, default=400)
    args = parser.parse_args()

    env = RiskEnv()
    network = build_network(env)
    if args.resume_from:
        load_checkpoint(Path(args.resume_from), network)

    config = SelfPlayConfig(
        num_simulations=args.mcts_sims,
        temperature=args.temperature,
        temperature_decay=args.temperature_decay,
        min_temperature=args.min_temperature,
        max_moves=args.max_moves,
        c_puct=args.mcts_cpuct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_frac=args.dirichlet_frac,
    )
    runner = SelfPlayRunner(network=network, env=env, config=config)
    random_players = parse_random_players(args.random_players, env.num_players)

    replay_buffer = deque(maxlen=args.buffer_size)
    output_dir = Path(args.output_dir)

    for iteration in range(1, args.iterations + 1):
        iteration_samples = []
        for _ in range(args.games_per_iteration):
            iteration_samples.extend(runner.play_game(random_players=random_players))
        replay_buffer.extend(iteration_samples)

        dataset = build_dataset(list(replay_buffer))
        metrics = train_network(
            network,
            dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        latest_path = output_dir / "model_latest.pt"
        save_checkpoint(latest_path, network)
        save_checkpoint(output_dir / f"model_iter_{iteration}.pt", network)

        if metrics:
            last = metrics[-1]
            print(
                "Iteration"
                f" {iteration}: policy={last.policy_loss:.4f},"
                f" value={last.value_loss:.4f}, total={last.total_loss:.4f}"
            )
        print(f"Samples in buffer: {len(replay_buffer)}")
        print(f"Saved model checkpoint to {latest_path}")


if __name__ == "__main__":
    main()
