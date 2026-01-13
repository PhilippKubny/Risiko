from __future__ import annotations

import argparse

import torch

from risiko.az.network import PolicyValueNet
from risiko.az.train import build_dataset, train_network
from risiko.game.env import RiskEnv
from risiko.game.map import TERRITORIES
from risiko.utils.serialization import load_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the policy/value network.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", type=str, default="data/model.pt")
    args = parser.parse_args()

    samples = load_samples(args.data)
    env = RiskEnv()
    action_dim = env.action_space.size
    input_dim = (
        env.num_players * len(TERRITORIES)
        + len(TERRITORIES)
        + 3
        + env.num_players
    )
    network = PolicyValueNet(input_dim=input_dim, action_dim=action_dim)
    dataset = build_dataset(samples)
    metrics = train_network(network, dataset, epochs=args.epochs)

    network.eval()
    network.cpu()
    network_state = {
        "state_dict": network.state_dict(),
        "input_dim": input_dim,
        "action_dim": action_dim,
    }
    torch.save(network_state, args.output)
    for epoch, metric in enumerate(metrics, start=1):
        print(
            f"Epoch {epoch}: policy={metric.policy_loss:.4f}, "
            f"value={metric.value_loss:.4f}, total={metric.total_loss:.4f}"
        )
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
