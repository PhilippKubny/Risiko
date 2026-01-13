from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .network import PolicyValueNet
from .self_play import SelfPlaySample


@dataclass
class TrainMetrics:
    policy_loss: float
    value_loss: float
    total_loss: float


def build_dataset(samples: Iterable[SelfPlaySample]) -> TensorDataset:
    states: List[List[float]] = []
    policies: List[List[float]] = []
    values: List[float] = []
    for sample in samples:
        states.append(sample.state)
        policies.append(sample.policy)
        values.append(sample.value)
    state_tensor = torch.tensor(np.array(states), dtype=torch.float32)
    policy_tensor = torch.tensor(np.array(policies), dtype=torch.float32)
    value_tensor = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(-1)
    return TensorDataset(state_tensor, policy_tensor, value_tensor)


def train_network(
    network: PolicyValueNet,
    dataset: TensorDataset,
    epochs: int = 4,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
) -> List[TrainMetrics]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    policy_loss_fn = nn.KLDivLoss(reduction="batchmean")
    value_loss_fn = nn.MSELoss()

    metrics: List[TrainMetrics] = []
    for _ in range(epochs):
        running_policy = 0.0
        running_value = 0.0
        running_total = 0.0
        batches = 0
        for states, policies, values in loader:
            optimizer.zero_grad()
            output = network(states)
            log_probs = torch.log_softmax(output.policy_logits, dim=-1)
            policy_loss = policy_loss_fn(log_probs, policies)
            value_loss = value_loss_fn(output.value, values)
            total_loss = policy_loss + value_loss
            total_loss.backward()
            optimizer.step()

            running_policy += float(policy_loss.item())
            running_value += float(value_loss.item())
            running_total += float(total_loss.item())
            batches += 1
        if batches == 0:
            break
        metrics.append(
            TrainMetrics(
                policy_loss=running_policy / batches,
                value_loss=running_value / batches,
                total_loss=running_total / batches,
            )
        )
    return metrics
