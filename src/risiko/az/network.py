from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class NetworkOutput:
    policy_logits: torch.Tensor
    value: torch.Tensor


class PolicyValueNet(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> NetworkOutput:
        features = self.backbone(x)
        policy_logits = self.policy_head(features)
        value = torch.tanh(self.value_head(features))
        return NetworkOutput(policy_logits=policy_logits, value=value)
