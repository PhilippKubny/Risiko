# file: risk_az_net.py
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RiskAZNet(nn.Module):
    """
    Schlankes, effizientes Policy+Value-Netz für Risk.
    Input:  flat_features (B, 181)
    Output: policy_logits (B, n_actions), value (B, 1)
    """

    def __init__(
        self,
        input_dim: int = 181,
        n_actions: int = 256,     # ggf. an deine Action-Space-Größe anpassen
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ):
        super().__init__()

        layers = []
        last_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            last_dim = hidden_dim

        self.body = nn.Sequential(*layers)

        # Policy-Head
        self.policy_head = nn.Linear(hidden_dim, n_actions)

        # Value-Head
        self.value_head = nn.Linear(hidden_dim, 1)

        # Optional: leichte Init
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: Tensor der Form (B, input_dim) mit float32 (flat_features).
        returns:
            policy_logits: (B, n_actions)
            value:         (B, 1), tanh-gebunden in [-1, 1]
        """
        # Safety: auf float32 casten
        if x.dtype != torch.float32:
            x = x.float()

        h = self.body(x)

        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h))

        return policy_logits, value


# Mini-Demo
if __name__ == "__main__":
    import torch

    batch_size = 4
    input_dim = 181
    n_actions = 256

    net = RiskAZNet(input_dim=input_dim, n_actions=n_actions)
    dummy = torch.randn(batch_size, input_dim)

    policy_logits, value = net(dummy)
    print("policy_logits shape:", policy_logits.shape)  # (4, 256)
    print("value shape:", value.shape)                  # (4, 1)
