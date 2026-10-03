"""Shared observation contract and small policy for visual following."""

from __future__ import annotations

import torch
from torch import nn

FEATURE_COLUMNS = (
    "bearing",
    "box_area",
    "box_height",
    "confidence",
    "bearing_rate",
    "centered",
    "holding_distance",
)
ACTION_COLUMNS = ("vx", "vyaw")
MAX_FORWARD_MPS = 0.40
MAX_YAW_RAD_S = 0.95


class FollowActor(nn.Module):
    """Map normalized YOLO target features to bounded body velocity commands."""

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.reshape(1, -1).float())
        self.register_buffer("scale", scale.reshape(1, -1).float().clamp_min(1e-4))
        self.network = nn.Sequential(
            nn.Linear(len(FEATURE_COLUMNS), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, len(ACTION_COLUMNS)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.mean) / self.scale
        raw = self.network(normalized)
        vx = torch.sigmoid(raw[..., 0]) * MAX_FORWARD_MPS
        vyaw = torch.tanh(raw[..., 1]) * MAX_YAW_RAD_S
        return torch.stack((vx, vyaw), dim=-1)
