"""Intent Prediction Network 模型定义 (IntentNet).

输入: (batch, 10, 7) - 10帧 × 7特征
输出: strength (batch,1), direction_vec (batch,2: [sinθ, cosθ])
"""

from typing import Tuple

import torch
import torch.nn as nn


class IntentNet(nn.Module):
    """轻量级 1D-CNN + MLP 结构."""

    def __init__(self, dropout: float = 0.2):
        super().__init__()

        # 1D 卷积层: 输入通道=7 特征维度, 时间步=10
        self.conv1 = nn.Conv1d(
            in_channels=7, out_channels=32, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool1d(kernel_size=2)  # 10 -> 5

        # Flatten 后维度: 64 channels × 5 frames = 320
        self.fc1 = nn.Linear(64 * 5, 64)
        self.dropout = nn.Dropout(dropout)
        self.relu3 = nn.ReLU(inplace=True)

        # 输出分支
        self.fc_strength = nn.Linear(64, 1)
        self.strength_act = nn.Sigmoid()

        self.fc_direction = nn.Linear(64, 2)  # [sinθ, cosθ]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 10, 7)
        Returns:
            strength: (batch, 1)
            direction_vec: (batch, 2)
        """
        # (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.maxpool(x)  # (B, 64, 5)

        x = x.flatten(start_dim=1)  # (B, 320)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu3(x)

        strength = self.strength_act(self.fc_strength(x))
        direction_vec = self.fc_direction(x)

        return strength, direction_vec


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = IntentNet()
    dummy = torch.randn(4, 10, 7)
    s, d = net(dummy)
    print("strength:", s.shape, "direction:", d.shape)
    print("params:", count_parameters(net))


