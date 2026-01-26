"""训练循环与损失函数 (Phase 2a IntentNet)."""

from typing import Tuple, Dict, List
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import IntentNet


def intent_loss(
    pred_strength: torch.Tensor,
    pred_direction: torch.Tensor,
    target_strength: torch.Tensor,
    target_direction: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """组合loss: strength MSE + direction(sin/cos) MSE."""
    # strength: (B,1) vs (B,)
    target_strength = target_strength.view_as(pred_strength)
    loss_strength = nn.MSELoss()(pred_strength, target_strength)

    # direction: 角度 -> sin/cos
    target_sin = torch.sin(target_direction)
    target_cos = torch.cos(target_direction)
    target_vec = torch.stack([target_sin, target_cos], dim=1)

    loss_direction = nn.MSELoss()(pred_direction, target_vec)

    lambda_strength = 1.0
    lambda_direction = 2.0
    total_loss = lambda_strength * loss_strength + lambda_direction * loss_direction

    return total_loss, loss_strength, loss_direction


def get_device() -> torch.device:
    """优先使用GPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print(f"Using device: {dev}")
    return dev


def train_model(
    model: IntentNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
) -> Tuple[IntentNet, Dict[str, List[float]], float]:
    """训练 IntentNet，带 early stopping."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_epoch = 0.0

        for batch in train_loader:
            inputs = batch["input"].to(device)  # (B,10,7)
            strength_labels = batch["strength_label"].to(device)
            direction_labels = batch["direction_label"].to(device)

            pred_strength, pred_direction = model(inputs)
            loss, _, _ = intent_loss(
                pred_strength, pred_direction, strength_labels, direction_labels
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()

        train_loss_epoch /= max(len(train_loader), 1)
        history["train_loss"].append(train_loss_epoch)

        # 验证
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(device)
                strength_labels = batch["strength_label"].to(device)
                direction_labels = batch["direction_label"].to(device)

                pred_strength, pred_direction = model(inputs)
                loss, _, _ = intent_loss(
                    pred_strength, pred_direction, strength_labels, direction_labels
                )
                val_loss_epoch += loss.item()

        val_loss_epoch /= max(len(val_loader), 1)
        history["val_loss"].append(val_loss_epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss_epoch:.4f} | "
            f"Val Loss: {val_loss_epoch:.4f}"
        )

        # early stopping
        if val_loss_epoch < best_val_loss - 1e-5:
            best_val_loss = val_loss_epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start_time
    print(f"训练总耗时: {elapsed:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, elapsed


if __name__ == "__main__":
    # 简单自测：不真正训练
    net = IntentNet()
    device = get_device()
    net.to(device)
    x = torch.randn(8, 10, 7).to(device)
    s, d = net(x)
    print("forward ok:", s.shape, d.shape)


