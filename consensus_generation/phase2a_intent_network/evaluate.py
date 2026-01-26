"""评估指标与里程碑检查 (Phase 2a)."""

from typing import Tuple
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


def _angle_to_sector(angle: float) -> int:
    """角度 -> 8扇区编号 [0..7]."""
    angle = angle % (2 * np.pi)
    sector = int((angle + np.pi / 8) / (np.pi / 4)) % 8
    return sector


def evaluate_direction_accuracy(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str,
) -> float:
    """计算方向预测8扇区 accuracy，并保存混淆矩阵图."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    all_pred_angles = []
    all_true_angles = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input"].to(device)
            true_dir = batch["direction_label"].cpu().numpy().squeeze()

            _, pred_dir_vec = model(inputs)
            pred_dir_vec = pred_dir_vec.cpu().numpy()
            pred_sin = pred_dir_vec[:, 0]
            pred_cos = pred_dir_vec[:, 1]
            pred_angles = np.arctan2(pred_sin, pred_cos)

            all_pred_angles.extend(pred_angles)
            all_true_angles.extend(true_dir)

    all_pred_angles = np.array(all_pred_angles)
    all_true_angles = np.array(all_true_angles)

    pred_sectors = np.array([_angle_to_sector(a) for a in all_pred_angles])
    true_sectors = np.array([_angle_to_sector(a) for a in all_true_angles])

    accuracy = (pred_sectors == true_sectors).mean()
    print(f"Direction Accuracy (8-sector): {accuracy * 100:.1f}%")

    cm = confusion_matrix(true_sectors, pred_sectors, labels=list(range(8)))
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Direction Confusion Matrix (Acc: {accuracy * 100:.1f}%)")
    fig_path = os.path.join(output_dir, "direction_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {fig_path}")

    return float(accuracy)


def evaluate_strength_correlation(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str,
) -> float:
    """计算 strength 预测的 Pearson 相关系数，并保存散点图."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    all_pred = []
    all_true = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input"].to(device)
            true_s = batch["strength_label"].cpu().numpy().squeeze()

            pred_s, _ = model(inputs)
            pred_s = pred_s.cpu().numpy().squeeze()

            all_true.extend(true_s)
            all_pred.extend(pred_s)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    corr, p_value = pearsonr(all_pred, all_true)
    print(f"Strength Correlation: r = {corr:.3f} (p = {p_value:.4f})")

    plt.figure(figsize=(7, 6))
    plt.scatter(all_true, all_pred, alpha=0.3, s=15)
    plt.plot([0, 1], [0, 1], "r--", label="Perfect")
    plt.xlabel("True Strength")
    plt.ylabel("Predicted Strength")
    plt.title(f"Strength Prediction (r = {corr:.3f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig_path = os.path.join(output_dir, "strength_correlation.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Strength correlation plot saved to {fig_path}")

    return float(corr)


def check_phase2a_milestones(direction_acc: float, strength_corr: float) -> bool:
    """根据指标检查 Phase 2a 是否达标."""
    print("\n" + "=" * 60)
    print("Phase 2a Milestone Check")
    print("=" * 60)

    # Direction
    print("\nMilestone 1: Direction Accuracy")
    print(f"  Result: {direction_acc * 100:.1f}%")
    if direction_acc > 0.80:
        print("  状态: EXCELLENT (>80%)")
        dir_status = "excellent"
    elif direction_acc > 0.75:
        print("  状态: ACCEPTABLE (>75%)")
        dir_status = "acceptable"
    else:
        print("  状态: FAILED (<75%)")
        dir_status = "failed"

    # Strength
    print("\nMilestone 2: Strength Correlation")
    print(f"  Result: r = {strength_corr:.3f}")
    if strength_corr > 0.7:
        print("  状态: PASS (>0.7)")
        str_status = "pass"
    else:
        print("  状态: FAILED (<0.7)")
        str_status = "failed"

    print("\n" + "=" * 60)
    print("Final Decision")
    print("=" * 60)

    if dir_status in ("excellent", "acceptable") and str_status == "pass":
        print("Phase 2a PASSED - Ready for Phase 2b")
        return True

    print("Phase 2a FAILED - Debug needed")
    if direction_acc < 0.75:
        print("  - 考虑增加网络容量/epoch 或窗口长度 (10 → 20)")
    if strength_corr < 0.7:
        print("  - 检查FSR标定，或尝试不同的strength label定义")
    return False


