"""Phase 2a 可视化：Intent pattern 案例 & 特征重要性."""

from typing import List, Dict
import os
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .data_preprocessing import IntentDataset


def visualize_intent_patterns(
    model: torch.nn.Module,
    test_samples: List[Dict],
    device: torch.device,
    output_dir: str,
    example_indices: List[int] | None = None,
) -> None:
    """选择若干案例展示 Force 历史与 Intent 预测."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    if example_indices is None:
        example_indices = [10, 50, 100]
    example_indices = [idx for idx in example_indices if idx < len(test_samples)]
    if not example_indices:
        print("可视化样本数量不足，跳过 intent patterns 图。")
        return

    fig, axes = plt.subplots(len(example_indices), 2, figsize=(12, 4 * len(example_indices)))

    if len(example_indices) == 1:
        axes = np.expand_dims(axes, 0)  # 保证二维

    for row_idx, sample_idx in enumerate(example_indices):
        sample = test_samples[sample_idx]

        # 输入
        inp = torch.from_numpy(sample["input"]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred_s, pred_dir_vec = model(inp)
        pred_strength = float(pred_s.item())
        pred_sin = float(pred_dir_vec[0, 0].item())
        pred_cos = float(pred_dir_vec[0, 1].item())
        pred_angle = float(np.arctan2(pred_sin, pred_cos))

        true_strength = float(sample["strength_label"])
        true_angle = float(sample["direction_label"])

        force_history = sample["input"][:, 2]  # ||F||

        # 左图: 力历史
        ax_left = axes[row_idx, 0]
        ax_left.plot(range(len(force_history)), force_history, "b-o", linewidth=2)
        ax_left.set_xlabel("Frame")
        ax_left.set_ylabel("Force Magnitude")
        ax_left.set_title(f"Case {row_idx+1}: Force Pattern (idx={sample_idx})")
        ax_left.grid(True, alpha=0.3)

        # 右图: Intent 向量
        ax_right = axes[row_idx, 1]
        ax_right.set_aspect("equal", adjustable="box")
        ax_right.set_xlim(-1.5, 1.5)
        ax_right.set_ylim(-1.5, 1.5)

        circle_pred = plt.Circle((0, 0), pred_strength, color="blue", alpha=0.3)
        circle_true = plt.Circle((0, 0), true_strength, color="green", fill=False, linewidth=2)
        ax_right.add_patch(circle_pred)
        ax_right.add_patch(circle_true)

        ax_right.arrow(
            0,
            0,
            pred_strength * np.cos(pred_angle),
            pred_strength * np.sin(pred_angle),
            head_width=0.1,
            head_length=0.1,
            fc="blue",
            ec="blue",
            linewidth=2,
            label="Pred",
        )
        ax_right.arrow(
            0,
            0,
            true_strength * np.cos(true_angle),
            true_strength * np.sin(true_angle),
            head_width=0.1,
            head_length=0.1,
            fc="green",
            ec="green",
            linestyle="--",
            linewidth=2,
            label="True",
        )

        ax_right.set_title(
            f"Intent\n"
            f"Strength: {pred_strength:.3f} vs {true_strength:.3f}\n"
            f"Direction: {np.rad2deg(pred_angle):.1f}° vs {np.rad2deg(true_angle):.1f}°"
        )
        ax_right.grid(True, alpha=0.3)

    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "intent_pattern_examples.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Intent pattern examples saved to {fig_path}")


def analyze_feature_importance(
    model: torch.nn.Module,
    test_samples: List[Dict],
    base_accuracy: float,
    device: torch.device,
    output_dir: str,
) -> Dict[str, float]:
    """使用 permutation importance 分析特征重要性."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    feature_names = ["Fx", "Fy", "||F||", "offset_x", "offset_y", "vel_x", "vel_y"]
    importances: Dict[str, float] = {}

    from .evaluate import evaluate_direction_accuracy

    print("\n进行特征重要性分析 (Permutation Importance)...")

    for feat_idx, feat_name in enumerate(feature_names):
        print(f"  测试特征: {feat_name}")

        corrupted_samples: List[Dict] = []
        for s in test_samples:
            cs = deepcopy(s)
            vals = cs["input"][:, feat_idx]
            np.random.shuffle(vals)
            cs["input"][:, feat_idx] = vals
            corrupted_samples.append(cs)

        loader = DataLoader(
            IntentDataset(corrupted_samples),
            batch_size=128,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        acc = evaluate_direction_accuracy(
            model, loader, device, output_dir=os.path.join(output_dir, "tmp")
        )
        importance = float(base_accuracy - acc)
        importances[feat_name] = importance
        print(f"    Accuracy drop: {importance * 100:.2f}%")

    # 可视化
    names = list(importances.keys())
    drops = [importances[n] for n in names]
    plt.figure(figsize=(7, 5))
    plt.barh(names, drops, color="steelblue", edgecolor="black")
    plt.xlabel("Accuracy Drop")
    plt.title("Feature Importance (Permutation)")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Feature importance plot saved to {fig_path}")

    print("\nFeature Importance Ranking:")
    for name in sorted(importances, key=importances.get, reverse=True):
        print(f"  {name}: {importances[name] * 100:.2f}%")

    return importances


