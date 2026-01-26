"""Phase 2a: Intent Prediction Network 训练主流程."""

import argparse
import json
import os

import numpy as np
import torch

from .data_preprocessing import (
    load_and_preprocess_data,
    split_data,
    create_dataloaders,
)
from .model import IntentNet, count_parameters
from .train import train_model, get_device
from .evaluate import (
    evaluate_direction_accuracy,
    evaluate_strength_correlation,
    check_phase2a_milestones,
)
from .visualize import visualize_intent_patterns, analyze_feature_importance


def save_results(
    model: IntentNet,
    norm_params: dict,
    history: dict,
    training_time: float,
    direction_acc: float,
    strength_corr: float,
    num_samples: dict,
    feature_importance: dict,
    output_dir: str,
) -> None:
    """保存模型与报告."""
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "intent_net.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "norm_params": norm_params,
            "train_loss_history": history.get("train_loss", []),
            "val_loss_history": history.get("val_loss", []),
        },
        model_path,
    )
    print(f"模型已保存: {model_path}")

    report = {
        "direction_accuracy": float(direction_acc),
        "strength_correlation": float(strength_corr),
        "training_time_sec": float(training_time),
        "num_parameters": int(count_parameters(model)),
        "num_samples": num_samples,
        "feature_importance": {k: float(v) for k, v in feature_importance.items()},
    }
    report_path = os.path.join(output_dir, "phase2a_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"报告已保存: {report_path}")


def run_phase2a(args: argparse.Namespace) -> bool:
    print("=" * 60)
    print("Phase 2a: Intent Network Training")
    print("=" * 60)

    device = get_device()

    # Step 1: 数据预处理
    print("\n[Step 1/5] Data Preprocessing...")
    all_samples, norm_params = load_and_preprocess_data(args.data_dir)
    train_samples, val_samples, test_samples = split_data(all_samples)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples,
        val_samples,
        test_samples,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # Step 2: 创建模型
    print("\n[Step 2/5] Model Creation...")
    model = IntentNet(dropout=args.dropout)
    print(f"Model parameters: {count_parameters(model):,}")

    # Step 3: 训练
    print("\n[Step 3/5] Training...")
    model, history, train_time = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
    )

    # Step 4: 评估
    print("\n[Step 4/5] Evaluation...")
    figures_dir = os.path.join(args.output_dir, "figures")
    direction_acc = evaluate_direction_accuracy(
        model, test_loader, device, output_dir=figures_dir
    )
    strength_corr = evaluate_strength_correlation(
        model, test_loader, device, output_dir=figures_dir
    )

    # Step 5: 可视化
    print("\n[Step 5/5] Visualization & Analysis...")
    visualize_intent_patterns(model, test_samples, device, output_dir=figures_dir)
    feature_importance = analyze_feature_importance(
        model, test_samples, base_accuracy=direction_acc, device=device, output_dir=figures_dir
    )

    # Milestone 检查
    passed = check_phase2a_milestones(direction_acc, strength_corr)

    num_samples = {
        "train": len(train_samples),
        "val": len(val_samples),
        "test": len(test_samples),
    }
    save_results(
        model,
        norm_params,
        history,
        train_time,
        direction_acc,
        strength_corr,
        num_samples,
        feature_importance,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("Phase 2a Complete!")
    print("=" * 60)
    return passed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 2a: Intent Prediction Network Training"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(
            "..",
            "code",
            "Mixed Reality",
            "Sensapex",
            "camera_image_processing",
            "robot_data",
        ),
        help="输入CSV数据目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="输出目录 (模型与图表)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_phase2a(args)


if __name__ == "__main__":
    main()


