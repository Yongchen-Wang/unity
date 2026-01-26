"""
主流程入口
整合所有模块，执行完整的consensus轨迹生成流程
"""

import argparse
import os
import json
import pickle
import sys
import numpy as np
from datetime import datetime

# 导入所有模块
from data_loader import load_and_preprocess_data
from trajectory_alignment import align_trajectories
from consensus_builder import build_consensus, compute_velocity_profile, calculate_confidence_coverage
from validation import validate_consensus
from visualization import visualize_all
from bc_data_generator import generate_bc_training_data, save_training_data


def generate_alignment_report(alignment_metadata: dict, output_path: str):
    """生成对齐质量报告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("轨迹对齐质量报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"对齐方法: {alignment_metadata.get('method', 'N/A')}\n")
        f.write(f"参考轨迹索引: {alignment_metadata.get('reference_idx', 'N/A')}\n")
        f.write(f"参考轨迹文件名: {alignment_metadata.get('reference_filename', 'N/A')}\n")
        f.write(f"重采样点数: {alignment_metadata.get('n_samples', 'N/A')}\n")
        f.write(f"DTW窗口比例: {alignment_metadata.get('window_ratio', 'N/A')}\n\n")
        
        f.write("对齐距离统计:\n")
        f.write("-" * 70 + "\n")
        f.write(f"平均DTW距离: {alignment_metadata.get('avg_dtw_distance', 0):.3f} mm\n")
        f.write(f"平均轨迹间DTW距离: {alignment_metadata.get('avg_pairwise_dtw', 0):.3f} mm\n")
        f.write(f"对齐质量分数: {alignment_metadata.get('alignment_quality', 0):.3f}\n\n")
        
        f.write("各轨迹对齐距离:\n")
        f.write("-" * 70 + "\n")
        distances = alignment_metadata.get('alignment_distances', [])
        for i, dist in enumerate(distances):
            f.write(f"轨迹 {i+1}: {dist:.3f} mm\n")


def save_results(consensus: dict, bc_data: dict, metrics: dict, 
                alignment_metadata: dict, output_dir: str):
    """保存所有结果"""
    print(f"\n{'='*70}")
    print("保存结果")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存consensus轨迹
    consensus_path = os.path.join(output_dir, 'consensus_trajectory.pkl')
    with open(consensus_path, 'wb') as f:
        pickle.dump(consensus, f)
        print(f"  [OK] Consensus轨迹: {consensus_path}")
    
    # 2. 保存BC训练数据
    bc_data_path = os.path.join(output_dir, 'bc_training_data.npy')
    save_training_data(bc_data, bc_data_path)
    print(f"  [OK] BC训练数据: {bc_data_path}")
    
    # 3. 保存对齐报告
    alignment_report_path = os.path.join(output_dir, 'alignment_report.txt')
    generate_alignment_report(alignment_metadata, alignment_report_path)
    print(f"  [OK] 对齐报告: {alignment_report_path}")
    
    # 4. 保存验证指标
    metrics_path = os.path.join(output_dir, 'validation_metrics.json')
    # 将numpy类型转换为Python原生类型以便JSON序列化
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            metrics_serializable[key] = {}
            for k, v in value.items():
                if isinstance(v, (np.floating, np.integer, np.bool_)):
                    metrics_serializable[key][k] = float(v) if not isinstance(v, np.bool_) else bool(v)
                elif isinstance(v, bool):
                    metrics_serializable[key][k] = v
                elif isinstance(v, np.ndarray):
                    metrics_serializable[key][k] = v.tolist()
                else:
                    metrics_serializable[key][k] = v
        elif isinstance(value, (np.floating, np.integer)):
            metrics_serializable[key] = float(value)
        elif isinstance(value, (np.bool_, bool)):
            metrics_serializable[key] = bool(value)
        elif isinstance(value, np.ndarray):
            metrics_serializable[key] = value.tolist()
        else:
            metrics_serializable[key] = value
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)
    print(f"  [OK] 验证指标: {metrics_path}")
    
    print(f"\n所有结果已保存到: {output_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Consensus轨迹生成系统 - 从多组导航数据生成consensus轨迹',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py --input ./robot_data --output ./consensus_output
  python main.py --input ./robot_data --output ./output --exclude 140802
  python main.py --input ./robot_data --output ./output --n-samples 150
        """
    )
    
    # 必需参数
    parser.add_argument('--input', type=str, required=True,
                       help='输入数据目录（包含CSV文件）')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录')
    
    # 可选参数
    parser.add_argument('--exclude', type=str, nargs='+', default=[],
                       help='要排除的文件名列表（例如: 140802）')
    parser.add_argument('--n-samples', type=int, default=100,
                       help='重采样点数（默认: 100）')
    parser.add_argument('--smooth-window', type=int, default=11,
                       help='平滑窗口长度（默认: 11）')
    parser.add_argument('--velocity-threshold', type=float, default=0.3,
                       help='速度阈值，低于此值的点将被移除（默认: 0.3 mm/s）')
    parser.add_argument('--min-length', type=float, default=20.0,
                       help='最小轨迹长度（默认: 20.0 mm）')
    parser.add_argument('--reference-method', type=str, default='median',
                       choices=['median', 'longest', 'shortest'],
                       help='参考轨迹选择方法（默认: median）')
    parser.add_argument('--consensus-method', type=str, default='median',
                       choices=['median', 'mean'],
                       help='Consensus生成方法（默认: median）')
    parser.add_argument('--window-ratio', type=float, default=0.2,
                       help='DTW窗口比例（默认: 0.2）')
    parser.add_argument('--state-dim', type=int, default=14,
                       help='BC训练状态维度（默认: 14）')
    parser.add_argument('--action-dim', type=int, default=2,
                       help='BC训练动作维度（默认: 2）')
    parser.add_argument('--no-visualization', action='store_true',
                       help='跳过可视化')
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("=" * 70)
    print("Consensus轨迹生成系统")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"排除文件: {args.exclude if args.exclude else '无'}")
    print("=" * 70)
    
    try:
        # 步骤1: 数据加载和预处理
        print("\n[步骤 1/7] 数据加载和预处理")
        trajectories, load_metadata = load_and_preprocess_data(
            data_dir=args.input,
            exclude=args.exclude,
            smooth_window=args.smooth_window,
            velocity_threshold=args.velocity_threshold,
            min_trajectory_length=args.min_length,
            align_start=True
        )
        
        if len(trajectories) == 0:
            print("\n[ERROR] 错误: 没有有效的轨迹数据")
            sys.exit(1)
        
        # 保存原始轨迹用于可视化
        original_trajectories = [t.copy() for t in trajectories]
        
        # 步骤2: 轨迹对齐
        print("\n[步骤 2/7] 轨迹对齐")
        aligned_trajectories, alignment_metadata = align_trajectories(
            trajectories=trajectories,
            method='dtw',
            reference_method=args.reference_method,
            n_samples=args.n_samples,
            window_ratio=args.window_ratio,
            resample_kind='cubic'
        )
        
        # 步骤3: Consensus生成
        print("\n[步骤 3/7] 生成Consensus轨迹")
        consensus = build_consensus(
            trajectories=aligned_trajectories,
            method=args.consensus_method,
            confidence_level=1.0,
            smooth_window=args.smooth_window,
            smooth_polyorder=3
        )
        
        # 步骤4: 计算速度Profile
        print("\n[步骤 4/7] 计算速度Profile")
        consensus = compute_velocity_profile(
            consensus=consensus,
            smooth=True,
            smooth_window=args.smooth_window,
            smooth_polyorder=3
        )
        
        # 计算置信区间覆盖
        coverage = calculate_confidence_coverage(consensus, aligned_trajectories, confidence_level=1.0)
        print(f"\n置信区间覆盖率 (±1 std): {coverage*100:.1f}%")
        consensus['confidence_coverage'] = coverage
        
        # 步骤5: 验证
        print("\n[步骤 5/7] 验证Consensus轨迹")
        thresholds = {
            'avg_std': 2.0,
            'dtw': 5.0,
            'avg_dtw_distance': 5.0,
            'start_error': 2.0,
            'end_error': 2.0
        }
        metrics = validate_consensus(
            consensus=consensus,
            aligned_trajectories=aligned_trajectories,
            alignment_metadata=alignment_metadata,
            thresholds=thresholds
        )
        
        # 步骤6: 可视化
        if not args.no_visualization:
            print("\n[步骤 6/7] 生成可视化图表")
            visualize_all(
                original_trajectories=original_trajectories,
                aligned_trajectories=aligned_trajectories,
                consensus=consensus,
                metrics=metrics,
                output_dir=args.output
            )
        else:
            print("\n[步骤 6/7] 跳过可视化")
        
        # 步骤7: 生成BC训练数据
        print("\n[步骤 7/7] 生成BC训练数据")
        bc_data = generate_bc_training_data(
            consensus=consensus,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            include_velocity=True,
            include_acceleration=False
        )
        
        # 保存所有结果
        save_results(
            consensus=consensus,
            bc_data=bc_data,
            metrics=metrics,
            alignment_metadata=alignment_metadata,
            output_dir=args.output
        )
        
        # 完成总结
        print("\n" + "=" * 70)
        print("处理完成！")
        print("=" * 70)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n生成的文件:")
        print(f"  - consensus_trajectory.pkl")
        print(f"  - bc_training_data.npy")
        print(f"  - alignment_report.txt")
        print(f"  - validation_metrics.json")
        if not args.no_visualization:
            print(f"  - all_trajectories.png")
            print(f"  - dtw_alignment.png")
            print(f"  - velocity_profile.png")
            print(f"  - confidence_heatmap.png")
            print(f"  - metrics_summary.png")
        print(f"\n输出目录: {args.output}")
        
        # 验证结果
        all_pass = metrics.get('all_pass', False)
        if all_pass:
            print("\n[PASS] 总体验证通过！")
        else:
            print("\n[WARNING] 总体验证未完全通过，请检查验证指标")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

