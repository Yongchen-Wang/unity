"""
可视化模块
生成各种轨迹和指标的可视化图表
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')


def plot_all_trajectories(original_trajectories: List[Dict],
                         aligned_trajectories: List[Dict],
                         consensus: Dict,
                         output_path: str = 'all_trajectories.png',
                         figsize: Tuple[int, int] = (12, 10)):
    """
    绘制所有轨迹 + Consensus轨迹
    
    Args:
        original_trajectories: 原始轨迹列表
        aligned_trajectories: 对齐后的轨迹列表
        consensus: consensus轨迹字典
        output_path: 输出文件路径
        figsize: 图表大小
    """
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))
    
    # 左图：原始轨迹
    ax1 = axes[0]
    for traj in original_trajectories:
        ax1.plot(traj['x'], traj['y'], alpha=0.3, linewidth=1, color='gray')
    ax1.set_title('原始轨迹', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (mm)', fontsize=12)
    ax1.set_ylabel('Y (mm)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # 右图：对齐后的轨迹 + Consensus
    ax2 = axes[1]
    
    # 绘制对齐后的轨迹（浅色半透明）
    for traj in aligned_trajectories:
        ax2.plot(traj['x'], traj['y'], alpha=0.2, linewidth=1, color='lightblue')
    
    # 绘制置信区间（阴影）
    consensus_x = consensus['x']
    consensus_y = consensus['y']
    x_std = consensus['x_std']
    y_std = consensus['y_std']
    
    # 使用百分位数作为置信区间（更鲁棒）
    x_q25 = consensus.get('x_q25', consensus_x - x_std)
    x_q75 = consensus.get('x_q75', consensus_x + x_std)
    y_q25 = consensus.get('y_q25', consensus_y - y_std)
    y_q75 = consensus.get('y_q75', consensus_y + y_std)
    
    # 填充置信区间（近似）
    ax2.fill_betweenx(consensus_y, x_q25, x_q75, alpha=0.2, color='red', label='置信区间')
    
    # 绘制Consensus轨迹（粗红线）
    ax2.plot(consensus_x, consensus_y, 'r-', linewidth=3, label='Consensus轨迹', zorder=10)
    
    # 标记起点和终点
    ax2.plot(consensus_x[0], consensus_y[0], 'go', markersize=10, label='起点', zorder=11)
    ax2.plot(consensus_x[-1], consensus_y[-1], 'ro', markersize=10, label='终点', zorder=11)
    
    ax2.set_title('对齐后的轨迹 + Consensus', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (mm)', fontsize=12)
    ax2.set_ylabel('Y (mm)', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  图表已保存: {output_path}")


def plot_dtw_alignment_comparison(original_trajectories: List[Dict],
                                  aligned_trajectories: List[Dict],
                                  output_path: str = 'dtw_alignment.png',
                                  figsize: Tuple[int, int] = (14, 10)):
    """
    绘制DTW对齐前后对比图
    
    Args:
        original_trajectories: 原始轨迹列表
        aligned_trajectories: 对齐后的轨迹列表
        output_path: 输出文件路径
        figsize: 图表大小
    """
    fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*2))
    
    # 上图：对齐前的轨迹
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(original_trajectories)))
    for i, traj in enumerate(original_trajectories):
        ax1.plot(traj['x'], traj['y'], alpha=0.7, linewidth=2, 
                color=colors[i], label=traj['filename'][:20])
    
    ax1.set_title('对齐前的轨迹', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (mm)', fontsize=12)
    ax1.set_ylabel('Y (mm)', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # 下图：对齐后的轨迹
    ax2 = axes[1]
    for i, traj in enumerate(aligned_trajectories):
        ax2.plot(traj['x'], traj['y'], alpha=0.7, linewidth=2, 
                color=colors[i], label=traj['filename'][:20])
    
    ax2.set_title('对齐后的轨迹', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (mm)', fontsize=12)
    ax2.set_ylabel('Y (mm)', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  图表已保存: {output_path}")


def plot_velocity_profile(consensus: Dict,
                         output_path: str = 'velocity_profile.png',
                         figsize: Tuple[int, int] = (12, 8)):
    """
    绘制Consensus速度曲线
    
    Args:
        consensus: consensus轨迹字典
        output_path: 输出文件路径
        figsize: 图表大小
    """
    if 'velocity_magnitude' not in consensus:
        print("  警告: consensus中无速度信息，跳过速度曲线绘制")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]))
    
    t = consensus['t']
    velocity = consensus['velocity_magnitude']
    
    # 上图：速度大小
    ax1 = axes[0]
    ax1.plot(t, velocity, 'b-', linewidth=2, label='速度大小')
    ax1.fill_between(t, 0, velocity, alpha=0.3, color='blue')
    ax1.set_xlabel('归一化时间 [0,1]', fontsize=12)
    ax1.set_ylabel('速度 (mm/s)', fontsize=12)
    ax1.set_title('Consensus速度曲线', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 添加统计信息
    max_vel = np.max(velocity)
    mean_vel = np.mean(velocity)
    ax1.axhline(mean_vel, color='r', linestyle='--', linewidth=1, label=f'平均速度: {mean_vel:.2f} mm/s')
    ax1.axhline(max_vel, color='g', linestyle='--', linewidth=1, label=f'最大速度: {max_vel:.2f} mm/s')
    ax1.legend(fontsize=10)
    
    # 下图：如果有加速度信息
    if 'acceleration' in consensus:
        acceleration = consensus['acceleration']
        ax2 = axes[1]
        ax2.plot(t, acceleration, 'r-', linewidth=2, label='加速度')
        ax2.fill_between(t, 0, acceleration, alpha=0.3, color='red')
        ax2.set_xlabel('归一化时间 [0,1]', fontsize=12)
        ax2.set_ylabel('加速度 (mm/s^2)', fontsize=12)
        ax2.set_title('加速度曲线', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
    else:
        # 如果没有加速度，绘制速度分量
        if 'vx' in consensus and 'vy' in consensus:
            ax2 = axes[1]
            ax2.plot(t, consensus['vx'], 'g-', linewidth=2, label='X方向速度', alpha=0.7)
            ax2.plot(t, consensus['vy'], 'm-', linewidth=2, label='Y方向速度', alpha=0.7)
            ax2.set_xlabel('归一化时间 [0,1]', fontsize=12)
            ax2.set_ylabel('速度 (mm/s)', fontsize=12)
            ax2.set_title('速度分量', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
        else:
            axes[1].remove()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  图表已保存: {output_path}")


def plot_confidence_heatmap(consensus: Dict,
                           output_path: str = 'confidence_heatmap.png',
                           figsize: Tuple[int, int] = (12, 10)):
    """
    绘制置信区间热力图
    
    Args:
        consensus: consensus轨迹字典
        output_path: 输出文件路径
        figsize: 图表大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = consensus['x']
    y = consensus['y']
    x_std = consensus['x_std']
    y_std = consensus['y_std']
    
    # 计算总标准差（欧氏距离）
    total_std = np.sqrt(x_std**2 + y_std**2)
    
    # 绘制轨迹，颜色表示标准差
    scatter = ax.scatter(x, y, c=total_std, cmap='YlOrRd', 
                        s=50, alpha=0.8, edgecolors='black', linewidths=0.5)
    
    # 绘制轨迹线
    ax.plot(x, y, 'k-', linewidth=2, alpha=0.5, zorder=0)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('标准差 (mm)', fontsize=12)
    
    # 标记起点和终点
    ax.plot(x[0], y[0], 'go', markersize=12, label='起点', zorder=10)
    ax.plot(x[-1], y[-1], 'ro', markersize=12, label='终点', zorder=10)
    
    ax.set_title('置信区间热力图（颜色表示标准差）', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  图表已保存: {output_path}")


def plot_metrics_summary(metrics: Dict,
                        output_path: str = 'metrics_summary.png',
                        figsize: Tuple[int, int] = (12, 8)):
    """
    绘制验证指标总结表格
    
    Args:
        metrics: 验证指标字典
        output_path: 输出文件路径
        figsize: 图表大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    table_data = []
    
    # 对齐质量指标
    table_data.append(['对齐质量指标', '', ''])
    table_data.append(['平均DTW距离', f"{metrics.get('avg_dtw_distance', np.nan):.3f} mm", 
                      '✓' if metrics.get('avg_dtw_distance', np.inf) < 5.0 else '✗'])
    table_data.append(['平均轨迹间DTW距离', f"{metrics.get('avg_pairwise_dtw', np.nan):.3f} mm", '-'])
    table_data.append(['对齐质量分数', f"{metrics.get('alignment_quality', np.nan):.3f}", '-'])
    table_data.append(['起点对齐误差', f"{metrics.get('start_error', np.nan):.3f} mm",
                      '✓' if metrics.get('start_error', np.inf) < 2.0 else '✗'])
    table_data.append(['终点对齐误差', f"{metrics.get('end_error', np.nan):.3f} mm",
                      '✓' if metrics.get('end_error', np.inf) < 2.0 else '✗'])
    
    table_data.append(['', '', ''])  # 空行
    
    # Consensus一致性指标
    table_data.append(['Consensus一致性指标', '', ''])
    table_data.append(['平均点标准差', f"{metrics.get('avg_std', np.nan):.3f} mm",
                      '✓' if metrics.get('avg_std', np.inf) < 2.0 else '✗'])
    table_data.append(['最大点标准差', f"{metrics.get('max_point_std', np.nan):.3f} mm", '-'])
    table_data.append(['平均偏差', f"{metrics.get('mean_deviation', np.nan):.3f} mm", '-'])
    table_data.append(['最大偏差', f"{metrics.get('max_deviation', np.nan):.3f} mm", '-'])
    
    table_data.append(['', '', ''])  # 空行
    
    # 平滑度指标
    table_data.append(['平滑度指标', '', ''])
    table_data.append(['平均曲率', f"{metrics.get('avg_curvature', np.nan):.6f}", '-'])
    table_data.append(['最大曲率', f"{metrics.get('max_curvature', np.nan):.6f}", '-'])
    table_data.append(['平均加速度', f"{metrics.get('avg_acceleration', np.nan):.3f} mm/s^2", '-'])
    
    table_data.append(['', '', ''])  # 空行
    
    # 速度连续性指标
    if 'avg_jerk' in metrics and not np.isnan(metrics['avg_jerk']):
        table_data.append(['速度连续性指标', '', ''])
        table_data.append(['平均Jerk', f"{metrics.get('avg_jerk', np.nan):.3f} mm/s^3", '-'])
        table_data.append(['最大Jerk', f"{metrics.get('max_jerk', np.nan):.3f} mm/s^3", '-'])
        table_data.append(['速度平滑度', f"{metrics.get('velocity_smoothness', np.nan):.3f}", '-'])
    
    # 创建表格
    table = ax.table(cellText=table_data,
                    colLabels=['指标', '值', '状态'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.5, 0.3, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置标题行样式
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置指标名称行样式（加粗）
    for i, row in enumerate(table_data):
        if row[1] == '' and row[2] == '' and row[0] != '':
            for j in range(3):
                table[(i+1, j)].set_facecolor('#E0E0E0')
                table[(i+1, j)].set_text_props(weight='bold')
    
    # 设置状态列颜色
    for i, row in enumerate(table_data):
        if row[2] == '✓':
            table[(i+1, 2)].set_facecolor('#C8E6C9')
        elif row[2] == '✗':
            table[(i+1, 2)].set_facecolor('#FFCDD2')
    
    # 总体状态
    all_pass = metrics.get('all_pass', False)
    status_text = '✓ 总体验证通过' if all_pass else '✗ 总体验证未通过'
    status_color = '#C8E6C9' if all_pass else '#FFCDD2'
    
    ax.text(0.5, 0.02, status_text, transform=ax.transAxes,
           fontsize=14, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.8))
    
    plt.title('验证指标总结', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  图表已保存: {output_path}")


def visualize_all(original_trajectories: List[Dict],
                 aligned_trajectories: List[Dict],
                 consensus: Dict,
                 metrics: Dict,
                 output_dir: str = './consensus_output'):
    """
    生成所有可视化图表
    
    Args:
        original_trajectories: 原始轨迹列表
        aligned_trajectories: 对齐后的轨迹列表
        consensus: consensus轨迹字典
        metrics: 验证指标字典
        output_dir: 输出目录
    """
    print(f"\n{'='*70}")
    print("生成可视化图表")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成所有图表
    plot_all_trajectories(
        original_trajectories, aligned_trajectories, consensus,
        os.path.join(output_dir, 'all_trajectories.png')
    )
    
    plot_dtw_alignment_comparison(
        original_trajectories, aligned_trajectories,
        os.path.join(output_dir, 'dtw_alignment.png')
    )
    
    plot_velocity_profile(
        consensus,
        os.path.join(output_dir, 'velocity_profile.png')
    )
    
    plot_confidence_heatmap(
        consensus,
        os.path.join(output_dir, 'confidence_heatmap.png')
    )
    
    plot_metrics_summary(
        metrics,
        os.path.join(output_dir, 'metrics_summary.png')
    )
    
    print(f"\n所有图表已保存到: {output_dir}")

