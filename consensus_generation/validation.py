"""
验证指标计算模块
计算对齐质量、一致性、平滑度等验证指标
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def calculate_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    计算轨迹的曲率
    
    Args:
        x: x坐标数组
        y: y坐标数组
    
    Returns:
        曲率数组
    """
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # 曲率公式: k = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**1.5
    
    # 避免除零
    denominator = np.clip(denominator, 1e-10, None)
    
    curvature = numerator / denominator
    
    return curvature


def calculate_alignment_quality(aligned_trajectories: List[Dict],
                                alignment_metadata: Dict) -> Dict:
    """
    计算对齐质量指标
    
    Args:
        aligned_trajectories: 对齐后的轨迹列表
        alignment_metadata: 对齐元数据
    
    Returns:
        对齐质量指标字典
    """
    print(f"\n{'='*70}")
    print("计算对齐质量指标")
    print(f"{'='*70}")
    
    # 从metadata中获取
    avg_dtw_distance = alignment_metadata.get('avg_dtw_distance', np.nan)
    avg_pairwise_dtw = alignment_metadata.get('avg_pairwise_dtw', np.nan)
    alignment_quality = alignment_metadata.get('alignment_quality', np.nan)
    
    # 计算额外的指标
    # 1. 对齐后轨迹的点标准差（在每个时间点）
    n_samples = len(aligned_trajectories[0]['x'])
    all_x = np.array([traj['x'] for traj in aligned_trajectories])
    all_y = np.array([traj['y'] for traj in aligned_trajectories])
    
    point_std_x = np.std(all_x, axis=0)
    point_std_y = np.std(all_y, axis=0)
    point_std_total = np.sqrt(point_std_x**2 + point_std_y**2)
    
    avg_point_std = np.mean(point_std_total)
    max_point_std = np.max(point_std_total)
    
    # 2. 起点和终点对齐误差
    start_positions_x = all_x[:, 0]
    start_positions_y = all_y[:, 0]
    end_positions_x = all_x[:, -1]
    end_positions_y = all_y[:, -1]
    
    start_error_x = np.std(start_positions_x)
    start_error_y = np.std(start_positions_y)
    end_error_x = np.std(end_positions_x)
    end_error_y = np.std(end_positions_y)
    
    start_error_total = np.sqrt(start_error_x**2 + start_error_y**2)
    end_error_total = np.sqrt(end_error_x**2 + end_error_y**2)
    
    metrics = {
        'avg_dtw_distance': avg_dtw_distance,
        'avg_pairwise_dtw': avg_pairwise_dtw,
        'alignment_quality': alignment_quality,
        'avg_point_std': avg_point_std,
        'max_point_std': max_point_std,
        'start_error': start_error_total,
        'end_error': end_error_total,
        'start_error_x': start_error_x,
        'start_error_y': start_error_y,
        'end_error_x': end_error_x,
        'end_error_y': end_error_y
    }
    
    print(f"  平均DTW距离: {avg_dtw_distance:.3f} mm")
    print(f"  平均轨迹间DTW距离: {avg_pairwise_dtw:.3f} mm")
    print(f"  对齐质量分数: {alignment_quality:.3f}")
    print(f"  平均点标准差: {avg_point_std:.3f} mm")
    print(f"  最大点标准差: {max_point_std:.3f} mm")
    print(f"  起点对齐误差: {start_error_total:.3f} mm")
    print(f"  终点对齐误差: {end_error_total:.3f} mm")
    
    return metrics


def calculate_consensus_consistency(consensus: Dict,
                                   aligned_trajectories: List[Dict]) -> Dict:
    """
    计算Consensus一致性指标
    
    Args:
        consensus: consensus轨迹字典
        aligned_trajectories: 对齐后的轨迹列表
    
    Returns:
        一致性指标字典
    """
    print(f"\n计算Consensus一致性指标...")
    
    # 从consensus中获取
    avg_std = consensus.get('avg_std', np.nan)
    x_std = consensus['x_std']
    y_std = consensus['y_std']
    
    # 计算每个轨迹与consensus的偏差
    consensus_x = consensus['x']
    consensus_y = consensus['y']
    
    deviations = []
    for traj in aligned_trajectories:
        traj_x = traj['x']
        traj_y = traj['y']
        
        # 计算每个点的偏差
        dx = traj_x - consensus_x
        dy = traj_y - consensus_y
        distances = np.sqrt(dx**2 + dy**2)
        
        deviations.append(distances)
    
    deviations = np.array(deviations)  # (n_traj, n_samples)
    
    # 统计信息
    mean_deviation = np.mean(deviations)
    max_deviation = np.max(deviations)
    median_deviation = np.median(deviations)
    
    # 计算每个时间点的偏差
    point_mean_deviation = np.mean(deviations, axis=0)
    point_max_deviation = np.max(deviations, axis=0)
    
    # 计算覆盖范围（80%轨迹点）
    percentile_80 = np.percentile(deviations, 80)
    
    metrics = {
        'avg_std': avg_std,
        'mean_deviation': mean_deviation,
        'median_deviation': median_deviation,
        'max_deviation': max_deviation,
        'percentile_80_deviation': percentile_80,
        'avg_x_std': np.mean(x_std),
        'avg_y_std': np.mean(y_std),
        'max_x_std': np.max(x_std),
        'max_y_std': np.max(y_std)
    }
    
    print(f"  平均标准差: {avg_std:.3f} mm")
    print(f"  平均偏差: {mean_deviation:.3f} mm")
    print(f"  中位数偏差: {median_deviation:.3f} mm")
    print(f"  最大偏差: {max_deviation:.3f} mm")
    print(f"  80%分位数偏差: {percentile_80:.3f} mm")
    
    return metrics


def calculate_smoothness(consensus: Dict) -> Dict:
    """
    计算轨迹平滑度指标
    
    Args:
        consensus: consensus轨迹字典
    
    Returns:
        平滑度指标字典
    """
    print(f"\n计算轨迹平滑度指标...")
    
    x = consensus['x']
    y = consensus['y']
    
    # 1. 曲率
    curvature = calculate_curvature(x, y)
    avg_curvature = np.mean(np.abs(curvature))
    max_curvature = np.max(np.abs(curvature))
    
    # 2. 曲率变化率（jerk）
    curvature_change = np.diff(curvature)
    avg_curvature_change = np.mean(np.abs(curvature_change))
    max_curvature_change = np.max(np.abs(curvature_change))
    
    # 3. 位置二阶导数（加速度）
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    acceleration_magnitude = np.sqrt(ddx**2 + ddy**2)
    avg_acceleration = np.mean(acceleration_magnitude)
    max_acceleration = np.max(acceleration_magnitude)
    
    # 4. 速度连续性（如果有速度信息）
    if 'velocity_magnitude' in consensus:
        velocity = consensus['velocity_magnitude']
        velocity_change = np.diff(velocity)
        avg_velocity_change = np.mean(np.abs(velocity_change))
        max_velocity_change = np.max(np.abs(velocity_change))
    else:
        avg_velocity_change = np.nan
        max_velocity_change = np.nan
    
    metrics = {
        'avg_curvature': avg_curvature,
        'max_curvature': max_curvature,
        'avg_curvature_change': avg_curvature_change,
        'max_curvature_change': max_curvature_change,
        'avg_acceleration': avg_acceleration,
        'max_acceleration': max_acceleration,
        'avg_velocity_change': avg_velocity_change,
        'max_velocity_change': max_velocity_change
    }
    
    print(f"  平均曲率: {avg_curvature:.6f}")
    print(f"  最大曲率: {max_curvature:.6f}")
    print(f"  平均曲率变化: {avg_curvature_change:.6f}")
    print(f"  平均加速度: {avg_acceleration:.3f} mm/s^2")
    
    return metrics


def calculate_velocity_continuity(consensus: Dict) -> Dict:
    """
    计算速度连续性指标
    
    Args:
        consensus: consensus轨迹字典（必须包含速度信息）
    
    Returns:
        速度连续性指标字典
    """
    print(f"\n计算速度连续性指标...")
    
    if 'velocity_magnitude' not in consensus:
        print("  警告: consensus中无速度信息，跳过速度连续性计算")
        return {
            'avg_acceleration': np.nan,
            'max_acceleration': np.nan,
            'avg_jerk': np.nan,
            'max_jerk': np.nan,
            'velocity_smoothness': np.nan
        }
    
    velocity = consensus['velocity_magnitude']
    
    # 计算加速度
    if 'acceleration' in consensus:
        acceleration = consensus['acceleration']
    else:
        dt = consensus['t'][1] - consensus['t'][0] if len(consensus['t']) > 1 else 1.0
        dv = np.diff(velocity)
        acceleration = np.concatenate([[0], dv / dt])
    
    avg_acceleration = np.mean(acceleration)
    max_acceleration = np.max(acceleration)
    
    # 计算jerk（加速度的变化率）
    dt = consensus['t'][1] - consensus['t'][0] if len(consensus['t']) > 1 else 1.0
    jerk = np.diff(acceleration) / dt
    jerk = np.concatenate([[0], jerk])
    
    avg_jerk = np.mean(np.abs(jerk))
    max_jerk = np.max(np.abs(jerk))
    
    # 速度平滑度：速度变化的方差（越小越平滑）
    velocity_variance = np.var(np.diff(velocity))
    
    metrics = {
        'avg_acceleration': avg_acceleration,
        'max_acceleration': max_acceleration,
        'avg_jerk': avg_jerk,
        'max_jerk': max_jerk,
        'velocity_smoothness': 1.0 / (1.0 + velocity_variance)  # 归一化平滑度分数
    }
    
    print(f"  平均加速度: {avg_acceleration:.3f} mm/s^2")
    print(f"  最大加速度: {max_acceleration:.3f} mm/s^2")
    print(f"  平均Jerk: {avg_jerk:.3f} mm/s^3")
    print(f"  最大Jerk: {max_jerk:.3f} mm/s^3")
    print(f"  速度平滑度: {metrics['velocity_smoothness']:.3f}")
    
    return metrics


def validate_consensus(consensus: Dict,
                      aligned_trajectories: List[Dict],
                      alignment_metadata: Dict,
                      thresholds: Optional[Dict] = None) -> Dict:
    """
    完整验证流程
    
    Args:
        consensus: consensus轨迹字典
        aligned_trajectories: 对齐后的轨迹列表
        alignment_metadata: 对齐元数据
        thresholds: 验证阈值字典，例如：
            {'avg_std': 2.0, 'dtw': 5.0, 'avg_dtw_distance': 5.0}
    
    Returns:
        完整验证指标字典，包含所有验证结果和pass/fail状态
    """
    if thresholds is None:
        thresholds = {
            'avg_std': 2.0,
            'dtw': 5.0,
            'avg_dtw_distance': 5.0,
            'start_error': 2.0,
            'end_error': 2.0
        }
    
    print(f"\n{'='*70}")
    print("验证Consensus轨迹")
    print(f"{'='*70}")
    
    # 1. 对齐质量
    alignment_metrics = calculate_alignment_quality(aligned_trajectories, alignment_metadata)
    
    # 2. Consensus一致性
    consistency_metrics = calculate_consensus_consistency(consensus, aligned_trajectories)
    
    # 3. 平滑度
    smoothness_metrics = calculate_smoothness(consensus)
    
    # 4. 速度连续性
    velocity_metrics = calculate_velocity_continuity(consensus)
    
    # 5. 合并所有指标
    all_metrics = {
        **alignment_metrics,
        **consistency_metrics,
        **smoothness_metrics,
        **velocity_metrics
    }
    
    # 6. 验证pass/fail
    validation_status = {
        'avg_std_pass': all_metrics.get('avg_std', np.inf) < thresholds.get('avg_std', 2.0),
        'dtw_pass': all_metrics.get('avg_dtw_distance', np.inf) < thresholds.get('dtw', 5.0),
        'avg_dtw_pass': all_metrics.get('avg_dtw_distance', np.inf) < thresholds.get('avg_dtw_distance', 5.0),
        'start_error_pass': all_metrics.get('start_error', np.inf) < thresholds.get('start_error', 2.0),
        'end_error_pass': all_metrics.get('end_error', np.inf) < thresholds.get('end_error', 2.0)
    }
    
    all_pass = all(validation_status.values())
    
    all_metrics['validation_status'] = validation_status
    all_metrics['all_pass'] = all_pass
    
    # 7. 打印验证结果
    print(f"\n{'='*70}")
    print("验证结果汇总")
    print(f"{'='*70}")
    print(f"平均DTW距离 < {thresholds.get('avg_dtw_distance', 5.0)}mm: "
          f"{'[PASS]' if validation_status['avg_dtw_pass'] else '[FAIL]'} "
          f"({all_metrics.get('avg_dtw_distance', np.nan):.3f}mm)")
    print(f"平均点标准差 < {thresholds.get('avg_std', 2.0)}mm: "
          f"{'[PASS]' if validation_status['avg_std_pass'] else '[FAIL]'} "
          f"({all_metrics.get('avg_std', np.nan):.3f}mm)")
    print(f"起点对齐误差 < {thresholds.get('start_error', 2.0)}mm: "
          f"{'[PASS]' if validation_status['start_error_pass'] else '[FAIL]'} "
          f"({all_metrics.get('start_error', np.nan):.3f}mm)")
    print(f"终点对齐误差 < {thresholds.get('end_error', 2.0)}mm: "
          f"{'[PASS]' if validation_status['end_error_pass'] else '[FAIL]'} "
          f"({all_metrics.get('end_error', np.nan):.3f}mm)")
    print(f"\n总体验证: {'[PASS]' if all_pass else '[FAIL]'}")
    
    return all_metrics

