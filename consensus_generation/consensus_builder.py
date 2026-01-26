"""
Consensus轨迹生成模块
从对齐后的轨迹生成consensus轨迹和速度profile
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.signal import savgol_filter


def build_consensus(trajectories: List[Dict],
                   method: str = 'median',
                   confidence_level: float = 1.0,
                   smooth_window: int = 11,
                   smooth_polyorder: int = 3) -> Dict:
    """
    从对齐后的轨迹生成consensus轨迹
    
    Args:
        trajectories: 对齐后的轨迹列表（所有轨迹长度相同）
        method: consensus方法 ('median', 'mean')
        confidence_level: 置信区间倍数（1.0 = ±1 std）
        smooth_window: 最终平滑窗口长度
        smooth_polyorder: 最终平滑多项式阶数
    
    Returns:
        consensus字典，包含：
        - 'x': consensus x坐标
        - 'y': consensus y坐标
        - 't': 归一化时间 [0, 1]
        - 'x_std': x方向标准差
        - 'y_std': y方向标准差
        - 'x_q25', 'x_q75': x方向第25和75百分位数
        - 'y_q25', 'y_q75': y方向第25和75百分位数
    """
    print(f"\n{'='*70}")
    print("生成Consensus轨迹")
    print(f"{'='*70}")
    
    if len(trajectories) == 0:
        raise ValueError("轨迹列表为空")
    
    # 检查所有轨迹长度是否相同
    n_samples = len(trajectories[0]['x'])
    for i, traj in enumerate(trajectories):
        if len(traj['x']) != n_samples:
            raise ValueError(f"轨迹 {i} 长度不一致: {len(traj['x'])} != {n_samples}")
    
    # 提取所有轨迹的x和y
    all_x = np.array([traj['x'] for traj in trajectories])  # (n_traj, n_samples)
    all_y = np.array([traj['y'] for traj in trajectories])  # (n_traj, n_samples)
    
    print(f"轨迹数量: {len(trajectories)}")
    print(f"每个轨迹点数: {n_samples}")
    print(f"Consensus方法: {method}")
    
    # 计算consensus（中位数或平均值）
    if method == 'median':
        consensus_x = np.median(all_x, axis=0)
        consensus_y = np.median(all_y, axis=0)
    else:  # 'mean'
        consensus_x = np.mean(all_x, axis=0)
        consensus_y = np.mean(all_y, axis=0)
    
    # 计算标准差
    x_std = np.std(all_x, axis=0)
    y_std = np.std(all_y, axis=0)
    
    # 计算百分位数（更鲁棒的置信区间）
    x_q25 = np.percentile(all_x, 25, axis=0)
    x_q75 = np.percentile(all_x, 75, axis=0)
    y_q25 = np.percentile(all_y, 25, axis=0)
    y_q75 = np.percentile(all_y, 75, axis=0)
    
    # 平滑consensus轨迹
    if n_samples >= smooth_window:
        try:
            consensus_x_smooth = savgol_filter(consensus_x, smooth_window, smooth_polyorder)
            consensus_y_smooth = savgol_filter(consensus_y, smooth_window, smooth_polyorder)
        except Exception as e:
            print(f"  警告: 平滑失败 - {e}, 使用原始consensus")
            consensus_x_smooth = consensus_x
            consensus_y_smooth = consensus_y
    else:
        consensus_x_smooth = consensus_x
        consensus_y_smooth = consensus_y
    
    # 归一化时间
    t_normalized = np.linspace(0, 1, n_samples)
    
    # 计算平均标准差
    avg_x_std = np.mean(x_std)
    avg_y_std = np.mean(y_std)
    avg_std = np.mean(np.sqrt(x_std**2 + y_std**2))
    
    print(f"  平均标准差: X={avg_x_std:.3f}mm, Y={avg_y_std:.3f}mm, 总={avg_std:.3f}mm")
    
    consensus = {
        'x': consensus_x_smooth,
        'y': consensus_y_smooth,
        't': t_normalized,
        'x_std': x_std,
        'y_std': y_std,
        'x_q25': x_q25,
        'x_q75': x_q75,
        'y_q25': y_q25,
        'y_q75': y_q75,
        'method': method,
        'n_trajectories': len(trajectories),
        'avg_std': avg_std
    }
    
    return consensus


def compute_velocity_profile(consensus: Dict,
                            smooth: bool = True,
                            smooth_window: int = 11,
                            smooth_polyorder: int = 3) -> Dict:
    """
    从consensus轨迹计算速度profile
    
    Args:
        consensus: consensus轨迹字典
        smooth: 是否平滑速度
        smooth_window: 平滑窗口长度
        smooth_polyorder: 平滑多项式阶数
    
    Returns:
        更新后的consensus字典，添加速度信息：
        - 'vx': x方向速度
        - 'vy': y方向速度
        - 'velocity_magnitude': 速度大小
        - 'acceleration': 加速度大小（可选）
    """
    print(f"\n计算速度Profile...")
    
    x = consensus['x']
    y = consensus['y']
    t = consensus['t']
    
    n = len(x)
    
    # 计算时间步长（假设归一化时间[0,1]）
    # 实际时间步长 = dt * total_time
    # 但我们只有归一化时间，所以假设单位时间
    dt = t[1] - t[0] if n > 1 else 1.0
    
    # 计算速度（差分）
    dx = np.diff(x)
    dy = np.diff(y)
    
    # 计算瞬时速度
    vx = dx / dt
    vy = dy / dt
    
    # 在末尾添加最后一个速度值（使用前向差分）
    vx = np.concatenate([vx, [vx[-1]]])
    vy = np.concatenate([vy, [vy[-1]]])
    
    # 平滑速度
    if smooth and n >= smooth_window:
        try:
            vx = savgol_filter(vx, smooth_window, smooth_polyorder)
            vy = savgol_filter(vy, smooth_window, smooth_polyorder)
        except Exception as e:
            print(f"  警告: 速度平滑失败 - {e}")
    
    # 计算速度大小
    velocity_magnitude = np.sqrt(vx**2 + vy**2)
    
    # 计算加速度（可选）
    dvx = np.diff(vx)
    dvy = np.diff(vy)
    acceleration = np.sqrt(dvx**2 + dvy**2) / dt
    acceleration = np.concatenate([[0], acceleration])  # 第一个点的加速度为0
    
    # 添加速度信息到consensus
    consensus['vx'] = vx
    consensus['vy'] = vy
    consensus['velocity_magnitude'] = velocity_magnitude
    consensus['acceleration'] = acceleration
    
    # 统计信息
    max_velocity = np.max(velocity_magnitude)
    mean_velocity = np.mean(velocity_magnitude)
    max_acceleration = np.max(acceleration)
    
    print(f"  最大速度: {max_velocity:.3f} mm/s")
    print(f"  平均速度: {mean_velocity:.3f} mm/s")
    print(f"  最大加速度: {max_acceleration:.3f} mm/s^2")
    
    return consensus


def calculate_confidence_coverage(consensus: Dict, 
                                 trajectories: List[Dict],
                                 confidence_level: float = 1.0) -> float:
    """
    计算置信区间覆盖原始轨迹点的比例
    
    Args:
        consensus: consensus轨迹字典
        trajectories: 原始轨迹列表
        confidence_level: 置信区间倍数
    
    Returns:
        覆盖比例 (0-1)
    """
    all_x = np.array([traj['x'] for traj in trajectories])
    all_y = np.array([traj['y'] for traj in trajectories])
    
    consensus_x = consensus['x']
    consensus_y = consensus['y']
    x_std = consensus['x_std']
    y_std = consensus['y_std']
    
    # 计算每个点是否在置信区间内
    x_lower = consensus_x - confidence_level * x_std
    x_upper = consensus_x + confidence_level * x_std
    y_lower = consensus_y - confidence_level * y_std
    y_upper = consensus_y + confidence_level * y_std
    
    # 检查所有轨迹点
    covered = 0
    total = 0
    
    for traj_x, traj_y in zip(all_x, all_y):
        x_in_range = (traj_x >= x_lower) & (traj_x <= x_upper)
        y_in_range = (traj_y >= y_lower) & (traj_y <= y_upper)
        in_range = x_in_range & y_in_range
        
        covered += np.sum(in_range)
        total += len(in_range)
    
    coverage = covered / total if total > 0 else 0.0
    
    return coverage

