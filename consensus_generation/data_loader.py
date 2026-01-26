"""
数据加载和预处理模块
从CSV文件中加载轨迹数据，进行平滑处理和有效性检查
"""

import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from typing import List, Tuple, Dict, Optional
import glob


def load_all_trajectories(data_dir: str, exclude: Optional[List[str]] = None) -> List[Dict]:
    """
    加载所有CSV文件中的轨迹数据
    
    Args:
        data_dir: CSV文件所在目录
        exclude: 要排除的文件名列表（例如 ['140802']）
    
    Returns:
        轨迹字典列表，每个字典包含：
        - 'x': x坐标数组
        - 'y': y坐标数组
        - 'time': 时间戳数组
        - 'velocity': 速度数组
        - 'filename': 文件名
        - 'raw_data': 原始DataFrame
    """
    if exclude is None:
        exclude = []
    
    csv_files = glob.glob(os.path.join(data_dir, 'robot_data_*.csv'))
    trajectories = []
    
    print(f"\n{'='*70}")
    print(f"加载轨迹数据: {data_dir}")
    print(f"{'='*70}")
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        
        # 检查是否需要排除
        should_exclude = any(exclude_str in filename for exclude_str in exclude)
        if should_exclude:
            print(f"  跳过: {filename} (在排除列表中)")
            continue
        
        try:
            df = pd.read_csv(csv_file)
            
            # 检查必需的列
            required_cols = ['position_x', 'position_y', 'time_elapsed', 'velocity_magnitude']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"  [FAIL] {filename}: 缺少必需的列 {missing_cols}")
                continue
            
            # 提取轨迹数据
            x = df['position_x'].values
            y = df['position_y'].values
            time_elapsed = df['time_elapsed'].values
            velocity = df['velocity_magnitude'].values
            
            # 移除NaN值
            valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(time_elapsed))
            x = x[valid_mask]
            y = y[valid_mask]
            time_elapsed = time_elapsed[valid_mask]
            velocity = velocity[valid_mask] if len(velocity) > 0 else np.zeros_like(time_elapsed)
            
            if len(x) < 10:
                print(f"  [FAIL] {filename}: 数据点不足 (<10)")
                continue
            
            trajectories.append({
                'x': x,
                'y': y,
                'time': time_elapsed,
                'velocity': velocity,
                'filename': filename,
                'raw_data': df
            })
            
            print(f"  [OK] {filename}: {len(x)} 个数据点, 时间范围 {time_elapsed[0]:.2f}-{time_elapsed[-1]:.2f}s")
            
        except Exception as e:
            print(f"  [FAIL] {filename}: 加载失败 - {e}")
            continue
    
    print(f"\n成功加载 {len(trajectories)} 条轨迹")
    return trajectories


def smooth_trajectory(x: np.ndarray, y: np.ndarray, 
                     window_length: int = 11, polyorder: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用Savitzky-Golay滤波平滑轨迹
    
    Args:
        x: x坐标数组
        y: y坐标数组
        window_length: 滤波窗口长度（必须为奇数）
        polyorder: 多项式阶数
    
    Returns:
        平滑后的x和y坐标
    """
    # 确保window_length为奇数
    if window_length % 2 == 0:
        window_length += 1
    
    # 确保window_length不超过数据长度
    if window_length >= len(x):
        window_length = len(x) if len(x) % 2 == 1 else len(x) - 1
        if window_length < 3:
            return x, y
    
    # 确保polyorder < window_length
    if polyorder >= window_length:
        polyorder = window_length - 1
    
    try:
        x_smooth = savgol_filter(x, window_length, polyorder)
        y_smooth = savgol_filter(y, window_length, polyorder)
        return x_smooth, y_smooth
    except Exception as e:
        print(f"  警告: 平滑失败 - {e}, 返回原始数据")
        return x, y


def remove_stationary_segments(x: np.ndarray, y: np.ndarray, 
                               time: np.ndarray, velocity: np.ndarray,
                               velocity_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    移除静止段（速度低于阈值）
    
    Args:
        x: x坐标数组
        y: y坐标数组
        time: 时间戳数组
        velocity: 速度数组
        velocity_threshold: 速度阈值 (mm/s)
    
    Returns:
        去除静止段后的x, y, time
    """
    moving_mask = velocity >= velocity_threshold
    
    # 至少保留首尾各一个点
    if len(moving_mask) > 0:
        moving_mask[0] = True
        moving_mask[-1] = True
    
    x_filtered = x[moving_mask]
    y_filtered = y[moving_mask]
    time_filtered = time[moving_mask]
    
    removed_ratio = 1.0 - np.sum(moving_mask) / len(moving_mask) if len(moving_mask) > 0 else 0.0
    if removed_ratio > 0.1:  # 如果移除了超过10%的点，给出警告
        print(f"    移除了 {removed_ratio*100:.1f}% 的静止段")
    
    return x_filtered, y_filtered, time_filtered


def check_trajectory_validity(x: np.ndarray, y: np.ndarray, 
                              min_length: float = 20.0,
                              max_jump: float = 50.0) -> Tuple[bool, str]:
    """
    检查轨迹有效性
    
    Args:
        x: x坐标数组
        y: y坐标数组
        min_length: 最小轨迹长度 (mm)
        max_jump: 最大允许跳变距离 (mm)
    
    Returns:
        (is_valid, error_message)
    """
    if len(x) < 2:
        return False, "数据点不足 (<2)"
    
    # 计算总长度
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    total_length = np.sum(distances)
    
    if total_length < min_length:
        return False, f"轨迹长度不足 ({total_length:.2f}mm < {min_length}mm)"
    
    # 检查是否有异常跳变
    max_distance = np.max(distances)
    if max_distance > max_jump:
        return False, f"检测到异常跳变 ({max_distance:.2f}mm > {max_jump}mm)"
    
    # 检查NaN值
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        return False, "包含NaN值"
    
    return True, "OK"


def detect_outliers(x: np.ndarray, y: np.ndarray, 
                   velocity: np.ndarray, max_velocity: float = 50.0) -> np.ndarray:
    """
    检测异常点（速度过高的点）
    
    Args:
        x: x坐标数组
        y: y坐标数组
        velocity: 速度数组
        max_velocity: 最大允许速度 (mm/s)
    
    Returns:
        异常点掩码 (True表示异常点)
    """
    velocity_outlier = velocity > max_velocity
    
    # 计算位置跳变
    if len(x) > 1:
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)
        
        # 假设采样率约50Hz，计算速度
        dt = np.diff(np.arange(len(x))) / 50.0  # 粗略估计
        dt = np.clip(dt, 0.001, None)  # 避免除零
        instant_velocity = distances / dt
        instant_velocity = np.concatenate([[0], instant_velocity])
        
        position_outlier = instant_velocity > max_velocity
    else:
        position_outlier = np.zeros_like(velocity_outlier, dtype=bool)
    
    outliers = velocity_outlier | position_outlier
    return outliers


def fix_outliers(x: np.ndarray, y: np.ndarray, outliers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    修复异常点（使用线性插值）
    
    Args:
        x: x坐标数组
        y: y坐标数组
        outliers: 异常点掩码
    
    Returns:
        修复后的x和y坐标
    """
    if not np.any(outliers):
        return x, y
    
    x_fixed = x.copy()
    y_fixed = y.copy()
    
    # 找到正常点的索引
    valid_indices = np.where(~outliers)[0]
    
    if len(valid_indices) < 2:
        # 如果正常点太少，返回原始数据
        return x, y
    
    # 对异常点进行插值
    x_fixed[outliers] = np.interp(np.where(outliers)[0], valid_indices, x[valid_indices])
    y_fixed[outliers] = np.interp(np.where(outliers)[0], valid_indices, y[valid_indices])
    
    return x_fixed, y_fixed


def align_start_points(trajectories: List[Dict]) -> List[Dict]:
    """
    对齐所有轨迹的起点
    
    Args:
        trajectories: 轨迹列表
    
    Returns:
        对齐后的轨迹列表
    """
    if len(trajectories) == 0:
        return trajectories
    
    # 计算所有起点的中位数
    start_x = np.array([traj['x'][0] for traj in trajectories])
    start_y = np.array([traj['y'][0] for traj in trajectories])
    
    median_start_x = np.median(start_x)
    median_start_y = np.median(start_y)
    
    # 对齐每条轨迹
    aligned_trajectories = []
    for traj in trajectories:
        dx = median_start_x - traj['x'][0]
        dy = median_start_y - traj['y'][0]
        
        # 只平移，不改变形状
        aligned_traj = traj.copy()
        aligned_traj['x'] = traj['x'] + dx
        aligned_traj['y'] = traj['y'] + dy
        
        aligned_trajectories.append(aligned_traj)
    
    print(f"  起点对齐: 中位数起点 ({median_start_x:.2f}, {median_start_y:.2f})")
    
    return aligned_trajectories


def load_and_preprocess_data(data_dir: str, 
                             exclude: Optional[List[str]] = None,
                             smooth_window: int = 11,
                             smooth_polyorder: int = 3,
                             velocity_threshold: float = 0.3,
                             min_trajectory_length: float = 20.0,
                             max_jump: float = 50.0,
                             max_velocity: float = 50.0,
                             align_start: bool = True) -> Tuple[List[Dict], Dict]:
    """
    完整的数据加载和预处理流程
    
    Args:
        data_dir: 数据目录
        exclude: 排除的文件列表
        smooth_window: 平滑窗口长度
        smooth_polyorder: 平滑多项式阶数
        velocity_threshold: 速度阈值
        min_trajectory_length: 最小轨迹长度
        max_jump: 最大跳变距离
        max_velocity: 最大速度
        align_start: 是否对齐起点
    
    Returns:
        (processed_trajectories, metadata)
    """
    print(f"\n{'='*70}")
    print("数据加载和预处理")
    print(f"{'='*70}")
    
    # 1. 加载所有轨迹
    trajectories = load_all_trajectories(data_dir, exclude)
    
    if len(trajectories) == 0:
        raise ValueError("未加载到任何有效轨迹")
    
    processed_trajectories = []
    metadata = {
        'total_loaded': len(trajectories),
        'processed': 0,
        'failed': 0,
        'failures': []
    }
    
    # 2. 预处理每条轨迹
    print(f"\n预处理轨迹...")
    for traj in trajectories:
        filename = traj['filename']
        x = traj['x'].copy()
        y = traj['y'].copy()
        time = traj['time'].copy()
        velocity = traj['velocity'].copy()
        
        try:
            # 2.1 平滑处理
            x, y = smooth_trajectory(x, y, smooth_window, smooth_polyorder)
            
            # 2.2 检测异常点
            outliers = detect_outliers(x, y, velocity, max_velocity)
            if np.any(outliers):
                x, y = fix_outliers(x, y, outliers)
                print(f"    {filename}: 修复了 {np.sum(outliers)} 个异常点")
            
            # 2.3 移除静止段
            x, y, time = remove_stationary_segments(x, y, time, velocity, velocity_threshold)
            
            # 2.4 有效性检查
            is_valid, error_msg = check_trajectory_validity(x, y, min_trajectory_length, max_jump)
            
            if not is_valid:
                print(f"    [FAIL] {filename}: {error_msg}")
                metadata['failed'] += 1
                metadata['failures'].append({'filename': filename, 'reason': error_msg})
                continue
            
            # 更新轨迹
            traj['x'] = x
            traj['y'] = y
            traj['time'] = time
            traj['length'] = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            
            processed_trajectories.append(traj)
            metadata['processed'] += 1
            
        except Exception as e:
            print(f"    [FAIL] {filename}: 处理失败 - {e}")
            metadata['failed'] += 1
            metadata['failures'].append({'filename': filename, 'reason': str(e)})
            continue
    
    # 3. 对齐起点
    if align_start and len(processed_trajectories) > 0:
        processed_trajectories = align_start_points(processed_trajectories)
    
    # 4. 添加统计信息到metadata
    if len(processed_trajectories) > 0:
        lengths = [traj['length'] for traj in processed_trajectories]
        metadata['avg_length'] = np.mean(lengths)
        metadata['median_length'] = np.median(lengths)
        metadata['min_length'] = np.min(lengths)
        metadata['max_length'] = np.max(lengths)
    
    print(f"\n预处理完成: {metadata['processed']}/{metadata['total_loaded']} 条轨迹有效")
    
    return processed_trajectories, metadata

