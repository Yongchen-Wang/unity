"""
轨迹对齐模块
使用DTW (Dynamic Time Warping) 对齐多条轨迹
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.interpolate import interp1d

try:
    from dtaidistance import dtw
    DTAIDISTANCE_AVAILABLE = True
except ImportError:
    DTAIDISTANCE_AVAILABLE = False
    print("警告: dtaidistance 未安装，将尝试使用 tslearn")
    try:
        from tslearn.metrics import dtw_path, dtw_distance
        TSLEARN_AVAILABLE = True
        DTAIDISTANCE_AVAILABLE = False
    except ImportError:
        TSLEARN_AVAILABLE = False
        print("错误: 未安装 dtaidistance 或 tslearn，DTW功能不可用")


def calculate_trajectory_length(x: np.ndarray, y: np.ndarray) -> float:
    """计算轨迹总长度"""
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    return np.sum(distances)


def select_reference_trajectory(trajectories: List[Dict], 
                                method: str = 'median') -> Tuple[int, Dict]:
    """
    选择参考轨迹
    
    Args:
        trajectories: 轨迹列表
        method: 选择方法 ('median', 'longest', 'shortest')
    
    Returns:
        (reference_index, reference_trajectory)
    """
    if len(trajectories) == 0:
        raise ValueError("轨迹列表为空")
    
    if method == 'longest':
        lengths = [traj['length'] for traj in trajectories]
        ref_idx = np.argmax(lengths)
    elif method == 'shortest':
        lengths = [traj['length'] for traj in trajectories]
        ref_idx = np.argmin(lengths)
    else:  # 'median' (default)
        lengths = [traj['length'] for traj in trajectories]
        median_length = np.median(lengths)
        ref_idx = np.argmin(np.abs(lengths - median_length))
    
    print(f"\n选择参考轨迹: {trajectories[ref_idx]['filename']} (索引 {ref_idx})")
    print(f"  方法: {method}, 长度: {trajectories[ref_idx]['length']:.2f}mm")
    
    return ref_idx, trajectories[ref_idx]


def dtw_align_dtaidistance(traj1: np.ndarray, traj2: np.ndarray, 
                          window: Optional[int] = None) -> Tuple[float, np.ndarray]:
    """
    使用 dtaidistance 进行DTW对齐
    
    Args:
        traj1: 参考轨迹 (N, 2)
        traj2: 待对齐轨迹 (M, 2)
        window: Sakoe-Chiba band窗口大小
    
    Returns:
        (distance, warping_path)
        warping_path: (2, K) 数组，第一行是traj1的索引，第二行是traj2的索引
    """
    if not DTAIDISTANCE_AVAILABLE:
        raise ImportError("dtaidistance 未安装")
    
    n, m = len(traj1), len(traj2)
    
    # dtaidistance只能处理1D序列，所以我们使用轨迹的累积弧长作为1D表示
    # 计算累积弧长
    def compute_arclength(traj):
        """计算轨迹的累积弧长"""
        diffs = np.diff(traj, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.concatenate([[0], np.cumsum(distances)])
    
    traj1_arclen = compute_arclength(traj1)
    traj2_arclen = compute_arclength(traj2)
    
    # 归一化到[0,1]以便比较
    if traj1_arclen[-1] > 0:
        traj1_arclen = traj1_arclen / traj1_arclen[-1]
    if traj2_arclen[-1] > 0:
        traj2_arclen = traj2_arclen / traj2_arclen[-1]
    
    try:
        # 使用dtaidistance计算DTW距离和路径
        distance, path_list = dtw.warping_path(traj1_arclen, traj2_arclen, window=window, use_pruning=True)
        
        # 转换路径格式
        if isinstance(path_list, (list, tuple)) and len(path_list) > 0:
            path_array = np.array(path_list).T
            # 确保索引在有效范围内
            path_array[0] = np.clip(path_array[0], 0, n - 1)
            path_array[1] = np.clip(path_array[1], 0, m - 1)
        else:
            # 如果没有路径，创建简单的线性映射
            min_len = min(n, m)
            path_array = np.array([np.linspace(0, n-1, min_len, dtype=int), 
                                  np.linspace(0, m-1, min_len, dtype=int)])
        
    except Exception as e:
        # 如果DTW失败，使用简单的线性插值对齐
        min_len = min(n, m)
        path_array = np.array([np.linspace(0, n-1, min_len, dtype=int), 
                              np.linspace(0, m-1, min_len, dtype=int)])
        # 计算平均欧氏距离
        aligned_indices = np.linspace(0, m-1, n, dtype=int)
        distance = np.mean([np.linalg.norm(traj1[i] - traj2[aligned_indices[i]]) 
                           for i in range(n)])
    
    return distance, path_array


def dtw_align_tslearn(traj1: np.ndarray, traj2: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    使用 tslearn 进行DTW对齐（备选方案）
    
    Args:
        traj1: 参考轨迹 (N, 2)
        traj2: 待对齐轨迹 (M, 2)
    
    Returns:
        (distance, warping_path)
    """
    if not TSLEARN_AVAILABLE:
        raise ImportError("tslearn 未安装")
    
    # tslearn需要 (n_samples, n_features) 格式
    path = dtw_path(traj1, traj2)
    distance = dtw_distance(traj1, traj2)
    
    # 转换path格式: [(i1, j1), (i2, j2), ...] -> (2, K)
    path_array = np.array(path).T
    
    return distance, path_array


def align_trajectory_to_reference(traj: np.ndarray, ref_traj: np.ndarray,
                                 window: Optional[int] = None) -> np.ndarray:
    """
    将一条轨迹对齐到参考轨迹
    
    Args:
        traj: 待对齐轨迹 (M, 2)
        ref_traj: 参考轨迹 (N, 2)
        window: DTW窗口大小
    
    Returns:
        对齐后的轨迹 (N, 2)
    """
    # 尝试使用 dtaidistance
    if DTAIDISTANCE_AVAILABLE:
        try:
            distance, path = dtw_align_dtaidistance(ref_traj, traj, window=window)
        except Exception as e:
            print(f"    dtaidistance对齐失败，尝试tslearn: {e}")
            if TSLEARN_AVAILABLE:
                distance, path = dtw_align_tslearn(ref_traj, traj)
            else:
                raise
    elif TSLEARN_AVAILABLE:
        distance, path = dtw_align_tslearn(ref_traj, traj)
    else:
        raise ImportError("未安装DTW库")
    
    # path格式: (2, K)，第一行是ref_traj索引，第二行是traj索引
    if path.shape[0] != 2:
        path = path.T
    
    ref_indices = path[0].astype(int)
    traj_indices = path[1].astype(int)
    
    # 使用线性插值将traj对齐到ref_traj的长度
    aligned_traj = np.zeros_like(ref_traj)
    
    for i in range(len(ref_traj)):
        # 找到ref_traj[i]对应的traj点
        matching_idx = np.where(ref_indices == i)[0]
        
        if len(matching_idx) > 0:
            # 如果有匹配点，取平均
            traj_points = traj[traj_indices[matching_idx]]
            aligned_traj[i] = np.mean(traj_points, axis=0)
        else:
            # 如果没有匹配点，使用最近的traj点
            closest_ref_idx = np.argmin(np.abs(ref_indices - i))
            aligned_traj[i] = traj[traj_indices[closest_ref_idx]]
    
    return aligned_traj


def resample_trajectory(x: np.ndarray, y: np.ndarray, 
                       n_samples: int = 100, kind: str = 'cubic') -> Tuple[np.ndarray, np.ndarray]:
    """
    重采样轨迹到固定点数
    
    Args:
        x: x坐标数组
        y: y坐标数组
        n_samples: 目标采样点数
        kind: 插值类型 ('linear', 'cubic', 'quadratic')
    
    Returns:
        重采样后的x和y坐标
    """
    if len(x) == n_samples:
        return x.copy(), y.copy()
    
    if len(x) < 4 and kind == 'cubic':
        kind = 'linear'  # cubic需要至少4个点
    
    # 创建归一化时间参数 [0, 1]
    t_old = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, n_samples)
    
    try:
        f_x = interp1d(t_old, x, kind=kind, bounds_error=False, fill_value='extrapolate')
        f_y = interp1d(t_old, y, kind=kind, bounds_error=False, fill_value='extrapolate')
        
        x_new = f_x(t_new)
        y_new = f_y(t_new)
        
        return x_new, y_new
    except Exception as e:
        print(f"    警告: {kind}插值失败，使用linear: {e}")
        f_x = interp1d(t_old, x, kind='linear', bounds_error=False, fill_value='extrapolate')
        f_y = interp1d(t_old, y, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        x_new = f_x(t_new)
        y_new = f_y(t_new)
        
        return x_new, y_new


def calculate_pairwise_dtw_distances(trajectories: List[np.ndarray]) -> np.ndarray:
    """
    计算所有轨迹对之间的DTW距离
    
    Args:
        trajectories: 轨迹列表，每个是 (N, 2) 数组
    
    Returns:
        距离矩阵 (n_traj, n_traj)
    """
    n = len(trajectories)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
                if DTAIDISTANCE_AVAILABLE:
                    try:
                        # 使用累积弧长计算DTW距离
                        def compute_arclength(traj):
                            diffs = np.diff(traj, axis=0)
                            dists = np.linalg.norm(diffs, axis=1)
                            arclen = np.concatenate([[0], np.cumsum(dists)])
                            return arclen / arclen[-1] if arclen[-1] > 0 else arclen
                        
                        traj_i_arclen = compute_arclength(trajectories[i])
                        traj_j_arclen = compute_arclength(trajectories[j])
                        dist, _ = dtw.warping_path(traj_i_arclen, traj_j_arclen, use_pruning=True)
                        distances[i, j] = dist
                        distances[j, i] = dist
                    except Exception as e:
                        # 如果DTW失败，使用平均欧氏距离
                        min_len = min(len(trajectories[i]), len(trajectories[j]))
                        dist = np.mean(np.linalg.norm(
                            trajectories[i][:min_len] - trajectories[j][:min_len], axis=1))
                        distances[i, j] = dist
                        distances[j, i] = dist
                elif TSLEARN_AVAILABLE:
                    try:
                        dist = dtw_distance(trajectories[i], trajectories[j])
                        distances[i, j] = dist
                        distances[j, i] = dist
                    except Exception as e:
                        print(f"    警告: DTW计算失败 (i={i}, j={j}): {e}")
                        distances[i, j] = np.inf
                        distances[j, i] = np.inf
                else:
                    # 使用欧氏距离作为备选
                    min_len = min(len(trajectories[i]), len(trajectories[j]))
                    dist = np.mean(np.linalg.norm(
                        trajectories[i][:min_len] - trajectories[j][:min_len], axis=1))
                    distances[i, j] = dist
                    distances[j, i] = dist
    
    return distances


def align_trajectories(trajectories: List[Dict],
                      method: str = 'dtw',
                      reference_method: str = 'median',
                      n_samples: int = 100,
                      window_ratio: float = 0.2,
                      resample_kind: str = 'cubic') -> Tuple[List[Dict], Dict]:
    """
    对齐所有轨迹
    
    Args:
        trajectories: 轨迹列表
        method: 对齐方法 ('dtw', 'none')
        reference_method: 参考轨迹选择方法 ('median', 'longest', 'shortest')
        n_samples: 重采样后的点数
        window_ratio: DTW窗口比例（相对于轨迹长度）
        resample_kind: 重采样插值类型
    
    Returns:
        (aligned_trajectories, alignment_metadata)
    """
    print(f"\n{'='*70}")
    print("轨迹对齐")
    print(f"{'='*70}")
    
    if len(trajectories) == 0:
        raise ValueError("轨迹列表为空")
    
    if method == 'none':
        # 仅重采样，不对齐
        print("方法: 仅重采样（不对齐）")
        aligned_trajectories = []
        for traj in trajectories:
            x_resampled, y_resampled = resample_trajectory(
                traj['x'], traj['y'], n_samples, resample_kind)
            aligned_traj = traj.copy()
            aligned_traj['x'] = x_resampled
            aligned_traj['y'] = y_resampled
            aligned_traj['t_normalized'] = np.linspace(0, 1, n_samples)
            aligned_trajectories.append(aligned_traj)
        
        metadata = {
            'method': 'none',
            'n_samples': n_samples,
            'alignment_quality': 1.0
        }
        return aligned_trajectories, metadata
    
    # DTW对齐
    if not DTAIDISTANCE_AVAILABLE and not TSLEARN_AVAILABLE:
        raise ImportError("需要安装 dtaidistance 或 tslearn 进行DTW对齐")
    
    print(f"方法: DTW对齐")
    
    # 1. 选择参考轨迹
    ref_idx, ref_traj = select_reference_trajectory(trajectories, reference_method)
    
    # 2. 重采样参考轨迹
    ref_x, ref_y = resample_trajectory(ref_traj['x'], ref_traj['y'], n_samples, resample_kind)
    ref_traj_2d = np.column_stack([ref_x, ref_y])
    
    # 3. 计算DTW窗口大小
    window = int(len(ref_traj_2d) * window_ratio) if window_ratio > 0 else None
    print(f"  DTW窗口: {window} (轨迹长度的 {window_ratio*100:.0f}%)")
    
    # 4. 对齐每条轨迹
    aligned_trajectories = []
    alignment_distances = []
    
    print(f"\n对齐轨迹...")
    for i, traj in enumerate(trajectories):
        if i == ref_idx:
            # 参考轨迹直接使用
            aligned_traj = traj.copy()
            aligned_traj['x'] = ref_x
            aligned_traj['y'] = ref_y
            aligned_traj['t_normalized'] = np.linspace(0, 1, n_samples)
            aligned_trajectories.append(aligned_traj)
            alignment_distances.append(0.0)
            print(f"  [{i+1}/{len(trajectories)}] {traj['filename']}: 参考轨迹")
        else:
            try:
                # 重采样到参考轨迹长度（临时，用于DTW对齐）
                traj_x, traj_y = resample_trajectory(traj['x'], traj['y'], 
                                                    len(ref_traj_2d), resample_kind)
                traj_2d = np.column_stack([traj_x, traj_y])
                
                # DTW对齐
                aligned_2d = align_trajectory_to_reference(traj_2d, ref_traj_2d, window)
                
                aligned_traj = traj.copy()
                aligned_traj['x'] = aligned_2d[:, 0]
                aligned_traj['y'] = aligned_2d[:, 1]
                aligned_traj['t_normalized'] = np.linspace(0, 1, n_samples)
                
                # 计算对齐距离
                if DTAIDISTANCE_AVAILABLE:
                    dist, _ = dtw.warping_path(ref_traj_2d, aligned_2d, 
                                              window=window, use_pruning=True)
                elif TSLEARN_AVAILABLE:
                    dist = dtw_distance(ref_traj_2d, aligned_2d)
                else:
                    dist = np.mean(np.linalg.norm(ref_traj_2d - aligned_2d, axis=1))
                
                alignment_distances.append(dist)
                aligned_trajectories.append(aligned_traj)
                
                print(f"  [{i+1}/{len(trajectories)}] {traj['filename']}: "
                      f"DTW距离 = {dist:.3f}mm")
                
            except Exception as e:
                print(f"  [FAIL] [{i+1}/{len(trajectories)}] {traj['filename']}: 对齐失败 - {e}")
                # 如果对齐失败，使用重采样作为备选
                traj_x, traj_y = resample_trajectory(traj['x'], traj['y'], 
                                                    n_samples, resample_kind)
                aligned_traj = traj.copy()
                aligned_traj['x'] = traj_x
                aligned_traj['y'] = traj_y
                aligned_traj['t_normalized'] = np.linspace(0, 1, n_samples)
                aligned_trajectories.append(aligned_traj)
                alignment_distances.append(np.inf)
    
    # 5. 计算对齐质量指标
    avg_dtw_distance = np.mean([d for d in alignment_distances if d != np.inf])
    
    # 计算对齐后轨迹之间的距离
    aligned_arrays = [np.column_stack([traj['x'], traj['y']]) 
                     for traj in aligned_trajectories]
    pairwise_distances = calculate_pairwise_dtw_distances(aligned_arrays)
    
    # 只计算有效距离（非对角线和非inf）
    valid_distances = pairwise_distances[
        ~np.isinf(pairwise_distances) & (pairwise_distances > 0)]
    avg_pairwise_dtw = np.mean(valid_distances) if len(valid_distances) > 0 else 0.0
    
    # 对齐质量：1.0表示完美对齐，0.0表示对齐失败
    # 使用平均距离的倒数（归一化）
    max_expected_distance = 10.0  # 假设最大期望距离为10mm
    alignment_quality = 1.0 / (1.0 + avg_pairwise_dtw / max_expected_distance)
    
    metadata = {
        'method': method,
        'reference_idx': ref_idx,
        'reference_filename': trajectories[ref_idx]['filename'],
        'n_samples': n_samples,
        'window_ratio': window_ratio,
        'avg_dtw_distance': avg_dtw_distance,
        'avg_pairwise_dtw': avg_pairwise_dtw,
        'alignment_distances': alignment_distances,
        'alignment_quality': alignment_quality
    }
    
    print(f"\n对齐完成:")
    print(f"  平均DTW距离: {avg_dtw_distance:.3f}mm")
    print(f"  平均轨迹间距离: {avg_pairwise_dtw:.3f}mm")
    print(f"  对齐质量: {alignment_quality:.3f}")
    
    return aligned_trajectories, metadata

