"""
BC训练数据生成模块
从consensus轨迹生成BC Policy训练所需的state-action对
"""

import numpy as np
from typing import Dict, List, Optional


def generate_bc_training_data(consensus: Dict,
                             state_dim: int = 14,
                             action_dim: int = 2,
                             include_velocity: bool = True,
                             include_acceleration: bool = False) -> Dict:
    """
    生成BC训练数据
    
    Args:
        consensus: consensus轨迹字典
        state_dim: 状态维度
        action_dim: 动作维度（通常是2，对应target_position的x和y）
        include_velocity: 是否包含速度信息
        include_acceleration: 是否包含加速度信息
    
    Returns:
        训练数据字典，包含：
        - 'states': 状态数组 (n_samples, state_dim)
        - 'actions': 动作数组 (n_samples, action_dim)
        - 'metadata': 元数据
    """
    print(f"\n{'='*70}")
    print("生成BC训练数据")
    print(f"{'='*70}")
    
    x = consensus['x']
    y = consensus['y']
    t = consensus['t']
    n_samples = len(x)
    
    # 初始化状态和动作数组
    # 状态: [position_x, position_y, velocity_x, velocity_y, ...]
    # 动作: [target_position_x, target_position_y]
    
    # 计算实际状态维度
    actual_state_dim = 2  # position_x, position_y
    if include_velocity:
        actual_state_dim += 2  # velocity_x, velocity_y
    if include_acceleration and 'acceleration' in consensus:
        actual_state_dim += 2  # acceleration_x, acceleration_y
    
    # 如果state_dim更大，填充其他状态变量（可以后续扩展）
    if state_dim < actual_state_dim:
        state_dim = actual_state_dim
        print(f"  警告: state_dim太小，自动调整为 {state_dim}")
    
    states = np.zeros((n_samples - 1, state_dim))  # 最后一个点没有下一个动作
    actions = np.zeros((n_samples - 1, action_dim))
    
    # 提取速度信息
    if include_velocity and 'vx' in consensus and 'vy' in consensus:
        vx = consensus['vx']
        vy = consensus['vy']
    else:
        # 从位置计算速度
        dt = t[1] - t[0] if len(t) > 1 else 1.0
        vx = np.gradient(x) / dt
        vy = np.gradient(y) / dt
    
    # 提取加速度信息
    if include_acceleration:
        if 'acceleration' in consensus:
            # 如果有加速度，需要计算加速度分量
            # 从速度计算加速度
            dt = t[1] - t[0] if len(t) > 1 else 1.0
            ax = np.gradient(vx) / dt
            ay = np.gradient(vy) / dt
        else:
            dt = t[1] - t[0] if len(t) > 1 else 1.0
            ax = np.gradient(vx) / dt
            ay = np.gradient(vy) / dt
    
    # 生成训练样本
    state_idx = 0
    
    for i in range(n_samples - 1):
        # State: 当前状态
        states[i, 0] = x[i]  # position_x
        states[i, 1] = y[i]  # position_y
        
        if include_velocity:
            states[i, 2] = vx[i]  # velocity_x
            states[i, 3] = vy[i]  # velocity_y
            state_idx = 4
        else:
            state_idx = 2
        
        if include_acceleration:
            states[i, state_idx] = ax[i]  # acceleration_x
            states[i, state_idx + 1] = ay[i]  # acceleration_y
            state_idx += 2
        
        # 填充剩余状态维度（可以用其他特征，这里用0）
        # 例如可以添加: distance_to_boundary, curvature, etc.
        # states[i, state_idx:] = ...
        
        # Action: 下一个目标位置
        actions[i, 0] = x[i + 1]  # target_position_x
        actions[i, 1] = y[i + 1]  # target_position_y
    
    # 生成元数据
    metadata = {
        'n_samples': n_samples - 1,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'include_velocity': include_velocity,
        'include_acceleration': include_acceleration,
        'actual_state_dim': actual_state_dim,
        'trajectory_length': np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)),
        'time_range': (t[0], t[-1])
    }
    
    print(f"  生成训练样本数: {n_samples - 1}")
    print(f"  状态维度: {state_dim} (实际使用: {actual_state_dim})")
    print(f"  动作维度: {action_dim}")
    print(f"  包含速度: {include_velocity}")
    print(f"  包含加速度: {include_acceleration}")
    
    # 统计信息
    print(f"\n状态统计:")
    print(f"  Position X: [{states[:, 0].min():.2f}, {states[:, 0].max():.2f}] mm")
    print(f"  Position Y: [{states[:, 1].min():.2f}, {states[:, 1].max():.2f}] mm")
    if include_velocity:
        print(f"  Velocity X: [{states[:, 2].min():.2f}, {states[:, 2].max():.2f}] mm/s")
        print(f"  Velocity Y: [{states[:, 3].min():.2f}, {states[:, 3].max():.2f}] mm/s")
    
    print(f"\n动作统计:")
    print(f"  Target X: [{actions[:, 0].min():.2f}, {actions[:, 0].max():.2f}] mm")
    print(f"  Target Y: [{actions[:, 1].min():.2f}, {actions[:, 1].max():.2f}] mm")
    
    training_data = {
        'states': states,
        'actions': actions,
        'metadata': metadata
    }
    
    return training_data


def save_training_data(training_data: Dict, output_path: str = 'bc_training_data.npy'):
    """
    保存训练数据为numpy格式
    
    Args:
        training_data: 训练数据字典
        output_path: 输出文件路径
    """
    np.save(output_path, training_data, allow_pickle=True)
    print(f"\n训练数据已保存: {output_path}")
    print(f"  文件大小: {training_data['states'].nbytes + training_data['actions'].nbytes} 字节")


def load_training_data(input_path: str = 'bc_training_data.npy') -> Dict:
    """
    加载训练数据
    
    Args:
        input_path: 输入文件路径
    
    Returns:
        训练数据字典
    """
    training_data = np.load(input_path, allow_pickle=True).item()
    print(f"训练数据已加载: {input_path}")
    print(f"  样本数: {training_data['metadata']['n_samples']}")
    print(f"  状态维度: {training_data['metadata']['state_dim']}")
    print(f"  动作维度: {training_data['metadata']['action_dim']}")
    return training_data

