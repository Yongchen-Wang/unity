"""数据预处理与样本生成
- 从 robot_data CSV 加载轨迹
- 生成滑动窗口样本 (WINDOW_SIZE=10, FUTURE_STEPS=5)
- 特征归一化
- 数据集划分 + DataLoader 封装
"""

import os
import glob
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


# 以当前文件所在目录为基准，构造绝对路径，避免工作目录不同导致找不到CSV
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
ROBOT_DATA_DIR = os.path.join(
    _PROJECT_ROOT,
    "code",
    "Mixed Reality",
    "Sensapex",
    "camera_image_processing",
    "robot_data",
)
EXCLUDE_FILES = ["robot_data_20251215_140802.csv"]

WINDOW_SIZE = 10
FUTURE_STEPS = 5
F_MAX = 1023.0  # FSR 最大值

REQUIRED_COLS = [
    "error_x_R",
    "error_y_R",
    "geomagic_velocity_x_R",
    "geomagic_velocity_y_R",
    "fsr_analog",
    "position_x",
    "position_y",
    "time_elapsed",
]


def list_robot_data_files(data_dir: str = ROBOT_DATA_DIR) -> List[str]:
    """列出有效的 robot_data CSV 文件（排除不健康文件）。"""
    pattern = os.path.join(data_dir, "robot_data_*.csv")
    files = sorted(glob.glob(pattern))
    valid_files = [f for f in files if os.path.basename(f) not in EXCLUDE_FILES]
    print(
        f"在 {data_dir} 找到 {len(valid_files)} 个CSV文件 (排除 {len(files) - len(valid_files)} 个)"
    )
    return valid_files


def extract_training_samples(df: pd.DataFrame) -> List[Dict]:
    """从单条轨迹 DataFrame 生成训练样本列表。"""
    # 保证必要列存在
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    # 只保留必要列，去掉包含NaN/Inf的行
    df = df[REQUIRED_COLS].replace([np.inf, -np.inf], np.nan).dropna(how="any")
    df = df.reset_index(drop=True)

    if len(df) < WINDOW_SIZE + FUTURE_STEPS + 1:
        # 数据点太少，直接返回空
        return []
    samples: List[Dict] = []

    for t in range(WINDOW_SIZE, len(df) - FUTURE_STEPS):
        window = df.iloc[t - WINDOW_SIZE : t]

        force_mag = window["fsr_analog"].values.astype(np.float32)
        offset_x = window["error_x_R"].values.astype(np.float32)
        offset_y = window["error_y_R"].values.astype(np.float32)
        vel_x = window["geomagic_velocity_x_R"].values.astype(np.float32)
        vel_y = window["geomagic_velocity_y_R"].values.astype(np.float32)

        # Fx, Fy 暂无，填 0
        fx = np.zeros_like(force_mag, dtype=np.float32)
        fy = np.zeros_like(force_mag, dtype=np.float32)

        input_features = np.stack(
            [fx, fy, force_mag, offset_x, offset_y, vel_x, vel_y], axis=1
        ).astype(np.float32)  # (10, 7)

        # 若窗口中仍有NaN/Inf，跳过该样本
        if not np.all(np.isfinite(input_features)):
            continue

        current_t = df.iloc[t]
        future_t = df.iloc[t + FUTURE_STEPS]

        current_force = float(current_t["fsr_analog"])
        if not np.isfinite(current_force):
            continue
        strength_label = np.clip(current_force / F_MAX, 0.0, 1.0).astype(np.float32)

        dx = float(future_t["position_x"] - current_t["position_x"])
        dy = float(future_t["position_y"] - current_t["position_y"])
        if not np.isfinite(dx) or not np.isfinite(dy) or (dx == 0.0 and dy == 0.0):
            # 方向无法定义或无位移，跳过
            continue
        direction_label = np.arctan2(dy, dx).astype(np.float32)

        samples.append(
            {
                "input": input_features,
                "strength_label": strength_label,
                "direction_label": direction_label,
            }
        )

    return samples


def normalize_features(all_samples: List[Dict]) -> Tuple[List[Dict], Dict]:
    """对所有样本输入特征做标准化。"""
    inputs = np.array([s["input"] for s in all_samples], dtype=np.float32)  # (N,10,7)

    # 使用 nanmean/nanstd 忽略NaN
    mean = np.nanmean(inputs, axis=(0, 1))
    std = np.nanstd(inputs, axis=(0, 1))
    # 避免除零或NaN
    std[std < 1e-6] = 1.0
    std[~np.isfinite(std)] = 1.0

    for s in all_samples:
        x = s["input"].astype(np.float32)
        # 先将NaN替换为均值，再标准化
        nan_mask = ~np.isfinite(x)
        if nan_mask.any():
            x[nan_mask] = np.take(mean, np.where(nan_mask)[1])
        x = (x - mean) / std
        # 再次将残余非有限值置为0
        x[~np.isfinite(x)] = 0.0
        s["input"] = x

    return all_samples, {"mean": mean, "std": std}


def load_and_preprocess_data(
    data_dir: str = ROBOT_DATA_DIR,
) -> Tuple[List[Dict], Dict]:
    """加载所有CSV并生成归一化后的样本列表与norm参数。"""
    csv_files = list_robot_data_files(data_dir)
    all_samples: List[Dict] = []

    for path in csv_files:
        try:
            df = pd.read_csv(path)
            samples = extract_training_samples(df)
            all_samples.extend(samples)
            print(f"  [OK] {os.path.basename(path)} -> {len(samples)} 个样本")
        except Exception as e:
            print(f"  [FAIL] {os.path.basename(path)}: {e}")

    if not all_samples:
        raise RuntimeError("未生成任何训练样本")

    # 归一化
    all_samples, norm_params = normalize_features(all_samples)
    print(
        f"总样本数: {len(all_samples)}, 特征均值: {norm_params['mean']}, 特征std: {norm_params['std']}"
    )
    return all_samples, norm_params


def split_data(
    all_samples: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """按比例划分 train/val/test。"""
    train_samples, temp_samples = train_test_split(
        all_samples, test_size=1 - train_ratio, random_state=random_state
    )
    val_size = val_ratio / (1 - train_ratio)
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=1 - val_size, random_state=random_state
    )
    print(
        f"训练集: {len(train_samples)}, 验证集: {len(val_samples)}, 测试集: {len(test_samples)}"
    )
    return train_samples, val_samples, test_samples


class IntentDataset(Dataset):
    """Intent 训练数据集封装。"""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "input": torch.from_numpy(s["input"]).float(),  # (10,7)
            "strength_label": torch.tensor(s["strength_label"], dtype=torch.float32),
            "direction_label": torch.tensor(s["direction_label"], dtype=torch.float32),
        }


def create_dataloaders(
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """根据样本创建 DataLoader。"""
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        IntentDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        IntentDataset(val_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        IntentDataset(test_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    samples, norm = load_and_preprocess_data()
    split_data(samples)


