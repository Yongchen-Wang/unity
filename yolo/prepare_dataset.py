"""
数据集准备脚本：将数据集分割为 train 和 val，并自动检测类别数量
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# 数据集路径
dataset_root = Path(__file__).parent / "yolo_dataset"
images_dir = dataset_root / "images"
labels_dir = dataset_root / "labels"

# 创建 train/val 目录
train_images_dir = dataset_root / "train" / "images"
train_labels_dir = dataset_root / "train" / "labels"
val_images_dir = dataset_root / "val" / "images"
val_labels_dir = dataset_root / "val" / "labels"

for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 获取所有图片文件名（不含扩展名）
image_files = [f.stem for f in images_dir.glob("*.jpg")]
print(f"找到 {len(image_files)} 张图片")

# 检测类别数量
classes = set()
for label_file in labels_dir.glob("*.txt"):
    with open(label_file, 'r') as f:
        for line in f:
            if line.strip():
                class_id = int(line.strip().split()[0])
                classes.add(class_id)
num_classes = len(classes)
print(f"检测到 {num_classes} 个类别: {sorted(classes)}")

# 按 80/20 分割 train/val
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

print(f"训练集: {len(train_files)} 张")
print(f"验证集: {len(val_files)} 张")

# 复制文件到对应目录
for file_stem in train_files:
    shutil.copy2(images_dir / f"{file_stem}.jpg", train_images_dir / f"{file_stem}.jpg")
    if (labels_dir / f"{file_stem}.txt").exists():
        shutil.copy2(labels_dir / f"{file_stem}.txt", train_labels_dir / f"{file_stem}.txt")

for file_stem in val_files:
    shutil.copy2(images_dir / f"{file_stem}.jpg", val_images_dir / f"{file_stem}.jpg")
    if (labels_dir / f"{file_stem}.txt").exists():
        shutil.copy2(labels_dir / f"{file_stem}.txt", val_labels_dir / f"{file_stem}.txt")

