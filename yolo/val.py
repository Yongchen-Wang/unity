"""
YOLO 验证脚本
"""
from ultralytics import YOLO
from pathlib import Path

if __name__ == '__main__':
    model = YOLO("runs/detect/runs/detect/yolo_custom2/weights/best.pt")
    
    # 在验证集上评估（降低置信度阈值，确保能检测到目标）
    metrics = model.val(
        data="dataset.yaml",
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        conf=0.001,  # 降低置信度阈值（默认0.25可能太高）
        iou=0.6,     # IoU阈值
    )
    
    # 主要指标
    print(f"\nmAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    