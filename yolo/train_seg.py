"""
YOLO 实例分割训练脚本
使用 YOLO v8/v9/v10 的 seg 模型进行实例分割训练
"""
from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)  # 切换到脚本目录
    
    # 加载预训练模型（会自动下载）
    model_name = "yolo26s-seg.pt"  # 可改为 yolov8s-seg.pt, yolov8m-seg.pt 等
    print(f"加载模型: {model_name}")
    model = YOLO(model_name)
    
    print("=" * 60)
    print("开始训练 YOLO 实例分割模型")
    print("=" * 60)
    
    # 数据集配置文件路径
    dataset_yaml = os.path.join(script_dir, "dataset.yaml")
    if not os.path.exists(dataset_yaml):
        print(f"错误: 找不到 dataset.yaml 文件: {dataset_yaml}")
        exit(1)
    
    # 开始训练
    results = model.train(
        data=dataset_yaml,        # 数据集配置文件（使用绝对路径）
        task="segment",           # 任务类型：segment（实例分割）
        epochs=30,               # 训练轮数（可以根据需要调整）
        imgsz=640,                # 输入图片尺寸（640, 1280等）
        batch=16,                 # 批次大小（根据GPU内存调整：8/16/32/64）
        device=0,                # GPU设备（0表示第一块GPU，使用CPU则设为'cpu'）
        workers=4,                # 数据加载进程数（Windows上建议设为0或较小值，Linux可以更大）
        project="runs/segment",   # 项目保存路径
        name="yolo_seg_magnet",   # 实验名称
        save=True,                # 保存检查点
        save_period=10,           # 每N个epoch保存一次检查点
        plots=True,               # 生成训练图表
        val=True,                 # 训练时进行验证
        # 数据增强参数
        hsv_h=0.015,             # 色调增强
        hsv_s=0.7,               # 饱和度增强
        hsv_v=0.4,               # 明度增强
        degrees=0.0,              # 旋转角度
        translate=0.1,           # 平移
        scale=0.5,                # 缩放
        flipud=0.0,               # 上下翻转概率
        fliplr=0.5,               # 左右翻转概率
        mosaic=1.0,               # Mosaic增强概率
        mixup=0.0,                # MixUp增强概率
        copy_paste=0.0,           # Copy-Paste增强概率
        # 优化器参数
        optimizer="auto",         # 优化器：SGD, Adam, AdamW, NAdam, RAdam, auto
        lr0=0.01,                 # 初始学习率
        lrf=0.01,                 # 最终学习率（lr0 * lrf）
        momentum=0.937,           # SGD动量
        weight_decay=0.0005,      # 权重衰减
        warmup_epochs=3.0,        # 预热轮数
        warmup_momentum=0.8,      # 预热动量
        warmup_bias_lr=0.1,       # 预热偏置学习率
        # 其他参数
        patience=50,              # Early stopping patience
        close_mosaic=10,          # 最后N个epoch关闭mosaic增强
    )
    
    print("=" * 60)
    print("训练完成！")
    print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
    print(f"最新模型保存在: {results.save_dir}/weights/last.pt")
    print("=" * 60)
    
    # # 可选：在验证集上评估
    # print("\n开始验证...")
    # metrics = model.val()
    # print(f"mAP50: {metrics.seg.map50:.4f}")
    # print(f"mAP50-95: {metrics.seg.map:.4f}")
