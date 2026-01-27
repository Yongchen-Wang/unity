"""YOLO 实例分割推理脚本"""
from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 模型路径
    model_path = "runs/segment/runs/segment/yolo_seg_magnet/weights/best.pt"
    
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 测试图片路径
    source = os.path.join(script_dir, "yolo_dataset/val/images")
    
    # 预测
    results = model.predict(
        source=source,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        save=True,
        save_txt=True,
        save_conf=True,
        show_labels=True,
        show_boxes=True,
        show_conf=True,
        project=os.path.join(script_dir, "runs/segment"),
        name="predict",
    )
    
    print("预测完成！结果保存在 runs/segment/predict/")
