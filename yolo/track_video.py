"""YOLO 实例分割视频跟踪"""
from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 加载训练好的模型
    model_path = "runs/segment/runs/segment/yolo_seg_magnet2/weights/best.pt"
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 视频路径
    video_path = os.path.join(script_dir, "../code/Mixed Reality/Sensapex/camera_image_processing/test1.mp4")
    
    # 跟踪参数
    results = model.track(
        source=video_path,
        imgsz=640,
        conf=0.392,              # 置信度阈值
        iou=0.45,               # IoU阈值
        show=True,              # 实时显示
        save=True,              # 保存结果
        tracker="bytetrack.yaml",  # 跟踪器：bytetrack.yaml 或 botsort.yaml
        project="runs/segment",
        name="track",
        show_boxes=False,           
    )
    
    print("跟踪结果保存在 runs/segment/track/")
