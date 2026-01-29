"""YOLO 实例分割相机实时跟踪"""
from ultralytics import YOLO
import os
import cv2

if __name__ == '__main__':
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 选择相机索引
    camera_index = 0  # 使用摄像头0
    
    # 验证相机
    print(f"\n尝试打开相机 {camera_index}...")
    test_cap = cv2.VideoCapture(camera_index)
    if not test_cap.isOpened():
        print(f"错误：无法打开相机 {camera_index}")
        print("请尝试修改 camera_index 为其他值（0, 2, 3等）")
        exit(1)
    test_cap.release()
    print("相机连接成功！")
    
    # 加载训练好的模型
    model_path = "runs/segment/runs/segment/yolo_seg_magnet2/weights/best.pt"
    print(f"\n加载模型: {model_path}")
    model = YOLO(model_path)
    
    print(f"\n开始实时跟踪...")
    print("提示：在跟踪窗口按 'q' 键退出")
    
    # 跟踪参数
    try:
        # stream=True 返回生成器，需要遍历处理每一帧
        results = model.track(
            source=camera_index,    # 使用相机索引
            imgsz=640,
            conf=0.392,              # 置信度阈值
            iou=0.45,               # IoU阈值
            show=True,              # 实时显示
            save=True,              # 保存结果
            tracker="bytetrack.yaml",  # 跟踪器：bytetrack.yaml 或 botsort.yaml
            project="runs/segment",
            name="track_camera",
            show_boxes=False,
            stream=True,            # 流式处理，适合实时相机
            verbose=True,           # 显示详细信息
        )
        
        # 处理每一帧结果
        for result in results:
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        print("请检查相机是否被其他程序占用")
    finally:
        cv2.destroyAllWindows()
    
    print("\n跟踪结束，结果保存在 runs/segment/track_camera/")
