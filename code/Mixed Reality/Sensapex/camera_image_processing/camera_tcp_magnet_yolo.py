"""
高精度圆柱体磁铁追踪系统 - TCP通信版本（YOLO版本）
说明：
  - 使用 YOLO 实例分割进行实时检测
  - 从分割掩码计算质心位置
  - 标记质心位置并发送到Unity
  - 摄像头分辨率强制设置为 1280x720，以保持与标定程序一致
"""

import cv2
import numpy as np
import socket
import time
import json
import os
from collections import deque
from ultralytics import YOLO


class KalmanFilter:
    """卡尔曼滤波器用于平滑位置追踪"""
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.001
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
        self.initialized = False
    
    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        
        if not self.initialized:
            self.kalman.statePre = np.array([[np.float32(x)],
                                             [np.float32(y)],
                                             [0], [0]], np.float32)
            self.kalman.statePost = np.array([[np.float32(x)],
                                              [np.float32(y)],
                                              [0], [0]], np.float32)
            self.initialized = True
            return x, y
        
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        return int(prediction[0]), int(prediction[1])


class MagnetTracker:
    """磁铁追踪器，集成卡尔曼滤波和平滑"""
    def __init__(self, max_distance=150, min_frames=2, history_size=20, smooth_window=5, min_movement=2):
        self.kalman = KalmanFilter()
        self.last_position = None
        self.last_sent_position = None
        self.max_distance = max_distance
        self.min_frames = min_frames
        self.min_movement = min_movement
        self.detection_count = 0
        self.lost_count = 0
        self.confirmed = False
        self.position_history = deque(maxlen=history_size)
        self.smooth_history = deque(maxlen=smooth_window)
        
    def update(self, detection):
        if detection is None:
            self.lost_count += 1
            self.detection_count = 0
            
            if self.lost_count < 5 and self.last_position is not None:
                return self.last_position, False
            
            if self.lost_count > 10:
                self.reset()
            return None, False
        
        cx, cy = detection
        
        if self.last_position is not None:
            distance = np.sqrt((cx - self.last_position[0])**2 +
                               (cy - self.last_position[1])**2)
            if distance > self.max_distance:
                if self.confirmed:
                    return self.last_position, True
                else:
                    self.detection_count = 0
                    return None, False
        
        self.detection_count += 1
        self.lost_count = 0
        
        if self.detection_count >= self.min_frames:
            self.confirmed = True
        
        # 卡尔曼滤波
        filtered_x, filtered_y = self.kalman.update(cx, cy)
        
        # 移动平均平滑
        self.smooth_history.append((filtered_x, filtered_y))
        if len(self.smooth_history) > 0:
            avg_x = int(np.mean([p[0] for p in self.smooth_history]))
            avg_y = int(np.mean([p[1] for p in self.smooth_history]))
        else:
            avg_x, avg_y = filtered_x, filtered_y
        
        # 最小移动阈值
        if self.last_sent_position is not None:
            movement = np.sqrt((avg_x - self.last_sent_position[0])**2 +
                             (avg_y - self.last_sent_position[1])**2)
            if movement < self.min_movement:
                return self.last_sent_position, self.confirmed
        
        self.last_position = (avg_x, avg_y)
        self.last_sent_position = (avg_x, avg_y)
        self.position_history.append((avg_x, avg_y))
        
        return (avg_x, avg_y), self.confirmed
    
    def reset(self):
        self.last_position = None
        self.last_sent_position = None
        self.detection_count = 0
        self.lost_count = 0
        self.confirmed = False
        self.kalman.initialized = False
        self.position_history.clear()
        self.smooth_history.clear()


def calculate_centroid_from_mask(mask):
    """
    从分割掩码计算质心位置
    
    Args:
        mask: 二值掩码 (numpy array)
    
    Returns:
        (cx, cy): 质心坐标，如果掩码为空则返回 None
    """
    if mask is None or mask.size == 0:
        return None
    
    # 计算掩码的矩
    moments = cv2.moments(mask)
    
    if moments["m00"] == 0:
        return None
    
    # 计算质心
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    
    return (cx, cy)


def load_calibration(calibration_file='calibration_params.json'):
    """加载标定参数"""
    if os.path.exists(calibration_file):
        with open(calibration_file, 'r', encoding='utf-8') as f:
            params = json.load(f)
            print(f"已加载标定文件: {calibration_file}")
            return params
    else:
        print(f"未找到标定文件: {calibration_file}，使用默认两点标定")
        return {
            'type': 'two_point',
            'pixel_points': [[0, 0], [1280, 720]],
            'space_points': [[-0.125, 0.875], [0, 1]]
        }


def pixel_to_space_transform(pixel_x, pixel_y, calibration_params):
    """像素坐标转空间坐标"""
    calib_type = calibration_params.get('type', 'two_point')
    
    if calib_type == 'perspective' and 'perspective_matrix' in calibration_params:
        matrix = np.array(calibration_params['perspective_matrix'])
        pixel_point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        space_point = cv2.perspectiveTransform(pixel_point, matrix)
        return float(space_point[0][0][0]), float(space_point[0][0][1])
    else:
        p1 = calibration_params['pixel_points'][0]
        p2 = calibration_params['pixel_points'][1]
        s1 = calibration_params['space_points'][0]
        s2 = calibration_params['space_points'][1]
        
        x_space = (pixel_x - p1[0]) / (p2[0] - p1[0]) * (s2[0] - s1[0]) + s1[0]
        y_space = (pixel_y - p1[1]) / (p2[1] - p1[1]) * (s2[1] - s1[1]) + s1[1]
        
        return x_space, y_space


class TCPConnection:
    """TCP连接管理器"""
    def __init__(self, unity_ip='127.0.0.1', unity_port=5007):
        self.unity_ip = unity_ip
        self.unity_port = unity_port
        self.sock = None
        self.connected = False
        self.reconnect_interval = 5.0
        self.last_reconnect_attempt = 0
        
    def connect(self):
        """连接到Unity TCP服务器"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.unity_ip, self.unity_port))
            self.connected = True
            print(f"[OK] 已连接到Unity TCP服务器 {self.unity_ip}:{self.unity_port}")
            return True
        except Exception as e:
            print(f"[ERROR] 连接失败: {e}")
            self.connected = False
            return False
    
    def send(self, message):
        """发送消息到Unity"""
        if not self.connected:
            current_time = time.time()
            if current_time - self.last_reconnect_attempt > self.reconnect_interval:
                print("尝试重新连接...")
                self.last_reconnect_attempt = current_time
                self.connect()
            return False
        
        try:
            self.sock.sendall(message.encode('utf-8'))
            return True
        except Exception as e:
            print(f"发送失败: {e}")
            self.connected = False
            self.close()
            return False
    
    def close(self):
        """关闭连接"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        self.connected = False


def main():
    # Unity连接设置
    unity_ip = '127.0.0.1'
    unity_port = 5007
    
    # 加载标定参数
    calibration_params = load_calibration()
    
    # 创建TCP连接
    tcp_conn = TCPConnection(unity_ip, unity_port)
    tcp_conn.connect()
    
    # 摄像头初始化：强制 1280x720
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头分辨率: {width}x{height}")
    
    # 加载 YOLO 模型
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 从 camera_image_processing 目录向上找到项目根目录，然后进入 yolo 目录
    project_root = os.path.join(script_dir, "..", "..", "..", "..")
    project_root = os.path.abspath(project_root)
    yolo_dir = os.path.join(project_root, "yolo")
    model_path = os.path.join(yolo_dir, "runs", "segment", "runs", "segment", "yolo_seg_magnet2", "weights", "best.pt")
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到YOLO模型文件: {model_path}")
        print("请确保模型文件存在")
        print(f"尝试的路径: {model_path}")
        return
    
    print(f"加载YOLO模型: {model_path}")
    # 切换到 yolo 目录加载模型（因为模型路径可能是相对的）
    original_cwd = os.getcwd()
    try:
        os.chdir(yolo_dir)
        model = YOLO("runs/segment/runs/segment/yolo_seg_magnet2/weights/best.pt")
    finally:
        os.chdir(original_cwd)
    print("YOLO模型加载完成")
    
    # 创建追踪器
    tracker = MagnetTracker(
        max_distance=150,
        min_frames=2,
        history_size=20,
        smooth_window=5,
        min_movement=2
    )
    
    print("=" * 70)
    print("高精度圆柱体磁铁追踪系统 - TCP版本（YOLO版本）")
    print("=" * 70)
    print(f"Unity服务器: {unity_ip}:{unity_port}")
    print(f"标定类型: {calibration_params['type']}")
    print("=" * 70)
    print("功能说明:")
    print("  - 使用YOLO实例分割检测磁铁")
    print("  - 从分割掩码计算质心位置")
    print("  - 质心坐标发送到Unity")
    print("  - TCP可靠传输，支持自动重连")
    print("=" * 70)
    print("快捷键:")
    print("  Q     - 退出程序")
    print("  R     - 重置追踪器")
    print("  C     - 重新连接TCP")
    print("=" * 70)
    
    last_print_time = time.time()
    frame_count = 0
    fps_time = time.time()
    fps = 0
    
    # YOLO检测参数
    conf_threshold = 0.392  # 置信度阈值
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        
        # 计算 FPS
        if frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - fps_time)
            fps_time = current_time
        
        # YOLO检测
        results = model.predict(
            frame,
            imgsz=640,
            conf=conf_threshold,
            iou=0.45,
            verbose=False
        )
        
        # 处理检测结果
        detection = None
        best_mask = None
        best_box = None
        
        if len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.data.cpu().numpy()
            
            # 选择置信度最高的检测结果
            if len(boxes) > 0:
                best_idx = 0
                best_conf = boxes[0][4] if len(boxes[0]) > 4 else 0
                for i in range(len(boxes)):
                    if len(boxes[i]) > 4 and boxes[i][4] > best_conf:
                        best_conf = boxes[i][4]
                        best_idx = i
                
                if best_idx < len(masks):
                    mask = masks[best_idx]
                    # 将掩码调整到原始图像尺寸
                    mask_resized = cv2.resize(mask, (width, height))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                    
                    # 计算质心
                    centroid = calculate_centroid_from_mask(mask_binary)
                    if centroid is not None:
                        detection = centroid
                        best_mask = mask_binary
                        best_box = boxes[best_idx]
        
        # 更新追踪器
        filtered_position, is_stable = tracker.update(detection)
        
        # 显示状态信息
        status_color = (0, 255, 0) if is_stable else (0, 165, 255)
        conn_color = (0, 255, 0) if tcp_conn.connected else (0, 0, 255)
        status_text = "TRACKING" if is_stable else "DETECTING" if tracker.detection_count > 0 else "SEARCHING"
        
        cv2.putText(frame, f"Status: {status_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"TCP: {'Connected' if tcp_conn.connected else 'Disconnected'}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, conn_color, 2)
        cv2.putText(frame, f"Frames: {tracker.detection_count}/{tracker.min_frames}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Conf: {conf_threshold:.2f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示检测状态
        if detection:
            cv2.putText(frame, f"Detected: YES (YOLO)", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Detected: NO", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 绘制检测结果
        if best_mask is not None:
            # 绘制分割掩码（半透明）
            mask_colored = cv2.applyColorMap(best_mask, cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)
        
        if best_box is not None and len(best_box) >= 4:
            # 绘制边界框
            x1, y1, x2, y2 = map(int, best_box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if len(best_box) > 4:
                conf = best_box[4]
                cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        if filtered_position:
            cx, cy = filtered_position
            
            # 只在稳定追踪时发送数据到Unity
            if is_stable:
                # 将质心坐标转换为空间坐标
                space_x, space_y = pixel_to_space_transform(cx, cy, calibration_params)
                # 为兼容 robot_data_logger.py，这里发送的是放大后的值
                scale_factor = 100.0
                x_scaled = space_x * scale_factor
                y_scaled = space_y * scale_factor
                # Unity / robot_data_logger 期望格式: "x,y,z\n"
                message = f"{x_scaled},{y_scaled},0\n"
                
                success = tcp_conn.send(message)
                
                current_time = time.time()
                if current_time - last_print_time >= 0.5:
                    status_symbol = "✓" if success else "✗"
                    print(f"{status_symbol} [稳定] 质心: ({cx}, {cy}) -> 空间: ({space_x:.4f}, {space_y:.4f})")
                    last_print_time = current_time
            
            # 绘制质心位置（大标记）
            cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)  # 红色实心圆
            cv2.circle(frame, (cx, cy), 18, (255, 255, 255), 3)  # 白色外圈
            cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (0, 255, 255), 3)  # 黄色十字
            cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (0, 255, 255), 3)
            
            # 显示质心坐标
            cv2.putText(frame, f"Centroid: ({cx}, {cy})",
                        (cx + 30, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 绘制历史轨迹
            if len(tracker.position_history) > 1:
                points = np.array(list(tracker.position_history), dtype=np.int32)
                cv2.polylines(frame, [points], False, (255, 0, 255), 2)
        
        # 如果是两点标定，画一下标定线
        if calibration_params['type'] == 'two_point':
            p1 = calibration_params['pixel_points'][0]
            p2 = calibration_params['pixel_points'][1]
            cv2.circle(frame, tuple(p1), 6, (255, 0, 0), -1)
            cv2.circle(frame, tuple(p2), 6, (255, 0, 0), -1)
            cv2.line(frame, tuple(p1), tuple(p2), (255, 0, 0), 2)
            cv2.putText(frame, "Calibration", (p1[0] - 50, p1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        cv2.imshow('TCP Magnet Detection (YOLO) [Q:quit R:reset C:connect]', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset()
            print("追踪器已重置")
        elif key == ord('c'):
            print("尝试重新连接TCP...")
            tcp_conn.close()
            tcp_conn.connect()
        
        time.sleep(0.01)
    
    cap.release()
    cv2.destroyAllWindows()
    tcp_conn.close()
    print("程序已退出")


if __name__ == "__main__":
    main()
