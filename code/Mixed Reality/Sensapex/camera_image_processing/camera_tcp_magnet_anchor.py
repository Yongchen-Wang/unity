"""
高精度圆柱体磁铁追踪系统 - TCP通信版本（Anchor 简化版）
说明：
  - 直接使用 camera_tcp_magnet.py 中的稳定检测与追踪算法
  - 不再依赖蓝色杆锚点、银色 HSV、ROI 等复杂检测手段
  - 仅保留：灰度阈值 + 形态学 + 面积/形状筛选 + 位置连续性 + Kalman 平滑
  - 摄像头分辨率仍强制设置为 1280x720，以保持与标定程序一致
"""

import cv2
import numpy as np
import socket
import time
import json
import os
from collections import deque


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
    """圆柱体磁铁追踪器，集成多种抗干扰措施"""
    def __init__(self, max_distance=100, min_frames=3, history_size=10, smooth_window=5, min_movement=2):
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
        self.size_history = deque(maxlen=5)
        
    def update(self, detection):
        if detection is None:
            self.lost_count += 1
            self.detection_count = 0
            
            if self.lost_count < 5 and self.last_position is not None:
                return self.last_position, False
            
            if self.lost_count > 10:
                self.reset()
            return None, False
        
        cx, cy, contour, x, y, w, h = detection
        
        if self.last_position is not None:
            distance = np.sqrt((cx - self.last_position[0])**2 +
                               (cy - self.last_position[1])**2)
            # 如果本帧候选点相对上一帧跳变太大，则认为是误检
            if distance > self.max_distance:
                if self.confirmed:
                    # 如果已经稳定追踪过，优先保持上一帧位置，避免瞬间跳变
                    return self.last_position, True
                else:
                    # 还未稳定时，大跳变直接忽略，并清空累计帧数
                    self.detection_count = 0
                    return None, False
        
        if len(self.size_history) > 2:
            avg_size = np.mean(self.size_history)
            current_size = (w + h) / 2
            size_ratio = current_size / avg_size if avg_size > 0 else 1
            
            if size_ratio < 0.3 or size_ratio > 3.0:
                if self.confirmed:
                    return self.last_position, True
                else:
                    return None, False
        
        self.detection_count += 1
        self.lost_count = 0
        
        filtered_x, filtered_y = self.kalman.update(cx, cy)
        self.smooth_history.append((filtered_x, filtered_y))
        
        # 平滑窗口越大，越稳定但延迟越大；这里只要>=2就开始做平均，减少延迟
        if len(self.smooth_history) >= 2:
            avg_x = int(np.mean([pos[0] for pos in self.smooth_history]))
            avg_y = int(np.mean([pos[1] for pos in self.smooth_history]))
            smoothed_pos = (avg_x, avg_y)
        else:
            smoothed_pos = (filtered_x, filtered_y)
        
        if self.last_sent_position is not None:
            distance = np.sqrt((smoothed_pos[0] - self.last_sent_position[0])**2 +
                               (smoothed_pos[1] - self.last_sent_position[1])**2)
            if distance < self.min_movement:
                smoothed_pos = self.last_sent_position
        
        self.position_history.append(smoothed_pos)
        self.size_history.append((w + h) / 2)
        self.last_position = smoothed_pos
        self.last_sent_position = smoothed_pos
        
        if self.detection_count >= self.min_frames:
            self.confirmed = True
        
        return smoothed_pos, self.confirmed
    
    def reset(self):
        self.kalman = KalmanFilter()
        self.last_position = None
        self.last_sent_position = None
        self.detection_count = 0
        self.lost_count = 0
        self.confirmed = False
        self.position_history.clear()
        self.smooth_history.clear()
        self.size_history.clear()


def is_point_in_calibration_area(pixel_x, pixel_y, calibration_params):
    """
    判断一个像素点是否在标定区域内
    - 支持两点和透视四点两种标定方式
    """
    if calibration_params is None:
        return True
    
    calib_type = calibration_params.get('type', 'two_point')
    
    if calib_type == 'perspective' and 'pixel_points' in calibration_params:
        pixel_points = calibration_params['pixel_points']
        if len(pixel_points) == 4:
            pts = np.array(pixel_points, dtype=np.int32)
            # pointPolygonTest: >0 在内部, =0 在边界, <0 在外部
            result = cv2.pointPolygonTest(pts, (float(pixel_x), float(pixel_y)), False)
            return result >= 0
    elif calib_type == 'two_point' and 'pixel_points' in calibration_params:
        pixel_points = calibration_params['pixel_points']
        if len(pixel_points) >= 2:
            p1 = pixel_points[0]
            p2 = pixel_points[1]
            min_x = min(p1[0], p2[0])
            max_x = max(p1[0], p2[0])
            min_y = min(p1[1], p2[1])
            max_y = max(p1[1], p2[1])
            return min_x <= pixel_x <= max_x and min_y <= pixel_y <= max_y
    
    return True


def apply_dynamic_threshold(gray, threshold_value):
    """
    动态阈值函数：
    - 在整体画面比较暗的时候，自动选择更合适的阈值，避免整幅图几乎全黑
    - 亮度很低时使用 OTSU 自适应阈值；亮度中等时使用 mean+offset 上限策略
    """
    # 计算当前灰度图的平均亮度
    mean_val = float(np.mean(gray))
    
    # 场景非常暗时，直接使用 OTSU 阈值，让那块最亮的磁铁区域自动变成白色
    if mean_val < 60.0:
        used_thresh, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        # 根据平均亮度给一个自适应上限：暗的时候自动降低阈值上限
        # 例如 mean=80 时，上限约为 100；mean=120 时，上限约为 140
        adaptive_cap = max(40.0, min(220.0, mean_val + 20.0))
        # 实际使用的阈值 = min(滑块阈值, 自适应上限)
        used_thresh = int(min(threshold_value, adaptive_cap))
        _, thresh = cv2.threshold(gray, used_thresh, 255, cv2.THRESH_BINARY)

    return thresh, used_thresh


def find_magnet_position(frame,
                         prev_position=None,
                         threshold_value=140,
                         min_area=60,
                         max_area=20000,
                         show_all_candidates=False,
                         calibration_params=None):
    """
    识别圆柱形磁铁（优化版本：面积+形状+位置连续性）
    返回圆柱体的正中心位置
    - 与 camera_tcp_magnet.py 中的实现完全一致
    """
    # 转换为灰度图并做轻微高斯滤波，抑制随机噪点但尽量保留边缘
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用动态阈值：在亮度很低时自动降低实际阈值，避免整幅变成全黑
    thresh, _ = apply_dynamic_threshold(gray, threshold_value)
    
    # 简单形态学操作
    # 使用稍小的卷积核，避免把磁铁和周围结构过度黏连
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_magnet = None
    best_score = 0
    all_candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 面积范围过滤（磁铁可能比小球大）
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算长宽比（圆柱体从侧面看应该是矩形）
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # 接受长宽比在一定范围内的物体（略放宽，但不过于极端）
            if 0.2 <= aspect_ratio <= 4.0:
                # 计算轮廓完整性（圆度）
                perimeter = cv2.arcLength(contour, True)
                circularity = 0
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # 过滤掉太不规则的形状（略微放宽圆度阈值，避免边缘不完整导致间歇丢失）
                if circularity >= 0.25:
                    cx = x + w // 2
                    cy = y + h // 2
                    
                    # 只在标定区域内的候选才参与后续判断
                    if calibration_params is not None:
                        if not is_point_in_calibration_area(cx, cy, calibration_params):
                            continue
        
                    # 简单评分系统：面积 + 位置连续性 + 形状
                    score = area  # 基础分：面积
                    
                    # 形状加分（圆度越高加分越多，最多+1000）
                    shape_bonus = circularity * 1000
                    score += shape_bonus
                    
                    # 如果有历史位置，优先选择靠近的
                    if prev_position is not None:
                        distance = np.sqrt((cx - prev_position[0])**2 +
                                           (cy - prev_position[1])**2)
                        # 距离越近加分越多（最多加5000分）
                        if distance < 200:
                            proximity_bonus = (200 - distance) / 200 * 5000
                            score += proximity_bonus
                    
                    # 记录所有候选
                    all_candidates.append({
                        'cx': cx, 'cy': cy, 'area': int(area),
                        'aspect_ratio': aspect_ratio, 'w': w, 'h': h,
                        'score': score, 'circularity': circularity
                    })
                    
                    # 选择得分最高的
                    if score > best_score:
                        best_score = score
                        # 使用矩计算更精确的中心点
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            best_magnet = (cx, cy, contour, x, y, w, h)
    
    if show_all_candidates:
        return best_magnet, all_candidates, thresh
    return best_magnet


def load_calibration(calibration_file='calibration_params.json'):
    """加载标定参数（与 camera_tcp_magnet.py 相同）"""
    if os.path.exists(calibration_file):
        with open(calibration_file, 'r', encoding='utf-8') as f:
            params = json.load(f)
            print(f"已加载标定文件: {calibration_file}")
            return params
    else:
        print(f"未找到标定文件: {calibration_file}，使用默认两点标定")
        return {
            'type': 'two_point',
            'pixel_points': [[119, 454], [548, 29]],
            'space_points': [[-0.125, 0.875], [0, 1]]
        }


def pixel_to_space_transform(pixel_x, pixel_y, calibration_params):
    """像素坐标转空间坐标（支持透视变换，与 camera_tcp_magnet.py 相同）"""
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
        # 端口改为 5007，与原 anchor 版本保持一致
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
            # 使用纯ASCII字符，避免在GBK终端中输出Unicode符号导致编码错误
            print(f"[OK] 已连接到Unity TCP服务器 {self.unity_ip}:{self.unity_port}")
            return True
        except Exception as e:
            # 同样避免使用 '✗' 等特殊符号
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
            # Unity的TCPReceiver期望格式: "x,y,z\n"
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
    # Unity连接设置（端口 5007，与原 anchor 版本保持一致）
    unity_ip = '127.0.0.1'
    unity_port = 5007
    
    # 加载标定参数
    calibration_params = load_calibration()
    
    # 创建TCP连接
    tcp_conn = TCPConnection(unity_ip, unity_port)
    tcp_conn.connect()
    
    # 摄像头初始化：与原 anchor 版本保持一致，强制 1280x720
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头分辨率: {width}x{height}")
    
    # 简单高效参数（根据当前相机 & 场景调整后的默认值）
    # 如果画面整体偏亮/反光强，可适当提高 threshold；如果磁铁偏暗，可适当降低
    threshold_value = 150  # 初始二值化阈值
    # 根据当前实验的磁铁尺寸调整：60–3000 是比较稳妥的起点范围
    min_area = 60          # 最小面积 - 过滤小噪点（小于此面积的白块直接忽略）
    max_area = 3000        # 最大面积 - 限制检测范围，避免把大块反光/边缘当成目标
    
    # 创建追踪器
    # 在保持较低延迟的前提下，略微提高稳定性（多一帧确认 + 更长的平滑窗口）
    tracker = MagnetTracker(
        max_distance=150,      # 最大移动距离
        min_frames=2,          # 最少连续帧数（2帧确认，减少误检）
        history_size=20,       # 历史轨迹长度
        smooth_window=5,       # 平滑窗口（增加一点稳定性）
        min_movement=2         # 最小移动阈值（2像素以下视为抖动）
    )
    
    print("=" * 70)
    print("高精度圆柱体磁铁追踪系统 - TCP版本（Anchor 简化版）")
    print("=" * 70)
    print(f"Unity服务器: {unity_ip}:{unity_port}")
    print(f"标定类型: {calibration_params['type']}")
    print("=" * 70)
    print("功能说明:")
    print("  - 追踪圆柱体磁铁的正中心位置")
    print("  - 中心坐标发送到Unity作为小球球心")
    print("  - TCP可靠传输，支持自动重连")
    print("=" * 70)
    print("抗抖动增强功能:")
    print("  - 卡尔曼滤波 + 移动平均双重平滑")
    print("  - 最小移动阈值（3像素死区）")
    print("  - 多帧验证确认检测")
    print("  - 增强形态学操作稳定轮廓")
    print("  - 位置连续性检查")
    print("  - 置信度评分系统")
    print("=" * 70)
    print("快捷键:")
    print("  Q     - 退出程序")
    print("  T     - 显示/隐藏阈值窗口")
    print("  D     - 开启/关闭调试模式（显示所有候选）")
    print("  R     - 重置追踪器")
    print("  C     - 重新连接TCP")
    print("=" * 70)
    
    last_print_time = time.time()
    show_threshold = True
    debug_mode = True  # 默认开启调试模式，显示所有候选
    frame_count = 0
    fps_time = time.time()
    fps = 0
    
    # 创建参数调节窗口（trackbar）
    cv2.namedWindow('Threshold & Parameters')
    cv2.createTrackbar('Threshold', 'Threshold & Parameters', threshold_value, 255, lambda x: None)
    cv2.createTrackbar('Min Area', 'Threshold & Parameters', min_area, 5000, lambda x: None)
    cv2.createTrackbar('Max Area', 'Threshold & Parameters', int(max_area / 10), 2000, lambda x: None)  # 滑块值×10，最大20000
    
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
        
        # 读取trackbar参数
        threshold_value = cv2.getTrackbarPos('Threshold', 'Threshold & Parameters')
        min_area = cv2.getTrackbarPos('Min Area', 'Threshold & Parameters')
        max_area = cv2.getTrackbarPos('Max Area', 'Threshold & Parameters') * 10  # 乘以10以扩大范围
        
        # 识别磁铁位置（增加：只在标定区域内的候选才被接受）
        prev_pos = tracker.last_position
        if debug_mode:
            magnet_detection, all_candidates, thresh = find_magnet_position(
                frame, prev_pos, threshold_value, min_area, max_area,
                show_all_candidates=True, calibration_params=calibration_params)
        else:
            magnet_detection = find_magnet_position(
                frame, prev_pos, threshold_value, min_area, max_area,
                show_all_candidates=False, calibration_params=calibration_params)
            all_candidates = []
            # 生成thresh用于显示（同样使用动态阈值，便于在低亮度下观察二值化结果）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            thresh, _ = apply_dynamic_threshold(blurred, threshold_value)
        
        # 更新追踪器
        filtered_position, is_stable = tracker.update(magnet_detection)
        
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
        
        # 显示参数
        cv2.putText(frame, f"Thresh: {threshold_value}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Area: {min_area}-{max_area}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示检测状态
        if magnet_detection:
            cv2.putText(frame, f"Detected: YES (MAGNET)", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Detected: NO", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 调试模式：显示所有候选
        if debug_mode and all_candidates:
            cv2.putText(frame, f"All Candidates: {len(all_candidates)}", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            for cand in all_candidates:
                cx, cy = cand['cx'], cand['cy']
                area = cand['area']
                score = cand.get('score', area)
                circularity = cand.get('circularity', 0)
                
                # 判断是否为选中的目标
                is_selected = (magnet_detection and cx == magnet_detection[0] and cy == magnet_detection[1])
                
                # 选中的用绿色大圆，其他用灰色小圆
                if is_selected:
                    color = (0, 255, 0)
                    radius = 15
                    thickness = 3
                else:
                    color = (128, 128, 128)
                    radius = 10
                    thickness = 2
                
                cv2.circle(frame, (cx, cy), radius, color, thickness)
                # 显示详细信息（面积、得分、圆度）
                cv2.putText(frame, f"A:{area} S:{int(score)}", (cx + 20, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(frame, f"C:{circularity:.2f}", (cx + 20, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        if filtered_position:
            cx, cy = filtered_position
            
            # 只在稳定追踪时发送数据到Unity
            if is_stable:
                # 将圆柱体中心坐标转换为空间坐标
                space_x, space_y = pixel_to_space_transform(cx, cy, calibration_params)
                # 为兼容 robot_data_logger.py，这里发送的是放大后的值
                # logger 端会使用相同的 scale_factor（默认 100）再除回来
                scale_factor = 100.0
                x_scaled = space_x * scale_factor
                y_scaled = space_y * scale_factor
                # Unity / robot_data_logger 期望格式: "x,y,z\n"
                message = f"{x_scaled},{y_scaled},0\n"
                
                success = tcp_conn.send(message)
                
                current_time = time.time()
                if current_time - last_print_time >= 0.5:
                    status_symbol = "✓" if success else "✗"
                    print(f"{status_symbol} [稳定] 磁铁中心: ({cx}, {cy}) -> 空间: ({space_x:.4f}, {space_y:.4f})")
                    last_print_time = current_time
            
            # 绘制检测结果
            if magnet_detection:
                _, _, contour, x, y, w, h = magnet_detection
                
                # 画轮廓（颜色根据状态变化）
                contour_color = (0, 255, 0) if is_stable else (0, 165, 255)
                cv2.drawContours(frame, [contour], -1, contour_color, 2)
                
                # 画边界框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            
            # 画滤波后的中心点（这就是发送到Unity的位置）
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
            cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
            cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)
            
            # 显示信息
            if magnet_detection:
                cv2.putText(frame, f"Center: ({cx}, {cy})",
                            (cx + 25, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制历史轨迹
            if len(tracker.position_history) > 1:
                points = np.array(list(tracker.position_history), dtype=np.int32)
                cv2.polylines(frame, [points], False, (255, 0, 255), 2)
            
            # 在阈值图上标记中心
            cv2.circle(thresh, (cx, cy), 8, (200, 200, 200), -1)
        
        # 如果是两点标定，画一下标定线（与原算法保持一致）
        if calibration_params['type'] == 'two_point':
            p1 = calibration_params['pixel_points'][0]
            p2 = calibration_params['pixel_points'][1]
            cv2.circle(frame, tuple(p1), 6, (255, 0, 0), -1)
            cv2.circle(frame, tuple(p2), 6, (255, 0, 0), -1)
            cv2.line(frame, tuple(p1), tuple(p2), (255, 0, 0), 2)
            cv2.putText(frame, "Calibration", (p1[0] - 50, p1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        cv2.imshow('TCP Magnet Detection [Q:quit T:thresh D:debug R:reset C:connect]', frame)
        if show_threshold:
            cv2.imshow('Threshold (T=hide)', thresh)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            show_threshold = not show_threshold
            if not show_threshold:
                cv2.destroyWindow('Threshold (T=hide)')
        elif key == ord('d'):
            debug_mode = not debug_mode
            status = "开启" if debug_mode else "关闭"
            print(f"调试模式已{status}（显示所有候选物体）")
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


