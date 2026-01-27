"""
视频检测测试脚本 - 用于测试 bimagnet.mp4 的磁铁检测效果
简练版本，仅保留核心检测功能
"""

import cv2
import numpy as np
import os
from collections import deque


class KalmanFilter:
    """卡尔曼滤波器用于平滑位置追踪"""
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.001
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
        self.initialized = False
    
    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.initialized:
            self.kalman.statePre = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], np.float32)
            self.initialized = True
            return x, y
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        return int(prediction[0][0]), int(prediction[1][0])           


class MagnetTracker:
    """磁铁追踪器"""
    def __init__(self, max_distance=150, min_frames=2, smooth_window=5):
        self.kalman = KalmanFilter()
        self.last_position = None
        self.max_distance = max_distance
        self.min_frames = min_frames
        self.detection_count = 0
        self.confirmed = False
        self.smooth_history = deque(maxlen=smooth_window)
        
    def update(self, detection):
        if detection is None:
            self.detection_count = 0
            if self.confirmed and self.last_position:
                return self.last_position, False
            return None, False
        
        cx, cy, contour, x, y, w, h = detection
        
        if self.last_position:
            distance = np.sqrt((cx - self.last_position[0])**2 + (cy - self.last_position[1])**2)
            if distance > self.max_distance:
                if self.confirmed:
                    return self.last_position, True
                return None, False
        
        self.detection_count += 1
        filtered_x, filtered_y = self.kalman.update(cx, cy)
        self.smooth_history.append((filtered_x, filtered_y))
        
        if len(self.smooth_history) >= 2:
            smoothed_pos = (int(np.mean([p[0] for p in self.smooth_history])), 
                           int(np.mean([p[1] for p in self.smooth_history])))
        else:
            smoothed_pos = (filtered_x, filtered_y)
        
        self.last_position = smoothed_pos
        if self.detection_count >= self.min_frames:
            self.confirmed = True
        
        return smoothed_pos, self.confirmed


def apply_dynamic_threshold(gray, threshold_value):
    """动态阈值"""
    mean_val = float(np.mean(gray))
    if mean_val < 60.0:
        used_thresh, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        adaptive_cap = max(40.0, min(220.0, mean_val + 20.0))
        used_thresh = int(min(threshold_value, adaptive_cap))
        _, thresh = cv2.threshold(gray, used_thresh, 255, cv2.THRESH_BINARY)
    return thresh, used_thresh


def find_magnet_position(frame, prev_position=None, threshold_value=150, min_area=60, max_area=3000):
    """识别磁铁位置"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh, _ = apply_dynamic_threshold(gray, threshold_value)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_magnet = None
    best_score = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            if 0.2 <= aspect_ratio <= 4.0:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                if circularity >= 0.25:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        score = area + circularity * 1000
                        if prev_position:
                            distance = np.sqrt((cx - prev_position[0])**2 + (cy - prev_position[1])**2)
                            if distance < 200:
                                score += (200 - distance) / 200 * 5000
                        
                        if score > best_score:
                            best_score = score
                            best_magnet = (cx, cy, contour, x, y, w, h)
    
    return best_magnet


def main():
    # 获取脚本所在目录，确保能找到视频文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "bimagnet.mp4")

    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 创建追踪器和参数窗口
    tracker = MagnetTracker()
    cv2.namedWindow('Detection')
    cv2.namedWindow('Threshold')
    cv2.createTrackbar('Threshold', 'Detection', 150, 255, lambda x: None)
    cv2.createTrackbar('Min Area', 'Detection', 60, 1000, lambda x: None)
    cv2.createTrackbar('Max Area', 'Detection', 300, 5000, lambda x: None)
    
    print("按 Q 退出，空格暂停/继续")
    
    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("视频播放完毕，重新开始")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        
        # 读取参数
        threshold_value = cv2.getTrackbarPos('Threshold', 'Detection')
        min_area = cv2.getTrackbarPos('Min Area', 'Detection')
        max_area = cv2.getTrackbarPos('Max Area', 'Detection')
        
        # 检测磁铁
        prev_pos = tracker.last_position
        magnet_detection = find_magnet_position(frame, prev_pos, threshold_value, min_area, max_area)
        filtered_pos, is_stable = tracker.update(magnet_detection)
        
        # 显示结果
        if magnet_detection:
            _, _, contour, x, y, w, h = magnet_detection
            cv2.drawContours(frame, [contour], -1, (0, 255, 0) if is_stable else (0, 165, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
        if filtered_pos:
            cx, cy = filtered_pos
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
            cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2)
            cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
            cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)
            cv2.putText(frame, f"Center: ({cx}, {cy})", (cx + 25, cy - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 状态信息
        status = "TRACKING" if is_stable else "DETECTING" if tracker.detection_count > 0 else "SEARCHING"
        cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示阈值图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh, _ = apply_dynamic_threshold(gray, threshold_value)
        if filtered_pos:
            cv2.circle(thresh, filtered_pos, 8, (200, 200, 200), -1)
        
        cv2.imshow('Detection', frame)
        cv2.imshow('Threshold', thresh)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()
    print("测试完成")


if __name__ == "__main__":
    main()
