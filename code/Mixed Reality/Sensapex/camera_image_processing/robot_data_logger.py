"""
Robot位置和速度记录器
记录4个值：位置(x, y) 和 速度(vx, vy)
从TCP端口接收位置数据，自动计算速度并保存到CSV文件
实时从distance_dataset.csv查询distance值
同时记录FSR力传感器数据（串口COM15）
"""

import socket
import time
import csv
import os
from datetime import datetime
import math
import numpy as np
from scipy.spatial import cKDTree
import threading
import json
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠ 警告: pyserial 未安装，FSR力数据记录功能将不可用")


# ==============================
# FSR AO(AD) -> 力(N) 换算（方法二：对数拟合）
# 质量(g) = -1039.1057 * ln(AD) + 7100.0651
# 力(N)   = 质量(g) * 9.81 / 1000
# ==============================
_FSR_A = -1039.1057
_FSR_B = 7100.0651
_FSR_MASS_MIN_G = 0.0
_FSR_MASS_MAX_G = 1500.0
_FSR_G = 9.81


def fsr_ad_to_force_n(ad_value):
    """将 FSR 的 AO/AD 值换算为力(N)。ad_value<=0 或 None 时返回 None。"""
    if ad_value is None:
        return None
    try:
        ad_int = int(ad_value)
    except (TypeError, ValueError):
        return None
    if ad_int <= 0:
        return None

    mass_g = _FSR_A * math.log(ad_int) + _FSR_B
    if mass_g < _FSR_MASS_MIN_G:
        mass_g = _FSR_MASS_MIN_G
    if mass_g > _FSR_MASS_MAX_G:
        mass_g = _FSR_MASS_MAX_G
    return mass_g * _FSR_G / 1000.0


class RobotDataLogger:
    """Robot数据记录器 - 记录位置和速度"""
    
    def __init__(self, tcp_host='127.0.0.1', tcp_port=5007, output_dir='robot_data', 
                 scale_factor=100, distance_dataset_path=None, 
                 fsr_port='COM15', fsr_baudrate=115200):
        """
        Args:
            tcp_host: TCP服务器地址（从camera_tcp_magnet_anchor.py接收数据）
            tcp_port: TCP端口（默认5007，与camera_tcp_magnet_anchor.py一致）
            output_dir: 输出目录
            scale_factor: 坐标放大倍数（与camera_tcp_magnet_anchor.py中的scale_factor一致，默认100）
            distance_dataset_path: distance_dataset.csv的路径（默认使用项目根目录下的文件）
            fsr_port: FSR串口名称（默认COM15）
            fsr_baudrate: FSR串口波特率（默认115200）
        """
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.output_dir = output_dir
        self.scale_factor = scale_factor
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 速度计算相关
        self.prev_x = None
        self.prev_y = None
        self.prev_time = None
        self.velocity_x = 0.0  # mm/s
        self.velocity_y = 0.0  # mm/s
        self.smoothing_alpha = 0.3  # 速度平滑系数
        self.smoothed_vx = 0.0
        self.smoothed_vy = 0.0
        
        # 数据记录
        self.csv_file = None
        self.csv_writer = None
        self.is_recording = False
        self.sock = None  # TCP客户端连接（连接到camera_tcp_magnet_anchor.py）
        self.tcp_server = None  # TCP服务器（监听来自camera_tcp_magnet_anchor.py的连接）
        self.tcp_client_conn = None  # 接受的客户端连接
        self.tcp_server_running = False
        
        # Distance查询相关
        self.distance_tree = None
        self.distance_values = None
        self.load_distance_dataset(distance_dataset_path)
        
        # FSR力传感器相关
        self.fsr_port = fsr_port
        self.fsr_baudrate = fsr_baudrate
        self.fsr_serial = None
        self.fsr_data_lock = threading.Lock()
        self.fsr_L_raw = None  # 左臂FSR原始值（A0）
        self.fsr_R_raw = None  # 右臂FSR原始值（A1）
        self.fsr_L = None  # 左臂FSR力（N）
        self.fsr_R = None  # 右臂FSR力（N）
        self.fsr_thread = None
        self.fsr_running = False
        self.fsr_last_update_time = None
        
        # 控制服务器相关（用于接收外部控制命令）
        self.control_server = None
        self.control_server_running = False
        self.control_port = 5008  # 控制命令端口
        
        # 差值数据服务器相关（用于接收位置差值数据和速度数据）
        self.error_server = None
        self.error_server_running = False
        self.error_port = 5009  # 差值数据端口
        self.error_data_lock = threading.Lock()  # 用于同步差值数据访问
        self.error_x_L = None  # 左 manipulator 的 x 差值（mm）
        self.error_y_L = None  # 左 manipulator 的 y 差值（mm）
        self.error_x_R = None  # 右 manipulator 的 x 差值（mm）
        self.error_y_R = None  # 右 manipulator 的 y 差值（mm）
        self.error_timestamp = None  # 差值数据的时间戳
        self.velocity_x_L = None  # 左 manipulator 的 x 方向速度（mm/s）
        self.velocity_y_L = None  # 左 manipulator 的 y 方向速度（mm/s）
        self.velocity_x_R = None  # 右 manipulator 的 x 方向速度（mm/s）
        self.velocity_y_R = None  # 右 manipulator 的 y 方向速度（mm/s）
        self.velocity_timestamp = None  # 速度数据的时间戳
        
        # ⭐ 独立时间戳（用于多源异步数据同步）
        self.camera_timestamp = None        # 摄像头数据时间戳
        self.geomagic_timestamp = None      # Geomagic差值数据时间戳
        self.fsr_timestamp = None           # FSR数据时间戳

        # Unity 转发相关（作为TCP客户端，将同一份位置数据转发给Unity）
        self.unity_host = '127.0.0.1'
        self.unity_port = 5006   # ⭐ Unity端口改为5006，只接收logger转发的数据
        self.unity_sock = None
        self.unity_connected = False

    # ==================== Unity 转发相关 ====================
    def connect_unity(self):
        """连接到Unity TCP接收端（作为客户端），用于转发位置数据"""
        if self.unity_connected and self.unity_sock is not None:
            return True
        try:
            if self.unity_sock:
                try:
                    self.unity_sock.close()
                except:
                    pass
            self.unity_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.unity_sock.connect((self.unity_host, self.unity_port))
            self.unity_connected = True
            print(f"✓ Unity 转发连接成功: {self.unity_host}:{self.unity_port}")
            return True
        except Exception as e:
            print(f"⚠ Unity 转发连接失败: {e}")
            self.unity_connected = False
            self.unity_sock = None
            return False

    def forward_to_unity(self, line: str):
        """
        将一行原始位置数据转发给Unity
        Args:
            line: 形如 'x,y,z' 的字符串（不含换行符）
        """
        if not line:
            return
        # 如果尚未连接，尝试连接一次（失败就跳过这帧）
        if not self.unity_connected or self.unity_sock is None:
            self.connect_unity()
            if not self.unity_connected:
                return
        try:
            msg = (line + '\n').encode('utf-8')
            self.unity_sock.sendall(msg)
        except Exception as e:
            print(f"⚠ 转发到Unity失败，将在下次尝试重连: {e}")
            try:
                if self.unity_sock:
                    self.unity_sock.close()
            except:
                pass
            self.unity_sock = None
            self.unity_connected = False
    
    def load_distance_dataset(self, distance_dataset_path=None):
        """
        加载distance_dataset.csv并构建KD树用于快速查询
        
        Args:
            distance_dataset_path: CSV文件路径，如果为None则使用默认路径
        """
        if distance_dataset_path is None:
            # 默认路径：项目根目录下的distance_dataset.csv
            # 从当前脚本位置推断项目根目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # 向上查找项目根目录（假设在 unity 目录下）
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
            distance_dataset_path = os.path.join(project_root, 'unity', 'distance_dataset.csv')
        
        if not os.path.exists(distance_dataset_path):
            print(f"⚠ 警告: distance_dataset.csv 不存在: {distance_dataset_path}")
            print("   将跳过distance查询功能")
            return
        
        try:
            print(f"正在加载 distance_dataset.csv: {distance_dataset_path}")
            # 使用numpy读取CSV（更快）
            data = np.genfromtxt(distance_dataset_path, delimiter=',', skip_header=1)
            
            # 提取x, y坐标和distance值
            x_coords = data[:, 0]
            y_coords = data[:, 1]
            distances = data[:, 2]
            
            # 构建KD树用于最近邻查询
            points = np.column_stack((x_coords, y_coords))
            self.distance_tree = cKDTree(points)
            self.distance_values = distances
            
            print(f"✓ 成功加载 {len(points)} 个distance数据点")
            print(f"   X范围: [{x_coords.min():.3f}, {x_coords.max():.3f}] mm")
            print(f"   Y范围: [{y_coords.min():.3f}, {y_coords.max():.3f}] mm")
            print(f"   Distance范围: [{distances.min():.3f}, {distances.max():.3f}] mm")
            
        except Exception as e:
            print(f"✗ 加载distance_dataset.csv失败: {e}")
            print("   将跳过distance查询功能")
            self.distance_tree = None
            self.distance_values = None
    
    def query_distance(self, x, y):
        """
        根据x, y坐标查询最近的distance值
        
        Args:
            x, y: 位置坐标（mm）
            
        Returns:
            distance值（mm），如果查询失败则返回None
        """
        if self.distance_tree is None or self.distance_values is None:
            return None
        
        try:
            # 使用KD树查找最近邻点
            query_point = np.array([x, y])
            distance, index = self.distance_tree.query(query_point, k=1)
            
            # 返回对应的distance值
            return float(self.distance_values[index])
        except Exception as e:
            # 查询失败时返回None
            return None
        
    def calculate_velocity(self, x, y, timestamp):
        """
        计算速度（mm/s）
        
        Args:
            x, y: 当前物理坐标（mm）
            timestamp: 当前时间戳（秒）
            
        Returns:
            (vx, vy) 速度（mm/s）
        """
        # 第一帧：只记录位置，不计算速度
        if self.prev_x is None or self.prev_y is None or self.prev_time is None:
            self.prev_x = x
            self.prev_y = y
            self.prev_time = timestamp
            return 0.0, 0.0
        
        # 计算时间差（秒）
        dt = timestamp - self.prev_time
        
        # 避免除零错误和异常大的时间差
        if dt <= 0 or dt > 1.0:  # 如果时间差异常，重置
            self.prev_x = x
            self.prev_y = y
            self.prev_time = timestamp
            return 0.0, 0.0
        
        # 计算位置差（mm）
        dx = x - self.prev_x
        dy = y - self.prev_y
        
        # 计算瞬时速度（mm/s）
        vx_instant = dx / dt
        vy_instant = dy / dt
        
        # 速度平滑（指数加权移动平均）
        if self.smoothed_vx == 0.0 and self.smoothed_vy == 0.0:
            # 第一次计算，直接使用瞬时速度
            self.smoothed_vx = vx_instant
            self.smoothed_vy = vy_instant
        else:
            # 平滑处理
            self.smoothed_vx = self.smoothing_alpha * vx_instant + (1 - self.smoothing_alpha) * self.smoothed_vx
            self.smoothed_vy = self.smoothing_alpha * vy_instant + (1 - self.smoothing_alpha) * self.smoothed_vy
        
        # 更新上一帧数据
        self.prev_x = x
        self.prev_y = y
        self.prev_time = timestamp
        
        return self.smoothed_vx, self.smoothed_vy
    
    def start_recording(self, filename=None):
        """开始记录数据"""
        if filename is None:
            # 自动生成文件名（带时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"robot_data_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 创建CSV文件
        self.csv_file = open(filepath, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        # 写入表头
        self.csv_writer.writerow([
            'timestamp',
            'time_elapsed',
            'position_x',
            'position_y',
            'velocity_x',
            'velocity_y',
            'velocity_magnitude',
            'distance_to_boundary',
            'timestamp_camera',
            'fsr_L_raw',       # 左臂FSR原始值（A0）
            'fsr_R_raw',       # 右臂FSR原始值（A1）
            'fsr_L',           # 左臂FSR力（N）
            'fsr_R',           # 右臂FSR力（N）
            'timestamp_fsr',
            'error_x_L',
            'error_y_L',
            'error_x_R',
            'error_y_R',
            'timestamp_geomagic',
            'geomagic_velocity_x_L',
            'geomagic_velocity_y_L',
            'geomagic_velocity_x_R',
            'geomagic_velocity_y_R',
            'timestamp_velocity'
        ])
        
        self.is_recording = True
        self.start_time = time.time()
        print(f"✓ 开始记录数据到: {filepath}")
        return filepath
    
    def stop_recording(self):
        """停止记录数据"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        self.is_recording = False
        print("✓ 数据记录已停止")
    
    def get_fsr_data(self):
        """
        获取最新的FSR数据（线程安全）
        
        Returns:
            (fsr_L_raw, fsr_R_raw, fsr_L, fsr_R) 元组
        """
        with self.fsr_data_lock:
            return self.fsr_L_raw, self.fsr_R_raw, self.fsr_L, self.fsr_R
    
    def get_latest_data(self):
        """
        获取最新的数据（供外部代码使用）
        
        Returns:
            dict: 包含最新数据的字典，如果数据不可用则返回None
            {
                'x': float,  # 位置X（mm）
                'y': float,  # 位置Y（mm）
                'vx': float,  # 速度X（mm/s）
                'vy': float,  # 速度Y（mm/s）
                'distance': float,  # 到边界的距离（mm）
                'fsr_L': float,  # 左臂FSR力（N）
                'fsr_R': float,  # 右臂FSR力（N）
                'timestamp': float  # 时间戳（秒）
            }
        """
        # 这个方法需要从 receive_and_log 中获取最新数据
        # 为了简化，我们返回当前计算的值
        # 注意：这个方法返回的是上一次接收到的数据
        if self.prev_x is None or self.prev_y is None:
            return None
        
        fsr_L_raw, fsr_R_raw, fsr_L, fsr_R = self.get_fsr_data()
        distance = self.query_distance(self.prev_x, self.prev_y)
        
        return {
            'x': self.prev_x,
            'y': self.prev_y,
            'vx': self.smoothed_vx,
            'vy': self.smoothed_vy,
            'distance': distance,
            'fsr_L_raw': fsr_L_raw,
            'fsr_R_raw': fsr_R_raw,
            'fsr_L': fsr_L,
            'fsr_R': fsr_R,
            'timestamp': self.prev_time if self.prev_time else time.time()
        }
    
    def get_error_data(self):
        """
        获取最新的差值数据和速度数据（线程安全）
        
        Returns:
            (error_x_L, error_y_L, error_x_R, error_y_R, error_timestamp,
             velocity_x_L, velocity_y_L, velocity_x_R, velocity_y_R, velocity_timestamp) 元组
        """
        with self.error_data_lock:
            return (self.error_x_L, self.error_y_L, self.error_x_R, self.error_y_R, self.error_timestamp,
                    self.velocity_x_L, self.velocity_y_L, self.velocity_x_R, self.velocity_y_R, self.velocity_timestamp)
    
    def record_data(self, x, y, vx, vy, distance=None, 
                    fsr_L_raw=None, fsr_R_raw=None, fsr_L=None, fsr_R=None, 
                    timestamp=None,
                    error_x_L=None, error_y_L=None, error_x_R=None, error_y_R=None,
                    geomagic_velocity_x_L=None, geomagic_velocity_y_L=None,
                    geomagic_velocity_x_R=None, geomagic_velocity_y_R=None):
        """
        记录一行数据
        
        Args:
            x, y: 位置（mm）
            vx, vy: 速度（mm/s）
            distance: 到边界的距离（mm）
            fsr_L_raw, fsr_R_raw: FSR原始值
            fsr_L, fsr_R: FSR力（N）
            timestamp: 时间戳（秒）
        """
        if not self.is_recording or self.csv_writer is None:
            return
        
        # 使用同一个时间戳记录所有数据（如果未提供则使用当前时间）
        if timestamp is None:
            current_time = time.time()
        else:
            current_time = timestamp
        elapsed_time = current_time - self.start_time
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        
        # 查询distance
        if distance is None:
            distance = self.query_distance(x, y)
        
        # 获取FSR数据
        if fsr_L_raw is None or fsr_R_raw is None or fsr_L is None or fsr_R is None:
            fsr_l_raw, fsr_r_raw, fsr_l, fsr_r = self.get_fsr_data()
            if fsr_L_raw is None:
                fsr_L_raw = fsr_l_raw
            if fsr_R_raw is None:
                fsr_R_raw = fsr_r_raw
            if fsr_L is None:
                fsr_L = fsr_l
            if fsr_R is None:
                fsr_R = fsr_r
        
        # 获取差值数据和速度数据（如果未提供）
        if (error_x_L is None or error_y_L is None or error_x_R is None or error_y_R is None or
            geomagic_velocity_x_L is None or geomagic_velocity_y_L is None or
            geomagic_velocity_x_R is None or geomagic_velocity_y_R is None):
            err_x_L, err_y_L, err_x_R, err_y_R, err_ts, vel_x_L, vel_y_L, vel_x_R, vel_y_R, vel_ts = self.get_error_data()
            if error_x_L is None:
                error_x_L = err_x_L
            if error_y_L is None:
                error_y_L = err_y_L
            if error_x_R is None:
                error_x_R = err_x_R
            if error_y_R is None:
                error_y_R = err_y_R
            if geomagic_velocity_x_L is None:
                geomagic_velocity_x_L = vel_x_L
            if geomagic_velocity_y_L is None:
                geomagic_velocity_y_L = vel_y_L
            if geomagic_velocity_x_R is None:
                    geomagic_velocity_x_R = vel_x_R
            if geomagic_velocity_y_R is None:
                geomagic_velocity_y_R = vel_y_R
        
        # 写入CSV
        try:
            self.csv_writer.writerow([
                f"{current_time:.6f}",
                f"{elapsed_time:.6f}",
                f"{x:.6f}",
                f"{y:.6f}",
                f"{vx:.6f}",
                f"{vy:.6f}",
                f"{velocity_magnitude:.6f}",
                f"{distance:.6f}" if distance is not None else "",
                f"{self.camera_timestamp:.6f}" if self.camera_timestamp is not None else "",
                f"{fsr_L_raw}" if fsr_L_raw is not None else "",
                f"{fsr_R_raw}" if fsr_R_raw is not None else "",
                f"{fsr_L:.6f}" if fsr_L is not None else "",
                f"{fsr_R:.6f}" if fsr_R is not None else "",
                f"{self.fsr_timestamp:.6f}" if self.fsr_timestamp is not None else "",
                f"{error_x_L:.6f}" if error_x_L is not None else "",
                f"{error_y_L:.6f}" if error_y_L is not None else "",
                f"{error_x_R:.6f}" if error_x_R is not None else "",
                f"{error_y_R:.6f}" if error_y_R is not None else "",
                f"{self.geomagic_timestamp:.6f}" if self.geomagic_timestamp is not None else "",
                f"{geomagic_velocity_x_L:.6f}" if geomagic_velocity_x_L is not None else "",
                f"{geomagic_velocity_y_L:.6f}" if geomagic_velocity_y_L is not None else "",
                f"{geomagic_velocity_x_R:.6f}" if geomagic_velocity_x_R is not None else "",
                f"{geomagic_velocity_y_R:.6f}" if geomagic_velocity_y_R is not None else "",
                f"{self.velocity_timestamp:.6f}" if self.velocity_timestamp is not None else ""
            ])
            self.csv_file.flush()
        except Exception as e:
            print(f"\n[ERROR] CSV写入失败: {e}")
            import traceback
            traceback.print_exc()
    
    def connect_fsr(self):
        """连接FSR串口"""
        if not SERIAL_AVAILABLE:
            print("⚠ FSR串口功能不可用（pyserial未安装）")
            return False
        
        try:
            self.fsr_serial = serial.Serial(self.fsr_port, self.fsr_baudrate, timeout=1)
            print(f"✓ FSR串口连接成功: {self.fsr_port} @ {self.fsr_baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"✗ FSR串口连接失败: {e}")
            print(f"   请检查串口 {self.fsr_port} 是否存在且未被占用")
            return False
        except Exception as e:
            print(f"✗ FSR串口连接失败: {e}")
            return False
    
    def start_fsr_thread(self):
        """启动FSR数据读取线程"""
        if self.fsr_serial is None:
            return False
        
        self.fsr_running = True
        self.fsr_thread = threading.Thread(target=self._fsr_read_loop, daemon=True)
        self.fsr_thread.start()
        print("✓ FSR数据读取线程已启动")
        return True
    
    def _fsr_read_loop(self):
        """FSR数据读取循环（在独立线程中运行）"""
        while self.fsr_running and self.fsr_serial is not None:
            try:
                line = self.fsr_serial.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                
                # 解析数据：格式 "A0=xxx,A1=yyy"
                pairs = {}
                for part in line.split(','):
                    if '=' not in part:
                        continue
                    key, value = part.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        pairs[key] = int(value)
                    except ValueError:
                        continue
                
                if 'A0' in pairs and 'A1' in pairs:
                    # 保存原始值和力值
                    a0_raw = pairs['A0']
                    a1_raw = pairs['A1']
                    fsr_L = fsr_ad_to_force_n(a0_raw)
                    fsr_R = fsr_ad_to_force_n(a1_raw)
                    
                    # 线程安全地更新数据
                    with self.fsr_data_lock:
                        self.fsr_L_raw = a0_raw
                        self.fsr_R_raw = a1_raw
                        self.fsr_L = fsr_L
                        self.fsr_R = fsr_R
                        self.fsr_last_update_time = time.time()
                        self.fsr_timestamp = time.time()
                        
            except serial.SerialException:
                break
            except Exception:
                continue
    
    def stop_fsr_thread(self):
        """停止FSR数据读取线程"""
        self.fsr_running = False
        if self.fsr_thread and self.fsr_thread.is_alive():
            self.fsr_thread.join(timeout=2.0)
    
    def start_tcp_server(self):
        """启动TCP服务器（监听来自camera_tcp_magnet_anchor.py的连接）"""
        def tcp_server_loop():
            """TCP服务器循环"""
            try:
                self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # ⭐ 检查端口是否已被占用
                test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    test_sock.bind(('127.0.0.1', self.tcp_port))
                    test_sock.close()
                except OSError:
                    print(f"✗ 错误: 端口 {self.tcp_port} 已被占用！")
                    print(f"   可能已有另一个 robot_data_logger.py 实例在运行")
                    print(f"   请先关闭其他实例，或使用任务管理器结束相关进程")
                    return
                
                self.tcp_server.bind(('127.0.0.1', self.tcp_port))
                self.tcp_server.listen(1)
                self.tcp_server.settimeout(1.0)  # 1秒超时，用于检查停止标志
                print(f"✓ TCP服务器已启动: 127.0.0.1:{self.tcp_port}")
                print("   等待 camera_tcp_magnet_anchor.py 连接...")
                
                while self.tcp_server_running:
                    try:
                        conn, addr = self.tcp_server.accept()
                        print(f"✓ 收到来自 {addr} 的TCP连接")
                        self.tcp_client_conn = conn
                        self.tcp_client_conn.settimeout(0.1)  # 接收数据时使用100ms超时
                        
                        # 保持连接，持续接收数据
                        while self.tcp_server_running and self.tcp_client_conn:
                            try:
                                data = self.tcp_client_conn.recv(1024)
                                if not data:
                                    # 连接断开
                                    print(f"[WARN] TCP客户端 {addr} 断开连接")
                                    self.tcp_client_conn.close()
                                    self.tcp_client_conn = None
                                    break
                                
                                # 处理接收到的数据
                                self._process_tcp_data(data)
                            except socket.timeout:
                                # 超时是正常的，继续循环
                                continue
                            except Exception as e:
                                print(f"[WARN] TCP接收数据错误: {e}")
                                self.tcp_client_conn.close()
                                self.tcp_client_conn = None
                                break
                    except socket.timeout:
                        # 超时是正常的，继续循环检查停止标志
                        continue
                    except Exception as e:
                        # 忽略连接错误，继续运行
                        continue
            except Exception as e:
                print(f"✗ TCP服务器错误: {e}")
            finally:
                if self.tcp_server:
                    try:
                        self.tcp_server.close()
                    except:
                        pass
                    self.tcp_server = None
        
        self.tcp_server_running = True
        tcp_thread = threading.Thread(target=tcp_server_loop, daemon=True)
        tcp_thread.start()
        return True
    
    def stop_tcp_server(self):
        """停止TCP服务器"""
        self.tcp_server_running = False
        if self.tcp_client_conn:
            try:
                self.tcp_client_conn.close()
            except:
                pass
            self.tcp_client_conn = None
        if self.tcp_server:
            try:
                self.tcp_server.close()
            except:
                pass
            self.tcp_server = None
    
    def _process_tcp_data(self, data):
        """处理接收到的TCP数据"""
        if not data:
            return
        
        # 解析数据：格式 "x,y,z\n"
        try:
            message = data.decode('utf-8').strip()
        except UnicodeDecodeError:
            print(f"\n✗ 数据解码失败: {data}")
            return
        
        # 处理可能的多行数据
        lines = message.split('\n')
        for line in lines:
            if not line.strip():
                continue
                
            parts = line.split(',')
            
            if len(parts) >= 2:
                try:
                    # 解析位置（注意：camera_tcp_magnet_anchor.py发送的是放大后的值）
                    x_scaled = float(parts[0])
                    y_scaled = float(parts[1])
                    
                    # 还原为实际物理坐标（mm）- 需要除以scale_factor
                    x = x_scaled / self.scale_factor
                    y = y_scaled / self.scale_factor
                    
                    # 获取时间戳
                    current_time = time.time()
                    
                    # 计算速度
                    vx, vy = self.calculate_velocity(x, y, current_time)
                    
                    # ⭐ 记录摄像头数据时间戳
                    self.camera_timestamp = current_time
                    
                    # 查询distance
                    distance = self.query_distance(x, y)
                    
                    # 获取FSR数据
                    fsr_L_raw, fsr_R_raw, fsr_L, fsr_R = self.get_fsr_data()
                    
                    # 获取差值数据和速度数据
                    error_x_L, error_y_L, error_x_R, error_y_R, error_ts, \
                    geomagic_velocity_x_L, geomagic_velocity_y_L, \
                    geomagic_velocity_x_R, geomagic_velocity_y_R, velocity_ts = self.get_error_data()
                    
                    # 记录数据
                    if self.is_recording:
                        self.record_data(x, y, vx, vy, distance, 
                                       fsr_L_raw, fsr_R_raw, fsr_L, fsr_R,
                                       timestamp=current_time,
                                       error_x_L=error_x_L, error_y_L=error_y_L,
                                       error_x_R=error_x_R, error_y_R=error_y_R,
                                       geomagic_velocity_x_L=geomagic_velocity_x_L,
                                       geomagic_velocity_y_L=geomagic_velocity_y_L,
                                       geomagic_velocity_x_R=geomagic_velocity_x_R,
                                       geomagic_velocity_y_R=geomagic_velocity_y_R)
                    
                    # 转发给Unity
                    self.forward_to_unity(line)

                    # 打印（每0.1秒打印一次）
                    if not hasattr(self, 'last_print_time'):
                        self.last_print_time = 0
                    if current_time - self.last_print_time > 0.1:
                        velocity_magnitude = np.sqrt(vx**2 + vy**2)
                        distance_str = f"{distance:.3f}" if distance is not None else "N/A"
                        if fsr_L is not None:
                            fsr_str = f"L:{fsr_L_raw}({fsr_L:.2f}N) R:{fsr_R_raw}({fsr_R:.2f}N)"
                        else:
                            fsr_str = "N/A"
                        print(f"位置: ({x:.2f}, {y:.2f}) mm | "
                              f"速度: ({vx:.2f}, {vy:.2f}) mm/s | "
                              f"速度大小: {velocity_magnitude:.2f} mm/s | "
                              f"距离: {distance_str} mm | "
                              f"FSR: {fsr_str}", end='\r')
                        self.last_print_time = current_time
                except ValueError as e:
                    print(f"\n✗ 解析数据失败: {e}, 数据: {line}")
            else:
                if line.strip():  # 只打印非空行
                    print(f"\n✗ 数据格式错误: {line}")
    
    def receive_and_log(self):
        """接收TCP数据并记录（已废弃，数据现在在TCP服务器线程中处理）"""
        # 这个方法现在不再使用，数据在TCP服务器线程的_process_tcp_data中处理
        # 保留这个方法是为了兼容性，但实际不做任何事情
        return False
    
    def start_control_server(self):
        """启动控制服务器（用于接收外部控制命令）"""
        def control_server_loop():
            """控制服务器循环"""
            try:
                self.control_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.control_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.control_server.bind(('127.0.0.1', self.control_port))
                self.control_server.listen(1)
                self.control_server.settimeout(1.0)  # 1秒超时，用于检查停止标志
                print(f"✓ 控制服务器已启动: 127.0.0.1:{self.control_port}")
                
                while self.control_server_running:
                    try:
                        conn, addr = self.control_server.accept()
                        data = conn.recv(1024).decode('utf-8').strip()
                        conn.close()
                        
                        if data == "START":
                            if not self.is_recording:
                                self.start_recording()
                        elif data == "STOP":
                            if self.is_recording:
                                self.stop_recording()
                        elif data == "STATUS":
                            status = "RECORDING" if self.is_recording else "IDLE"
                            print(f"[控制] 状态: {status}")
                    except socket.timeout:
                        # 超时是正常的，继续循环检查停止标志
                        continue
                    except Exception as e:
                        # 忽略连接错误，继续运行
                        continue
            except Exception as e:
                print(f"✗ 控制服务器错误: {e}")
            finally:
                if self.control_server:
                    try:
                        self.control_server.close()
                    except:
                        pass
        
        self.control_server_running = True
        control_thread = threading.Thread(target=control_server_loop, daemon=True)
        control_thread.start()
        return True
    
    def stop_control_server(self):
        """停止控制服务器"""
        self.control_server_running = False
        if self.control_server:
            try:
                self.control_server.close()
            except:
                pass
            self.control_server = None
    
    def start_error_server(self):
        """启动差值数据服务器（用于接收位置差值数据）"""
        def error_server_loop():
            """差值数据服务器循环"""
            try:
                self.error_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.error_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.error_server.bind(('127.0.0.1', self.error_port))
                self.error_server.listen(1)
                self.error_server.settimeout(1.0)  # 1秒超时，用于检查停止标志
                print(f"✓ 差值数据服务器已启动: 127.0.0.1:{self.error_port}")
                
                while self.error_server_running:
                    try:
                        conn, addr = self.error_server.accept()
                        data = conn.recv(1024).decode('utf-8').strip()
                        conn.close()
                        
                        # 解析差值数据：格式 "ERROR,hand,error_x_mm,error_y_mm,timestamp"
                        if data.startswith("ERROR,"):
                            parts = data.split(',')
                            if len(parts) == 5:
                                try:
                                    _, hand, error_x_str, error_y_str, timestamp_str = parts
                                    error_x = float(error_x_str)
                                    error_y = float(error_y_str)
                                    error_ts = float(timestamp_str)
                                    
                                    # 线程安全地更新差值数据
                                    with self.error_data_lock:
                                        if hand == "L":
                                            self.error_x_L = error_x
                                            self.error_y_L = error_y
                                        elif hand == "R":
                                            self.error_x_R = error_x
                                            self.error_y_R = error_y
                                        self.error_timestamp = error_ts
                                        self.geomagic_timestamp = time.time()
                                except (ValueError, IndexError):
                                    # 解析失败，忽略
                                    continue
                        
                        # 解析速度数据：格式 "VELOCITY,hand,vx_mm_s,vy_mm_s,timestamp"
                        elif data.startswith("VELOCITY,"):
                            parts = data.split(',')
                            if len(parts) == 5:
                                try:
                                    _, hand, vx_str, vy_str, timestamp_str = parts
                                    vx = float(vx_str)
                                    vy = float(vy_str)
                                    vel_ts = float(timestamp_str)
                                    
                                    # 线程安全地更新速度数据
                                    with self.error_data_lock:
                                        if hand == "L":
                                            self.velocity_x_L = vx
                                            self.velocity_y_L = vy
                                        elif hand == "R":
                                            self.velocity_x_R = vx
                                            self.velocity_y_R = vy
                                        self.velocity_timestamp = vel_ts
                                except (ValueError, IndexError):
                                    # 解析失败，忽略
                                    continue
                    except socket.timeout:
                        # 超时是正常的，继续循环检查停止标志
                        continue
                    except Exception as e:
                        # 忽略连接错误，继续运行
                        continue
            except Exception as e:
                print(f"✗ 差值数据服务器错误: {e}")
            finally:
                if self.error_server:
                    try:
                        self.error_server.close()
                    except:
                        pass
        
        self.error_server_running = True
        error_thread = threading.Thread(target=error_server_loop, daemon=True)
        error_thread.start()
        return True
    
    def stop_error_server(self):
        """停止差值数据服务器"""
        self.error_server_running = False
        if self.error_server:
            try:
                self.error_server.close()
            except:
                pass
            self.error_server = None
    
    def close(self):
        """关闭连接和文件"""
        # 停止控制服务器
        self.stop_control_server()
        
        # 停止差值数据服务器
        self.stop_error_server()
        
        # 停止FSR线程
        self.stop_fsr_thread()
        
        # 关闭FSR串口
        if self.fsr_serial:
            try:
                self.fsr_serial.close()
            except:
                pass
            self.fsr_serial = None
        
        # 停止TCP服务器
        self.stop_tcp_server()

        # 关闭Unity转发socket
        if self.unity_sock:
            try:
                self.unity_sock.close()
            except:
                pass
            self.unity_sock = None
            self.unity_connected = False
        
        # 停止记录
        self.stop_recording()


def main():
    """主函数 - 只读取数据，不自动存储"""
    print("=" * 70)
    print("    Robot位置和速度数据读取器")
    print("=" * 70)
    print("\n功能说明:")
    print("  - 从TCP端口5007接收位置数据（来自camera_tcp_magnet_anchor.py）")
    print("  - 自动计算速度（vx, vy）")
    print("  - 实时查询distance_to_boundary（从distance_dataset.csv）")
    print("  - 同时读取FSR力传感器数据（串口COM15）")
    print("  - 所有数据使用同一个时间戳，确保同步")
    print("  - 当前模式：只读取数据，不自动存储")
    print("  - 存储功能由外部代码通过 start_recording() 和 stop_recording() 控制")
    print("\n操作说明:")
    print("  - 按 Ctrl+C 退出程序")
    print("  - 数据存储由外部代码控制")
    print("=" * 70)
    
    # 创建记录器
    logger = RobotDataLogger(
        tcp_host='127.0.0.1', 
        tcp_port=5007,
        output_dir='robot_data',
        scale_factor=100,  # 与camera_tcp_magnet_anchor.py中的scale_factor一致
        distance_dataset_path=r'D:\project\individual_project\unity\distance_dataset.csv'
    )
    
    try:
        # 启动TCP服务器（等待camera_tcp_magnet_anchor.py连接）
        print("\n正在启动TCP服务器...")
        logger.start_tcp_server()
        print("  - TCP端口: 5007")
        print("  - 等待 camera_tcp_magnet_anchor.py 连接...")
        
        # 连接到FSR串口
        print("\n正在连接到FSR串口...")
        fsr_connected = logger.connect_fsr()
        if fsr_connected:
            logger.start_fsr_thread()
        else:
            print("⚠ 警告: FSR数据读取功能将不可用")
        
        # 启动控制服务器（用于接收外部控制命令）
        print("\n正在启动控制服务器...")
        logger.start_control_server()
        print("  - 控制端口: 5008")
        print("  - 命令: START (开始记录), STOP (停止记录)")
        
        # 启动差值数据服务器（用于接收位置差值数据）
        print("\n正在启动差值数据服务器...")
        logger.start_error_server()
        print("  - 差值数据端口: 5009")
        print("  - 接收 Geomagic Touch 与 UMP 的坐标差值")
        
        print("\n" + "=" * 70)
        print("TCP服务器已启动，等待数据...")
        print("等待外部控制命令（START/STOP）...")
        print("按 Ctrl+C 退出")
        print("=" * 70 + "\n")
        
        # 主循环：TCP数据在服务器线程中处理，这里只需要保持程序运行
        # 外部代码可以通过发送START命令来开始记录
        while True:
            time.sleep(0.1)  # 主循环只需要保持程序运行
            
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.close()
        print("\n程序结束")


if __name__ == "__main__":
    main()