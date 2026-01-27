import os
import csv
import time
import socket
import threading
from datetime import datetime
from sensapex import UMP
import keyboard

# === 配置参数 ===
UDP_SEND_IP = "127.0.0.1"
UDP_SEND_PORT = 5006
UDP_RECV_PORT = 5005  # 接收 geomagic touch 数据的端口
SCALE_FACTOR = 200  # ⭐ 缩放比例：Geomagic Touch 移动 1 单位 → 微控制器移动 0.5mm
SEND_INTERVAL = 1.0 / 60.0  # 60Hz 发送频率
UDP_TIMEOUT = 0.15  # ⭐ UDP接收超时时间（秒），Unity每0.1s发送一次，设置0.15s确保能检测到按钮释放
DATA_LOGGER_CONTROL_PORT = 5008  # robot_data_logger 控制端口
DATA_LOGGER_ERROR_PORT = 5009  # robot_data_logger 差值数据端口

# === UMP 初始位置和单位转换 ===
UMP_INITIAL_POS = (7000, 7000, 20000)  # UMP 初始位置 (x, y, z)
UMP_TO_MM = 18 / 17000  # 单位转换：17000 UMP 单位 = 18mm

# === 创建保存文件夹 ===
os.makedirs("left", exist_ok=True)
os.makedirs("right", exist_ok=True)

# === 初始化 Sensapex 设备 ===
ump = UMP.get_ump()
manipulator_L = ump.get_device(1)
manipulator_R = ump.get_device(2)

# === 建立 UDP 通信 ===
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind(("0.0.0.0", UDP_RECV_PORT))
sock_recv.settimeout(UDP_TIMEOUT)  # ⭐ 设置超时，用于检测按钮释放

# === 状态共享变量 ===
last_pos_L = None
last_pos_R = None
stop_event = threading.Event()
start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# ⭐ 按钮状态标志（用于检测按钮是否按下）
button_pressed_L = False
button_pressed_R = False
button_lock = threading.Lock()  # 用于线程安全访问按钮状态
# ⭐ 上一次按钮状态（用于检测边沿）
prev_button_pressed_L = False
prev_button_pressed_R = False
# ⭐ 控制激活状态（切换模式）
control_active = False  # False=停止控制, True=开始控制
control_lock = threading.Lock()  # 用于线程安全访问控制状态

# ⭐ Geomagic Touch 累积位移（相对于按下按钮时的位置）
cumulative_dx_L = 0.0  # 左 manipulator 的 x 方向累积位移
cumulative_dz_L = 0.0  # 左 manipulator 的 z 方向累积位移（对应 UMP 的 y）
cumulative_dx_R = 0.0  # 右 manipulator 的 x 方向累积位移
cumulative_dz_R = 0.0  # 右 manipulator 的 z 方向累积位移（对应 UMP 的 y）
cumulative_lock = threading.Lock()  # 用于线程安全访问累积值

# ⭐ Geomagic Touch 速度计算变量
prev_time_L = None  # 左 manipulator 上一次的时间戳
prev_time_R = None  # 右 manipulator 上一次的时间戳
prev_dx_L = 0.0  # 左 manipulator 上一次的 dx
prev_dz_L = 0.0  # 左 manipulator 上一次的 dz
prev_dx_R = 0.0  # 右 manipulator 上一次的 dx
prev_dz_R = 0.0  # 右 manipulator 上一次的 dz
velocity_lock = threading.Lock()  # 用于线程安全访问速度计算变量

# ⭐ 启动延迟标志（避免程序启动时误触发按钮边沿检测）
startup_delay_passed = False
startup_delay_time = 2.0  # 启动后2秒内忽略按钮边沿检测
program_start_time = time.time()
# ⭐ 等待按钮释放标志（延迟结束后，如果按钮是按下状态，等待释放）
waiting_for_button_release = False
waiting_for_button_release_start_time = None  # 进入等待释放模式的时间
BUTTON_RELEASE_WAIT_TIMEOUT = 5.0  # 等待按钮释放的超时时间（秒）

# === 文件句柄和Writer初始化 ===
left_file = open(f"left/{start_time}.csv", "w", newline='')
right_file = open(f"right/{start_time}.csv", "w", newline='')
left_writer = csv.writer(left_file)
right_writer = csv.writer(right_file)
left_writer.writerow(['timestamp', 'dx', 'dy'])
right_writer.writerow(['timestamp', 'dx', 'dy'])

# === 坐标转换函数 ===
def map_position(raw_value):
    return (((raw_value - 1) / (20000 - 1)) * 20 - 10) / 100

# === 发送控制命令到数据记录器 ===
def send_data_logger_command(command):
    """发送控制命令到 robot_data_logger"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)  # 0.5秒超时
        sock.connect(('127.0.0.1', DATA_LOGGER_CONTROL_PORT))
        sock.sendall(command.encode('utf-8'))
        sock.close()
        return True
    except Exception as e:
        # 如果连接失败，可能是数据记录器未运行，静默失败
        return False

# === 发送位置差值到数据记录器 ===
def send_position_error_to_logger(hand, error_x_mm, error_y_mm, timestamp):
    """发送位置差值数据到 robot_data_logger"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)  # 0.1秒超时
        sock.connect(('127.0.0.1', DATA_LOGGER_ERROR_PORT))
        # 格式: "ERROR,hand,error_x_mm,error_y_mm,timestamp"
        message = f"ERROR,{hand},{error_x_mm:.6f},{error_y_mm:.6f},{timestamp:.6f}\n"
        sock.sendall(message.encode('utf-8'))
        sock.close()
        return True
    except Exception as e:
        # 如果连接失败，静默失败（数据记录器可能未运行）
        return False

# === 发送速度数据到数据记录器 ===
def send_velocity_to_logger(hand, vx_mm_s, vy_mm_s, timestamp):
    """发送速度数据到 robot_data_logger"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)  # 0.1秒超时
        sock.connect(('127.0.0.1', DATA_LOGGER_ERROR_PORT))
        # 格式: "VELOCITY,hand,vx_mm_s,vy_mm_s,timestamp"
        message = f"VELOCITY,{hand},{vx_mm_s:.6f},{vy_mm_s:.6f},{timestamp:.6f}\n"
        sock.sendall(message.encode('utf-8'))
        sock.close()
        return True
    except Exception as e:
        # 如果连接失败，静默失败（数据记录器可能未运行）
        return False

# === 线程1: 读取位移并发送 + 记录 ===
def read_and_send():
    global last_pos_L, last_pos_R
    while not stop_event.is_set():
        start_loop = time.time()

        raw_L = manipulator_L.get_pos()
        raw_R = manipulator_R.get_pos()
        pos_L = tuple(map_position(x) for x in raw_L[:3])
        pos_R = tuple(map_position(x) for x in raw_R[:3])

        if last_pos_L is not None and last_pos_R is not None:
            # 只计算并发送 x、y 两个方向的位移
            disp_L = (pos_L[0] - last_pos_L[0], pos_L[1] - last_pos_L[1])
            disp_R = (pos_R[0] - last_pos_R[0], pos_R[1] - last_pos_R[1])

            # 发送 UDP（只包含 x、y）
            msg_L = f"L,{disp_L[0]},{disp_L[1]}"
            msg_R = f"R,{disp_R[0]},{disp_R[1]}"
            sock_send.sendto(msg_L.encode(), (UDP_SEND_IP, UDP_SEND_PORT))
            sock_send.sendto(msg_R.encode(), (UDP_SEND_IP, UDP_SEND_PORT))

            # 写入 CSV（只记录 x、y）
            timestamp = time.time()
            left_writer.writerow([timestamp, disp_L[0], disp_L[1]])
            right_writer.writerow([timestamp, disp_R[0], disp_R[1]])
            left_file.flush()
            right_file.flush()

        last_pos_L = pos_L
        last_pos_R = pos_R

        elapsed = time.time() - start_loop
        time.sleep(max(0, SEND_INTERVAL - elapsed))

# === 线程2: 接收 Geomagic Touch 数据并控制微控制器 ===
def receive_geomagic_and_control():
    global button_pressed_L, button_pressed_R, prev_button_pressed_L, prev_button_pressed_R, control_active
    global cumulative_dx_L, cumulative_dz_L, cumulative_dx_R, cumulative_dz_R
    global prev_time_L, prev_time_R, prev_dx_L, prev_dz_L, prev_dx_R, prev_dz_R
    global startup_delay_passed, waiting_for_button_release, waiting_for_button_release_start_time
    while not stop_event.is_set():
        try:
            # ⭐ 尝试接收UDP数据（带超时）
            data, _ = sock_recv.recvfrom(1024)
            decoded = data.decode().split(",")
            
            # ⭐ 支持两种格式：
            # 格式1: hand,dx,dz,dy,button (5个字段，包含按钮状态)
            # 格式2: hand,dx,dz,dy (4个字段，收到数据即表示按钮按下)
            if len(decoded) == 5:
                # 格式1：包含按钮状态
                hand, dx, dz, dy, button = decoded[0], float(decoded[1]), float(decoded[2]), float(decoded[3]), int(decoded[4])
                button_pressed = (button == 1)
            elif len(decoded) == 4:
                # 格式2：收到数据即表示按钮按下（Unity只在按钮按下时发送）
                hand, dx, dz, dy = decoded[0], float(decoded[1]), float(decoded[2]), float(decoded[3])
                button_pressed = True
            else:
                continue
            
            # ⭐ 检测按钮按下边沿（从释放到按下），切换控制状态
            with button_lock:
                # ⭐ 检查启动延迟是否已过
                global startup_delay_passed, waiting_for_button_release
                just_passed_delay = False
                if not startup_delay_passed:
                    elapsed = time.time() - program_start_time
                    if elapsed >= startup_delay_time:
                        startup_delay_passed = True
                        just_passed_delay = True
                        # 启动延迟刚结束时，检查按钮状态
                        if button_pressed_L or button_pressed_R:
                            # 如果按钮当前是按下状态，进入等待释放模式
                            waiting_for_button_release = True
                            waiting_for_button_release_start_time = time.time()  # 记录进入等待模式的时间
                            prev_button_pressed_L = button_pressed_L
                            prev_button_pressed_R = button_pressed_R
                            print("\n[系统] 启动延迟已过")
                            print(f"[系统] 检测到按钮当前为按下状态，等待按钮释放...（超时时间：{BUTTON_RELEASE_WAIT_TIMEOUT}秒）")
                        else:
                            # 如果按钮当前是释放状态，直接开始正常检测
                            waiting_for_button_release = False
                            waiting_for_button_release_start_time = None
                            prev_button_pressed_L = False
                            prev_button_pressed_R = False
                            print("\n[系统] 启动延迟已过，可以开始使用按钮控制")
                
                # ⭐ 检查是否在等待按钮释放模式
                if waiting_for_button_release:
                    # 检查是否超时
                    if waiting_for_button_release_start_time is not None:
                        elapsed_wait_time = time.time() - waiting_for_button_release_start_time
                        if elapsed_wait_time >= BUTTON_RELEASE_WAIT_TIMEOUT:
                            # 超时，强制退出等待释放模式
                            waiting_for_button_release = False
                            waiting_for_button_release_start_time = None
                            prev_button_pressed_L = False
                            prev_button_pressed_R = False
                            print(f"\n[系统] 等待按钮释放超时（{BUTTON_RELEASE_WAIT_TIMEOUT}秒），强制退出等待模式")
                            print("[系统] 可以开始使用按钮控制（下次按下按钮将开始控制）")
                    
                    # 等待释放模式：只更新按钮状态，不检测边沿
                    if hand == "L":
                        button_pressed_L = button_pressed
                    elif hand == "R":
                        button_pressed_R = button_pressed
                    # 如果按钮已释放（当前状态为 False），退出等待释放模式
                    if not button_pressed_L and not button_pressed_R:
                        waiting_for_button_release = False
                        waiting_for_button_release_start_time = None
                        prev_button_pressed_L = False
                        prev_button_pressed_R = False
                        print("[系统] 按钮已释放，可以开始使用按钮控制")
                # ⭐ 只有在启动延迟过后且不在等待释放模式时才处理按钮状态和边沿检测
                # ⭐ 如果刚刚通过延迟，跳过本次循环的边沿检测，避免立即触发
                elif startup_delay_passed and not just_passed_delay:
                    # 检测任意按钮的按下边沿
                    prev_any_button_pressed = prev_button_pressed_L or prev_button_pressed_R
                    current_any_button_pressed = False
                    
                    if hand == "L":
                        current_any_button_pressed = button_pressed
                        button_pressed_L = button_pressed
                    elif hand == "R":
                        current_any_button_pressed = button_pressed
                        button_pressed_R = button_pressed
                    
                    # 检测按下边沿：之前按钮释放，现在按钮按下
                    button_edge_detected = (current_any_button_pressed and 
                                           not prev_any_button_pressed)
                    
                    # 调试信息（可选，用于诊断）
                    # print(f"[调试] prev={prev_any_button_pressed}, current={current_any_button_pressed}, edge={button_edge_detected}")
                    
                    if button_edge_detected:
                        # 切换控制状态
                        with control_lock:
                            control_active = not control_active
                            new_state = control_active
                        
                        # 根据新状态发送控制命令和重置累积值
                        if new_state:
                            # 第一次按下：开始控制和记录
                            # 重置累积值（Geomagic Touch 的 dx 和 dz 从 0 开始累积）
                            with cumulative_lock:
                                cumulative_dx_L = 0.0
                                cumulative_dz_L = 0.0
                                cumulative_dx_R = 0.0
                                cumulative_dz_R = 0.0
                            
                            # 重置速度计算变量
                            with velocity_lock:
                                prev_time_L = None
                                prev_time_R = None
                                prev_dx_L = 0.0
                                prev_dz_L = 0.0
                                prev_dx_R = 0.0
                                prev_dz_R = 0.0
                            
                            send_data_logger_command("START")
                            print("\n[控制] 实验开始：开始控制UMP并记录数据")
                        else:
                            # 第二次按下：停止控制和记录
                            send_data_logger_command("STOP")
                            print("\n[控制] 实验停止：停止控制UMP并停止记录数据")
                    
                    # 更新上一次按钮状态（只有在启动延迟过后才更新）
                    prev_button_pressed_L = button_pressed_L
                    prev_button_pressed_R = button_pressed_R
                else:
                    # 启动延迟期间：只更新按钮状态，但不更新 prev 状态，也不检测边沿
                    if hand == "L":
                        button_pressed_L = button_pressed
                    elif hand == "R":
                        button_pressed_R = button_pressed
                    # 不更新 prev_button_pressed，这样延迟结束后第一次真正的按钮按下才会触发边沿
            
            # ⭐ 只有控制激活且按钮按下时才执行控制
            # ⭐ 同时需要确保启动延迟已过
            with control_lock:
                is_control_active = control_active
            
            if not is_control_active or not button_pressed or not startup_delay_passed:
                continue
            
            # 解析数据：hand, dx, dz, dy
            # dx = geomagic touch 的 x 坐标
            # dz = geomagic touch 的 z 坐标
            # dy = geomagic touch 的 y 坐标（忽略，不使用）

            if hand == "L" and button_pressed_L:
                manipulator_L.stop()
                current = list(manipulator_L.get_pos())
                # geomagic touch 的 x -> 微控制器的 x
                new_x = min(max(0, current[0] - dx * SCALE_FACTOR), 20000)
                # geomagic touch 的 z+ -> 微控制器的 y- (左臂y方向反转)
                new_y = min(max(0, current[1] - dz * SCALE_FACTOR), 20000)
                # z 轴保持当前位置不变
                manipulator_L.goto_pos((new_x, new_y, current[2], current[3]), speed=5000)
                
                current_time = time.time()
                
                # ⭐ 计算速度
                vx_mm_s = 0.0
                vy_mm_s = 0.0
                with velocity_lock:
                    if prev_time_L is not None:
                        dt = current_time - prev_time_L
                        if dt > 0:
                            # 计算速度：dx 和 dz 是增量值，需要除以时间间隔
                            # dx 和 dz 是 Geomagic Touch 坐标系中的值，需要转换为 UMP 坐标系，再转换为 mm
                            dx_ump = dx * SCALE_FACTOR  # 转换为 UMP 单位
                            dz_ump = dz * SCALE_FACTOR  # 转换为 UMP 单位
                            dx_mm = dx_ump * UMP_TO_MM  # 转换为 mm
                            dz_mm = dz_ump * UMP_TO_MM  # 转换为 mm
                            vx_mm_s = dx_mm / dt  # mm/s
                            vy_mm_s = -dz_mm / dt  # mm/s（注意：dz 对应 y 方向，左臂反转）
                    
                    # 更新上一次的值
                    prev_time_L = current_time
                    prev_dx_L = dx
                    prev_dz_L = dz
                
                # ⭐ 计算位置差值
                # 累积 Geomagic Touch 的位移
                with cumulative_lock:
                    cumulative_dx_L += dx
                    cumulative_dz_L += dz
                    cum_dx = cumulative_dx_L
                    cum_dz = cumulative_dz_L
                
                # ⭐ 计算位置差值
                # Geomagic Touch 在 UMP 坐标系中的位移（累积值）
                geomagic_dx_ump = cum_dx * SCALE_FACTOR
                geomagic_dy_ump = -cum_dz * SCALE_FACTOR  # 反转y方向（与控制映射一致）
                
                # UMP 相对于固定初始位置的位移
                ump_dx = current[0] - UMP_INITIAL_POS[0]
                ump_dy = current[1] - UMP_INITIAL_POS[1]
                
                # 差值（UMP 单位）= Geomagic Touch 位移 - UMP 位移
                error_x_ump = geomagic_dx_ump - ump_dx
                error_y_ump = geomagic_dy_ump - ump_dy
                
                # 转换为 mm（17000 UMP 单位 = 18mm）
                error_x_mm = error_x_ump * UMP_TO_MM
                error_y_mm = error_y_ump * UMP_TO_MM
                
                # 发送差值数据和速度数据到数据记录器
                send_position_error_to_logger("L", error_x_mm, error_y_mm, current_time)
                send_velocity_to_logger("L", vx_mm_s, vy_mm_s, current_time)

            elif hand == "R" and button_pressed_R:
                manipulator_R.stop()
                current = list(manipulator_R.get_pos())
                # geomagic touch 的 x -> 微控制器的 x
                new_x = min(max(0, current[0] + dx * SCALE_FACTOR), 20000)
                # geomagic touch 的 z+ -> 微控制器的 y+
                new_y = min(max(0, current[1] + dz * SCALE_FACTOR), 20000)
                # z 轴保持当前位置不变
                manipulator_R.goto_pos((new_x, new_y, current[2], current[3]), speed=5000)
                
                current_time = time.time()
                
                # ⭐ 计算速度
                vx_mm_s = 0.0
                vy_mm_s = 0.0
                with velocity_lock:
                    if prev_time_R is not None:
                        dt = current_time - prev_time_R
                        if dt > 0:
                            # 计算速度：dx 和 dz 是增量值，需要除以时间间隔
                            # dx 和 dz 是 Geomagic Touch 坐标系中的值，需要转换为 UMP 坐标系，再转换为 mm
                            dx_ump = dx * SCALE_FACTOR  # 转换为 UMP 单位
                            dz_ump = dz * SCALE_FACTOR  # 转换为 UMP 单位
                            dx_mm = dx_ump * UMP_TO_MM  # 转换为 mm
                            dz_mm = dz_ump * UMP_TO_MM  # 转换为 mm
                            vx_mm_s = dx_mm / dt  # mm/s
                            vy_mm_s = dz_mm / dt  # mm/s（注意：dz 对应 y 方向）
                    
                    # 更新上一次的值
                    prev_time_R = current_time
                    prev_dx_R = dx
                    prev_dz_R = dz
                
                # ⭐ 计算位置差值
                # 累积 Geomagic Touch 的位移
                with cumulative_lock:
                    cumulative_dx_R += dx
                    cumulative_dz_R += dz
                    cum_dx = cumulative_dx_R
                    cum_dz = cumulative_dz_R
                
                # ⭐ 计算位置差值
                # Geomagic Touch 在 UMP 坐标系中的位移（累积值）
                geomagic_dx_ump = cum_dx * SCALE_FACTOR
                geomagic_dy_ump = cum_dz * SCALE_FACTOR
                
                # UMP 相对于固定初始位置的位移
                ump_dx = current[0] - UMP_INITIAL_POS[0]
                ump_dy = current[1] - UMP_INITIAL_POS[1]
                
                # 差值（UMP 单位）= Geomagic Touch 位移 - UMP 位移
                error_x_ump = geomagic_dx_ump - ump_dx
                error_y_ump = geomagic_dy_ump - ump_dy
                
                # 转换为 mm（17000 UMP 单位 = 18mm）
                error_x_mm = error_x_ump * UMP_TO_MM
                error_y_mm = error_y_ump * UMP_TO_MM
                
                # 发送差值数据和速度数据到数据记录器
                send_position_error_to_logger("R", error_x_mm, error_y_mm, current_time)
                send_velocity_to_logger("R", vx_mm_s, vy_mm_s, current_time)
                
        except socket.timeout:
            # ⭐ 超时表示没有收到数据，按钮已释放
            # Unity只在按钮按下时发送数据，超时说明按钮已释放
            with button_lock:
                button_pressed_L = False
                button_pressed_R = False
                # ⭐ 如果在等待释放模式，退出该模式
                if waiting_for_button_release:
                    waiting_for_button_release = False
                    waiting_for_button_release_start_time = None
                    prev_button_pressed_L = False
                    prev_button_pressed_R = False
                    print("[系统] 按钮已释放（超时检测），可以开始使用按钮控制")
                # ⭐ 只有在启动延迟过后才更新上一次按钮状态
                elif startup_delay_passed:
                    prev_button_pressed_L = False
                    prev_button_pressed_R = False
            
            continue
        except Exception as e:
            # 打印错误以便调试（可选）
            # print(f"接收数据错误: {e}")
            continue

# === 监听退出键 ===
def listen_for_quit():
    print("=" * 70)
    print("Geomagic Touch 控制 UMP 系统")
    print("=" * 70)
    print("\nRecording started. Press 'q' to stop.")
    keyboard.wait('q')
    stop_event.set()

# === 启动线程 ===
t1 = threading.Thread(target=read_and_send)
t2 = threading.Thread(target=receive_geomagic_and_control)
t3 = threading.Thread(target=listen_for_quit)

t1.start()
t2.start()
t3.start()

t1.join()
t2.join()
t3.join()

left_file.close()
right_file.close()
sock_send.close()
sock_recv.close()
print("Recording stopped and files saved.")

