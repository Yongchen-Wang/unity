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
UDP_RECV_PORT = 5005
SCALE_FACTOR = 80
SEND_INTERVAL = 1.0 / 60.0  # 60Hz 发送频率

# === 创建保存文件夹 ===
os.makedirs("left", exist_ok=True)
os.makedirs("right", exist_ok=True)

# === 初始化 Sensapex 设备 ===
ump = UMP.get_ump()
manipulator_L = ump.get_device(2)
manipulator_R = ump.get_device(1)

# === 建立 UDP 通信 ===
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind(("0.0.0.0", UDP_RECV_PORT))

# === 状态共享变量 ===
last_pos_L = None
last_pos_R = None
stop_event = threading.Event()
start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# === 文件句柄和Writer初始化 ===
left_file = open(f"left/{start_time}.csv", "w", newline='')
right_file = open(f"right/{start_time}.csv", "w", newline='')
left_writer = csv.writer(left_file)
right_writer = csv.writer(right_file)
left_writer.writerow(['timestamp', 'dx', 'dy', 'dz'])
right_writer.writerow(['timestamp', 'dx', 'dy', 'dz'])

# === 坐标转换函数 ===
def map_position(raw_value):
    return (((raw_value - 1) / (20000 - 1)) * 20 - 10) / 100

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
            disp_L = tuple(pos - last for pos, last in zip(pos_L, last_pos_L))
            disp_R = tuple(pos - last for pos, last in zip(pos_R, last_pos_R))

            # 发送 UDP
            msg_L = f"L,{disp_L[0]},{disp_L[1]},{disp_L[2]}"
            msg_R = f"R,{disp_R[0]},{disp_R[1]},{disp_R[2]}"
            sock_send.sendto(msg_L.encode(), (UDP_SEND_IP, UDP_SEND_PORT))
            sock_send.sendto(msg_R.encode(), (UDP_SEND_IP, UDP_SEND_PORT))

            # 写入 CSV
            timestamp = time.time()
            left_writer.writerow([timestamp] + list(disp_L))
            right_writer.writerow([timestamp] + list(disp_R))
            left_file.flush()
            right_file.flush()

        last_pos_L = pos_L
        last_pos_R = pos_R

        elapsed = time.time() - start_loop
        time.sleep(max(0, SEND_INTERVAL - elapsed))

# === 线程2: 接收 Unity 指令并控制机器人 ===
def receive_and_control():
    while not stop_event.is_set():
        try:
            data, _ = sock_recv.recvfrom(1024)
            decoded = data.decode().split(",")
            if len(decoded) != 4:
                continue
           
            hand, dx, dz, dy = decoded[0], float(decoded[1]), float(decoded[2]), float(decoded[3])

            if hand == "L":
                manipulator_L.stop()
                current = list(manipulator_L.get_pos())
                new_x = min(max(0, current[0] - dx * SCALE_FACTOR), 20000)
                new_y = min(max(0, current[1] + dy * SCALE_FACTOR), 20000)
                new_z = min(max(0, current[2] - dz * SCALE_FACTOR), 20000)
                manipulator_L.goto_pos((new_x, new_y, new_z, current[3]), speed=5000)

            elif hand == "R":
                manipulator_R.stop()
                current = list(manipulator_R.get_pos())
                new_x = min(max(0, current[0] + dx * SCALE_FACTOR), 20000)
                new_y = min(max(0, current[1] - dy * SCALE_FACTOR), 20000)
                new_z = min(max(0, current[2] - dz * SCALE_FACTOR), 20000)
                manipulator_R.goto_pos((new_x, new_y, new_z, current[3]), speed=5000)
        except:
            continue

# === 监听退出键 ===
def listen_for_quit():
    print("Recording started. Press 'q' to stop.")
    keyboard.wait('q')
    stop_event.set()

# === 启动线程 ===
t1 = threading.Thread(target=read_and_send)
t2 = threading.Thread(target=receive_and_control)
t3 = threading.Thread(target=listen_for_quit)

t1.start()
t2.start()
t3.start()

t1.join()
t2.join()
t3.join()

left_file.close()
right_file.close()
print("Recording stopped and files saved.")
