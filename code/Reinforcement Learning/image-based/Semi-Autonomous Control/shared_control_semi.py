import socket
from sensapex import UMP
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from real_env_shared_control_semi import DiscreteMazeEnv  # 引入你之前的自定义环境

# 服务器设置
server_ip = "0.0.0.0"  # 监听所有网络接口
server_port = 12345  # 接收数据的端口，同Unity脚本中的端口
buffer_size = 1024

# 目标服务器设置（发送数据的目标）
# target_ip = '192.168.137.137'
target_ip = '127.0.0.1'
target_port = 5005  # 发送数据的目标端口

# 创建服务器TCP套接字并开始监听
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.listen(1)
print("TCP server up and listening")

# 尝试接受连接，直到成功建立连接
while True:
    try:
        conn, addr = server_socket.accept()
        print(f"Connected to {addr}")
        break  # 成功连接，跳出循环
    except Exception as e:
        print(f"Accepting connection failed: {e}. Retrying...")
        time.sleep(1)

# 创建客户端TCP套接字，用于发送数据
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 不断尝试连接发送端，直到连接成功
while True:
    try:
        client_socket.connect((target_ip, target_port))
        print("Connected to target server.")
        break  # 成功连接，跳出循环
    except socket.error as err:
        print("Connection failed. Retrying...")
        time.sleep(1)  # 等待一段时间后重试

# 获取UMP实例和设备
ump = UMP.get_ump()
manipulator = ump.get_device(1)

# 初始位置和设置
previousx, previousy, previousz = 2500, 2500, 11000  # 强化学习完成前的初始位置
max_limit = 18000
min_limit = 0
update_interval = 0.01  # 发送位置更新的间隔

# 记录位置的列表
position_log = []

# 打开文件用于写入记录数据
output_file = "robot_position_log.csv"
with open(output_file, 'w') as f:
    f.write("Time,Position_X,Position_Y,Position_Z\n")

# 加载强化学习模型
model_path = "ppo_logs/models/latest_model.zip"
env = DummyVecEnv([lambda: Monitor( DiscreteMazeEnv(grid_map_path='C:/Users/maoyudong/Desktop/PPO/PPO_TD3_DDPG_SAC/discrete_grid_map.npy',
            original_image_path='C:/Users/maoyudong/Desktop/PPO/PPO_TD3_DDPG_SAC/map.png',
            render_mode='none' , # 设置为 'none'，你可以根据需要调整
            seed=0
        ))])
model = PPO.load(model_path, env=env)

# 初始导航状态
navigation_done = False

try:
    start_time = time.time()  # 记录开始时间

    # 强化学习导航阶段
    obs = env.reset()
    while not navigation_done:
        action, _ = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = env.step(action)
        current_pos = env.envs[0].get_position_no_convert()  # 获取当前位置，假设它返回的是[x, y]

        # 打印和记录当前位置
        timestamp = time.time() - start_time
        position_log.append((timestamp, current_pos))
        # 记录文件时使用11000作为第三个坐标值
        with open(output_file, 'a') as f:
            f.write(f"{timestamp:.2f},{current_pos[0]},{current_pos[1]},11000\n")
        print(f"RL Navigation Position: {current_pos}")

        # 发送当前位置到Unity
        message = ','.join(map(str, current_pos)) + ',11000'  # 添加11000作为第三个坐标值
        message = message.encode()  # 格式化位置数据为字符串
        client_socket.sendall(message)  # 发送数据

        # 判断是否到达终点
        if dones[0]:
            
            navigation_done = True
            env.navigation_done = True
            print("Reinforcement Learning navigation complete.")
            previousx, previousy, previousz = current_pos[0], current_pos[1], 11000  # 更新手动控制的起始位置
            print(f"Starting manual control from position: {previousx}, {previousy}, {previousz}")
            time.sleep(0.1)  # 等待2秒切换到手动模式

    # 开始手动遥操作阶段
    while True:
        # 尝试从连接接收数据
        conn.settimeout(0.01)  # 设置非阻塞接收，如果在0.01秒内没有数据，则引发timeout异常
        try:
            message = conn.recv(buffer_size)
            if message:
                message_decoded = message.decode('utf-8')
                try:
                    index_tip_position_str = message_decoded.split(',')
                    if len(index_tip_position_str) != 3:
                        raise ValueError("Invalid data format.")
                    deltax, deltay, deltaz = [float(index_tip_position_str[i]) for i in [0, 1, 2]]

                    # # MR控制模式
                    # temp_previousx = previousx + 100000 * deltax
                    # temp_previousz = previousz
                    # temp_previousy = previousy - 100000 * deltay

                    # mouse控制模式
                    temp_previousx = previousx + 60000 * deltax
                    temp_previousz = previousz
                    temp_previousy = previousy - 60000 * deltay

                    # 使用中间变量应用范围限制，以决定控制机器人的坐标
                    controlx = max(min(temp_previousx, max_limit), min_limit)
                    controly = max(min(temp_previousy, max_limit), min_limit)
                    controlz = max(min(temp_previousz, max_limit), min_limit)

                    # 如果中间变量在范围内，则更新实际坐标变量
                    if min_limit <= temp_previousx <= max_limit:
                        previousx = temp_previousx
                    if min_limit <= temp_previousy <= max_limit:
                        previousy = temp_previousy
                    if min_limit <= temp_previousz <= max_limit:
                        previousz = temp_previousz

                    # 控制机器人到新的位置，使用控制变量确保坐标在指定范围内
                    manipulator.goto_pos((controlx, controly, controlz, 10000), 8000)
                    print(f"Moved to new position: {(controlx, controly, controlz)}")
                except ValueError as e:
                    print(f"Data error: {e}, skipping...")
                    continue  # 跳过这次循环，继续下一个
        except socket.timeout:
            # 没有接收到数据，继续执行
            pass

        # 获取当前位置并发送
        current_pos = manipulator.get_pos()
        print(f"Real-time Position: {current_pos[0:3]}")

        # 记录当前位置和时间戳
        timestamp = time.time() - start_time
        position_log.append((timestamp, current_pos[0:3]))

        # 写入文件
        with open(output_file, 'a') as f:
            f.write(f"{timestamp:.2f},{current_pos[0]},{current_pos[1]},{current_pos[2]}\n")

        message = ','.join(map(str, current_pos[0:3])).encode()
        client_socket.sendall(message)
        time.sleep(update_interval)  # 等待指定的更新间隔时间

except KeyboardInterrupt:
    print("Operation stopped by user.")

finally:
    conn.close()  # 关闭接收端连接
    client_socket.close()  # 关闭发送端socket
    server_socket.close()  # 关闭服务器socket

    # 输出记录的轨迹数据
    for entry in position_log:
        timestamp, position = entry
        print(f"Time: {timestamp:.2f}s, Position: {position}")
