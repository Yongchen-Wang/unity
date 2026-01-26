import socket
from sensapex import UMP
import time

# 服务器设置
server_ip = "0.0.0.0"  # 监听所有网络接口
left_server_port = 12345  # 左手接收数据的端口
right_server_port = 22345  # 右手接收数据的端口
buffer_size = 1024

# 目标服务器设置（发送数据的目标）
# target_ip = '192.168.137.137'
target_ip = '127.0.0.1'
left_target_port = 5005  # 左手发送数据的目标端口
right_target_port = 5225  # 右手发送数据的目标端口

# 创建服务器TCP套接字并开始监听 - 左手
left_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
left_server_socket.bind((server_ip, left_server_port))
left_server_socket.listen(1)
print("TCP server up and listening for left hand")

# 创建服务器TCP套接字并开始监听 - 右手
right_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
right_server_socket.bind((server_ip, right_server_port))
right_server_socket.listen(1)
print("TCP server up and listening for right hand")

# 尝试接受连接，直到成功建立连接 - 左手
while True:
    try:
        left_conn, left_addr = left_server_socket.accept()
        print(f"Connected to {left_addr} for left hand")
        break
    except Exception as e:
        print(f"Accepting connection for left hand failed: {e}. Retrying...")
        time.sleep(1)

# 尝试接受连接，直到成功建立连接 - 右手
while True:
    try:
        right_conn, right_addr = right_server_socket.accept()
        print(f"Connected to {right_addr} for right hand")
        break
    except Exception as e:
        print(f"Accepting connection for right hand failed: {e}. Retrying...")
        time.sleep(1)

# 创建客户端TCP套接字，用于发送数据 - 左手
left_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 创建客户端TCP套接字，用于发送数据 - 右手
right_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 不断尝试连接发送端，直到连接成功 - 左手
while True:
    try:
        left_client_socket.connect((target_ip, left_target_port))
        print("Connected to target server for left hand.")
        break
    except socket.error as err:
        print("Connection for left hand failed. Retrying...")
        time.sleep(1)

# 不断尝试连接发送端，直到连接成功 - 右手
while True:
    try:
        right_client_socket.connect((target_ip, right_target_port))
        print("Connected to target server for right hand.")
        break
    except socket.error as err:
        print("Connection for right hand failed. Retrying...")
        time.sleep(1)

# 获取UMP实例和设备 - 左手
ump = UMP.get_ump()
left_manipulator = ump.get_device(1)

# 获取UMP实例和设备 - 右手
right_manipulator = ump.get_device(2)  # 假设右手控制的机器人ID为2

# 定义初始位置、限制和更新间隔
previousx_left, previousy_left, previousz_left = 10000, 10000, 10000
previousx_right, previousy_right, previousz_right = 10000, 10000, 10000
max_limit = 18000
min_limit = 2000
update_interval = 0.1  # 发送位置更新的间隔

try:
    while True:
        # 处理左手数据
        left_conn.settimeout(0.01)  # 设置非阻塞接收
        try:
            left_message = left_conn.recv(buffer_size)
            if left_message:
                left_message_decoded = left_message.decode('utf-8')
                try:
                    left_index_tip_position_str = left_message_decoded.split(',')
                    if len(left_index_tip_position_str) != 3:
                        raise ValueError("Invalid data format for left hand.")
                    deltax_left, deltay_left, deltaz_left = [float(left_index_tip_position_str[i]) for i in [0, 2, 1]]

                    temp_previousx_left = previousx_left + 64000/2 * deltax_left
                    temp_previousz_left = previousz_left - 64000/2 * deltaz_left
                    temp_previousy_left = previousy_left - 96000/2 * deltay_left

                    controlx_left = max(min(temp_previousx_left, max_limit), min_limit)
                    controly_left = max(min(temp_previousy_left, max_limit), min_limit)
                    controlz_left = max(min(temp_previousz_left, max_limit), min_limit)

                    if min_limit <= temp_previousx_left <= max_limit:
                        previousx_left = temp_previousx_left
                    if min_limit <= temp_previousy_left <= max_limit:
                        previousy_left = temp_previousy_left
                    if min_limit <= temp_previousz_left <= max_limit:
                        previousz_left = temp_previousz_left

                    left_manipulator.goto_pos((controlx_left, controly_left, controlz_left, 10000), 8000)
                    print(f"Left hand moved to new position: {(controlx_left, controly_left, controlz_left)}")
                except ValueError as e:
                    print(f"Data error for left hand: {e}, skipping...")
                    continue  # 跳过这次循环，继续下一个
        except socket.timeout:
            pass

        # 处理右手数据
        right_conn.settimeout(0.01)  # 设置非阻塞接收
        try:
            right_message = right_conn.recv(buffer_size)
            if right_message:
                right_message_decoded = right_message.decode('utf-8')
                try:
                    right_index_tip_position_str = right_message_decoded.split(',')
                    if len(right_index_tip_position_str) != 3:
                        raise ValueError("Invalid data format for right hand.")
                    deltax_right, deltay_right, deltaz_right = [float(right_index_tip_position_str[i]) for i in [0, 2, 1]]

                    temp_previousx_right = previousx_right - 64000/2 * deltax_right
                    temp_previousz_right = previousz_right - 64000/2 * deltaz_right
                    temp_previousy_right = previousy_right + 96000/2 * deltay_right

                    controlx_right = max(min(temp_previousx_right, max_limit), min_limit)
                    controly_right = max(min(temp_previousy_right, max_limit), min_limit)
                    controlz_right = max(min(temp_previousz_right, max_limit), min_limit)

                    if min_limit <= temp_previousx_right <= max_limit:
                        previousx_right = temp_previousx_right
                    if min_limit <= temp_previousy_right <= max_limit:
                        previousy_right = temp_previousy_right
                    if min_limit <= temp_previousz_right <= max_limit:
                        previousz_right = temp_previousz_right

                    right_manipulator.goto_pos((controlx_right, controly_right, controlz_right, 10000), 8000)
                    print(f"Right hand moved to new position: {(controlx_right, controly_right, controlz_right)}")
                except ValueError as e:
                    print(f"Data error for right hand: {e}, skipping...")
                    continue  # 跳过这次循环，继续下一个
        except socket.timeout:
            pass
        # 获取左手机器人当前位置并发送
        current_pos_left = left_manipulator.get_pos()
        left_message = ','.join(map(str, current_pos_left[0:3])).encode()
        left_client_socket.sendall(left_message)

        # 获取右手机器人当前位置并发送
        current_pos_right = right_manipulator.get_pos()
        right_message = ','.join(map(str, current_pos_right[0:3])).encode()
        right_client_socket.sendall(right_message)

        # 等待指定的更新间隔时间
        time.sleep(update_interval)

except KeyboardInterrupt:
    print("Operation stopped by user.")
finally:
    # 关闭所有连接和套接字
    left_conn.close()
    right_conn.close()
    left_client_socket.close()
    right_client_socket.close()
    left_server_socket.close()
    right_server_socket.close()