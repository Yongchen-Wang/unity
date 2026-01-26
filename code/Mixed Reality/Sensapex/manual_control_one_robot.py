import socket
from sensapex import UMP
import time

# 服务器设置
server_ip = "0.0.0.0"  # 监听所有网络接口
server_port = 12345  # 接收数据的端口，同Unity脚本中的端口
buffer_size = 1024

# 目标服务器设置（发送数据的目标）
# target_ip = '127.0.0.1'
target_ip = '192.168.137.137'
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
previousx, previousy, previousz = 10000, 10000, 10000
max_limit = 18000
min_limit = 2000
update_interval = 0.1  # 发送位置更新的间隔

try:
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
                    deltax, deltay, deltaz = [float(index_tip_position_str[i]) for i in [0, 2, 1]]
                    
                    # 更新坐标，并乘以一个因子（如2000）来调整增量的大小
                    temp_previousx = previousx + 64000 * deltax
                    temp_previousz = previousz - 64000 * deltaz
                    temp_previousy = previousy - 96000 * deltay

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
        message = ','.join(map(str, current_pos[0:3])).encode()
        client_socket.sendall(message)
        time.sleep(update_interval)  # 等待指定的更新间隔时间

except KeyboardInterrupt:
    print("Operation stopped by user.")

finally:
    conn.close()  # 关闭接收端连接
    client_socket.close()  # 关闭发送端socket
    server_socket.close()  # 关闭服务器socket
