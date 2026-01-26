import socket
from sensapex import UMP
import time

server_ip = "0.0.0.0"  # 监听所有网络接口
server_port = 22345  # 同Unity脚本中的端口
buffer_size = 1024

# 创建TCP套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定到地址和ip
server_socket.bind((server_ip, server_port))

# 开始监听
server_socket.listen(1)
print("TCP server up and listening")

# 获取UMP实例
ump = UMP.get_ump()

# 获取特定的操纵杆设备
manipulator = ump.get_device(2)

# 初始位置
previousx, previousy, previousz = 10000, 10000, 10000

# 设置范围限制
max_limit = 18000
min_limit = 2000

# 接受一个新连接
conn, addr = server_socket.accept()
print(f"Connected to {addr}")

try:
    while True:
        message = conn.recv(buffer_size)
        if not message:
            break  # 如果没有接收到数据，跳出循环
        message_decoded = message.decode('utf-8')

        # 解析接收到的坐标数据，按照新的顺序x, z, y调整
        index_tip_position_str = message_decoded.split(',')
        # 按照x, z, y的顺序排列
        deltax, deltay, deltaz = [float(index_tip_position_str[i]) for i in [0, 2, 1]]
        
        # 更新坐标，并乘以一个因子（如2000）来调整增量的大小
        temp_previousx = previousx - 4000 * deltax
        temp_previousz = previousz - 4000 * deltaz  # 根据新顺序更新
        temp_previousy = previousy + 6000 * deltay
        
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
        manipulator.goto_pos((controlx, controly, controlz, 10000), 5000)
        print(f"Moved to new position: {(controlx, controly, controlz)}")

except KeyboardInterrupt:
    print("Server stopped by user")

finally:
    conn.close()  # 关闭连接
    server_socket.close()  # 关闭套接字
