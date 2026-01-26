from sensapex import UMP
import time
import socket

# 获取UMP实例
ump = UMP.get_ump()
# dev_ids = ump.list_devices()
# print(dev_ids)

# 获取特定的操纵杆设备
manipulator = ump.get_device(2)
# manipulator.calibrate_zero_position()
# # # manipulator.goto_pos((10000,10000,10000,10000),2000)

# # # 设置更新位置的间隔时间
update_interval = 0.1

# 创建TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 目标IP地址和端口（这里用的是假设的HoloLens监听的端口12345，实际应用时替换成正确的IP地址和端口）
#target_address = ('192.168.137.137', 5005)  # 修改为HoloLens的IP地址和端口
target_address = ('127.0.0.1', 5225)

try:
    # 连接到服务器
    sock.connect(target_address)
    print("Connected to server.")

    # 开始持续更新位置
    while True:
        # 获取当前位置
        current_pos = manipulator.get_pos()
        print(f"Real-time Position: {current_pos[0:3]}")

        # 将位置数据转换为字节串并通过TCP发送
        message = ','.join(map(str, current_pos[0:3])).encode()
        sock.sendall(message)

        # 等待指定的更新间隔时间
        time.sleep(update_interval)
except KeyboardInterrupt:
    print("Stopped real-time position updates.")
finally:
    # 关闭socket连接
    sock.close()