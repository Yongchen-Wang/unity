# import cv2

# # 定义一个回调函数，用于获取鼠标点击的坐标
# def get_coordinates(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击事件
#         # param中包含了缩放比例，用于将坐标转换回原始分辨率
#         scale_x, scale_y = param
#         original_x = int(x * scale_x)
#         original_y = int(y * scale_y)
#         print(f"Clicked Coordinates on Display: ({x}, {y})")
#         print(f"Corresponding Coordinates in Original Resolution: ({original_x}, {original_y})")

# def main():
#     # 打开默认摄像头
#     camera = cv2.VideoCapture(0)
    
#     if not camera.isOpened():
#         print("Failed to open the camera.")
#         return
    
#     # 设置摄像头的分辨率为1600x1200
#     original_width = 1600
#     original_height = 1200
#     camera.set(cv2.CAP_PROP_FRAME_WIDTH, original_width)
#     camera.set(cv2.CAP_PROP_FRAME_HEIGHT, original_height)
    
#     # 定义缩放后的分辨率
#     resized_width = 800
#     resized_height = 600
    
#     # 计算缩放比例
#     scale_x = original_width / resized_width
#     scale_y = original_height / resized_height
    
#     # 创建一个窗口并绑定鼠标回调函数
#     cv2.namedWindow("Camera Feed")
#     cv2.setMouseCallback("Camera Feed", get_coordinates, param=(scale_x, scale_y))
    
#     while True:
#         # 读取一帧图像
#         ret, frame = camera.read()
        
#         if not ret:
#             print("Failed to capture an image from the camera.")
#             break
        
#         # 将图像缩放到800x600
#         resized_frame = cv2.resize(frame, (resized_width, resized_height))
        
#         # 显示缩放后的图像
#         cv2.imshow("Camera Feed", resized_frame)
        
#         # 按下'q'键退出
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # 释放摄像头并关闭所有窗口
#     camera.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



# ####裁剪后的图像
# import cv2

# # 定义一个回调函数，用于获取鼠标点击的坐标
# def get_coordinates(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击事件
#         print(f"Selected Coordinates: ({x}, {y})")

# def main():
#     # 打开默认摄像头
#     camera = cv2.VideoCapture(0)
    
#     if not camera.isOpened():
#         print("Failed to open the camera.")
#         return
    
#     # 设置摄像头分辨率为1600x1200
#     camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
#     camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    
#     # 创建一个窗口并绑定鼠标回调函数
#     cv2.namedWindow("Camera Feed")
#     cv2.setMouseCallback("Camera Feed", get_coordinates)
    
#     while True:
#         # 读取一帧图像
#         ret, frame = camera.read()
        
#         if not ret:
#             print("Failed to capture an image from the camera.")
#             break
        
#         # 将图像缩放到800x600
#         resized_frame = cv2.resize(frame, (800, 600))
        
#         # 裁剪图像，范围是从(52, 2)到(527, 477)
#         cropped_frame = resized_frame[0:600, 100:700]  # 记住数组的范围是半开区间 [start, end)，所以需要+1
        
#         # 显示裁剪后的图像
#         cv2.imshow("Cropped Camera Feed", cropped_frame)
        
#         # 按下'q'键退出
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # 释放摄像头并关闭所有窗口
#     camera.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()







import cv2

# 定义全局变量来保存起点和终点
# start_pos = (1100, 300) 
# goal_pos = (100, 1650) 

# start_pos = (1200, 1400) 
# goal_pos = (1700, 100)

# start_pos = (1500, 200)
# goal_pos = (300, 1400)


start_pos = (1500, 200)
goal_pos = (300, 1400) 


# 鼠标回调函数，用于获取用户点击的坐标
def get_coordinates(event, x, y, flags, param):
    global start_pos, goal_pos

    if event == cv2.EVENT_LBUTTONDOWN:
        if start_pos is None:
            start_pos = (x, y)
            print(f"Start position set to: {start_pos}")
        elif goal_pos is None:
            goal_pos = (x, y)
            print(f"Goal position set to: {goal_pos}")

def main():
    global start_pos, goal_pos

    # 打开默认摄像头
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Failed to open the camera.")
        return

    # 设置摄像头分辨率为1600x1200
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

    # 创建一个窗口并绑定鼠标回调函数
    cv2.namedWindow("Camera Feed")
    cv2.setMouseCallback("Camera Feed", get_coordinates)

    while True:
        # 读取一帧图像
        ret, frame = camera.read()

        if not ret:
            print("Failed to capture an image from the camera.")
            break

        # 将图像缩放到800x600
        resized_frame = cv2.resize(frame, (800, 600))

        # 裁剪图像，范围是从(100, 0)到(700, 600)
        cropped_frame = resized_frame[0:600, 100:700]

        # 如果起点和终点都已设置，则在图像上绘制它们
        if start_pos is not None and goal_pos is not None:
            # 计算缩放比例（从 1800x1800 到 600x600）
            scale_x = 600 / 1800
            scale_y = 600 / 1800

            # 根据缩放比例调整起点和终点的位置
            start_pos_scaled = (int(start_pos[0] * scale_x), int(start_pos[1] * scale_y))
            goal_pos_scaled = (int(goal_pos[0] * scale_x), int(goal_pos[1] * scale_y))

            # 定义正方形的大小（边长为20个像素）
            square_size = 20
            color_start = (0, 255, 0)  # 绿色
            color_goal = (0, 0, 255)   # 红色
            thickness = 2  # 线条粗细为2个像素

            # 在裁剪后的图像上绘制起点（空心正方形）
            cv2.rectangle(
                cropped_frame,
                (start_pos_scaled[0] - square_size, start_pos_scaled[1] - square_size),
                (start_pos_scaled[0] + square_size, start_pos_scaled[1] + square_size),
                color_start,
                thickness
            )

            # 在裁剪后的图像上绘制终点（空心正方形）
            cv2.rectangle(
                cropped_frame,
                (goal_pos_scaled[0] - square_size, goal_pos_scaled[1] - square_size),
                (goal_pos_scaled[0] + square_size, goal_pos_scaled[1] + square_size),
                color_goal,
                thickness
            )

        # 显示裁剪后的图像
        cv2.imshow("Cropped Camera Feed", cropped_frame)

        # 按下'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭所有窗口
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
