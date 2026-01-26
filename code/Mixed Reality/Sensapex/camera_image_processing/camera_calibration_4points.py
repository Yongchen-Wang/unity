"""
四点透视标定工具
使用此工具标定摄像头视野与真实空间的映射关系
支持透视变换，提供比两点线性标定更高的精度
"""

import cv2
import numpy as np
import json

class CalibrationTool:
    def __init__(self):
        self.pixel_points = []
        self.space_points = []
        self.current_frame = None
        self.point_names = ['左上角', '右上角', '右下角', '左下角']
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pixel_points) < 4:
                self.pixel_points.append([x, y])
                point_idx = len(self.pixel_points) - 1
                print(f"已标记 {self.point_names[point_idx]}: 像素坐标 ({x}, {y})")
                
                if len(self.pixel_points) == 4:
                    print("\n已标记所有4个点！")
                    print("按 'S' 保存标定参数，按 'R' 重新开始")
    
    def draw_points(self, frame):
        """绘制已标记的点"""
        display_frame = frame.copy()
        
        # 绘制已标记的点
        for i, (x, y) in enumerate(self.pixel_points):
            cv2.circle(display_frame, (x, y), 8, self.colors[i], -1)
            cv2.circle(display_frame, (x, y), 10, (255, 255, 255), 2)
            cv2.putText(display_frame, self.point_names[i], (x + 15, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[i], 2)
        
        # 如果标记了多个点，连线形成多边形
        if len(self.pixel_points) >= 2:
            pts = np.array(self.pixel_points, dtype=np.int32)
            cv2.polylines(display_frame, [pts], len(self.pixel_points) == 4, 
                         (0, 255, 255), 2)
        
        # 显示提示信息
        status_y = 30
        if len(self.pixel_points) < 4:
            tip = f"请点击标记 {self.point_names[len(self.pixel_points)]}"
            cv2.putText(display_frame, tip, (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "按 'S' 保存, 'R' 重新开始, 'Q' 退出", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(display_frame, f"已标记: {len(self.pixel_points)}/4", (10, status_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_frame
    
    def input_space_coordinates(self):
        """输入真实空间坐标"""
        print("\n" + "="*70)
        print("请输入四个标定点的真实物理坐标（单位：毫米 mm）")
        print("="*70)
        print("建议：")
        print("  1. 使用卷尺或游标卡尺测量")
        print("  2. 建立坐标系：左下角为原点(0, 0)")
        print("  3. X轴向右，Y轴向上")
        print("="*70)
        
        self.space_points = []
        
        for i, name in enumerate(self.point_names):
            print(f"\n{name} (像素坐标: {self.pixel_points[i]}):")
            while True:
                try:
                    x_str = input(f"  X坐标 (mm): ")
                    y_str = input(f"  Y坐标 (mm): ")
                    x = float(x_str)
                    y = float(y_str)
                    self.space_points.append([x, y])
                    print(f"  ✓ 已记录: ({x}, {y}) mm")
                    break
                except ValueError:
                    print("  ✗ 输入错误，请输入数字！")
        
        print("\n" + "="*70)
        print("标定点汇总:")
        print("="*70)
        for i, name in enumerate(self.point_names):
            print(f"{name}:")
            print(f"  像素坐标: {self.pixel_points[i]}")
            print(f"  物理坐标: {self.space_points[i]} mm")
        print("="*70)
    
    def calculate_transform(self):
        """计算透视变换矩阵"""
        pixel_pts = np.array(self.pixel_points, dtype=np.float32)
        space_pts = np.array(self.space_points, dtype=np.float32)
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(pixel_pts, space_pts)
        
        print("\n透视变换矩阵:")
        print(matrix)
        
        return matrix
    
    def save_calibration(self, matrix, filename='calibration_params.json'):
        """保存标定参数到文件"""
        calibration_data = {
            'type': 'perspective',
            'pixel_points': self.pixel_points,
            'space_points': self.space_points,
            'perspective_matrix': matrix.tolist(),
            'note': '四点透视标定，像素坐标顺序：左上、右上、右下、左下'
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 标定参数已保存到: {filename}")
        return filename
    
    def verify_calibration(self, matrix):
        """验证标定精度"""
        print("\n" + "="*70)
        print("标定精度验证:")
        print("="*70)
        
        max_error = 0
        total_error = 0
        
        for i, (pixel_pt, space_pt) in enumerate(zip(self.pixel_points, self.space_points)):
            # 使用变换矩阵转换
            pixel_array = np.array([[pixel_pt]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pixel_array, matrix)
            calc_x, calc_y = transformed[0][0]
            
            # 计算误差
            error_x = abs(calc_x - space_pt[0])
            error_y = abs(calc_y - space_pt[1])
            error_total = np.sqrt(error_x**2 + error_y**2)
            
            max_error = max(max_error, error_total)
            total_error += error_total
            
            print(f"{self.point_names[i]}:")
            print(f"  期望: ({space_pt[0]:.2f}, {space_pt[1]:.2f}) mm")
            print(f"  计算: ({calc_x:.2f}, {calc_y:.2f}) mm")
            print(f"  误差: X={error_x:.3f} mm, Y={error_y:.3f} mm, 总={error_total:.3f} mm")
        
        avg_error = total_error / 4
        print("="*70)
        print(f"平均误差: {avg_error:.3f} mm")
        print(f"最大误差: {max_error:.3f} mm")
        
        if max_error < 1.0:
            print("✓ 标定质量: 优秀 (误差 < 1mm)")
        elif max_error < 3.0:
            print("✓ 标定质量: 良好 (误差 < 3mm)")
        elif max_error < 5.0:
            print("⚠ 标定质量: 一般 (误差 < 5mm)")
        else:
            print("✗ 标定质量: 较差 (误差 >= 5mm)，建议重新标定")
        print("="*70)

def main():
    print("="*70)
    print("四点透视标定工具")
    print("="*70)
    print("使用说明:")
    print("1. 在容器的四个角放置标记点（如胶带、贴纸）")
    print("2. 用鼠标依次点击画面中的四个角：左上 -> 右上 -> 右下 -> 左下")
    print("3. 输入每个点的真实物理坐标（毫米）")
    print("4. 程序会自动计算透视变换矩阵并保存")
    print("="*70)
    print("快捷键:")
    print("  鼠标左键 - 标记点")
    print("  S       - 保存标定参数")
    print("  R       - 重新开始标定")
    print("  Q       - 退出程序")
    print("="*70)
    input("按回车开始...")
    
    # 打开摄像头
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    # 设置更高分辨率（与检测程序保持一致）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n摄像头分辨率: {width}x{height}")
    
    # 创建标定工具
    calibration_tool = CalibrationTool()
    
    # 创建窗口并设置鼠标回调
    window_name = 'Camera Calibration - Click 4 corners'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, calibration_tool.mouse_callback)
    
    print("\n开始标定...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取摄像头画面")
            break
        
        calibration_tool.current_frame = frame
        display_frame = calibration_tool.draw_points(frame)
        
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n退出程序")
            break
        
        elif key == ord('r'):
            # 重新开始
            calibration_tool.pixel_points = []
            calibration_tool.space_points = []
            print("\n重新开始标定...")
        
        elif key == ord('s'):
            # 保存标定
            if len(calibration_tool.pixel_points) == 4:
                # 输入空间坐标
                calibration_tool.input_space_coordinates()
                
                # 计算变换矩阵
                matrix = calibration_tool.calculate_transform()
                
                # 验证精度
                calibration_tool.verify_calibration(matrix)
                
                # 保存
                filename = calibration_tool.save_calibration(matrix)
                
                print(f"\n✓ 标定完成！")
                print(f"标定文件: {filename}")
                print(f"现在可以运行 camera_tcp_sphere.py 使用此标定")
                
                # 询问是否继续
                print("\n按 'Q' 退出，按 'R' 重新标定...")
            else:
                print("\n请先标记所有4个点！")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

