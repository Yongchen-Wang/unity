"""
统一启动脚本 - 同时运行控制程序、检测程序和数据记录器（YOLO版本）
使用Python subprocess实现
"""

import subprocess
import time
import sys
import os
from sensapex import UMP

ump = UMP.get_ump()

def verify_ump_devices():
    """验证两个UMP设备已连接"""
    try:
        ump = UMP.get_ump()
        dev_ids = ump.list_devices()
        if len(dev_ids) != 2:
            print(f" 错误: 2个UMP设备，检测到 {len(dev_ids)} 个")
            return False
        print(f"✓ UMP ids: {dev_ids}")
        return True
    except Exception as e:
        print(f"UMP设备连接失败: {e}")
        return False

def run_keyboard_preset():
    """
    预先运行 keyboard_control.py 一次：
    - 自动发送命令: R 10000 10000 20000, L 10000 10000 20000
    - 等待一小段时间让机械臂运动完成
    - 再发送 'q' 退出脚本
    运行完成后，用户按 Enter 再继续启动后面的三个程序
    """
    print("\n" + "="*50)
    print("[0/3] 预设 Sensapex 位置...")
    print("="*50)
    
    print("  目标位置: R(10000, 10000, 20000), L(10000, 10000, 20000)")
    

    # 获取设备
    umpL = ump.get_device(1)  # 左臂
    umpR = ump.get_device(2)  # 右臂
    
    # 获取当前位置
    pos_L = umpL.get_pos()
    pos_R = umpR.get_pos()
    print(f"  左臂当前位置: ({pos_L[0]}, {pos_L[1]}, {pos_L[2]})")
    print(f"  右臂当前位置: ({pos_R[0]}, {pos_R[1]}, {pos_R[2]})")
    
    # 发送移动命令
    print("  移动右臂")
    umpR.goto_pos((20000, 20000, 14500, pos_R[3]), speed=1000)
    time.sleep(1.0)  # 间隔1秒
    
    print("  移动左臂")
    umpL.goto_pos((0, 3000, 14500, pos_L[3]), speed=1000)
    
    time.sleep(8.0)

    print("✓ 预设位置命令已执行完毕")
    return True


def start_control_program():
    """启动Sensapex控制程序"""
    print("\n" + "="*50)
    print("[1/3] 启动控制程序 (geomagic_control.py)...")
    print("="*50)
    
    control_dir = r"D:\project\individual_project\unity"
    control_script = "geomagic_control.py"
    control_path = os.path.join(control_dir, control_script)
    
    # 检查文件是否存在
    if not os.path.exists(control_path):
        print(f"✗ 错误: 找不到文件 {control_path}")
        return False
    
    print(f"  目录: {control_dir}")
    print(f"  脚本: {control_script}")
    
    try:
        # 在新的命令行窗口启动
        if sys.platform == "win32":
            process = subprocess.Popen(
                ["python", control_script],
                cwd=control_dir,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            print(f"  进程ID: {process.pid}")
        else:
            # Linux/Mac
            process = subprocess.Popen(
                ["python", control_script],
                cwd=control_dir
            )
            print(f"  进程ID: {process.pid}")
        
        time.sleep(0.5)  # 等待一下，让进程启动
        if process.poll() is not None:
            print(f"✗ 进程立即退出，返回码: {process.returncode}")
            return False
        
        print("✓ 控制程序已启动")
        return True
    except FileNotFoundError:
        print("✗ 错误: 找不到 'python' 命令，请确保 Python 已安装并在 PATH 中")
        return False
    except Exception as e:
        print(f"✗ 启动控制程序失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def start_camera_program():
    """启动摄像头检测程序（YOLO版本）"""
    print("\n" + "="*50)
    print("[3/3] 启动摄像头检测程序 (camera_tcp_magnet_yolo.py - YOLO版本)...")
    print("="*50)
    
    # 使用相对路径，基于当前脚本位置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    camera_dir = os.path.join(script_dir, "code", "Mixed Reality", "Sensapex", "camera_image_processing")
    camera_script = "camera_tcp_magnet_yolo.py"
    camera_path = os.path.join(camera_dir, camera_script)
    
    # 检查文件是否存在
    if not os.path.exists(camera_path):
        print(f"✗ 错误: 找不到文件 {camera_path}")
        return False
    
    if not os.path.exists(camera_dir):
        print(f"✗ 错误: 找不到目录 {camera_dir}")
        return False
    
    print(f"  目录: {camera_dir}")
    print(f"  脚本: {camera_script}")
    
    try:
        # 在新的命令行窗口启动
        if sys.platform == "win32":
            # 先测试运行以捕获可能的错误
            print("  正在测试脚本...")
            test_process = subprocess.Popen(
                ["python", "-u", camera_script],  # -u 表示无缓冲输出
                cwd=camera_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )
            # 等待一小段时间看是否有立即错误
            time.sleep(2.0)  # YOLO加载模型需要更长时间
            return_code = test_process.poll()
            
            if return_code is not None:
                # 进程已退出，获取错误信息
                stdout, stderr = test_process.communicate()
                print(f"✗ 进程立即退出，返回码: {return_code}")
                if stderr:
                    print(f"  错误信息 (stderr):")
                    error_lines = stderr.strip().split('\n')
                    for line in error_lines[:15]:  # 显示前15行
                        if line.strip():
                            print(f"    {line}")
                if stdout:
                    print(f"  输出信息 (stdout):")
                    output_lines = stdout.strip().split('\n')
                    for line in output_lines[:15]:  # 显示前15行
                        if line.strip():
                            print(f"    {line}")
                if not stderr and not stdout:
                    print("  (没有错误输出，可能是导入错误或配置问题)")
                print(f"\n  提示: 可以手动运行以下命令查看详细错误:")
                print(f"    cd \"{camera_dir}\"")
                print(f"    python {camera_script}")
                return False
            
            # 如果测试成功，关闭测试进程，在新窗口启动
            print("  测试通过，在新窗口启动...")
            test_process.terminate()
            try:
                test_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                test_process.kill()
            
            process = subprocess.Popen(
                ["python", camera_script],
                cwd=camera_dir,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            print(f"  进程ID: {process.pid}")
        else:
            # Linux/Mac
            process = subprocess.Popen(
                ["python", camera_script],
                cwd=camera_dir
            )
            print(f"  进程ID: {process.pid}")
        
        # 检查进程是否成功启动
        time.sleep(0.5)  # 等待一下，让进程启动
        if process.poll() is not None:
            print(f"✗ 进程立即退出，返回码: {process.returncode}")
            return False
        
        print("✓ 摄像头检测程序（YOLO版本）已启动")
        return True
    except FileNotFoundError:
        print("✗ 错误: 找不到 'python' 命令，请确保 Python 已安装并在 PATH 中")
        return False
    except Exception as e:
        print(f"✗ 启动摄像头检测程序失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def start_data_logger():
    """启动数据记录器 (robot_data_logger.py)"""
    print("\n" + "="*50)
    print("[2/3] 启动数据记录器 (robot_data_logger.py)...")
    print("="*50)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logger_dir = os.path.join(script_dir, "code", "Mixed Reality", "Sensapex", "camera_image_processing")
    logger_script = "robot_data_logger.py"
    logger_path = os.path.join(logger_dir, logger_script)
    
    # 检查文件是否存在
    if not os.path.exists(logger_path):
        print(f"✗ 错误: 找不到文件 {logger_path}")
        return False
    
    if not os.path.exists(logger_dir):
        print(f"✗ 错误: 找不到目录 {logger_dir}")
        return False
    
    print(f"  目录: {logger_dir}")
    print(f"  脚本: {logger_script}")
    
    try:
        # 在新的命令行窗口启动
        if sys.platform == "win32":
            process = subprocess.Popen(
                ["python", logger_script],
                cwd=logger_dir,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            print(f"  进程ID: {process.pid}")
        else:
            # Linux/Mac
            process = subprocess.Popen(
                ["python", logger_script],
                cwd=logger_dir
            )
            print(f"  进程ID: {process.pid}")
        
        time.sleep(0.5)  # 等待一下，让进程启动
        if process.poll() is not None:
            print(f"✗ 进程立即退出，返回码: {process.returncode}")
            return False
        
        print("✓ 数据记录器已启动")
        return True
    except FileNotFoundError:
        print("✗ 错误: 找不到 'python' 命令，请确保 Python 已安装并在 PATH 中")
        return False
    except Exception as e:
        print(f"✗ 启动数据记录器失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("="*60)
    print("   磁性圆柱体控制和检测系统 - 启动脚本（YOLO版本）")
    print("="*60)

    # 验证UMP设备连接
    if not verify_ump_devices():
        input("\n按Enter键退出...")
        sys.exit(1)

    # 第一步：预先运行 keyboard_control.py，设置 R 7000 7000 20000
    preset_ok = run_keyboard_preset()
    if not preset_ok:
        print("\n⚠ 预设位置步骤失败，可以检查 keyboard_control.py 后重试。")
    else:
        print("\n预设位置步骤完成。")
    
    # 等待用户确认后再启动后面的三个程序
    input("\n按 Enter 键开始启动控制程序 / 数据记录器 / 摄像头检测程序...")

    # 启动控制程序
    success1 = start_control_program()
    time.sleep(2)  # 等待2秒
    
    # 启动数据记录器（TCP服务器，需要先启动）
    success2 = start_data_logger()
    time.sleep(3)  # 等待3秒，确保TCP服务器完全启动并绑定端口
    
    # 启动摄像头检测程序（TCP客户端，在服务器启动后连接）
    success3 = start_camera_program()
    
    print("\n" + "="*60)
    if success1 and success2 and success3:
        print("✓ 三个程序已成功启动！")
        print("\n窗口说明:")
        print("  - 窗口1: Sensapex控制程序 (geomagic_control.py)")
        print("  - 窗口2: 数据记录器 (robot_data_logger.py) - TCP服务器")
        print("  - 窗口3: 摄像头检测程序 (camera_tcp_magnet_yolo.py) - YOLO版本 - TCP客户端")
        print("\n提示:")
        print("  - 在各自窗口按 'q' 或 'Q' 停止程序")
        print("  - 或直接关闭窗口")
        print("  - 数据记录器窗口需要按 Enter 开始记录")
        print("  - YOLO检测会显示质心位置标记")
        print("  - 现在可以启动Unity场景了！")
    else:
        print("✗ 部分程序启动失败，请检查路径和环境")
    print("="*60)
    
    input("\n按Enter键退出启动脚本...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n启动脚本被中断")
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        input("按Enter键退出...")
