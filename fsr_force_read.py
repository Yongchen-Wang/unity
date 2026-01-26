import serial
import math

# 把 'COM15' 换成你的串口名，波特率要与 Arduino 一致
ser = serial.Serial('COM15', 115200, timeout=1)

print("串口连接成功，开始读取数据...")
print("按 Ctrl+C 退出\n")

# ==============================
# AO(AD) -> 力(N) 换算（方法二：对数拟合）
# 质量(g) = -1039.1057 * ln(AD) + 7100.0651
# 力(N)   = 质量(g) * 9.81 / 1000
# ==============================
_A = -1039.1057
_B = 7100.0651
_MASS_MIN_G = 0.0
_MASS_MAX_G = 1500.0
_G = 9.81


def ad_to_force_n(ad_value: int) -> float:
    """将 AO/AD 值换算为力(N)。ad_value<=0 时返回 0。"""
    if ad_value is None or ad_value <= 0:
        return 0.0
    mass_g = _A * math.log(ad_value) + _B
    if mass_g < _MASS_MIN_G:
        mass_g = _MASS_MIN_G
    if mass_g > _MASS_MAX_G:
        mass_g = _MASS_MAX_G
    return mass_g * _G / 1000.0


while True:
    try:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        
        # 跳过空行
        if not line:
            continue
        
        # 解析数据：格式 "A0=1023,A1=1023"
        pairs = {}
        for part in line.split(','):
            if '=' not in part:
                continue
            key, value = part.split('=', 1)
            key = key.strip()
            value = value.strip()
            try:
                pairs[key] = int(value)
            except ValueError:
                continue

        if 'A0' not in pairs or 'A1' not in pairs:
            # 可选：打印调试信息
            # print(f"警告: 数据格式不正确，缺少A0/A1: {line}")
            continue

        a0 = pairs['A0']
        a1 = pairs['A1']
        f0 = ad_to_force_n(a0)
        f1 = ad_to_force_n(a1)
        print(f"A0={a0}, A1={a1}, F0_N={f0:.3f}, F1_N={f1:.3f}")
            
    except KeyboardInterrupt:
        print("\n程序中断")
        break
    except Exception as e:
        print(f"错误: {e}")
        continue

ser.close()
print("串口已关闭")