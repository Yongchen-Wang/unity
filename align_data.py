"""
多源数据对齐和分析工具
用于处理包含独立时间戳的CSV数据

功能:
1. 重采样对齐到统一采样率
2. 数据新鲜度分析
3. 可视化时间戳分布
4. 数据质量报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import argparse


def load_data(csv_file):
    """加载CSV数据"""
    print(f"正在加载数据: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"[OK] 加载完成: {len(df)} 行数据")
    return df


def analyze_timestamps(df):
    """分析时间戳分布和数据新鲜度"""
    print("\n" + "=" * 70)
    print("时间戳分析")
    print("=" * 70)
    
    # 计算采样率（使用 timestamp 作为主时间戳）
    main_ts_col = 'timestamp' if 'timestamp' in df.columns else 'timestamp_main'
    if main_ts_col in df.columns:
        dt = df[main_ts_col].diff().dropna()
        main_rate = 1.0 / dt.mean() if dt.mean() > 0 else 0
        print(f"\n主时钟采样率: {main_rate:.1f} Hz")
        print(f"  平均间隔: {dt.mean()*1000:.1f} ms")
        print(f"  标准差: {dt.std()*1000:.1f} ms")
    
    # 分析各数据源的更新频率
    sources = {
        'Camera': 'timestamp_camera',
        'Geomagic': 'timestamp_geomagic',
        'Velocity': 'timestamp_velocity',
        'FSR': 'timestamp_fsr'
    }
    
    print("\n各数据源采样率:")
    for name, col in sources.items():
        if col in df.columns:
            # 去除重复时间戳(表示数据未更新)
            unique_ts = df[col].dropna().drop_duplicates()
            if len(unique_ts) > 1:
                dt = unique_ts.diff().dropna()
                rate = 1.0 / dt.mean() if dt.mean() > 0 else 0
                print(f"  {name:12s}: {rate:6.1f} Hz  (更新 {len(unique_ts)} 次)")
            else:
                print(f"  {name:12s}:   无数据或未更新")
    
    # 数据新鲜度统计
    if 'freshness_geomagic' in df.columns:
        print("\n数据新鲜度统计 (延迟):")
        for metric in ['freshness_geomagic', 'freshness_velocity', 'freshness_fsr']:
            if metric in df.columns:
                delays = df[metric].dropna() * 1000  # 转换为ms
                if len(delays) > 0:
                    name = metric.replace('freshness_', '').capitalize()
                    print(f"  {name:12s}: 平均={delays.mean():6.1f}ms, "
                          f"最大={delays.max():6.1f}ms, "
                          f"标准差={delays.std():6.1f}ms")


def resample_and_align(df, target_rate=50, method='linear'):
    """
    重采样并对齐数据到统一采样率
    
    Args:
        df: 原始数据DataFrame
        target_rate: 目标采样率 (Hz)
        method: 插值方法 ('linear', 'cubic', 'nearest')
    
    Returns:
        对齐后的DataFrame
    """
    print("\n" + "=" * 70)
    print(f"数据重采样: 目标采样率 {target_rate} Hz, 插值方法 '{method}'")
    print("=" * 70)
    
    # 创建统一时间网格（使用 timestamp 作为主时间戳）
    main_ts_col = 'timestamp' if 'timestamp' in df.columns else 'timestamp_main'
    t_min = df[main_ts_col].min()
    t_max = df[main_ts_col].max()
    dt = 1.0 / target_rate
    
    unified_time = np.arange(t_min, t_max, dt)
    print(f"\n时间范围: {t_min:.3f} - {t_max:.3f} 秒 ({t_max - t_min:.1f} 秒)")
    print(f"原始采样点: {len(df)}")
    print(f"目标采样点: {len(unified_time)}")
    
    # 创建结果DataFrame
    result = pd.DataFrame({'timestamp': unified_time})
    result['time_elapsed'] = result['timestamp'] - t_min
    
    # 定义数据组
    data_groups = {
        'camera': {
            'timestamp_col': 'timestamp_camera',
            'data_cols': ['position_x', 'position_y', 'velocity_x', 'velocity_y', 
                         'velocity_magnitude', 'distance_to_boundary']
        },
        'geomagic': {
            'timestamp_col': 'timestamp_geomagic',
            'data_cols': ['error_x_L', 'error_y_L', 'error_x_R', 'error_y_R']
        },
        'velocity': {
            'timestamp_col': 'timestamp_velocity',
            'data_cols': ['geomagic_velocity_x_L', 'geomagic_velocity_y_L',
                         'geomagic_velocity_x_R', 'geomagic_velocity_y_R']
        },
        'fsr': {
            'timestamp_col': 'timestamp_fsr',
            'data_cols': ['fsr_analog', 'fsr_digital']
        }
    }
    
    # 对每组数据进行插值
    for group_name, group_info in data_groups.items():
        ts_col = group_info['timestamp_col']
        data_cols = group_info['data_cols']
        
        # 检查时间戳列是否存在
        if ts_col not in df.columns:
            print(f"  跳过 {group_name}: 缺少时间戳列 '{ts_col}'")
            continue
        
        # 提取有效数据(去除NaN)
        valid_cols = [ts_col] + [col for col in data_cols if col in df.columns]
        group_df = df[valid_cols].dropna(subset=[ts_col])
        
        if len(group_df) < 2:
            print(f"  跳过 {group_name}: 数据点不足 (<2)")
            continue
        
        print(f"\n插值 {group_name}:")
        print(f"  原始点: {len(group_df)}")
        
        # 去除重复时间戳(保留最后一个)
        group_df = group_df.drop_duplicates(subset=[ts_col], keep='last')
        print(f"  去重后: {len(group_df)}")
        
        # 对每个数据列进行插值
        for col in data_cols:
            if col not in df.columns:
                continue
            
            # 提取非NaN数据点
            valid_data = group_df[[ts_col, col]].dropna()
            
            if len(valid_data) < 2:
                print(f"    {col}: 跳过(数据不足)")
                continue
            
            try:
                # 创建插值函数
                if method == 'linear':
                    f = interpolate.interp1d(
                        valid_data[ts_col], 
                        valid_data[col],
                        kind='linear',
                        bounds_error=False,
                        fill_value='extrapolate'
                    )
                elif method == 'cubic':
                    f = interpolate.interp1d(
                        valid_data[ts_col],
                        valid_data[col],
                        kind='cubic',
                        bounds_error=False,
                        fill_value='extrapolate'
                    )
                elif method == 'nearest':
                    f = interpolate.interp1d(
                        valid_data[ts_col],
                        valid_data[col],
                        kind='nearest',
                        bounds_error=False,
                        fill_value='extrapolate'
                    )
                
                # 插值到统一时间网格
                result[col] = f(unified_time)
                print(f"    {col}: [OK]")
                
            except Exception as e:
                print(f"    {col}: [FAIL] 插值失败 ({e})")
    
    return result


def visualize_timestamps(df, output_path='timestamp_analysis.png'):
    """可视化时间戳分布"""
    print(f"\n生成时间戳可视化图: {output_path}")
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 图1: 各数据源的时间戳分布
    ax1 = axes[0]
    main_ts_col = 'timestamp' if 'timestamp' in df.columns else 'timestamp_main'
    t_min = df[main_ts_col].min()
    
    if 'timestamp_camera' in df.columns:
        camera_unique = df[[main_ts_col, 'timestamp_camera']].drop_duplicates('timestamp_camera')
        ax1.scatter(camera_unique.index, 
                   (camera_unique['timestamp_camera'] - t_min) * 1000,
                   s=10, label='Camera', alpha=0.6)
    
    if 'timestamp_geomagic' in df.columns:
        geo_unique = df[[main_ts_col, 'timestamp_geomagic']].dropna().drop_duplicates('timestamp_geomagic')
        ax1.scatter(geo_unique.index,
                   (geo_unique['timestamp_geomagic'] - t_min) * 1000,
                   s=10, label='Geomagic', alpha=0.6)
    
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Data Source Timestamps Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 数据新鲜度(延迟)
    ax2 = axes[1]
    if 'freshness_geomagic' in df.columns:
        delays = df['freshness_geomagic'].dropna() * 1000
        if len(delays) > 0:
            ax2.plot(delays.index, delays.values, label='Geomagic Delay', alpha=0.7)
    
    if 'freshness_velocity' in df.columns:
        delays = df['freshness_velocity'].dropna() * 1000
        if len(delays) > 0:
            ax2.plot(delays.index, delays.values, label='Velocity Delay', alpha=0.7)
    
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Delay (ms)')
    ax2.set_title('Data Freshness (Delay) Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 采样间隔
    ax3 = axes[2]
    main_ts_col = 'timestamp' if 'timestamp' in df.columns else 'timestamp_main'
    if main_ts_col in df.columns:
        dt = df[main_ts_col].diff() * 1000  # ms
        ax3.plot(dt.index, dt.values, label='Main Clock Interval', alpha=0.7)
        ax3.axhline(dt.mean(), color='r', linestyle='--', 
                   label=f'Mean: {dt.mean():.1f} ms')
    
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Interval (ms)')
    ax3.set_title('Sampling Interval')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[OK] 图表已保存: {output_path}")
    plt.close()


def generate_report(df, output_path='data_quality_report.txt'):
    """生成数据质量报告"""
    print(f"\n生成数据质量报告: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("多源异步数据质量报告\n")
        f.write("=" * 70 + "\n\n")
        
        # 基本统计
        main_ts_col = 'timestamp' if 'timestamp' in df.columns else 'timestamp_main'
        t_min = df[main_ts_col].min()
        t_max = df[main_ts_col].max()
        f.write(f"总记录数: {len(df)}\n")
        f.write(f"时间范围: {t_min:.3f} - {t_max:.3f} 秒\n")
        f.write(f"持续时间: {t_max - t_min:.1f} 秒\n\n")
        
        # 各列的数据完整性
        f.write("数据完整性:\n")
        f.write("-" * 70 + "\n")
        for col in df.columns:
            non_null = df[col].notna().sum()
            percentage = (non_null / len(df)) * 100
            f.write(f"  {col:30s}: {non_null:5d} / {len(df)} ({percentage:5.1f}%)\n")
        
        f.write("\n")
        
        # 数据新鲜度统计
        if 'freshness_geomagic' in df.columns:
            f.write("数据新鲜度统计 (延迟):\n")
            f.write("-" * 70 + "\n")
            for metric in ['freshness_geomagic', 'freshness_velocity', 'freshness_fsr']:
                if metric in df.columns:
                    delays = df[metric].dropna() * 1000
                    if len(delays) > 0:
                        name = metric.replace('freshness_', '')
                        f.write(f"  {name:15s}:\n")
                        f.write(f"    平均: {delays.mean():6.2f} ms\n")
                        f.write(f"    中位数: {delays.median():6.2f} ms\n")
                        f.write(f"    最大: {delays.max():6.2f} ms\n")
                        f.write(f"    标准差: {delays.std():6.2f} ms\n")
            f.write("\n")
        
        # 异常值检测
        f.write("异常值检测:\n")
        f.write("-" * 70 + "\n")
        
        # 检查时间戳倒退
        main_ts_col = 'timestamp' if 'timestamp' in df.columns else 'timestamp_main'
        if main_ts_col in df.columns:
            backward = (df[main_ts_col].diff() < 0).sum()
            f.write(f"  时间戳倒退: {backward} 次\n")
        
        # 检查过大的延迟
        if 'freshness_geomagic' in df.columns:
            large_delay = (df['freshness_geomagic'] > 1.0).sum()  # >1秒
            f.write(f"  Geomagic延迟>1秒: {large_delay} 次\n")
        
        f.write("\n报告生成完成\n")
    
    print(f"[OK] 报告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='多源数据对齐和分析工具')
    parser.add_argument('csv_file', help='输入CSV文件路径')
    parser.add_argument('--rate', type=int, default=50, help='目标采样率(Hz), 默认50')
    parser.add_argument('--method', choices=['linear', 'cubic', 'nearest'], 
                       default='linear', help='插值方法, 默认linear')
    parser.add_argument('--no-align', action='store_true', help='跳过数据对齐')
    parser.add_argument('--no-plot', action='store_true', help='跳过可视化')
    
    args = parser.parse_args()
    
    # 加载数据
    df = load_data(args.csv_file)
    
    # 分析时间戳
    analyze_timestamps(df)
    
    # 生成报告
    generate_report(df)
    
    # 可视化
    if not args.no_plot:
        visualize_timestamps(df)
    
    # 数据对齐
    if not args.no_align:
        aligned_df = resample_and_align(df, target_rate=args.rate, method=args.method)
        
        # 保存对齐后的数据
        output_path = args.csv_file.replace('.csv', f'_aligned_{args.rate}Hz.csv')
        aligned_df.to_csv(output_path, index=False)
        print(f"\n[OK] 对齐后的数据已保存: {output_path}")
    
    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)


if __name__ == "__main__":
    # 如果直接运行(无命令行参数),使用示例数据
    import sys
    if len(sys.argv) == 1:
        print("用法示例:")
        print("  python align_data.py robot_data_20251214_125649.csv")
        print("  python align_data.py robot_data.csv --rate 100 --method cubic")
        print("  python align_data.py robot_data.csv --no-align  # 只分析不对齐")
    else:
        main()
