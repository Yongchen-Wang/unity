# Consensus轨迹生成系统

从11组导航数据中生成consensus轨迹，作为BC Policy训练的ground truth。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
cd consensus_generation
python main.py --input "../code/Mixed Reality/Sensapex/camera_image_processing/robot_data" --output ./consensus_output
```

### 高级选项

```bash
# 排除特定文件
python main.py --input ./robot_data --output ./output --exclude 140802

# 自定义重采样点数
python main.py --input ./robot_data --output ./output --n-samples 150

# 跳过可视化
python main.py --input ./robot_data --output ./output --no-visualization
```

### 命令行参数

- `--input`: 输入数据目录（包含CSV文件）
- `--output`: 输出目录
- `--exclude`: 要排除的文件名列表（例如: 140802）
- `--n-samples`: 重采样点数（默认: 100）
- `--smooth-window`: 平滑窗口长度（默认: 11）
- `--velocity-threshold`: 速度阈值（默认: 0.3 mm/s）
- `--min-length`: 最小轨迹长度（默认: 20.0 mm）
- `--reference-method`: 参考轨迹选择方法（median/longest/shortest）
- `--consensus-method`: Consensus生成方法（median/mean）
- `--window-ratio`: DTW窗口比例（默认: 0.2）
- `--state-dim`: BC训练状态维度（默认: 14）
- `--action-dim`: BC训练动作维度（默认: 2）
- `--no-visualization`: 跳过可视化

## 输出文件

- `consensus_trajectory.pkl` - 完整consensus数据
- `bc_training_data.npy` - BC训练数据
- `alignment_report.txt` - 对齐质量报告
- `validation_metrics.json` - 验证指标
- `all_trajectories.png` - 所有轨迹总览图
- `dtw_alignment.png` - DTW对齐前后对比图
- `velocity_profile.png` - 速度曲线图
- `confidence_heatmap.png` - 置信区间热力图
- `metrics_summary.png` - 验证指标总结表格

## 验证标准

生成的consensus轨迹必须满足：
- ✅ 平均DTW距离 < 5mm
- ✅ 平均点标准差 < 2mm
- ✅ 速度曲线平滑（无突变）
- ✅ 置信区间覆盖80%原始轨迹点

