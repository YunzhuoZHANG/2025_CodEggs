#!/usr/bin/env python3
"""
为8个海区位置绘制时间序列图
基于AWI/analysis中的绘图策略，适配压缩维度的NetCDF数据
"""

import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math
from datetime import datetime
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# 文件路径
ROOT = '/public/home/yunzhuozhang/AWI'
ANADIR = os.path.join(ROOT, 'location_analysis')
COMPRESSED_FILES = {
    'fixed': os.path.join(ROOT, 'codarea_mhw_fixedbaseline_compressed.nc'),
    'moving': os.path.join(ROOT, 'codarea_mhw_movingbaseline_compressed.nc'),
}

# 时间参数
TIME_LEN = 37960
NYEARS = 104
NDAY = 365
YEAR0 = 1982

# 温度分组（包含所有组）
BOUNDS = [-math.inf, -0.92, 1.352, 2.119, 2.834, math.inf]
LABELS_FULL = ['< -0.92', 'optimum (-0.92~1.352)', 'Moderate (1.352~2.119)', 'High (2.119~2.834)', 'Severe (>=2.834)']
PLOT_GROUP_INDICES = list(range(len(LABELS_FULL)))  # 包含所有索引 (包括 < -0.92)
LABELS = [LABELS_FULL[i] for i in PLOT_GROUP_INDICES]
NG = len(LABELS)

# 8个位置（海区）命名：按用户指定顺序 1..8
LOCATION_NAMES = [
    'Franklin Bay',
    'Western Greenland',
    'Novaya Zemlya',
    'Svalbard',
    'ChukchiSea',
    'East Greenland',
    'Hudson Bay',
    'Severnaya Zemlya'
]

# 简化名称（用于图例/坐标轴标签）
LOCATION_SHORT_NAMES = [
    'Franklin Bay',
    'Western Greenland',
    'Novaya Zemlya',
    'Svalbard',
    'ChukchiSea',
    'East Greenland',
    'Hudson Bay',
    'Severnaya Zemlya'
]

# 颜色方案（5个温度组）——加深一度，并调整“高温”为橙红色、“极端”为深红色
# 顺序：< -0.92 / optimum / Moderate / High (orange-red) / Severe (deep red)
COLORS = [
    (0.35, 0.35, 0.35),  # deeper gray
    (0.45, 0.70, 0.95),  # deeper light blue
    (0.98, 0.88, 0.35),  # deeper light yellow
    (0.95, 0.40, 0.20),  # orange-red
    (0.80, 0.10, 0.10),  # deep red
]

def process_compressed_data(nc_file):
    """
    处理压缩的NetCDF文件，计算每个位置的年度统计
    返回: (locations, years, groups, mhw_flags) 的数据
    """
    print(f"处理文件: {os.path.basename(nc_file)}")
    
    with nc.Dataset(nc_file, 'r') as ds:
        # 读取数据
        temp_class = ds.variables['temperature_class'][:]  # (8, 37960)
        mhw_class = ds.variables['mhw_class'][:]           # (8, 37960)
        
        # 获取位置映射信息
        if 'original_dim1' in ds.variables:
            dim1_map = ds.variables['original_dim1'][:]
            dim2_map = ds.variables['original_dim2'][:]
        else:
            # 默认映射
            dim1_map = np.array([0,0,0,0,1,1,1,1])
            dim2_map = np.array([0,1,2,3,0,1,2,3])
        
        print(f"  数据形状: temp_class={temp_class.shape}, mhw_class={mhw_class.shape}")
        
        # 准备结果数组: (8位置, 104年, 5组, 2标志)
        location_stats = np.zeros((8, NYEARS, len(LABELS_FULL), 2), dtype=np.int64)
        
        # 按年处理数据
        for year_idx in range(NYEARS):
            start_day = year_idx * NDAY
            end_day = start_day + NDAY
            
            if end_day > temp_class.shape[1]:
                print(f"  警告: 年 {year_idx} 超出数据范围")
                break
                
            for location in range(8):
                # 提取当年数据
                temp_year = temp_class[location, start_day:end_day]
                mhw_year = mhw_class[location, start_day:end_day]
                
                # 统计每天的温度组和MHW状态
                for day in range(min(NDAY, len(temp_year))):
                    temp_group = temp_year[day]
                    mhw_flag = mhw_year[day]
                    
                    # 只处理有效数据
                    if 0 <= temp_group < len(LABELS_FULL) and 0 <= mhw_flag <= 1:
                        location_stats[location, year_idx, temp_group, mhw_flag] += 1
        
        print(f"  统计完成")
        return location_stats, dim1_map, dim2_map

def plot_location_time_series(data_fixed, data_moving, dim1_map, dim2_map):
    """
    为每个位置绘制时间序列图（2x3 布局：上行 Fixed，下行 Moving；列为 MHW=0、MHW=1、SUM）
    线条全部使用实线；图例汇总到图像底部一行（标题：Categories）。
    """
    years = np.arange(YEAR0, YEAR0 + NYEARS)
    
    print("开始绘制位置时间序列图...")
    
    # 为每个位置创建单独的图像
    for location in range(8):
        location_name = LOCATION_NAMES[location]
        short_name = LOCATION_SHORT_NAMES[location]
        original_pos = f"[{dim1_map[location]},{dim2_map[location]}]"
        
        print(f"  绘制位置 {location}: {short_name} {original_pos}")
        
        # 提取该位置的数据，包含所有温度组
        # data_fixed 形状: (8_locations, 104_years, 5_groups, 2_flags)
        # 需要: (104_years, 5_groups, 2_flags)
        fixed_data = data_fixed[location, :, :, :][:, PLOT_GROUP_INDICES, :]  # (years, 5_groups, 2_flags)
        moving_data = data_moving[location, :, :, :][:, PLOT_GROUP_INDICES, :]
        
        # 计算总和（MHW=0 + MHW=1）
        fixed_sum = fixed_data[:, :, 0] + fixed_data[:, :, 1]
        moving_sum = moving_data[:, :, 0] + moving_data[:, :, 1]
        
        # 创建图像 (2行3列: Fixed, Moving × MHW=0, MHW=1, SUM)
        # 参照 SST 图比例，适当增加宽度；增大底部边距以容纳底部 legend
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0.3, wspace=0.25, bottom=0.22, top=0.88)
        
        # 存储图例信息
        legend_lines = []
        legend_labels = []
        
        # 计算y轴最大值
        ymax = max(
            fixed_data.max() if fixed_data.size > 0 else 0,
            moving_data.max() if moving_data.size > 0 else 0,
            fixed_sum.max() if fixed_sum.size > 0 else 0,
            moving_sum.max() if moving_sum.size > 0 else 0
        )
        if ymax <= 0:
            ymax = 1.0

        # 布局信息
        row_info = [('Fixed', fixed_data, fixed_sum), ('Moving', moving_data, moving_sum)]
        col_info = [('MHW=0', 0), ('MHW=1', 1), ('SUM', 'sum')]

        for r, (scheme, scheme_data, scheme_sum) in enumerate(row_info):
            for c, (col_label, flag) in enumerate(col_info):
                ax = axes[r, c]
                if flag == 'sum':
                    arr = scheme_sum
                else:
                    arr = scheme_data[:, :, flag]

                for gi in range(NG):
                    color = COLORS[gi % len(COLORS)]
                    # 线型全部为实线
                    line, = ax.plot(years, arr[:, gi], color=color, linewidth=1.8)
                    if r == 0 and c == 0:
                        # 图例仅使用温度组（不添加 (MHW=0) 等提示）
                        legend_lines.append(line)
                        legend_labels.append(LABELS[gi])

                # 顶部列标题（仅第一行）
                if r == 0:
                    ax.set_title(col_label)
                # y 轴标签：按行显示 Baseline 类型与纵轴数据说明（Days Count）
                if c == 0:
                    ax.set_ylabel(f"{scheme} Baseline\nDays Count")

                ax.set_ylim(0, ymax * 1.05)
                ax.grid(True, alpha=0.3)
        
        # 设置x轴标签
        for c in range(3):
            axes[-1, c].set_xlabel('Year')
        
        # 添加底部图例一行，标题 Categories（下移以避免遮挡 x 轴标签）
        fig.legend(legend_lines, legend_labels, loc='lower center', ncol=NG, title='Categories', bbox_to_anchor=(0.5, 0.06))
        
        # 设置总标题
        fig.suptitle(
            f'Region {location+1}: {location_name} {original_pos}\nTemperature Group Days by MHW State',
            fontsize=12, fontweight='bold'
        )
        
        # 保存图像
        filename = f"location_{location:02d}_{short_name.replace('-', '_').lower()}_time_series.png"
        filepath = os.path.join(ANADIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"    保存: {filename}")

def plot_all_locations_comparison(data_fixed, data_moving):
    """
    绘制所有8个位置的对比图
    """
    years = np.arange(YEAR0, YEAR0 + NYEARS)
    
    print("绘制所有位置对比图...")
    
    # 计算每个位置的总天数（所有温度组和MHW状态的总和）
    fixed_totals = np.sum(data_fixed[:, :, PLOT_GROUP_INDICES, :], axis=(2, 3))  # (8, years)
    moving_totals = np.sum(data_moving[:, :, PLOT_GROUP_INDICES, :], axis=(2, 3))
    
    # 计算每个位置的MHW天数
    fixed_mhw = np.sum(data_fixed[:, :, PLOT_GROUP_INDICES, 1], axis=2)  # (8, years)
    moving_mhw = np.sum(data_moving[:, :, PLOT_GROUP_INDICES, 1], axis=2)
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # 颜色映射
    location_colors = plt.cm.tab10(np.linspace(0, 1, 8))
    
    # 1. 固定基线 - 总天数
    ax = axes[0, 0]
    for loc in range(8):
        ax.plot(years, fixed_totals[loc, :], color=location_colors[loc], 
               label=LOCATION_SHORT_NAMES[loc], linewidth=1.5)
    ax.set_title('Fixed Baseline - Total Days')
    ax.set_ylabel('Days per year')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. 移动基线 - 总天数
    ax = axes[0, 1]
    for loc in range(8):
        ax.plot(years, moving_totals[loc, :], color=location_colors[loc], 
               label=LOCATION_SHORT_NAMES[loc], linewidth=1.5)
    ax.set_title('Moving Baseline - Total Days')
    ax.set_ylabel('Days per year')
    ax.grid(True, alpha=0.3)
    
    # 3. 固定基线 - MHW天数
    ax = axes[1, 0]
    for loc in range(8):
        ax.plot(years, fixed_mhw[loc, :], color=location_colors[loc], 
               label=LOCATION_SHORT_NAMES[loc], linewidth=1.5)
    ax.set_title('Fixed Baseline - MHW Days')
    ax.set_xlabel('Year')
    ax.set_ylabel('MHW days per year')
    ax.grid(True, alpha=0.3)
    
    # 4. 移动基线 - MHW天数
    ax = axes[1, 1]
    for loc in range(8):
        ax.plot(years, moving_mhw[loc, :], color=location_colors[loc], 
               label=LOCATION_SHORT_NAMES[loc], linewidth=1.5)
    ax.set_title('Moving Baseline - MHW Days')
    ax.set_xlabel('Year')
    ax.set_ylabel('MHW days per year')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('8 Locations Comparison - Temperature Group Analysis\n(All Temperature Groups)', 
                 fontsize=14, fontweight='bold')
    
    # 保存对比图
    filename = "all_locations_comparison.png"
    filepath = os.path.join(ANADIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  保存: {filename}")

def plot_overall_average(data_fixed, data_moving):
    """绘制8个位置的总体平均时间序列图（2x3 布局，上 Fixed 下 Moving；列为 MHW=0/MHW=1/SUM）"""
    print("绘制总体平均时间序列图...")
    years = np.arange(YEAR0, YEAR0 + NYEARS)

    # data_* 形状: (8_locations, years, groups, flags)
    # 按位置求平均
    fixed_avg = np.mean(data_fixed, axis=0)   # (years, groups, flags)
    moving_avg = np.mean(data_moving, axis=0) # (years, groups, flags)

    # 计算总和（MHW=0 + MHW=1）
    fixed_sum = fixed_avg[:, :, 0] + fixed_avg[:, :, 1]
    moving_sum = moving_avg[:, :, 0] + moving_avg[:, :, 1]

    # 创建图像 (2行3列)；适当增加宽度与底部边距
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.25, bottom=0.22, top=0.85)

    legend_lines = []
    legend_labels = []

    ymax = max(
        np.nanmax(fixed_avg) if fixed_avg.size else 0,
        np.nanmax(moving_avg) if moving_avg.size else 0,
        np.nanmax(fixed_sum) if fixed_sum.size else 0,
        np.nanmax(moving_sum) if moving_sum.size else 0,
    )
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0

    row_info = [('Fixed', fixed_avg, fixed_sum), ('Moving', moving_avg, moving_sum)]
    col_info = [('MHW=0', 0), ('MHW=1', 1), ('SUM', 'sum')]

    for r, (scheme, scheme_avg, scheme_sum) in enumerate(row_info):
        for c, (col_label, flag) in enumerate(col_info):
            ax = axes[r, c]
            arr = scheme_sum if flag == 'sum' else scheme_avg[:, :, flag]
            for gi in range(NG):
                color = COLORS[gi % len(COLORS)]
                line, = ax.plot(years, arr[:, gi], color=color, linewidth=1.8)
                if r == 0 and c == 0:
                    legend_lines.append(line)
                    legend_labels.append(LABELS[gi])

            if r == 0:
                ax.set_title(col_label)
            if c == 0:
                ax.set_ylabel(f"{scheme} Baseline\nDays Count")
            ax.set_ylim(0, ymax * 1.05)
            ax.grid(True, alpha=0.3)

    for c in range(3):
        axes[-1, c].set_xlabel('Year')

    # 底部合并图例，标题 Categories；下移避免遮挡 x 轴标签
    fig.legend(legend_lines, legend_labels, loc='lower center', ncol=NG, title='Categories', bbox_to_anchor=(0.5, 0.06))

    fig.suptitle('Trend in Annual Counts of Different Temperature Categories | Overall Average',
                 fontsize=14, fontweight='bold')

    filename = "overall_average_time_series.png"
    filepath = os.path.join(ANADIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  保存: {filename}")

def plot_temperature_groups_heatmap(data_fixed, data_moving):
    """
    绘制温度组的热图（两张图）：
    - 一张 Fixed：两行分别为 MHW=0 / MHW=1；每个子图为 9 条带（Overall + 8 地区）
    - 一张 Moving：同上
    2x5 布局（列为 5 个温度类别）。
    """
    print("绘制温度组热图...")

    years = np.arange(YEAR0, YEAR0 + NYEARS)

    def build_blocks(baseline_data, flag):
        # baseline_data: (8, years, 5_groups, 2_flags)
        # 返回长度为 5 的列表，每个元素为 (9, years) [Overall + 8 地区]
        blocks = []
        for gi in range(len(LABELS)):
            mat = baseline_data[:, :, gi, flag]  # (8, years)
            overall = np.mean(mat, axis=0)       # (years,)
            blocks.append(np.vstack([overall[None, :], mat]))
        return blocks

    def plot_for_baseline(name, baseline_data, outfile):
        blocks0 = build_blocks(baseline_data, flag=0)
        blocks1 = build_blocks(baseline_data, flag=1)
        # 计算每行（MHW=0 和 MHW=1）的统一色标范围（基于该行所有子图的最大值）
        row_max0 = max(float(np.nanmax(b)) if b.size else 0.0 for b in blocks0)
        row_max1 = max(float(np.nanmax(b)) if b.size else 0.0 for b in blocks1)
        # 非负保障
        row_max0 = max(row_max0, 0.0)
        row_max1 = max(row_max1, 0.0)
        # 如果某行全为零，避免色标无效
        if row_max0 == 0:
            row_max0 = 1.0
        if row_max1 == 0:
            row_max1 = 1.0

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        # 右侧留出空间给竖直 colorbar
        fig.subplots_adjust(hspace=0.18, wspace=0.10, right=0.88)

        for gi, group_label in enumerate(LABELS):
            # 行 0: MHW=0
            ax = axes[0, gi]
            im0 = ax.imshow(blocks0[gi], aspect='auto', cmap='viridis', vmin=0, vmax=row_max0,
                           extent=[YEAR0, YEAR0+NYEARS-1, 8.5, -0.5])
            if gi == 0:
                ax.set_ylabel('MHW=0')
                ax.set_yticks(range(9))
                ax.set_yticklabels(['Overall'] + LOCATION_SHORT_NAMES)
            else:
                ax.set_yticks(range(9))
                ax.set_yticklabels([''] * 9)
            ax.set_title(group_label)
            ax.hlines(0.5, YEAR0, YEAR0 + NYEARS - 1, colors='orange', linestyles='--', linewidth=1.2)

            # 行 1: MHW=1
            ax = axes[1, gi]
            # 对 Moving 基线的 MHW=1 行强制设置 clim=0~50
            vmax_row1 = 50.0 if name.lower() == 'moving' else row_max1
            im1 = ax.imshow(blocks1[gi], aspect='auto', cmap='viridis', vmin=0, vmax=vmax_row1,
                           extent=[YEAR0, YEAR0+NYEARS-1, 8.5, -0.5])
            if gi == 0:
                ax.set_ylabel('MHW=1')
                ax.set_yticks(range(9))
                ax.set_yticklabels(['Overall'] + LOCATION_SHORT_NAMES)
            else:
                ax.set_yticks(range(9))
                ax.set_yticklabels([''] * 9)
            ax.set_xlabel('Year')
            ax.hlines(0.5, YEAR0, YEAR0 + NYEARS - 1, colors='orange', linestyles='--', linewidth=1.2)

        # 为每行分别添加竖直 colorbar，放在图像右侧；每行统一 colormap 和 clim（两行均为 viridis）
        # 行 0（MHW=0）：viridis
        norm0 = Normalize(vmin=0, vmax=row_max0)
        sm0 = ScalarMappable(norm=norm0, cmap='viridis')
        sm0.set_array([])
        cbar0 = fig.colorbar(sm0, ax=axes[0, :], orientation='vertical', fraction=0.05, pad=0.02)
        cbar0.set_label('Days per year')
        # 行 1（MHW=1）：viridis；对于 Moving 基线，统一到 0~50
        vmax_row1_cbar = 50.0 if name.lower() == 'moving' else row_max1
        norm1 = Normalize(vmin=0, vmax=vmax_row1_cbar)
        sm1 = ScalarMappable(norm=norm1, cmap='viridis')
        sm1.set_array([])
        cbar1 = fig.colorbar(sm1, ax=axes[1, :], orientation='vertical', fraction=0.05, pad=0.02)
        cbar1.set_label('Days per year')

        fig.suptitle(f'Heatmap of Annual Counts of Temperature Categories at Different Locations | {name} Baseline',
                     fontsize=14, fontweight='bold')

        filepath = os.path.join(ANADIR, outfile)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  保存: {outfile}")

    # 绘制两张图：Fixed / Moving
    plot_for_baseline('Fixed', data_fixed, 'temperature_groups_heatmap_fixed.png')
    plot_for_baseline('Moving', data_moving, 'temperature_groups_heatmap_moving.png')

def create_summary_statistics(data_fixed, data_moving, dim1_map, dim2_map):
    """
    创建统计摘要
    """
    print("创建统计摘要...")
    
    # 计算统计数据
    years = np.arange(YEAR0, YEAR0 + NYEARS)
    
    # 总天数统计
    fixed_totals = np.sum(data_fixed[:, :, PLOT_GROUP_INDICES, :], axis=(2, 3))
    moving_totals = np.sum(data_moving[:, :, PLOT_GROUP_INDICES, :], axis=(2, 3))
    
    # MHW天数统计
    fixed_mhw = np.sum(data_fixed[:, :, PLOT_GROUP_INDICES, 1], axis=2)
    moving_mhw = np.sum(data_moving[:, :, PLOT_GROUP_INDICES, 1], axis=2)
    
    # 创建摘要文件
    summary_file = os.path.join(ANADIR, "location_analysis_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("8 Locations Temperature Analysis Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Analysis period: {YEAR0}-{YEAR0+NYEARS-1} ({NYEARS} years)\n")
        f.write(f"Temperature groups analyzed: {', '.join(LABELS)}\n")
        f.write("(Including all temperature groups)\n\n")
        
        f.write("Location Mapping:\n")
        for loc in range(8):
            f.write(f"  Location {loc}: {LOCATION_NAMES[loc]} -> Original[{dim1_map[loc]},{dim2_map[loc]}]\n")
        f.write("\n")
        
        f.write("Average Annual Statistics:\n")
        f.write("-" * 30 + "\n")
        
        for loc in range(8):
            f.write(f"\nLocation {loc} ({LOCATION_SHORT_NAMES[loc]}):\n")
            
            # 固定基线统计
            fixed_avg_total = np.mean(fixed_totals[loc, :])
            fixed_avg_mhw = np.mean(fixed_mhw[loc, :])
            fixed_mhw_ratio = (fixed_avg_mhw / fixed_avg_total * 100) if fixed_avg_total > 0 else 0
            
            # 移动基线统计
            moving_avg_total = np.mean(moving_totals[loc, :])
            moving_avg_mhw = np.mean(moving_mhw[loc, :])
            moving_mhw_ratio = (moving_avg_mhw / moving_avg_total * 100) if moving_avg_total > 0 else 0
            
            f.write(f"  Fixed Baseline:\n")
            f.write(f"    Average total days/year: {fixed_avg_total:.1f}\n")
            f.write(f"    Average MHW days/year: {fixed_avg_mhw:.1f}\n")
            f.write(f"    MHW percentage: {fixed_mhw_ratio:.1f}%\n")
            
            f.write(f"  Moving Baseline:\n")
            f.write(f"    Average total days/year: {moving_avg_total:.1f}\n")
            f.write(f"    Average MHW days/year: {moving_avg_mhw:.1f}\n")
            f.write(f"    MHW percentage: {moving_mhw_ratio:.1f}%\n")
        
        # 趋势分析
        f.write("\nTrend Analysis (Linear):\n")
        f.write("-" * 25 + "\n")
        
        for loc in range(8):
            # 计算线性趋势
            fixed_trend = np.polyfit(years, fixed_mhw[loc, :], 1)[0]  # 斜率
            moving_trend = np.polyfit(years, moving_mhw[loc, :], 1)[0]
            
            f.write(f"\nLocation {loc} ({LOCATION_SHORT_NAMES[loc]}) MHW trend:\n")
            f.write(f"  Fixed baseline: {fixed_trend:.3f} days/year per year\n")
            f.write(f"  Moving baseline: {moving_trend:.3f} days/year per year\n")
    
    print(f"  保存统计摘要: location_analysis_summary.txt")

def main():
    """主函数"""
    print("8个位置的海区时间序列分析")
    print("="*50)
    
    # 确保输出目录存在
    os.makedirs(ANADIR, exist_ok=True)
    
    # 处理数据
    print("\n1. 处理压缩数据...")
    data_fixed, dim1_map, dim2_map = process_compressed_data(COMPRESSED_FILES['fixed'])
    data_moving, _, _ = process_compressed_data(COMPRESSED_FILES['moving'])
    
    # 绘制图像
    print("\n2. 绘制位置时间序列图...")
    plot_location_time_series(data_fixed, data_moving, dim1_map, dim2_map)
    
    print("\n3. 绘制位置对比图...")
    plot_all_locations_comparison(data_fixed, data_moving)
    
    print("\n4. 绘制温度组热图...")
    plot_temperature_groups_heatmap(data_fixed, data_moving)
    
    print("\n5. 绘制总体平均图...")
    plot_overall_average(data_fixed, data_moving)

    print("\n6. 创建统计摘要...")
    create_summary_statistics(data_fixed, data_moving, dim1_map, dim2_map)
    
    print(f"\n分析完成！所有图像和摘要已保存到: {ANADIR}")
    print("\n生成的文件:")
    print("- location_XX_*.png: 各位置的详细时间序列图")
    print("- all_locations_comparison.png: 所有位置对比图")
    print("- temperature_groups_heatmap_by_location.png: 温度组热图")
    print("- location_analysis_summary.txt: 统计摘要")

if __name__ == "__main__":
    main()
