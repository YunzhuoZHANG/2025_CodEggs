#!/usr/bin/env python3
"""
调试数据维度问题
"""

import netCDF4 as nc
import numpy as np

# 文件路径
ROOT = '/public/home/yunzhuozhang/AWI'
COMPRESSED_FILES = {
    'fixed': ROOT + '/codarea_mhw_fixedbaseline_compressed.nc',
}

# 参数
TIME_LEN = 37960
NYEARS = 104
NDAY = 365
YEAR0 = 1982
LABELS_FULL = ['< -0.92', 'optimum (-0.92~1.352)', 'Moderate (1.352~2.119)', 'High (2.119~2.834)', 'Severe (>=2.834)']
PLOT_GROUP_INDICES = list(range(1, len(LABELS_FULL)))

def debug_data_processing():
    print("调试数据处理...")
    
    with nc.Dataset(COMPRESSED_FILES['fixed'], 'r') as ds:
        temp_class = ds.variables['temperature_class'][:]  # (8, 37960)
        mhw_class = ds.variables['mhw_class'][:]           # (8, 37960)
        
        print(f"原始数据形状:")
        print(f"  temp_class: {temp_class.shape}")
        print(f"  mhw_class: {mhw_class.shape}")
        
        # 检查前100个时间点的数据分布
        print(f"\n前100个时间点的温度类别分布:")
        for location in range(8):
            temp_sample = temp_class[location, :100]
            unique, counts = np.unique(temp_sample, return_counts=True)
            print(f"  Location {location}: {dict(zip(unique, counts))}")
        
        # 准备结果数组: (8位置, 104年, 5组, 2标志)
        location_stats = np.zeros((8, NYEARS, len(LABELS_FULL), 2), dtype=np.int64)
        
        print(f"\n统计数组形状: {location_stats.shape}")
        
        # 处理第一年数据作为测试
        year_idx = 0
        start_day = year_idx * NDAY
        end_day = start_day + NDAY
        
        print(f"\n处理第一年 (天数 {start_day} 到 {end_day}):")
        
        for location in range(8):
            temp_year = temp_class[location, start_day:end_day]
            mhw_year = mhw_class[location, start_day:end_day]
            
            print(f"  Location {location}:")
            print(f"    temp_year 形状: {temp_year.shape}")
            print(f"    mhw_year 形状: {mhw_year.shape}")
            
            # 统计当年数据
            for day in range(min(NDAY, len(temp_year))):
                temp_group = temp_year[day]
                mhw_flag = mhw_year[day]
                
                if 0 <= temp_group < len(LABELS_FULL) and 0 <= mhw_flag <= 1:
                    location_stats[location, year_idx, temp_group, mhw_flag] += 1
            
            # 显示统计结果
            year_stats = location_stats[location, year_idx, :, :]
            print(f"    年度统计 (组, MHW标志): {year_stats.shape}")
            for group in range(len(LABELS_FULL)):
                for flag in range(2):
                    count = year_stats[group, flag]
                    if count > 0:
                        print(f"      组 {group}, MHW={flag}: {count} 天")
        
        print(f"\n排除最低组后的索引: {PLOT_GROUP_INDICES}")
        
        # 测试切片
        location = 0
        fixed_data = location_stats[location, :, PLOT_GROUP_INDICES, :]
        print(f"\nLocation 0 切片后数据形状: {fixed_data.shape}")
        print(f"  期望形状: (年数={NYEARS}, 组数={len(PLOT_GROUP_INDICES)}, 标志=2)")
        
        # 测试各组的数据
        for gi in range(len(PLOT_GROUP_INDICES)):
            group_data = fixed_data[:, gi, :]
            print(f"  组 {gi} 数据形状: {group_data.shape}")
            print(f"  组 {gi} MHW=0 前5年: {group_data[:5, 0]}")
            print(f"  组 {gi} MHW=1 前5年: {group_data[:5, 1]}")

if __name__ == "__main__":
    debug_data_processing()
