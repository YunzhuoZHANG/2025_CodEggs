#!/usr/bin/env python3
"""
调试版本的位置绘图脚本
"""

import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 文件路径
ROOT = '/public/home/yunzhuozhang/AWI'
ANADIR = os.path.join(ROOT, 'analysis')
COMPRESSED_FILES = {
    'fixed': os.path.join(ROOT, 'codarea_mhw_fixedbaseline_compressed.nc'),
    'moving': os.path.join(ROOT, 'codarea_mhw_movingbaseline_compressed.nc'),
}

def test_data_access():
    """测试数据访问"""
    print("测试数据访问...")
    
    for name, filepath in COMPRESSED_FILES.items():
        print(f"\n检查文件: {name} - {filepath}")
        
        if not os.path.exists(filepath):
            print(f"  ❌ 文件不存在!")
            continue
            
        try:
            with nc.Dataset(filepath, 'r') as ds:
                print(f"  ✅ 文件可以打开")
                print(f"  维度: {dict(ds.dimensions)}")
                print(f"  变量: {list(ds.variables.keys())}")
                
                # 检查关键变量
                if 'temperature_class' in ds.variables:
                    temp_class = ds.variables['temperature_class']
                    print(f"  temperature_class 形状: {temp_class.shape}")
                    print(f"  temperature_class 数据类型: {temp_class.dtype}")
                    
                    # 读取一小部分数据测试
                    sample_data = temp_class[0, :10]
                    print(f"  样本数据: {sample_data}")
                    
                    # 检查唯一值
                    unique_vals = np.unique(temp_class[:])
                    print(f"  温度类别唯一值: {unique_vals}")
                
                if 'mhw_class' in ds.variables:
                    mhw_class = ds.variables['mhw_class']
                    print(f"  mhw_class 形状: {mhw_class.shape}")
                    unique_mhw = np.unique(mhw_class[:])
                    print(f"  MHW类别唯一值: {unique_mhw}")
                    
        except Exception as e:
            print(f"  ❌ 读取文件时出错: {e}")
            import traceback
            traceback.print_exc()

def create_simple_plot():
    """创建一个简单的测试图"""
    print("\n创建简单测试图...")
    
    try:
        # 确保目录存在
        os.makedirs(ANADIR, exist_ok=True)
        
        # 读取数据
        with nc.Dataset(COMPRESSED_FILES['fixed'], 'r') as ds:
            temp_class = ds.variables['temperature_class'][:]  # (8, 37960)
            
        print(f"读取到的数据形状: {temp_class.shape}")
        
        # 创建简单的时间序列图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制前8天每个位置的温度类别
        for location in range(8):
            data_sample = temp_class[location, :365]  # 第一年的数据
            valid_data = data_sample[data_sample >= 0]
            
            if len(valid_data) > 0:
                ax.plot(range(len(valid_data)), valid_data, 
                       label=f'Location {location}', marker='o', markersize=2)
        
        ax.set_xlabel('Day of Year')
        ax.set_ylabel('Temperature Class')
        ax.set_title('Temperature Classes for First Year by Location')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 保存图像
        test_filename = os.path.join(ANADIR, "test_location_plot.png")
        fig.savefig(test_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ 测试图像保存: {test_filename}")
        
    except Exception as e:
        print(f"❌ 创建测试图时出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("调试8个位置的数据访问")
    print("="*50)
    
    test_data_access()
    create_simple_plot()
    
    print("\n调试完成!")

if __name__ == "__main__":
    main()
