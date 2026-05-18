#!/usr/bin/env python3
"""
绘制 SST 的阶段（historical/current/future）逐月平均时间序列：

版式：2x3 子图
- 行：Fixed 基线、Moving 基线
- 列：MHW=0、MHW=1、SUM（不区分 MHW）

每个子图上用三条线表示各阶段（historical/current/future）的逐月平均 SST；
横坐标为月份（1..12），纵坐标为 SST（°C）。
另外以横线标出温度分类阈值（-0.92, 1.352, 2.119, 2.834），并在阈值区间间加入浅色半透明背景：
< -0.92: 灰色；optimum: 浅蓝；moderate: 浅黄；high: 橙红；severe: 深红。
"""

import os
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ------------ 常量与路径 ---------------
ROOT = '/public/home/yunzhuozhang/AWI'
ANADIR = os.path.join(ROOT, 'location_analysis')
COMPRESSED_FILES = {
    'fixed': os.path.join(ROOT, 'codarea_mhw_fixedbaseline_compressed.nc'),
    'moving': os.path.join(ROOT, 'codarea_mhw_movingbaseline_compressed.nc'),
}

TIME_LEN = 37960
NYEARS = 104
NDAY = 365
YEAR0 = 1982

# 阶段
PHASES = {
    'historical': lambda y: y < 2010,
    'current':    lambda y: 2010 <= y <= 2050,
    'future':     lambda y: y > 2050,
}

PHASE_COLORS = {
    'historical': '#1f77b4',  # blue
    'current':    '#ff7f0e',  # orange
    'future':     '#2ca02c',  # green
}

# 带年份的阶段标签
PHASE_LABELS = {
    'historical': 'historical (1982–2009)',
    'current':    'current (2010–2050)',
    'future':     'future (2051–2085)',
}

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

THRESHOLDS = [-0.92, 1.352, 2.119, 2.834]
# 略微加深背景与图例色；同时将“高温”改为橙红、“极端”改为深红
SHADE_BANDS = [
    (-np.inf, THRESHOLDS[0], (0.35, 0.35, 0.35), 0.16),        # deeper gray
    (THRESHOLDS[0], THRESHOLDS[1], (0.45, 0.70, 0.95), 0.16),  # deeper light blue
    (THRESHOLDS[1], THRESHOLDS[2], (0.98, 0.88, 0.35), 0.16),  # deeper light yellow
    (THRESHOLDS[2], THRESHOLDS[3], (1.00, 0.27, 0.00), 0.16),  # OrangeRed (更偏橙红)
    (THRESHOLDS[3], np.inf, (0.80, 0.10, 0.10), 0.16),         # deep red
]


def load_arrays(nc_path):
    with nc.Dataset(nc_path, 'r') as ds:
        sst = ds.variables['sst_all'][:]       # (8, TIME_LEN)
        mhw = ds.variables['mhw_class'][:]     # (8, TIME_LEN)
        if 'original_dim1' in ds.variables:
            dim1_map = ds.variables['original_dim1'][:]
            dim2_map = ds.variables['original_dim2'][:]
        else:
            dim1_map = np.array([0,0,0,0,1,1,1,1])
            dim2_map = np.array([0,1,2,3,0,1,2,3])
    return sst, mhw, dim1_map, dim2_map


def doy_indices_for_phase(year_mask):
    """为阶段内每个日序 d 返回所有绝对时间索引。
    返回 idx_by_doy: list 长度 365，每个元素是绝对索引的一维数组。
    """
    idx_by_doy = []
    year_ids = np.where(year_mask)[0]
    for d in range(NDAY):
        # 该 DOY 的全部年对应的绝对索引
        idx = year_ids * NDAY + d
        idx = idx[(idx >= 0) & (idx < TIME_LEN)]
        idx_by_doy.append(idx)
    return idx_by_doy


def _circular_running_mean(values: np.ndarray, window: int = 11) -> np.ndarray:
    """NaN 感知的环形（跨 DOY）移动平均。
    - values: (NDAY,) 数组，可含 NaN
    - window: 奇数窗口，默认 11
    返回同尺寸平滑结果；如果某点窗口内全为 NaN，则该点为 NaN。
    """
    if window < 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    v = values.astype(float)
    mask = np.isfinite(v).astype(float)
    v_filled = np.where(np.isfinite(v), v, 0.0)

    # 环状扩展
    v_ext = np.concatenate([v_filled[-half:], v_filled, v_filled[:half]])
    m_ext = np.concatenate([mask[-half:], mask, mask[:half]])
    kernel = np.ones(window, dtype=float)

    num = np.convolve(v_ext, kernel, mode='valid')  # 长度 NDAY
    den = np.convolve(m_ext, kernel, mode='valid')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    out[den == 0] = np.nan
    return out


def _fill_nan_circular(y: np.ndarray) -> np.ndarray:
    """环形线性插值填补 NaN；若全部为 NaN，则返回原值。"""
    x = np.arange(y.size)
    finite = np.isfinite(y)
    if not np.any(finite):
        return y
    # 为了环形，复制首尾点
    xf = x[finite]
    yf = y[finite]
    # 扩展用于环形插值
    xf_ext = np.concatenate([xf - y.size, xf, xf + y.size])
    yf_ext = np.concatenate([yf, yf, yf])
    yi = np.interp(x, xf_ext, yf_ext)
    return yi


def compute_phase_doy_means(sst, mhw, phase, smooth_window: int = 11):
    """计算给定阶段在每个 DOY 上的 SST 平均，分三条曲线：MHW=0、MHW=1、SUM。
    返回数组形状 (3, NDAY)：顺序 [MHW0, MHW1, SUM]，并进行跨 DOY 平滑与缺样填充。
    """
    years = YEAR0 + np.arange(NYEARS)
    sel = np.array([PHASES[phase](int(y)) for y in years], dtype=bool)
    idx_by_doy = doy_indices_for_phase(sel)

    out = np.full((3, NDAY), np.nan, dtype=float)

    for d in range(NDAY):
        idx = idx_by_doy[d]
        if idx.size == 0:
            continue
        # 所有位置由外部循环控制，这里传入的是 (time,) 或 (loc,time)。此函数按位置单独调用。
        s_d = sst[idx]
        m_d = mhw[idx]

        # SUM: 无条件均值
        out[2, d] = np.nanmean(s_d) if s_d.size else np.nan

        # MHW=0 与 MHW=1 条件均值
        for f, row in [(0, 0), (1, 1)]:
            cond = (m_d == f)
            if np.any(cond):
                out[row, d] = np.nanmean(s_d[cond])
            else:
                out[row, d] = np.nan
    # 跨 DOY 平滑并填补缺样，确保不断线
    for i in range(3):
        sm = _circular_running_mean(out[i], window=smooth_window)
        if np.any(~np.isfinite(sm)):
            sm = _fill_nan_circular(sm)
        out[i] = sm
    return out


def compute_phase_month_means(sst, mhw, phase, fill_nan: bool = True):
    """计算给定阶段在每个月上的 SST 平均，分三条曲线：MHW=0、MHW=1、SUM。
    返回数组形状 (3, 12)：顺序 [MHW0, MHW1, SUM]。
    - sst, mhw 为单个位置的一维时间序列（长度 TIME_LEN）。
    - 阶段通过 PHASES 定义，按年选择。
    """
    years = YEAR0 + np.arange(NYEARS)
    sel_years = np.array([PHASES[phase](int(y)) for y in years], dtype=bool)
    year_ids = np.where(sel_years)[0]

    # no-leap 月份天数
    month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=int)
    month_starts = np.concatenate([[0], np.cumsum(month_days)[:-1]])  # 每月起始 DOY（0-based）

    out = np.full((3, 12), np.nan, dtype=float)

    for m in range(12):
        d0 = int(month_starts[m])
        d1 = int(d0 + month_days[m])  # exclusive
        days = np.arange(d0, d1, dtype=int)  # 当月内的 DOY

        if year_ids.size == 0 or days.size == 0:
            continue
        # 组合成绝对时间索引：所有选中年份 × 当月天
        idx = (year_ids[:, None] * NDAY + days[None, :]).reshape(-1)
        idx = idx[(idx >= 0) & (idx < TIME_LEN)]
        if idx.size == 0:
            continue

        s_m = sst[idx]
        m_m = mhw[idx]

        # SUM: 无条件均值
        out[2, m] = np.nanmean(s_m) if s_m.size else np.nan

        # MHW=0 与 MHW=1 条件均值
        for f, row in [(0, 0), (1, 1)]:
            cond = (m_m == f)
            out[row, m] = np.nanmean(s_m[cond]) if np.any(cond) else np.nan

    if fill_nan:
        # 使用环形线性插值填补缺样，避免折线断裂
        for i in range(3):
            yi = _fill_nan_circular(out[i])
            out[i] = yi
    return out


def plot_timeseries_for_location(loc, fixed_sst, fixed_mhw, moving_sst, moving_mhw, dim1_map, dim2_map):
    """绘制单个位置的 2x3 时间序列图，三列分别为 MHW=0、MHW=1、SUM（按月平均）。"""
    name = LOCATION_NAMES[loc]
    safe = name.replace(' ', '_').replace('-', '_').lower()
    original_pos = f"[{int(dim1_map[loc])},{int(dim2_map[loc])}]"

    # 计算每个阶段曲线（按月平均）
    curves = {}
    for phase in PHASES.keys():
        curves[(phase, 'fixed')] = compute_phase_month_means(fixed_sst, fixed_mhw, phase)
        curves[(phase, 'moving')] = compute_phase_month_means(moving_sst, moving_mhw, phase)

    # 统一 y 轴范围
    all_vals = []
    for k, arr in curves.items():
        all_vals.append(arr)
    all_vals = np.concatenate(all_vals, axis=None)
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)
    pad = 0.05 * (vmax - vmin if np.isfinite(vmax - vmin) and (vmax - vmin) > 0 else 1.0)
    ylo, yhi = (vmin - pad, vmax + pad)

    x = np.arange(1, 12 + 1)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    # 顶部留给标题，底部留给两行图例
    fig.subplots_adjust(hspace=0.25, wspace=0.15, top=0.9, bottom=0.22)

    col_info = [('MHW=0', 0), ('MHW=1', 1), ('SUM', 2)]
    row_info = [('Fixed Baseline', 'fixed'), ('Moving Baseline', 'moving')]

    legend_lines = []
    legend_labels = []
    band_handles = []
    band_labels = []

    for r, (rlabel, scheme) in enumerate(row_info):
        for c, (clabel, idx_row) in enumerate(col_info):
            ax = axes[r, c]
            # 背景阈值区间（先画背景）
            for lower, upper, color, alpha in SHADE_BANDS:
                y1 = max(ylo, lower if np.isfinite(lower) else ylo)
                y2 = min(yhi, upper if np.isfinite(upper) else yhi)
                if y2 > y1:
                    ax.axhspan(y1, y2, facecolor=color, alpha=alpha, zorder=0)

            # 各阶段曲线
            for phase, color in PHASE_COLORS.items():
                y = curves[(phase, scheme)][idx_row]
                line, = ax.plot(x, y, color=color, linewidth=1.8, label=PHASE_LABELS[phase], zorder=3)
                if r == 0 and c == 0:
                    legend_lines.append(line)
                    legend_labels.append(PHASE_LABELS[phase])

            # 阈值横线
            for thr in THRESHOLDS:
                ax.axhline(thr, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)

            if r == 0:
                ax.set_title(clabel)
            if c == 0:
                ax.set_ylabel(f"{rlabel}\nSST (°C)")
            ax.grid(True, alpha=0.3)

    for c in range(3):
        axes[1, c].set_xlabel('Month (1-12)')

    # y 轴范围
    for ax in axes.ravel():
        ax.set_ylim(ylo, yhi)

    # 区间颜色图例（仅添加一次）
    band_info = [
        ('< -0.92', SHADE_BANDS[0][2]),
        ('optimum (-0.92~1.352)', SHADE_BANDS[1][2]),
        ('Moderate (1.352~2.119)', SHADE_BANDS[2][2]),
        ('High (2.119~2.834)', SHADE_BANDS[3][2]),
        ('Severe (>=2.834)', SHADE_BANDS[4][2]),
    ]
    for lbl, col in band_info:
        band_handles.append(Patch(facecolor=col, edgecolor='none', alpha=SHADE_BANDS[0][3]))
        band_labels.append(lbl)

    # 分两行图例（底部）：第一行阶段（带说明），第二行温度区间（带说明）
    leg1 = fig.legend(legend_lines, legend_labels, loc='lower center', ncol=3,
                      bbox_to_anchor=(0.5, 0.08), title='Phases')
    leg2 = fig.legend(band_handles, band_labels, loc='lower center', ncol=5,
                      bbox_to_anchor=(0.5, 0.02), title='Temperature bands')
    for lg in (leg1, leg2):
        lg.get_title().set_fontsize(10)
    fig.suptitle(f"Region {loc+1}: {name} {original_pos} | SST Monthly Mean", fontsize=13, fontweight='bold', y=0.98)

    os.makedirs(ANADIR, exist_ok=True)
    out = os.path.join(ANADIR, f"sst_timeseries_region_{loc:02d}_{safe}.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_timeseries_overall(fixed_sst, fixed_mhw, moving_sst, moving_mhw):
    """对 8 个位置取平均后绘制总图（按月平均）。"""
    # 先计算每个位置的阶段曲线后再平均，保证条件筛选在位置级别完成
    curves = {}
    for phase in PHASES.keys():
        # 收集每个位置的曲线，然后逐曲线平均（沿 axis=0）
        fixed_list = []
        moving_list = []
        for loc in range(8):
            fixed_list.append(compute_phase_month_means(fixed_sst[loc], fixed_mhw[loc], phase))
            moving_list.append(compute_phase_month_means(moving_sst[loc], moving_mhw[loc], phase))
        curves[(phase, 'fixed')] = np.nanmean(np.stack(fixed_list, axis=0), axis=0)
        curves[(phase, 'moving')] = np.nanmean(np.stack(moving_list, axis=0), axis=0)

    # y 轴范围
    all_vals = np.concatenate([arr for arr in curves.values()], axis=None)
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)
    pad = 0.05 * (vmax - vmin if np.isfinite(vmax - vmin) and (vmax - vmin) > 0 else 1.0)
    ylo, yhi = (vmin - pad, vmax + pad)

    x = np.arange(1, 12 + 1)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.25, wspace=0.15, top=0.9, bottom=0.22)

    col_info = [('MHW=0', 0), ('MHW=1', 1), ('SUM', 2)]
    row_info = [('Fixed Baseline', 'fixed'), ('Moving Baseline', 'moving')]

    legend_lines = []
    legend_labels = []
    band_handles = []
    band_labels = []

    for r, (rlabel, scheme) in enumerate(row_info):
        for c, (clabel, idx_row) in enumerate(col_info):
            ax = axes[r, c]
            # 背景阈值区间
            for lower, upper, color_band, alpha in SHADE_BANDS:
                y1 = max(ylo, lower if np.isfinite(lower) else ylo)
                y2 = min(yhi, upper if np.isfinite(upper) else yhi)
                if y2 > y1:
                    ax.axhspan(y1, y2, facecolor=color_band, alpha=alpha, zorder=0)

            for phase, color in PHASE_COLORS.items():
                y = curves[(phase, scheme)][idx_row]
                line, = ax.plot(x, y, color=color, linewidth=1.8, label=PHASE_LABELS[phase], zorder=3)
                if r == 0 and c == 0:
                    legend_lines.append(line)
                    legend_labels.append(PHASE_LABELS[phase])
            for thr in THRESHOLDS:
                ax.axhline(thr, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)
            if r == 0:
                ax.set_title(clabel)
            if c == 0:
                ax.set_ylabel(f"{rlabel}\nSST (°C)")
            ax.grid(True, alpha=0.3)

    for c in range(3):
        axes[1, c].set_xlabel('Month (1-12)')
    for ax in axes.ravel():
        ax.set_ylim(ylo, yhi)

    band_info = [
        ('< -0.92', SHADE_BANDS[0][2]),
        ('optimum (-0.92~1.352)', SHADE_BANDS[1][2]),
        ('Moderate (1.352~2.119)', SHADE_BANDS[2][2]),
        ('High (2.119~2.834)', SHADE_BANDS[3][2]),
        ('Severe (>=2.834)', SHADE_BANDS[4][2]),
    ]
    for lbl, col in band_info:
        band_handles.append(Patch(facecolor=col, edgecolor='none', alpha=SHADE_BANDS[0][3]))
        band_labels.append(lbl)
    # 分两行图例（底部）：第一行阶段（带说明），第二行温度区间（带说明）
    leg1 = fig.legend(legend_lines, legend_labels, loc='lower center', ncol=3,
                      bbox_to_anchor=(0.5, 0.08), title='Phases')
    leg2 = fig.legend(band_handles, band_labels, loc='lower center', ncol=5,
                      bbox_to_anchor=(0.5, 0.02), title='Temperature bands')
    for lg in (leg1, leg2):
        lg.get_title().set_fontsize(10)
    fig.suptitle("SST Monthly Mean | Overall Average", fontsize=13, fontweight='bold', y=0.98)

    os.makedirs(ANADIR, exist_ok=True)
    out = os.path.join(ANADIR, f"sst_timeseries_overall.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    print("SST 阶段时间序列（逐月平均）绘图（阈值背景）...")
    os.makedirs(ANADIR, exist_ok=True)

    # 读取数据
    print("读取 Fixed 压缩 NC...")
    fixed_sst, fixed_mhw, dim1_map, dim2_map = load_arrays(COMPRESSED_FILES['fixed'])
    print("读取 Moving 压缩 NC...")
    moving_sst, moving_mhw, _, _ = load_arrays(COMPRESSED_FILES['moving'])

    # 各位置绘图
    for loc in range(8):
        print(f"绘制位置 {loc}...")
        plot_timeseries_for_location(loc, fixed_sst[loc], fixed_mhw[loc], moving_sst[loc], moving_mhw[loc], dim1_map, dim2_map)

    # 总体平均
    print("绘制总体平均...")
    plot_timeseries_overall(fixed_sst, fixed_mhw, moving_sst, moving_mhw)

    print(f"完成，输出目录：{ANADIR}")


if __name__ == '__main__':
    main()
