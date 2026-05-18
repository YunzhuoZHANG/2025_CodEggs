#!/usr/bin/env python3
"""Export meanspace.mat to CSV and generate polished validation plots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.io import loadmat


@dataclass(frozen=True)
class VariableSpec:
    name: str
    start_year: int
    end_year: int
    source: str
    baseline: str
    metric: str
    metric_label: str
    unit: str


ROOT = Path(__file__).resolve().parent
MAT_PATH = ROOT / "meanspace.mat"
WIDE_CSV_PATH = ROOT / "meanspace_wide.csv"
LONG_CSV_PATH = ROOT / "meanspace_long.csv"
PNG_PATH = ROOT / "meanspace_validation_comparison.png"
PDF_PATH = ROOT / "meanspace_validation_comparison.pdf"

VARIABLE_SPECS = [
    VariableSpec(
        name="mtp_oisst_space",
        start_year=1982,
        end_year=2023,
        source="OISST",
        baseline="observed",
        metric="mean_temperature",
        metric_label="Mean temperature during MHW",
        unit="degC",
    ),
    VariableSpec(
        name="meand_oisst_space",
        start_year=1982,
        end_year=2023,
        source="OISST",
        baseline="observed",
        metric="mean_duration",
        metric_label="Mean duration",
        unit="days",
    ),
    VariableSpec(
        name="mtp_awim_space",
        start_year=1982,
        end_year=2085,
        source="AWI",
        baseline="moving",
        metric="mean_temperature",
        metric_label="Mean temperature during MHW",
        unit="degC",
    ),
    VariableSpec(
        name="meand_awim_space",
        start_year=1982,
        end_year=2085,
        source="AWI",
        baseline="moving",
        metric="mean_duration",
        metric_label="Mean duration",
        unit="days",
    ),
    VariableSpec(
        name="mtp_awif_space",
        start_year=1982,
        end_year=2085,
        source="AWI",
        baseline="fixed",
        metric="mean_temperature",
        metric_label="Mean temperature during MHW",
        unit="degC",
    ),
    VariableSpec(
        name="meand_awif_space",
        start_year=1982,
        end_year=2085,
        source="AWI",
        baseline="fixed",
        metric="mean_duration",
        metric_label="Mean duration",
        unit="days",
    ),
]

FULL_YEARS = np.arange(1982, 2086)
COLORS = {
    "oisst": "#2B6CB0",
    "awi": "#D1495B",
    "future_fill": "#EEF2F7",
    "grid": "#D5DEE8",
    "text": "#243B53",
    "spine": "#9FB3C8",
}


def load_flat_variable(data: dict[str, np.ndarray], spec: VariableSpec) -> np.ndarray:
    values = np.asarray(data[spec.name], dtype=float).squeeze()
    if values.ndim != 1:
        raise ValueError(f"{spec.name} is not a 1D vector after squeeze(): {values.shape}")

    expected_length = spec.end_year - spec.start_year + 1
    if values.size != expected_length:
        raise ValueError(
            f"{spec.name} has length {values.size}, expected {expected_length} "
            f"for {spec.start_year}-{spec.end_year}"
        )
    return values


def export_tables(mat_path: Path, wide_csv_path: Path, long_csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_data = loadmat(mat_path)

    wide_df = pd.DataFrame({"year": FULL_YEARS})
    long_records: list[dict[str, object]] = []

    for spec in VARIABLE_SPECS:
        values = load_flat_variable(raw_data, spec)
        years = np.arange(spec.start_year, spec.end_year + 1)

        aligned = pd.Series(values, index=years).reindex(FULL_YEARS)
        wide_df[spec.name] = aligned.to_numpy()

        for year, value in zip(years, values, strict=True):
            long_records.append(
                {
                    "year": int(year),
                    "value": float(value),
                    "variable": spec.name,
                    "source": spec.source,
                    "baseline": spec.baseline,
                    "metric": spec.metric,
                    "metric_label": spec.metric_label,
                    "unit": spec.unit,
                    "period": "overlap" if year <= 2023 else "future_extension",
                }
            )

    long_df = pd.DataFrame.from_records(long_records).sort_values(
        ["metric", "source", "baseline", "year"]
    )

    wide_df.to_csv(wide_csv_path, index=False, float_format="%.6f")
    long_df.to_csv(long_csv_path, index=False, float_format="%.6f")
    return wide_df, long_df


def apply_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Noto Serif CJK SC"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.labelcolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": COLORS["spine"],
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.9,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "savefig.bbox": "tight",
        }
    )


def panel_columns(metric: str, baseline: str) -> tuple[str, str]:
    metric_prefix = "mtp" if metric == "mean_temperature" else "meand"
    awi_suffix = "awim" if baseline == "moving" else "awif"
    return f"{metric_prefix}_oisst_space", f"{metric_prefix}_{awi_suffix}_space"


def set_panel_limits(ax: plt.Axes, observed: np.ndarray, simulated: np.ndarray, metric: str) -> None:
    values = np.concatenate([observed[np.isfinite(observed)], simulated[np.isfinite(simulated)]])
    if values.size == 0:
        return

    vmin = float(values.min())
    vmax = float(values.max())
    span = max(vmax - vmin, 1.0)
    padding = 0.10 * span

    if metric == "mean_duration":
        lower = max(0.0, vmin - 0.03 * span)
    else:
        lower = vmin - padding

    upper = vmax + padding
    ax.set_ylim(lower, upper)


def plot_panel(ax: plt.Axes, wide_df: pd.DataFrame, metric: str, baseline: str, panel_label: str) -> None:
    oisst_col, awi_col = panel_columns(metric, baseline)
    years = wide_df["year"].to_numpy()
    oisst = wide_df[oisst_col].to_numpy(dtype=float)
    awi = wide_df[awi_col].to_numpy(dtype=float)

    overlap_mask = years <= 2023
    oisst_mask = np.isfinite(oisst)
    awi_mask = np.isfinite(awi)
    oisst_overlap = oisst[overlap_mask & oisst_mask]
    awi_overlap = awi[overlap_mask & awi_mask]

    ax.axvspan(2023.5, 2085.5, color=COLORS["future_fill"], zorder=0)
    ax.axvline(2023.5, color=COLORS["spine"], linestyle=":", linewidth=1.0, zorder=1)

    ax.plot(
        years[oisst_mask],
        oisst[oisst_mask],
        color=COLORS["oisst"],
        label="OISST",
        zorder=3,
    )
    ax.plot(
        years[awi_mask],
        awi[awi_mask],
        color=COLORS["awi"],
        label="AWI",
        zorder=2,
    )

    if oisst_overlap.size:
        ax.hlines(
            y=float(np.nanmean(oisst_overlap)),
            xmin=1982,
            xmax=2023,
            color=COLORS["oisst"],
            linestyle=(0, (4, 2)),
            linewidth=1.6,
            alpha=0.9,
            zorder=1.5,
        )
    if awi_overlap.size:
        ax.hlines(
            y=float(np.nanmean(awi_overlap)),
            xmin=1982,
            xmax=2023,
            color=COLORS["awi"],
            linestyle=(0, (4, 2)),
            linewidth=1.6,
            alpha=0.9,
            zorder=1.5,
        )

    metric_title = "Mean temperature" if metric == "mean_temperature" else "Mean duration"
    baseline_title = "moving baseline" if baseline == "moving" else "fixed baseline"
    ax.set_title(f"{panel_label} {metric_title} vs AWI {baseline_title}", loc="left", pad=8)

    if metric == "mean_temperature":
        ax.set_ylabel("Mean temperature during MHW [°C]")
    else:
        ax.set_ylabel("Mean duration [days]")

    ax.set_xlim(1982, 2085)
    ax.set_xticks([1982, 2000, 2020, 2040, 2060, 2080])
    ax.set_xlabel("Year")
    set_panel_limits(ax, oisst, awi, metric)


def create_figure(wide_df: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    apply_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.22, wspace=0.18, hspace=0.28)

    plot_panel(axes[0, 0], wide_df, "mean_temperature", "moving", "(a)")
    plot_panel(axes[0, 1], wide_df, "mean_duration", "moving", "(b)")
    plot_panel(axes[1, 0], wide_df, "mean_temperature", "fixed", "(c)")
    plot_panel(axes[1, 1], wide_df, "mean_duration", "fixed", "(d)")

    legend_handles = [
        Line2D([0], [0], color=COLORS["oisst"], linewidth=2.0, label="OISST"),
        Line2D([0], [0], color=COLORS["awi"], linewidth=2.0, label="AWI"),
        Line2D(
            [0],
            [0],
            color=COLORS["oisst"],
            linewidth=1.8,
            linestyle=(0, (4, 2)),
            label="OISST overlap mean (1982-2023)",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["awi"],
            linewidth=1.8,
            linestyle=(0, (4, 2)),
            label="AWI overlap mean (1982-2023)",
        ),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.085))
    fig.suptitle("Spawning-ground marine heatwave metrics: OISST and AWI comparison", y=0.965, fontsize=14)
    fig.text(
        0.5,
        0.025,
        "Grey shading marks the AWI-only extension beyond the OISST observational period.",
        ha="center",
        va="bottom",
        fontsize=9,
        color=COLORS["text"],
    )

    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=MAT_PATH, help="Path to the input meanspace.mat file.")
    parser.add_argument(
        "--wide-csv",
        type=Path,
        default=WIDE_CSV_PATH,
        help="Path to the wide CSV output.",
    )
    parser.add_argument(
        "--long-csv",
        type=Path,
        default=LONG_CSV_PATH,
        help="Path to the long CSV output.",
    )
    parser.add_argument("--png", type=Path, default=PNG_PATH, help="Path to the PNG figure output.")
    parser.add_argument("--pdf", type=Path, default=PDF_PATH, help="Path to the PDF figure output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wide_df, long_df = export_tables(args.mat, args.wide_csv, args.long_csv)
    create_figure(wide_df, args.png, args.pdf)

    print(f"Input MAT file: {args.mat}")
    print(f"Wide CSV saved to: {args.wide_csv}")
    print(f"Long CSV saved to: {args.long_csv}")
    print(f"PNG figure saved to: {args.png}")
    print(f"PDF figure saved to: {args.pdf}")
    print()
    print("Rows in wide CSV:", len(wide_df))
    print("Rows in long CSV:", len(long_df))


if __name__ == "__main__":
    main()
