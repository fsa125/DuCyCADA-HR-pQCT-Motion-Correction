"""
plot_results.py
---------------
Generates box-plot figures from the CSV produced by evaluate.py.

Two plot modes:
    single  -- one CSV, plot PSNR / SSIM / VIF columns.
    compare -- one CSV with paired columns (e.g. PSNR_proxrad vs PSNR_distrad),
               draws side-by-side box plots for each metric.

Usage (single CSV)
------------------
python plot_results.py \\
    --mode   single \\
    --csv    results/metrics_Dist_rad.csv \\
    --output results/boxplot_Dist_rad.png

Usage (comparison CSV)
----------------------
python plot_results.py \\
    --mode    compare \\
    --csv     results/Metric.csv \\
    --cols_a  PSNR_proxrad SSIM_proxrad IFC_proxrad \\
    --cols_b  PSNR_distrad SSIM_distrad IFC_distrad \\
    --output  results/boxplot_compare.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Generate box-plot figures from metric CSVs.")

    parser.add_argument("--mode",   type=str, choices=["single", "compare"], default="single",
                        help="'single': plot one CSV.  'compare': side-by-side from paired columns.")
    parser.add_argument("--csv",    type=str, required=True,
                        help="Path to the input CSV file.")
    parser.add_argument("--output", type=str, default="results/boxplot.png",
                        help="Path to save the output figure.")
    parser.add_argument("--dpi",    type=int, default=150)

    # single mode: which columns to plot (default: PSNR, SSIM, VIF)
    parser.add_argument("--metrics", nargs="+",
                        default=["PSNR", "SSIM", "VIF"],
                        help="(single mode) Column names to include in the box plot.")

    # compare mode: paired column groups
    parser.add_argument("--cols_a", nargs="+",
                        default=["PSNR_proxrad", "SSIM_proxrad", "IFC_proxrad"],
                        help="(compare mode) First group column names.")
    parser.add_argument("--cols_b", nargs="+",
                        default=["PSNR_distrad", "SSIM_distrad", "IFC_distrad"],
                        help="(compare mode) Second group column names.")
    parser.add_argument("--metric_labels", nargs="+",
                        default=["PSNR", "SSIM", "IFC"],
                        help="(compare mode) Display names for each metric subplot.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Plot helpers  (unchanged logic from notebook cells 61 / 62 / 63)
# ---------------------------------------------------------------------------

def plot_single(df, metrics, output, dpi):
    """Single CSV box plot with seaborn styling."""
    df_plot = df[metrics].dropna()

    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("Set2")
    sns.boxplot(data=df_plot, ax=ax, palette=palette, linewidth=2, fliersize=5)

    plt.title("Box Plot of Image Quality Metrics",
              fontsize=16, fontname="Times New Roman", fontweight="bold")
    plt.ylabel("Values", fontsize=14, fontname="Times New Roman")
    plt.xticks(fontsize=12, fontname="Times New Roman")
    plt.yticks(fontsize=12, fontname="Times New Roman")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {output}")


def plot_compare(df, cols_a, cols_b, metric_labels, output, dpi):
    """Side-by-side box plots for each metric pair (unchanged from notebook cell 63)."""
    title_font = {"fontsize": 14, "fontname": "Times New Roman"}
    colors     = ["lightblue", "lightgreen"]

    fig, axes = plt.subplots(1, len(metric_labels), figsize=(6 * len(metric_labels), 6))
    if len(metric_labels) == 1:
        axes = [axes]

    for ax, col_a, col_b, label in zip(axes, cols_a, cols_b, metric_labels):
        data = df[[col_a, col_b]].dropna()
        bp   = ax.boxplot(
            [data[col_a], data[col_b]],
            labels=[col_a, col_b],
            patch_artist=True,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title(label, **title_font)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.yaxis.label.set_size(12)
        ax.yaxis.label.set_fontname("Times New Roman")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    df   = pd.read_csv(args.csv)
    if "Batch" in df.columns:
        df = df.drop(columns=["Batch"])

    if args.mode == "single":
        plot_single(df, args.metrics, args.output, args.dpi)
    else:
        assert len(args.cols_a) == len(args.cols_b) == len(args.metric_labels), \
            "--cols_a, --cols_b and --metric_labels must all have the same length."
        plot_compare(df, args.cols_a, args.cols_b, args.metric_labels, args.output, args.dpi)


if __name__ == "__main__":
    main()
