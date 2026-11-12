"""
plot_results.py
结果可视化：生成报告所需的图表。

输出（到 results/figures/）：
  1. scatter_pred_vs_true.png   —— 预测 vs 实验 pKd 散点图（含对角线和相关系数）
  2. comparison_bar.png         —— 各模型 Pearson/Spearman/RMSE 对比柱状图
  3. ablation_bar.png           —— 消融实验柱状图（若 comparison 含足够项）

运行：python src/plot_results.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def plot_scatter():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "improved_predictions.csv"))
    y_true = df["true_pKd"].values
    y_pred = df["pred_pKd"].values
    r = pearsonr(y_true, y_pred)[0]
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    fig, ax = plt.subplots(figsize=(6, 5))
    # 浅色马卡龙配色：与 Figure 3 中 "RF (+ descriptor)" 模型一致（浅橙 + 深橙描边）
    ax.scatter(y_true, y_pred, s=34, alpha=0.75, color="#F5C4B3",
               edgecolors="#D85A30", linewidths=0.6)
    lims = [min(y_true.min(), y_pred.min()) - 0.5, max(y_true.max(), y_pred.max()) + 0.5]
    # 对角线改为深灰虚线（低饱和度，与整体浅色风格协调）
    ax.plot(lims, lims, "--", color="#6b7280", linewidth=1.4, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Experimental pKd", fontsize=11)
    ax.set_ylabel("Predicted pKd", fontsize=11)
    ax.set_title(f"Predicted vs Experimental binding affinity\n(CASF-2016 core set, n={len(y_true)})",
                 fontsize=11, fontweight="normal", loc="left")
    # 统计信息：简洁无边框文本框，与 Figure 3 风格协调
    ax.text(0.03, 0.97, f"Pearson R = {r:.3f}\nRMSE = {rmse:.3f}",
            transform=ax.transAxes, va="top", fontsize=10, color="#111827")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_aspect("equal", adjustable="box")
    # 只保留左/下轴线，去掉顶部和右侧（与 Figure 3 一致）
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "scatter_pred_vs_true.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 散点图已保存 -> {out}")


def _ylim_auto(vals, pad_ratio=0.12):
    """根据数据自动计算柱状图 y 轴范围（避免新数据下柱子被切掉）。"""
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span == 0:
        span = abs(hi) * 0.1 or 1.0
    return (lo - span * pad_ratio, hi + span * pad_ratio)


def plot_comparison():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "comparison.csv"))
    n = len(df)

    pearson = df["pearson"].tolist()
    spearman = df["spearman"].tolist()
    rmse = df["rmse"].tolist()

    # 简短、不重叠的模型标签
    labels = [
        "RF-Score\n(36 contact)",
        "XGBoost\n(36 contact)",
        "XGBoost\n(+ descriptor)",
        "RF\n(+ descriptor)",
    ]
    # 浅色马卡龙配色（蓝/绿/紫/橙）
    colors = ["#B5D4F4", "#9FE1CB", "#CECBF6", "#F5C4B3"]
    edges = ["#378ADD", "#1D9E75", "#7F77DD", "#D85A30"]

    best_pearson = int(np.argmax(pearson))
    best_spearman = int(np.argmax(spearman))
    best_rmse = int(np.argmin(rmse))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    def draw(ax, vals, best_idx, ylabel, title, dy):
        for i in range(n):
            ax.bar(i, vals[i], color=colors[i], edgecolor=edges[i],
                   linewidth=1.0, width=0.66)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(*_ylim_auto(vals))
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="normal", loc="left")
        for i, v in enumerate(vals):
            ax.text(i, v + dy, f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if i == best_idx else "normal")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    draw(axes[0], pearson, best_pearson, "Pearson R", "(a) Scoring power", 0.002)
    draw(axes[1], spearman, best_spearman, "Spearman rho", "(b) Ranking power", 0.002)
    draw(axes[2], rmse, best_rmse, "RMSE", "(c) Error (lower is better)", 0.004)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "comparison_bar.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 对比柱状图已保存 -> {out}")


def plot_ablation():
    """消融实验柱状图：仅接触特征 vs 仅描述子 vs 两者结合（随机森林）。"""
    path = os.path.join(RESULTS_DIR, "ablation.csv")
    if not os.path.exists(path):
        print("[图] 未找到 ablation.csv，跳过消融图")
        return

    df = pd.read_csv(path)
    n = len(df)
    pearson = df["pearson"].tolist()
    spearman = df["spearman"].tolist()
    rmse = df["rmse"].tolist()

    labels = [
        "Contact only\n(36)",
        "Descriptor only\n(12)",
        "Combined\n(48)",
    ]
    colors = ["#B5D4F4", "#9FE1CB", "#F5C4B3"]
    edges = ["#378ADD", "#1D9E75", "#D85A30"]

    best_pearson = int(np.argmax(pearson))
    best_spearman = int(np.argmax(spearman))
    best_rmse = int(np.argmin(rmse))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))

    def draw(ax, vals, best_idx, ylabel, title, dy):
        for i in range(n):
            ax.bar(i, vals[i], color=colors[i], edgecolor=edges[i],
                   linewidth=1.0, width=0.55)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(*_ylim_auto(vals))
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="normal", loc="left")
        for i, v in enumerate(vals):
            ax.text(i, v + dy, f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if i == best_idx else "normal")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    draw(axes[0], pearson, best_pearson, "Pearson R", "(a) Scoring power", 0.002)
    draw(axes[1], spearman, best_spearman, "Spearman rho", "(b) Ranking power", 0.002)
    draw(axes[2], rmse, best_rmse, "RMSE", "(c) Error (lower is better)", 0.004)

    fig.suptitle("Ablation study (Random Forest)", fontsize=12, y=1.02, x=0.5)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "ablation_bar.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] 消融柱状图已保存 -> {out}")


def main():
    plot_scatter()
    plot_comparison()
    plot_ablation()


if __name__ == "__main__":
    main()
