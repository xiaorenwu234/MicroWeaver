import json
import os

import matplotlib.pyplot as plt
import numpy as np

from microweaver.visualization.config import VisualizationConfig


def extract_values(metric_key: str, algorithms, report) -> list[float]:
    """Extract specified metric values from each algorithm, return NaN if missing."""
    values = []
    for algo in algorithms:
        raw = report[algo].get(metric_key)
        values.append(float(raw) if raw is not None else np.nan)
    return values


def compute_best_indices(values_matrix, higher_flags):
    """
    Return a list of indices of the optimal schemes for each metric (allow ties).
    values_matrix: List[List[float]], rows = metrics, columns = algorithms
    """
    best = []
    for vals, hb in zip(values_matrix, higher_flags):
        arr = np.array(vals, dtype=float)
        valid_mask = ~np.isnan(arr)
        if not valid_mask.any():
            best.append([])
            continue
        candidate = arr[valid_mask]
        best_val = candidate.max() if hb else candidate.min()
        winners = np.where((arr == best_val) & valid_mask)[0].tolist()
        best.append(winners)
    return best


def add_value_and_best_labels(rects, algo_idx, ax, better_indices, better_label_color, better_label_fontsize):
    for idx, rect in enumerate(rects):
        height = rect.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            f"{height:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
        )

        if algo_idx in better_indices[idx]:
            ax.annotate(
                "Best",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 15),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=better_label_fontsize,
                color=better_label_color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="green", alpha=0.7),
            )


def main(config: VisualizationConfig):
    with open(config.report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    metrics = [
        "Semantic Cohesion (SC)",
        "Service Coupling (SCP)",
        "Service Boundary Clarity (SBC)",
        "Service Size Balance (SSB)",
        "Structural Instability Index (SII)",
        "Internal Call Proportion (ICP)",
        "Modularity",
    ]

    higher_better = [
        True,   # SC: Higher is better
        False,  # SCP: Lower is better
        True,   # SBC: Higher is better
        False,  # SSB: Lower is better
        False,  # SII: Lower is better
        True,   # ICP: Higher is better
        True,   # Modularity: Higher is better
    ]

    algorithms = list(report.keys())

    metric_values = []
    for key in metrics:
        if "(" in key and ")" in key:
            core_key = key.split("(")[1].rstrip(")")
        else:
            core_key = key
        metric_values.append(extract_values(core_key, algorithms, report))

    better_indices = compute_best_indices(metric_values, higher_better)

    x = np.arange(len(metrics))
    algo_count = len(algorithms)
    width = min(0.8 / max(algo_count, 1), 0.22)
    total_width = width * algo_count

    colors = plt.cm.tab10.colors  # Use built-in colormap
    better_label_color = "darkgreen"
    better_label_fontsize = 12

    fig_width = max(18, int(10 + algo_count * 1.2))
    fig, ax = plt.subplots(figsize=(fig_width, 9))

    rect_groups = []
    for i, algo in enumerate(algorithms):
        offsets = x - total_width / 2 + width * i + width / 2
        heights = [metric_values[m_idx][i] for m_idx in range(len(metrics))]
        rects = ax.bar(offsets, heights, width, label=algo, color=colors[i % len(colors)])
        rect_groups.append(rects)

    ax.set_title("Microservice Partitioning Metrics Comparison", fontsize=22, pad=20)
    ax.set_ylabel("Metric Value", fontsize=16)
    ax.set_xlabel("Evaluation Metrics", fontsize=16)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=11)

    for algo_idx, rects in enumerate(rect_groups):
        add_value_and_best_labels(rects, algo_idx, ax, better_indices, better_label_color, better_label_fontsize)

    y_limits_source = [val for vals in metric_values for val in vals if not np.isnan(val)]
    y_max = max(y_limits_source) * 1.2 if y_limits_source else 1
    ax.set_ylim(0, y_max)

    for i, hb in enumerate(higher_better):
        direction = "↑ Higher is better" if hb else "↓ Lower is better"
        ax.text(
            i,
            y_max * 1.03,
            direction,
            ha="center",
            va="bottom",
            fontsize=12,
            color="darkred",
            fontweight="bold",
        )

    ax.legend(loc="upper right", fontsize=14)

    plt.tight_layout()

    os.makedirs(os.path.dirname(config.chart_save_path), exist_ok=True)
    plt.savefig(config.chart_save_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main(VisualizationConfig())