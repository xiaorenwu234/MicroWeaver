import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

from microweaver.visualization.config import VisualizationConfig


def main(config: VisualizationConfig):
    num_bold = FontProperties(family="DejaVu Sans", weight="bold")
    num_normal = FontProperties(family="DejaVu Sans", weight="regular")

    plt.rcParams["axes.unicode_minus"] = False

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
        True,  # SC: Higher is better
        False,  # SCP: Lower is better
        True,  # SBC: Higher is better
        False,  # SSB: Lower is better
        False,  # SII: Lower is better
        True,  # ICP: Higher is better
        True,  # Modularity: Higher is better
    ]
    metric_keys = [m.split("(")[1].rstrip(")") if "(" in m else m for m in metrics]

    algorithms = list(report.keys())

    algo_values = {}
    for algo in algorithms:
        values = []
        for key in metric_keys:
            raw = report[algo].get(key)
            values.append(float(raw) if raw is not None else np.nan)
        algo_values[algo] = values

    better_indices = compute_best_indices(algo_values, higher_better)

    fig_width = max(22, 12 + len(algorithms) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    ax.axis("off")

    col_labels = [f"{m}\n({'↑' if hb else '↓'})" for m, hb in zip(metrics, higher_better)]
    row_labels = algorithms

    cell_text = [
        [f"{v:.4f}" if not np.isnan(v) else "-" for v in algo_values[algo]]
        for algo in algorithms
    ]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        rowLabels=row_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
        edges="closed",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8)
        cell.set_facecolor("white")

        text = cell.get_text()

        if row == 0:
            text.set_fontweight("bold")
            text.set_fontsize(12)

        elif col == -1:
            text.set_fontweight("bold")
            text.set_fontsize(12)

        elif row >= 1 and col >= 0:
            idx_list = better_indices[col]
            if (row - 1) in idx_list:
                text.set_fontproperties(num_bold)
            else:
                text.set_fontproperties(num_normal)
            text.set_fontsize(11)

    plt.title("Microservice Decomposition Metrics Comparison", fontsize=15, pad=20, fontweight="bold")

    plt.figtext(
        0.5,
        0.02,
        "Note: ↑ indicates higher is better, ↓ indicates lower is better; "
        "Bold values represent the best results for each metric",
        ha="center",
        fontsize=10,
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    os.makedirs(os.path.dirname(config.table_save_path), exist_ok=True)
    plt.savefig(config.table_save_path, dpi=300, bbox_inches="tight")


def compute_best_indices(values_per_algo, higher_flags):
    """
    values_per_algo: Dict[str, List[float]]
    returns winners: List[List[int]] - indices of best algorithms per metric
    """
    algo_list = list(values_per_algo.keys())
    best_list = []
    for col_idx, hb in enumerate(higher_flags):
        col_vals = np.array([values_per_algo[a][col_idx] for a in algo_list], dtype=float)
        valid = ~np.isnan(col_vals)
        if not valid.any():
            best_list.append([])
            continue
        best_val = col_vals[valid].max() if hb else col_vals[valid].min()
        winners = np.where((col_vals == best_val) & valid)[0].tolist()
        best_list.append(winners)
    return best_list


if __name__ == "__main__":
    config = VisualizationConfig()
    main(config)