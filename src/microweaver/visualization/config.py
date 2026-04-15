import os

from microweaver.config import BaseConfig


class VisualizationConfig(BaseConfig):
    report_path = os.path.join(BaseConfig.evaluate_result_folder_path, "report.json")
    chart_save_path = os.path.join(BaseConfig.visualize_result_folder_path, "evaluate_chart.png")
    table_save_path = os.path.join(BaseConfig.visualize_result_folder_path, "evaluate_table.png")
