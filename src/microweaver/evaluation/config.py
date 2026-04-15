import os

from microweaver.config import BaseConfig

class EvaluateConfig(BaseConfig):
    """评估配置"""
    repeat_times: int = 3
    evaluate_result_path = os.path.join(BaseConfig.evaluate_result_folder_path, "report.json")