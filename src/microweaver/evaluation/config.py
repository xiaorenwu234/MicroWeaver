import os

from microweaver.config import BaseConfig

class EvaluateConfig(BaseConfig):
    """Evaluation configuration"""
    repeat_times: int = 3
    evaluate_result_path = os.path.join(BaseConfig.evaluate_result_folder_path, "report.json")