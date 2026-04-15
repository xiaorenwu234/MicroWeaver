import os.path

import json

from microweaver.evaluation.config import EvaluateConfig
from microweaver.evaluation.evaluator import calculate_evaluation_metrics

import warnings

warnings.filterwarnings("ignore")


def main():
    results = calculate_evaluation_metrics()
    results = {
        key: value.__dict__ for key, value in results.items()
    }
    report_path = EvaluateConfig.evaluate_result_path
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    return results


if __name__ == '__main__':
    main()
