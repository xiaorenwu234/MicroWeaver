import os

from microweaver.evaluation.config import EvaluateConfig
from microweaver.evaluation.model import EvaluateResult
from microweaver.evaluation.semantic.comparative_evaluate import main as semantic_main
from microweaver.evaluation.structural.structured_evaluate import main as structured_main


def calculate_evaluation_metrics():
    evaluation_config = EvaluateConfig()
    structured_result = structured_main()
    semantic_result = semantic_main(evaluation_config.partition_result_folder_path, evaluation_config.repeat_times)
    report = {}
    count = 0
    for folder in os.listdir(evaluation_config.partition_result_folder_path):
        folder_path = os.path.join(evaluation_config.partition_result_folder_path, folder)
        if os.path.isdir(folder_path):
            report[folder] = EvaluateResult(
                SC=semantic_result["SC"][count]/100,
                SCP=semantic_result["SCP"][count]/100,
                SBC=semantic_result["SBC"][count]/100,
                judge_result=semantic_result["judge_result"][count],
                SSB=structured_result[count]["SSB"],
                SII=structured_result[count]["SII"],
                ICP=structured_result[count]["ICP"],
                Modularity=structured_result[count]["Modularity"],
            )
            count+=1

    return report