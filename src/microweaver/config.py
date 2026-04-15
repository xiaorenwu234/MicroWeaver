import os

from microweaver.util.env import get_env_numeric


class BaseConfig:
    base_dir = os.getenv("BASE_DIR", r"G:\monolithic-to-microservice")
    app_name = str(os.getenv("APP_NAME", "daytrader"))
    data_path = os.path.join(base_dir, "data", "inputs", app_name, "data.json")
    data_folder_path = os.path.dirname(data_path)
    resource_path = os.path.join(base_dir, "data", "resource_datasets", app_name)
    partition_result_folder_path = os.path.join(base_dir, "results", "splits", app_name)
    result_path = os.path.join(partition_result_folder_path, "microweaver", "result.json")
    evaluate_result_folder_path = os.path.join(base_dir, "results", "reports", app_name)
    visualize_result_folder_path = os.path.join(base_dir, "results", "viz", app_name)
    num_clusters = int(get_env_numeric("NUM_CLUSTERS", 5))