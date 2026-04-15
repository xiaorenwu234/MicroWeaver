from microweaver.config import BaseConfig
import os


class InputConfig(BaseConfig):
    static_json_path = os.path.join(BaseConfig.data_folder_path, "static.json")
    dynamic_json_path = os.path.join(BaseConfig.data_folder_path, "dynamic.json")
    class_info_json_path = os.path.join(BaseConfig.data_folder_path, "class_info.json")
    static_jar_path = os.path.join(BaseConfig.base_dir, "src/microweaver/input_builder/static_analyze/dependency-extractor/target/class-dependency-analyzer.jar")
    merge_json = False
    generate_description = False
