import json
import os

from microweaver.input_builder.config import InputConfig
from microweaver.util.file_op import load_json, save_json


def parse_class_name(qualified_name: str) -> str:
    class_name = qualified_name
    if '(' in qualified_name:
        method_name = qualified_name.split('(')[0]
        class_name = '.'.join(method_name.split('.')[:-1])
    return class_name


def merge_class_info(dynamic_json_path: str, static_json_path: str, merged_path: str):
    static_data = load_json(static_json_path)
    if not os.path.exists(dynamic_json_path):
        pass

    else:
        dep_data = load_json(dynamic_json_path)

        name_id_map = {item["qualifiedName"]: item["id"] for item in static_data}
        dep_id_name_map = {item["id"]: item["qualifiedName"] for item in dep_data}

        for item in dep_data:
            qualified_name = item["qualifiedName"]
            class_name = parse_class_name(qualified_name)
            if class_name in name_id_map:
                id = name_id_map[class_name]
                for dependency in item["dependencies"]:
                    dep_name = dep_id_name_map.get(dependency)
                    if not dep_name:
                        continue
                    dep_class_name = parse_class_name(dep_name)
                    dep_id = name_id_map.get(dep_class_name)
                    if dep_id is not None and dep_id not in static_data[id]["dependencies"]:
                        static_data[id]["dependencies"].append(dep_id)
                        static_data[id]["edge_types"].append("call")

    save_json(static_data, merged_path)


def main(config: InputConfig):
    merge_class_info(
        dynamic_json_path=config.dynamic_json_path,
        static_json_path=config.static_json_path,
        merged_path=config.data_path
    )
