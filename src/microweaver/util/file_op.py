import json
import os

def load_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(json_data, json_path: str):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)
