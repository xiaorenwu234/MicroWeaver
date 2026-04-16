"""
Script for merging class dependencies, inheritance, and JavaDoc information
Usage: python static_dependencies_mapper.py <target_path> [target_package]
Note: target_package is optional, if empty, package name parameter is not passed
"""

import json
import sys
import subprocess
import os

from microweaver.input_builder.config import InputConfig


def run_java_analyzer(target_path, jar_path, class_info_path):
    if not os.path.exists(jar_path):
        print(f"Error: JAR file not found {jar_path}")
        sys.exit(1)
    
    cmd = ["java", "-jar", jar_path, target_path, class_info_path]
    print(f"Executing: java -jar {jar_path} {target_path} {class_info_path}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print("Java analyzer executed successfully")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: Java analyzer execution failed")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error message: {e.stderr}")
        sys.exit(1)


def parse_json_file(config: InputConfig):
    """Parse JSON file, convert classInfo.json format to data.json format"""
    with open(config.class_info_json_path, "r", encoding="utf-8") as f:
        classInfo = json.load(f)
    
    name_to_id = {}
    for idx, (qualified_name, info) in enumerate(classInfo.items()):
        name_to_id[qualified_name] = idx
    
    result = []
    for idx, (qualified_name, info) in enumerate(classInfo.items()):
        # Get extendsAndImplements and dependencies
        extends_and_implements = info.get("extendsAndImplements", [])
        dependencies = info.get("dependencies", [])
        
        # Process extendsAndImplements first (edge_type is "extends")
        dep_ids = []
        edge_types = []
        
        # Record processed dependencies (avoid duplicates)
        processed_deps = set()
        
        # Process extendsAndImplements
        for dep_name in extends_and_implements:
            if dep_name in name_to_id and dep_name not in processed_deps:
                dep_ids.append(name_to_id[dep_name])
                edge_types.append("extends")
                processed_deps.add(dep_name)
        
        # Process dependencies (exclude those already in extendsAndImplements)
        for dep_name in dependencies:
            if dep_name in name_to_id and dep_name not in processed_deps:
                dep_ids.append(name_to_id[dep_name])
                edge_types.append("call")
                processed_deps.add(dep_name)
        
        # Build result object
        item = {
            "id": idx,
            "name": qualified_name.split(".")[-1],
            "qualifiedName": qualified_name,
            "description": "",
            "methods": info.get("methods", []),
            "dependencies": dep_ids,
            "edge_types": edge_types,
            "javaDoc": info.get("javaDoc", ""),
            "filePath": info.get("filePath", ""),
            "typeKind": info.get("typeKind", "")
        }
        result.append(item)
    
    output_file = config.static_json_path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"Conversion completed! Total {len(result)} classes")
    print(f"Result saved to {output_file}")


def main(config: InputConfig):
    print("=" * 60)
    print(f"Target path: {config.resource_path}")
    print()
    
    run_java_analyzer(config.resource_path, config.static_jar_path, config.class_info_json_path)
    print()
    
    parse_json_file(config)
    
    print("=" * 60)
    print("Completed!")
    print("=" * 60)
