"""
合并类依赖、继承和JavaDoc信息的脚本
使用方法: python static_dependencies_mapper.py <目标路径> [目标包名]
注意: 目标包名为可选参数，如果为空则不传递包名参数
"""

import json
import sys
import subprocess
import os

from microweaver.input_builder.config import InputConfig


def run_java_analyzer(target_path, jar_path, class_info_path):
    if not os.path.exists(jar_path):
        print(f"错误: 找不到JAR文件 {jar_path}")
        sys.exit(1)
    
    cmd = ["java", "-jar", jar_path, target_path, class_info_path]
    print(f"正在执行: java -jar {jar_path} {target_path} {class_info_path}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print("Java分析器执行成功")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"错误: Java分析器执行失败")
        print(f"返回码: {e.returncode}")
        if e.stdout:
            print(f"输出: {e.stdout}")
        if e.stderr:
            print(f"错误信息: {e.stderr}")
        sys.exit(1)


def parse_json_file(config: InputConfig):
    """解析JSON文件，将classInfo.json格式转换为data.json格式"""
    with open(config.class_info_json_path, "r", encoding="utf-8") as f:
        classInfo = json.load(f)
    
    name_to_id = {}
    for idx, (qualified_name, info) in enumerate(classInfo.items()):
        name_to_id[qualified_name] = idx
    
    result = []
    for idx, (qualified_name, info) in enumerate(classInfo.items()):
        # 获取 extendsAndImplements 和 dependencies
        extends_and_implements = info.get("extendsAndImplements", [])
        dependencies = info.get("dependencies", [])
        
        # 先处理 extendsAndImplements（edge_type 为 "extends"）
        dep_ids = []
        edge_types = []
        
        # 记录已经处理过的依赖（避免重复）
        processed_deps = set()
        
        # 处理 extendsAndImplements
        for dep_name in extends_and_implements:
            if dep_name in name_to_id and dep_name not in processed_deps:
                dep_ids.append(name_to_id[dep_name])
                edge_types.append("extends")
                processed_deps.add(dep_name)
        
        # 处理 dependencies（排除已经在 extendsAndImplements 中的）
        for dep_name in dependencies:
            if dep_name in name_to_id and dep_name not in processed_deps:
                dep_ids.append(name_to_id[dep_name])
                edge_types.append("call")
                processed_deps.add(dep_name)
        
        # 构建结果对象
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
    
    print(f"转换完成！共 {len(result)} 个类")
    print(f"结果已保存到 {output_file}")


def main(config: InputConfig):
    print("=" * 60)
    print(f"目标路径: {config.resource_path}")
    print()
    
    run_java_analyzer(config.resource_path, config.static_jar_path, config.class_info_json_path)
    print()
    
    parse_json_file(config)
    
    print("=" * 60)
    print("完成!")
    print("=" * 60)
