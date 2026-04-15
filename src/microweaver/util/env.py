import os

def get_env_numeric(env_name: str, default: float) -> float:
    """
    读取环境变量并转换为浮点数（处理空值/非数字情况）
    :param env_name: 环境变量名
    :param default: 默认值（转换失败时返回）
    :return: 浮点型数值
    """
    env_value = os.getenv(env_name)
    if env_value is None:
        return default
    try:
        return float(env_value)
    except ValueError:
        print(f"警告：环境变量{env_name}的值{env_value}不是有效数字，使用默认值{default}")
        return default

def get_env_boolean(env_name: str, default: bool = False) -> bool:
    env_value = os.getenv(env_name, "").strip().lower()
    true_values = {"true", "1", "yes", "on"}
    false_values = {"false", "0", "no", "off"}
    if env_value in true_values:
        return True
    elif env_value in false_values:
        return False
    else:
        print(f"警告：{env_name}={env_value} 无效，使用默认值{default}")
        return default


def get_env_boolean(env_name: str, default: bool = False) -> bool:
    env_value = os.getenv(env_name, "").strip().lower()
    true_values = {"true", "1", "yes", "on"}
    false_values = {"false", "0", "no", "off"}
    if env_value in true_values:
        return True
    elif env_value in false_values:
        return False
    else:
        print(f"警告：{env_name}={env_value} 无效，使用默认值{default}")
        return default