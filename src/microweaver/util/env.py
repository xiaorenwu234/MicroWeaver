import os

def get_env_numeric(env_name: str, default: float) -> float:
    """
    Read environment variable and convert to float (handle empty/non-numeric cases)
    :param env_name: Environment variable name
    :param default: Default value (returned when conversion fails)
    :return: Float value
    """
    env_value = os.getenv(env_name)
    if env_value is None:
        return default
    try:
        return float(env_value)
    except ValueError:
        print(f"Warning: Environment variable {env_name} value {env_value} is not a valid number, using default {default}")
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
        print(f"Warning: {env_name}={env_value} is invalid, using default {default}")
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
        print(f"Warning: {env_name}={env_value} is invalid, using default {default}")
        return default