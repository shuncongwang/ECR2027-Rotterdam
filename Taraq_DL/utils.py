from pathlib import Path
import yaml


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError("Config file not found: %s" % config_path.resolve())

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise TypeError("config.yaml must be parsed as a YAML mapping/dictionary.")

    return cfg


def project_path(config, key):
    project_dir = Path(config["project"]["dir"])
    path_value = Path(config["paths"][key])
    if path_value.is_absolute():
        return path_value
    return project_dir / path_value
