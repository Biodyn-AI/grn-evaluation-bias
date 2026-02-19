from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path_value: str | Path, base_dir: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()


def resolve_paths(config: Dict[str, Any], config_path: str | Path) -> Dict[str, Any]:
    base_dir = Path(config_path).resolve().parent
    paths = config.get("paths", {})
    resolved = {}
    for key, value in paths.items():
        if value is None:
            resolved[key] = None
        else:
            resolved[key] = resolve_path(value, base_dir)
    config["paths"] = resolved
    config["_config_dir"] = base_dir
    return config
