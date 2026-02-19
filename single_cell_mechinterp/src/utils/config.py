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
