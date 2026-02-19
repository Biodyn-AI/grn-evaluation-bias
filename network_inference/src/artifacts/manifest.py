from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable


def _stringify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _stringify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify(v) for v in value]
    return value


def write_manifest(
    path: str | Path,
    config: Dict[str, Any],
    edges_count: int | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": _stringify(config),
    }
    if edges_count is not None:
        payload["edges_count"] = edges_count
    if extra:
        payload["extra"] = _stringify(extra)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
