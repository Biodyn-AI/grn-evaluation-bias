from __future__ import annotations

from typing import Any, Dict

import torch


def move_to_device(batch: Dict[str, Any], device: str):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved
