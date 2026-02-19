from __future__ import annotations

from pathlib import Path
import sys


def ensure_mechinterp_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    mechinterp_path = repo_root / "single_cell_mechinterp"
    if mechinterp_path.exists() and str(mechinterp_path) not in sys.path:
        # Make single_cell_mechinterp/src importable as top-level "src".
        sys.path.insert(0, str(mechinterp_path))
    return mechinterp_path
