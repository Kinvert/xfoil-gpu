"""Emit a reproducible CPU runtime manifest for the evaluator."""

from __future__ import annotations

import json
import shutil
import platform
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple


def _resolve_executable(path_or_name: str, repo_root: Path) -> Tuple[str, bool]:
    """Resolve an executable candidate and report availability without mutating env."""
    if not path_or_name:
        return "", False

    candidate = path_or_name
    if os.path.sep in candidate and not os.path.isabs(candidate):
        candidate = str(repo_root / candidate)

    if os.path.isabs(candidate) and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate, True

    resolved = shutil.which(candidate)
    return (resolved or "", bool(resolved))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    configured = os.environ.get("XFOIL_EXE", "")
    configured_resolved, configured_available = _resolve_executable(configured, repo_root)
    which_xfoil = shutil.which("xfoil")
    xfoil_available = bool(which_xfoil) or configured_available

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "xfoil_executable_configured": os.environ.get("XFOIL_EXE", ""),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "xfoil": {
            "binary": configured_resolved if configured_resolved else (which_xfoil or ""),
            "configured_binary": configured_resolved,
            "configured_available": configured_available,
            "available": xfoil_available,
        },
        "paths": {
            "cwd": str(Path.cwd()),
            "repo_root": str(repo_root),
        },
        "env": {
            "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
            "CUDA_PATH": os.environ.get("CUDA_PATH", ""),
        },
    }
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
