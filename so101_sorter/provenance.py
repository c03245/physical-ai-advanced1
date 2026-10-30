"""実行条件とソースの指紋を記録する（モデルやシミュレータは起動しない）。"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def execution_context() -> dict:
    files = [ROOT / name for name in ("main.py", "pyproject.toml", "uv.lock", ".python-version")]
    for folder, pattern in (("so101_sorter", "*.py"), ("scripts", "*.py"), ("tools", "*.py"), ("assets", "*")):
        files.extend(p for p in (ROOT / folder).rglob(pattern) if p.is_file())
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0")
        digest.update(path.read_bytes())
    versions = {}
    for name in ("genesis-world", "torch", "transformers", "ollama", "anthropic", "numpy", "pydantic"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {
        "source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "packages": versions,
        "environment": {
            k: v
            for k, v in sorted(os.environ.items())
            if k
            in (
                "SO101_GRIP_TORQUE",
                "SO101_PEDESTAL_H",
                "SO101_CONTACT_TIMECONST",
                "SO101_DECOMP",
                "SO101_VIEWER",
                "OLLAMA_MODEL",
                "OLLAMA_THINK",
                "PLANNER_PROMPT_V2",
            )
        },
    }


def run_metadata(options: dict) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        **execution_context(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in options.items()},
    }
