"""Portable default paths for public-data workflows."""

from __future__ import annotations

import os
from pathlib import Path


SOURCE_ROOT_ENV = "PANCANCER_SOURCE_ROOT"


def source_project_root(repository_root: Path) -> Path:
    """Return the configured legacy-input root without a machine-specific path."""
    configured = os.environ.get(SOURCE_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return (repository_root / "data/external/rl-genrisk-main").resolve()
