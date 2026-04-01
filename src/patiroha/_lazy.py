"""Lazy import helper for optional dependencies."""

from __future__ import annotations

import importlib
from types import ModuleType


def require(package: str, extra: str) -> ModuleType:
    """Import a package or raise ImportError with install hint.

    Args:
        package: The package name to import.
        extra: The pip extra name (e.g. "embeddings", "clustering").

    Returns:
        The imported module.

    Raises:
        ImportError: If the package is not installed.
    """
    try:
        return importlib.import_module(package)
    except ImportError:
        raise ImportError(
            f"{package} is required for this feature. Install it with: pip install patiroha[{extra}]"
        ) from None
