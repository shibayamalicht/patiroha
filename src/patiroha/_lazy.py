# Copyright 2026 しばやま (shibayamalicht)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
