"""Auto-discovery of nodes via importlib/pkgutil."""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conductor.registry import NodeRegistry


def discover_nodes(package_name: str, registry: NodeRegistry) -> int:
    """Import every module in a package so the registrations in them run.

    Returns how many definitions the registry gained.
    """
    count_before = len(registry.definitions())
    package = importlib.import_module(package_name)

    for _importer, modname, _ispkg in pkgutil.walk_packages(
        package.__path__, prefix=package.__name__ + "."
    ):
        importlib.import_module(modname)

    return len(registry.definitions()) - count_before
