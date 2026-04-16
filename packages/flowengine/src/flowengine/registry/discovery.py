"""Auto-discovery of nodes via importlib/pkgutil."""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowengine.registry import NodeRegistry


def discover_nodes(package_name: str, registry: NodeRegistry) -> int:
    """Import all modules in a package to trigger @node decorators.

    Returns the number of newly registered nodes.
    """
    count_before = len(registry.all())
    package = importlib.import_module(package_name)

    for _importer, modname, _ispkg in pkgutil.walk_packages(
        package.__path__, prefix=package.__name__ + "."
    ):
        importlib.import_module(modname)

    return len(registry.all()) - count_before
