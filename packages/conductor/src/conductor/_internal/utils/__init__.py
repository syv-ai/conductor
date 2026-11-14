"""Small helpers with no idea of a graph, a node or a run; nothing here imports anything of conductor's.

``text.py`` is the English that messages and reprs are written with: ``comma_separated``, ``plural`` and
``qualified_name``.
"""

from .text import comma_separated, plural, qualified_name

__all__ = ["comma_separated", "plural", "qualified_name"]
