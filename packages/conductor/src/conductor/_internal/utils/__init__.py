"""Small helpers that know nothing about graphs, nodes or runs.

Every other package may use them, so this one imports nothing from the rest of conductor. ``text.py`` has the
bits of English that conductor's messages are written with.
"""

from .text import comma_separated, plural, qualified_name

__all__ = ["comma_separated", "plural", "qualified_name"]
