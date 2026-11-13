"""The graph a host uses: ``type.py`` is ``Graph``, a compiled graph that can start a run; ``make.py`` is
``make_graph``, a graph record written from Python. This package stands on every other one; the record a graph is
compiled from is ``record/``.
"""

from .make import make_graph
from .type import Graph, read_json

__all__ = ["Graph", "make_graph", "read_json"]
