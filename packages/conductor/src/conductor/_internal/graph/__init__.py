"""The graph a host uses: ``type.py`` is ``Graph``, a compiled graph that can start a run; ``make.py`` is
``make_graph``, a graph record written from Python. This package stands on every other one; the record a graph is
compiled from is ``record/``.
"""

from conductor._internal.graph.make import make_graph
from conductor._internal.graph.type import Graph
