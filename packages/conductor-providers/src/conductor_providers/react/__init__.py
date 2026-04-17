"""ReactFlow provider — translates conductor graphs to/from ReactFlow JSON.

Wire format produced by ``graph_to_react``:

    {
      "nodes": [
        {
          "id": "n1",
          "type": "text-uppercase@1",
          "position": {"x": 0, "y": 0},
          "data": {
            "data": {...},            # conductor's static-data dict
            "produces": {...},        # present only if non-empty
            "consumes": {"text": ["n0", "result"]},   # tuples as lists
          }
        }
      ],
      "edges": [
        {
          "id": "e1",
          "source": "n0",
          "target": "n1",
          "sourceHandle": "result",
          "targetHandle": "text"
        }
      ]
    }

Notes:
- ReactFlow uses camelCase for ``sourceHandle`` / ``targetHandle``.
- Conductor stores ``consumes`` values as tuples; JSON has no tuples, so
  the wire form uses lists and ``react_to_graph`` restores tuples.
- Positions are ReactFlow-only; if they aren't provided on the inbound
  side, conductor doesn't care. ``graph_to_react`` fills them in with a
  simple topological layout so nodes don't stack at the origin.
- The palette (node-type metadata for a sidebar) is available via
  ``palette_from_registry``, which wraps ``conductor.registry.schema``.
"""

from conductor_providers.react.graph import graph_to_react, react_to_graph
from conductor_providers.react.layout import topological_positions
from conductor_providers.react.schema import palette_from_registry

__all__ = [
    "graph_to_react",
    "react_to_graph",
    "palette_from_registry",
    "topological_positions",
]
