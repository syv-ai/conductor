"""ReactFlow provider — translates a conductor ``Flow`` to/from ReactFlow JSON.

Wire format produced by ``graph_to_react``:

    {
      "nodes": [
        {
          "id": "n1",
          "type": "text-uppercase",
          "position": {"x": 0, "y": 0},
          "data": {                    # the placement record, dumped
            "id": "n1",
            "type": "text-uppercase",
            "version": 1,
            "bindings": {"text": {"refs": ["n0.result"]}},
            "locked": [], "title": "", "description": "", "fields": {}
          }
        }
      ],
      "edges": [                       # one cable per ref, derived
        {
          "id": "n0.result->n1.text",
          "source": "n0",
          "target": "n1",
          "sourceHandle": "result",
          "targetHandle": "text"
        }
      ]
    }

Notes:
- ReactFlow uses camelCase for ``sourceHandle`` / ``targetHandle``.
- Positions are ReactFlow's; ``graph_to_react`` reads one from the
  placement's ``display`` and lays out the rest, ``react_to_graph`` writes
  the canvas's back into ``display``.
- The palette (node-type metadata for a sidebar) is available via
  ``palette_from_registry``.
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
