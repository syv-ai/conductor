"""``decision``: send a value one of two ways.

The node gives ``value`` on ``if_true`` or ``if_false`` by ``when``, and
``SKIPPED`` on the other, so what is wired to the way not taken doesn't run on
that row. A choice between more than two ways is a chain of decisions.

``when`` is a yes or no, not an expression to parse: other nodes give one
(``text-contains``, ``regex-match``, ``logic-not``), so the condition is wired
in rather than written in a language this node would have to own.

``value`` takes any value, since the node passes it on without reading it.
The ``compute_outputs`` hook gives both outputs the type that arrives on it.
"""

from typing import Annotated, Any

from conductor import SKIPPED, NodeRegistry, Param

from conductor_nodes._inputs import Flag

registry = NodeRegistry(category="control")


def _either_way(value: type) -> dict[str, Any]:
    """Both outputs of the type arriving on ``value``, ``Any`` until it is wired; one holds a value on each row."""
    return {
        "if_true": Annotated[value, Param(title="If true", exclusive_group="when")],
        "if_false": Annotated[value, Param(title="If false", exclusive_group="when")],
    }


@registry.node("decision", compute_outputs=_either_way)
def decision(value: Any, when: Flag = True) -> dict[str, Any]:
    """Sends a value one of two ways by `when`."""
    if when:
        return {"if_true": value, "if_false": SKIPPED}
    return {"if_true": SKIPPED, "if_false": value}
