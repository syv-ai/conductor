"""``logic-not``: turn a yes into a no and a no into a yes."""

from typing import Annotated

from conductor import NodeRegistry, Param

from conductor_nodes._inputs import Flag

registry = NodeRegistry(category="logic")


@registry.node("logic-not", title="Not", description="Turns yes into no and no into yes.")
def negate(value: Flag) -> Annotated[bool, Param(title="Negated")]:
    """Turns yes into no and no into yes."""
    return not value
