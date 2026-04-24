"""Subprocess-call marker node.

See :mod:`conductor.compound.subprocess` for the runtime. Bind a
``SubprocessRegistry`` when compiling:

    from conductor import compile
    from conductor.compound.subprocess import SUBPROCESS, SubprocessRegistry
    from conductor_nodes import subprocess

    subprocess.register(registry)
    sub_registry = SubprocessRegistry()
    sub_registry.register(other_flow)
    compiled = compile(
        flow=caller_flow, registry=registry,
        compound_types=[SUBPROCESS],
        subprocess_registry=sub_registry,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from conductor.types import NodeCategory
from conductor.widgets import Number, Output, SchemaBuilder, Text

if TYPE_CHECKING:
    from conductor import NodeRegistry


def register(registry: "NodeRegistry") -> None:
    """Register the subprocess-call marker node."""

    @registry.node(
        "subprocess-call", version=1, name="Subprocess",
        description=(
            "Calls another flow by id and version. The sub-flow runs "
            "synchronously; its results are returned as this node's output."
        ),
        category=NodeCategory.CONTROL,
    )
    def subprocess_call(
        flow_id: Annotated[str, Text(label="Flow id")] = "",
        flow_version: Annotated[int, Number(label="Flow version", integer_only=True)] = 1,
        inputs: Annotated[dict, SchemaBuilder(label="Input mapping")] = None,  # type: ignore[assignment]
    ) -> Annotated[Any, Output(label="Sub-flow results")]:
        raise NotImplementedError("Handled by the SUBPROCESS compound node")
