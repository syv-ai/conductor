"""Two failure families, and what people are told.

A node's failure is internal unless the outside world caused it. An
``ExternalFailure`` — raised by the node itself, or a foreign exception the
version's ``Policy(retry_on=...)`` names — retries under the policy; any
other failure runs once. A cause carries a generic message per code unless
the node wrote one for people; a foreign exception's text goes to the
log and never reaches an event.
"""

from typing import Annotated

import pytest
from conductor import (
    CompiledGraph,
    DType,
    ExternalFailure,
    Graph,
    GraphNode,
    NodeDefinition,
    NodeError,
    NodeRegistry,
    Param,
    Policy,
    Result,
    Static,
)
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "failures-test-text"
    display_title = "Text"


In = Annotated[Txt, Param(title="In", widget=Textarea())]
Out = Annotated[Txt, Result(title="Out")]


def _compiled(node_cls: type[NodeDefinition], *more: type[NodeDefinition]) -> CompiledGraph:
    """One node of ``node_cls`` as ``n1``, its text typed in."""
    reg = NodeRegistry()
    for cls in (node_cls, *more):
        reg = reg.extended_with([cls])
    return CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="n1", type=node_cls.id, version=1, bindings={"text": Static("x")})]), reg
    )


def _error(events: list[dict]) -> dict:
    return next(e for e in events if e.type == "node_failed")


# -- the families -----------------------------------------------------------------


# -- the foreign exception's text -----------------------------------------------


def test_node_error_has_no_retryable_flag():
    """The family is the class: an ``ExternalFailure`` retries, nothing else does."""
    assert not hasattr(NodeError, "retryable")
    assert issubclass(ExternalFailure, NodeError)


# -- the policy's bounds ------------------------------------------------------------


@pytest.mark.parametrize("raised", [{"concurrency": 0}, {"retries": -1}, {"delay": -0.5}, {"timeout": 0}])
def test_a_policy_out_of_bounds_is_refused(raised):
    with pytest.raises(ValueError):
        Policy(**raised)


def test_a_policy_keeps_retry_on_out_of_what_a_palette_reads():
    """Which exceptions a node retries on is the engine's business; a palette
    dumps the policy to JSON, and exception classes have no JSON form."""
    policy = Policy(retries=2, retry_on=(ConnectionError, TimeoutError))

    assert policy.retry_on == (ConnectionError, TimeoutError)
    assert policy.model_dump(mode="json") == {"retries": 2, "retry_delay": 1.0, "attempt_timeout": None, "concurrency": 8, "reuse_when_inputs_match": True}
    assert "retry_on" not in Policy.model_json_schema()["properties"]


# -- retries and rows ------------------------------------------------------------------


class Shape(DType, dict):
    id = "failures-shape"
    display_title = "Shape"


class Square(Shape):
    """A narrower type whose validator lets any ``Shape`` through as it is —
    the hole a host type can have: in memory it accepts what it would never
    build from JSON."""

    id = "failures-square"
    display_title = "Square"

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema

        return core_schema.no_info_plain_validator_function(
            lambda value: value if isinstance(value, Shape) else cls(value)
        )


