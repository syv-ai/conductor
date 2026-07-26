"""RegistryView: a read-only overlay serving dynamic NodeDefinitions."""

from typing import Annotated

import pytest
from conductor.registry import NodeRegistry
from conductor.registry.definition import NodeDefinition
from conductor.registry.view import RegistryView
from conductor.widgets import Output, Text


def _definition(node_id: str) -> NodeDefinition:
    return NodeDefinition(
        id=node_id, base_id=node_id, version=1, name=node_id, description=""
    )


class _Source:
    def __init__(self, known: dict[str, NodeDefinition]) -> None:
        self.known = known

    def get_definition(self, node_type: str) -> NodeDefinition | None:
        return self.known.get(node_type)


def _base_with_one_node() -> NodeRegistry:
    reg = NodeRegistry()

    @reg.node("static", version=1, name="Static", description="A static node")
    def static(v: Annotated[str, Text(label="V")] = "hi") -> Annotated[str, Output(label="Out")]:
        return v

    return reg


def test_static_registry_wins() -> None:
    base = _base_with_one_node()
    # A source that also claims "static@1" must not shadow the base node.
    shadow = _definition("static@1")
    view = RegistryView(base, [_Source({"static@1": shadow})])
    assert view.get("static@1") is base.get("static@1")
    assert view.get("static@1") is not shadow


def test_source_serves_unknown_types() -> None:
    base = _base_with_one_node()
    dyn = _definition("flow_version:abc")
    view = RegistryView(base, [_Source({"flow_version:abc": dyn})])
    assert view.get("flow_version:abc") is dyn


def test_contains_consults_sources() -> None:
    base = _base_with_one_node()
    dyn = _definition("flow_version:abc")
    view = RegistryView(base, [_Source({"flow_version:abc": dyn})])
    assert view.contains("static@1") is True  # from base
    assert view.contains("flow_version:abc") is True  # from source
    assert view.contains("nope") is False


def test_unknown_everywhere_is_none() -> None:
    base = _base_with_one_node()
    view = RegistryView(base, [_Source({})])
    assert view.get("flow_version:missing") is None
    assert view.contains("flow_version:missing") is False


def test_first_matching_source_wins() -> None:
    base = _base_with_one_node()
    first = _definition("flow_version:abc")
    second = _definition("flow_version:abc")
    view = RegistryView(
        base,
        [_Source({"flow_version:abc": first}), _Source({"flow_version:abc": second})],
    )
    assert view.get("flow_version:abc") is first


def test_delegates_everything_else_to_base() -> None:
    base = _base_with_one_node()
    view = RegistryView(base, [_Source({})])
    # Non-overlaid methods pass straight through to the base registry.
    assert view.all() == base.all()
    assert view.get_latest("static") is base.get_latest("static")
    assert view.is_deprecated("static@1") is False
    # Dynamic definitions are NOT enumerated — the overlay is point-lookup only.
    dyn_view = RegistryView(base, [_Source({"flow_version:abc": _definition("flow_version:abc")})])
    assert dyn_view.all() == base.all()


def test_missing_base_raises_attribute_error_not_recursion() -> None:
    # An instance without __init__ (unpickle / __new__) must not recurse
    # forever in __getattr__ when the delegate attribute is absent.
    view = RegistryView.__new__(RegistryView)
    with pytest.raises(AttributeError):
        _ = view.all()
