"""RegistryView: a read-only overlay serving host-supplied node classes."""

from typing import Annotated

import pytest
from conductor.dtype import DType
from conductor.node import NodeDefinition
from conductor.registry import NodeRegistry
from conductor.registry.view import RegistryView
from conductor.returns import Result
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "registry-view-test-text"
    title = "Text"


Out = Annotated[Txt, Result(title="Out")]


def _definition(node_id: str) -> type[NodeDefinition]:
    """A fresh node class under ``node_id``, as a source would build one."""

    class Dynamic(NodeDefinition):
        id = node_id
        title = node_id
        description = "d"
        category = "test"

        def run(self, v: Annotated[Txt, Textarea(title="V")] = Txt("")) -> Out:
            return v

    return Dynamic


class _Source:
    def __init__(self, known: dict[str, type[NodeDefinition]]) -> None:
        self.known = known

    def get_definition(self, node_type: str) -> type[NodeDefinition] | None:
        return self.known.get(node_type)


class Static(NodeDefinition):
    id = "static"
    title = "Static"
    description = "A static node"
    category = "test"

    def run(self, v: Annotated[Txt, Textarea(title="V")] = Txt("hi")) -> Out:
        return v


def _base_with_one_node() -> NodeRegistry:
    reg = NodeRegistry()
    reg.register(Static)
    return reg


def test_static_registry_wins() -> None:
    base = _base_with_one_node()
    # A source that also claims "static" must not shadow the base node.
    shadow = _definition("static")
    view = RegistryView(base, [_Source({"static": shadow})])
    assert view.get("static") is base.get("static")
    assert view.get("static") is not shadow


def test_source_serves_unknown_types() -> None:
    base = _base_with_one_node()
    dyn = _definition("flow_version:abc")
    view = RegistryView(base, [_Source({"flow_version:abc": dyn})])
    assert view.get("flow_version:abc") is dyn


def test_contains_consults_sources() -> None:
    base = _base_with_one_node()
    dyn = _definition("flow_version:abc")
    view = RegistryView(base, [_Source({"flow_version:abc": dyn})])
    assert view.contains("static") is True  # from base
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
    assert view.definitions() == base.definitions()
    # Dynamic definitions are NOT enumerated — the overlay is point-lookup only.
    dyn_view = RegistryView(base, [_Source({"flow_version:abc": _definition("flow_version:abc")})])
    assert dyn_view.definitions() == base.definitions()


def test_missing_base_raises_attribute_error_not_recursion() -> None:
    # An instance without __init__ (unpickle / __new__) must not recurse
    # forever in __getattr__ when the delegate attribute is absent.
    view = RegistryView.__new__(RegistryView)
    with pytest.raises(AttributeError):
        _ = view.definitions()
