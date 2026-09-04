"""A flow file is the record, dumped."""

from conductor.flow_format import flow_to_dict, flow_to_yaml, load_flow, yaml_to_flow
from conductor.graph.binding import Sources, Static
from conductor.graph.model import FieldContent, Flow, GraphNode
from conductor.ref import Ref


def _flow():
    return Flow(
        nodes=[
            GraphNode(id="a", type="echo", version=1, bindings={"x": Static(value="hi")}, locked=("x",), title="A", fields={"x": FieldContent(title="X")}),
            GraphNode(id="b", type="echo", version=1, bindings={"x": Sources(refs=(Ref("a", "result"),))}, display={"x": 1}),
        ],
    )


def test_dict_roundtrip():
    assert load_flow(flow_to_dict(_flow())) == _flow()


def test_yaml_roundtrip():
    assert yaml_to_flow(flow_to_yaml(_flow())) == _flow()


def test_the_dict_is_the_record():
    """And a ref stores as the address, the one form it has anywhere: the
    same string a flow-level `Input` is named by, so the stored graph and
    the derived interface cannot spell one wire two ways."""
    data = flow_to_dict(_flow())

    assert data["nodes"][1]["bindings"]["x"] == {"refs": ["a.result"]}
    assert data["nodes"][0]["locked"] == ["x"]
    assert "inputs" not in data and "outputs" not in data and "edges" not in data


def test_a_node_without_a_type_is_refused():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        load_flow({"nodes": [{"id": "a", "version": 1}]})
