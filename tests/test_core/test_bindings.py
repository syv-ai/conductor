import pytest
from conductor.graph.binding import Ref, Sources, Static, static_values


def test_a_wire_carries_ordered_refs():
    wire = Sources(refs=(Ref("a", "result"), Ref("b", "result")))

    assert [r.node_id for r in wire.refs] == ["a", "b"]


def test_wire_order_is_operand_order():
    first = Sources(refs=(Ref("a", "r"), Ref("b", "r")))
    second = Sources(refs=(Ref("b", "r"), Ref("a", "r")))
    assert first != second


def test_there_are_exactly_two_variants():
    """No Port (a caller's value replaces a Static for one run), no Guard
    (a branch is an output), no per-ref enable flag."""
    import conductor.graph.binding as bindings

    assert not hasattr(bindings, "Port")
    assert not hasattr(bindings, "Guard")
    assert not hasattr(Sources(refs=()), "when")


def test_absence_means_the_declared_default_applies():
    """One state, not two — there is nothing to dismiss."""
    bindings = {"b": Static(value=3)}

    assert "a" not in bindings


def test_static_values_extracts_only_static_bindings():
    bindings = {
        "text": Static(value="hi"),
        "number": Static(value=3),
        "source": Sources(refs=(Ref("a", "result"),)),
    }
    assert static_values(bindings) == {"text": "hi", "number": 3}


def test_bindings_are_frozen():
    with pytest.raises(Exception):
        Static(value=1).value = 2
