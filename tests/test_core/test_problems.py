"""What is wrong with a graph, as data."""

import pytest
from conductor.graph.problem import Problem


def test_a_problem_names_a_code_a_message_and_a_place():
    p = Problem(
        code="unknown_input",
        message="The node has no such field.",
        fatal=True,
        node_id="letter",
        field="template",
    )

    assert p.code == "unknown_input"
    assert p.node_id == "letter"
    assert p.field == "template"


def test_a_problem_about_a_whole_node_names_no_field():
    p = Problem(code="cycle", message="The flow loops back on itself.", fatal=True, node_id="letter")

    assert p.node_id == "letter"
    assert p.field is None


def test_every_problem_is_about_a_node():
    """There is no graph-level problem: a graph is only ever wrong
    somewhere, so the anchor has two states and not three."""
    with pytest.raises(TypeError):
        Problem(code="empty", message="The flow is empty.", fatal=True)

def test_fatal_is_a_boolean_not_a_two_valued_enum():
    """One fact, once. There are exactly two audiences: the editor
    shows everything, a run stops at the first fatal one."""
    assert Problem(code="c", message="m", fatal=True, node_id="n").fatal is True
    assert Problem(code="c", message="m", fatal=False, node_id="n").fatal is False


def test_a_problem_is_frozen():
    p = Problem(code="c", message="m", fatal=True, node_id="n")

    with pytest.raises(Exception):
        p.code = "other"


def test_problems_compare_by_value():
    a = Problem(code="c", message="m", fatal=True, node_id="n")
    b = Problem(code="c", message="m", fatal=True, node_id="n")

    assert a == b
