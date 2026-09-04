"""The contract every node in the standard library satisfies.

Checked against the whole catalog rather than node by node, so a change
to one module cannot quietly drop a title, a category or an id.
"""

from collections import defaultdict
from typing import Any, get_args

import pytest
from conductor.node import NodeDefinition
from conductor_nodes import get_default_registry
from conductor_nodes.types import Flag, Json, Number, Text

#: Every node id the catalog is expected to hold, checked in both directions
#: against ``registry.definitions()``: a node missing here or an unlisted
#: node in the registry both fail.
EXPECTED_IDS = {
    "text-uppercase", "text-lowercase", "text-trim", "text-length",
    "text-concat", "text-replace", "text-contains", "text-split",
    "text-join", "text-reverse",
    "math-add", "math-subtract", "math-multiply", "math-divide",
    "math-modulo", "math-round", "math-min", "math-max", "math-abs",
    "logic-if-empty", "logic-if-equals", "logic-not",
    "json-parse", "json-stringify", "json-get",
    "regex-match", "regex-replace", "regex-extract",
    "decision",
}

#: The types the catalog is declared with. A field's dtype is one of these,
#: a ``Series`` of one, or ``Any`` on a node that routes a value it does not
#: read.
VOCABULARY = {Text, Number, Flag, Json}


@pytest.fixture(scope="module")
def registry():
    return get_default_registry()


def _current(node_cls):
    """The interface of the version the catalog currently offers."""
    return node_cls.versions[node_cls.current].interface


def _is_vocabulary(dtype) -> bool:
    element = getattr(dtype, "element", None) or dtype
    return element in VOCABULARY


def test_the_catalog_is_exactly_what_is_expected(registry):
    """Both directions: a subset check would let an unreviewed node into the catalog."""
    assert {n.id for n in registry.definitions()} == EXPECTED_IDS


def test_the_markers_and_the_signals_are_gone(registry):
    """The old compound and signal node types are not in the catalog.

    Iteration is a ``Series`` reaching a scalar input, an embedded flow is
    a definition the host builds, and a pause is a node returning ``Asks``,
    so none of these needs a node of its own.
    """
    for gone in (
        "for-each-start", "for-each-end", "while-start", "while-end", "subprocess-call",
        "signal-wait", "signal-timer",
    ):
        assert not registry.contains(gone), gone


def test_every_node_declares_a_title_a_description_and_a_category(registry):
    from conductor_nodes.types import Category

    for node_cls in registry.definitions():
        assert node_cls.title, f"{node_cls.id} has no title"
        assert node_cls.description, f"{node_cls.id} has no description"
        assert node_cls.category in get_args(Category), f"{node_cls.id} is filed nowhere"


def test_every_node_numbers_its_versions_from_one(registry):
    for node_cls in registry.definitions():
        assert min(node_cls.versions) == 1, f"{node_cls.id} starts at {min(node_cls.versions)}"


def test_every_field_declares_a_title(registry):
    for node_cls in registry.definitions():
        iface = _current(node_cls)
        for field in (*iface.inputs, *iface.outputs):
            assert field.title, f"{node_cls.id}.{field.name} has no title"


def test_every_field_is_declared_in_the_vocabulary_or_is_any(registry):
    """Every field's type is one of ``conductor_nodes.types``, a ``Series`` of
    one, or ``Any`` on a node that routes a value it does not read."""
    for node_cls in registry.definitions():
        iface = _current(node_cls)
        for field in (*iface.inputs, *iface.outputs):
            assert field.dtype is Any or _is_vocabulary(field.dtype), (
                f"{node_cls.id}.{field.name} is {field.dtype!r}"
            )


def test_only_the_gate_declares_any(registry):
    """``decision`` routes a value it does not read, so it declares ``Any``
    in and ``Any`` on both outputs, and its ``compute_outputs`` fills in the
    arriving type. No other node declares ``Any``."""
    vague = {
        node_cls.id
        for node_cls in registry.definitions()
        if any(f.dtype is Any for f in (*_current(node_cls).inputs, *_current(node_cls).outputs))
    }
    assert vague == {"decision"}
    gate = _current(registry.get("decision"))
    assert gate.inputs[0].dtype is Any and all(o.dtype is Any for o in gate.outputs)


def test_branches_are_one_choice(registry):
    """Outputs sharing a ``choice`` are exclusive alternatives, exactly one
    produced per run. The three routing nodes declare one group each;
    nothing else declares any."""
    groups = {}
    for node_cls in registry.definitions():
        by_choice = defaultdict(list)
        for out in _current(node_cls).outputs:
            if out.choice is not None:
                by_choice[out.choice].append(out.name)
        if by_choice:
            groups[node_cls.id] = dict(by_choice)
    assert groups == {
        "decision": {"when": ["if_true", "if_false"]},
        "logic-if-empty": {"emptiness": ["not_empty", "empty"]},
        "logic-if-equals": {"equality": ["equal", "not_equal"]},
    }


def test_no_node_says_what_the_engine_must_do(registry):
    """No node carries a ``role``: a branch not taken is ``SKIPPED``, a
    value the engine acts on, and nothing on the class announces it."""
    for node_cls in registry.definitions():
        assert not hasattr(node_cls, "role"), node_cls.id


def test_no_stdlib_node_overrides_a_shaping_hook(registry):
    """No node's inputs or outputs depend on a value, with one exception:
    ``decision`` overrides ``compute_outputs`` to put the type arriving on
    ``value`` onto both outputs.

    There is no ``validate`` hook to override: a value's constraints are
    its dtype's constructor rules.
    """
    for node_cls in registry.definitions():
        assert getattr(node_cls, "compute_inputs") is getattr(NodeDefinition, "compute_inputs"), (
            f"{node_cls.id} overrides compute_inputs — record why it must"
        )
        if node_cls.id != "decision":
            assert getattr(node_cls, "compute_outputs") is getattr(NodeDefinition, "compute_outputs"), (
                f"{node_cls.id} overrides compute_outputs — record why it must"
            )
        assert not hasattr(node_cls, "validate"), f"{node_cls.id} has a validate hook; there is none"


def test_every_node_describes(registry):
    """``describe()`` builds for every node."""
    for node_cls in registry.definitions():
        assert node_cls.describe().id == node_cls.id
