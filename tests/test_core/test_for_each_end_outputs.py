"""Tests for ``compute_for_each_end_outputs`` — the default ``compute_outputs``
hook on the stdlib ``for-each-end`` marker.

Exercises the hook as a standalone function (building ``ComputeOutputsContext``
directly) so the typing/labelling/ordering rules are pinned independently of a
full compile.
"""

from __future__ import annotations

from conductor import compute_for_each_end_outputs
from conductor.metadata import OutputMetadata
from conductor.registry.dynamic_outputs import (
    ComputeOutputsContext,
    IncomingBinding,
)


def _binding(
    target_handle: str,
    source_node_id: str,
    source_handle: str,
    type_str: str,
    label: str = "",
) -> IncomingBinding:
    return IncomingBinding(
        target_handle=target_handle,
        source_node_id=source_node_id,
        source_handle=source_handle,
        source_output=OutputMetadata(
            name=source_handle, type_str=type_str, label=label
        ),
    )


def _ctx(*bindings: IncomingBinding) -> ComputeOutputsContext:
    return ComputeOutputsContext(
        data={},
        incoming=tuple(bindings),
        node_id="end",
        defaults=(OutputMetadata(name="output_1", type_str="list", label="Collected"),),
    )


def test_single_source_wraps_element_type_in_list() -> None:
    out = compute_for_each_end_outputs(
        _ctx(_binding("items", "body", "output_1", "str", "Filnavn"))
    )
    assert len(out) == 1
    assert out[0].name == "output_1"
    assert out[0].type_str == "list[str]"
    assert out[0].label == "Filnavn"


def test_list_source_unwraps_one_level() -> None:
    # A body output already typed ``list[str]`` collects into ``list[str]``,
    # not ``list[list[str]]`` — one level is unwrapped.
    out = compute_for_each_end_outputs(
        _ctx(_binding("items", "body", "output_1", "list[str]", "Tekster"))
    )
    assert out[0].type_str == "list[str]"


def test_multiple_sources_get_ordered_output_slots() -> None:
    out = compute_for_each_end_outputs(
        _ctx(
            _binding("items", "a", "output_1", "str", "Filnavn"),
            _binding("items", "b", "output_1", "file", "Dokument"),
        )
    )
    assert [(o.name, o.type_str, o.label) for o in out] == [
        ("output_1", "list[str]", "Filnavn"),
        ("output_2", "list[file]", "Dokument"),
    ]


def test_dedup_by_source_and_handle() -> None:
    # Same (source, handle) wired via both the new ``items`` target and a
    # legacy ``item_2`` target collapses to a single slot.
    out = compute_for_each_end_outputs(
        _ctx(
            _binding("items", "a", "output_1", "str", "X"),
            _binding("item_2", "a", "output_1", "str", "X"),
        )
    )
    assert len(out) == 1


def test_legacy_item_handle_accepted() -> None:
    out = compute_for_each_end_outputs(
        _ctx(_binding("item", "body", "output_1", "int", "Antal"))
    )
    assert out[0].type_str == "list[int]"


def test_non_collection_edges_ignored() -> None:
    # A secondary control input that is not an end-collection edge must not
    # produce an output slot — with no collection edges the hook returns the
    # static defaults unchanged.
    ctx = _ctx(_binding("control", "x", "output_1", "str", "Nope"))
    assert compute_for_each_end_outputs(ctx) == list(ctx.defaults)


def test_no_bindings_returns_defaults() -> None:
    ctx = _ctx()
    out = compute_for_each_end_outputs(ctx)
    assert out == list(ctx.defaults)


def test_missing_type_falls_back_to_any() -> None:
    out = compute_for_each_end_outputs(
        _ctx(_binding("items", "body", "output_1", "", "Ukendt"))
    )
    assert out[0].type_str == "list[any]"


def test_label_sub_output_prefix_stripped() -> None:
    out = compute_for_each_end_outputs(
        _ctx(_binding("items", "body", "result.Filnavn", "str", "result.Filnavn"))
    )
    assert out[0].label == "Filnavn"
