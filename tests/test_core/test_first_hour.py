"""The programs a new developer writes first, through the three doors."""

from __future__ import annotations

import json
import logging

from conductor import CompiledGraph, DType, Graph, GraphNode, NodeRegistry, node


class Text(DType, str):
    id = "text"


@node("boom")
def _boom() -> Text:
    raise ValueError("kaboom: the secret")


def test_a_nodes_exception_reaches_the_log_with_its_traceback_and_stays_off_the_state(caplog):
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="a", type="boom", version=1)]), NodeRegistry([_boom]))

    with caplog.at_level(logging.ERROR, logger="conductor"):
        state = compiled.run_sync()

    assert state.status == "failed"
    (logged,) = [record for record in caplog.records if record.name == "conductor"]
    assert logged.exc_info is not None and logged.exc_info[0] is ValueError
    assert "'a'" in logged.getMessage()
    assert state.failure.ctx == {"exception": "ValueError"}
    assert "kaboom" not in json.dumps(state.model_dump(mode="json"))


def test_a_text_built_on_str_can_call_str_title():
    assert Text("hello world").title() == "Hello World"
