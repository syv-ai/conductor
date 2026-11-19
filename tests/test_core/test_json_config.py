"""Tests for ``conductor.JSON_CONFIG``, the config a host builds its own JSON models from."""

import conductor
from conductor import JSON_CONFIG, Graph, NodeRegistry, host_type, make_graph, node
from pydantic import BaseModel, ConfigDict


def test_json_config_is_a_public_name():
    assert "JSON_CONFIG" in conductor.__all__


def test_a_host_model_built_from_json_config_is_taken_and_reads_bytes_as_base64():
    @host_type("attachment", title="Attachment")
    class Attachment(BaseModel):
        model_config = ConfigDict(**JSON_CONFIG)
        data: bytes

    @node("attachment-size", title="Attachment size", description="Measures a file.")
    def attachment_size(file: Attachment) -> int:
        return len(file.data)

    registry = NodeRegistry()
    registry.add(attachment_size)
    graph = Graph.compile(make_graph(attachment_size), registry)
    assert graph.problems == []
    run = graph.run({"file": {"data": b"hello"}})
    assert run.outputs["result"] == 5
    assert graph.run({"file": {"data": "aGVsbG8="}}).outputs["result"] == 5
