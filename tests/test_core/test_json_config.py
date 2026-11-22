"""Tests for ``conductor.JSON_CONFIG``, the config a host builds its own JSON models from."""

from typing import TypedDict

import conductor
import pytest
from conductor import JSON_CONFIG, Graph, host_type, make_graph, node
from pydantic import BaseModel, ConfigDict, with_config
from typing_extensions import TypedDict as ExtensionsTypedDict

from tests.test_core.nodes import registry_of


def test_json_config_is_a_public_name():
    assert "JSON_CONFIG" in conductor.__all__


def test_a_host_model_built_from_json_config_is_taken_and_reads_bytes_as_base64():
    @host_type("attachment", title="Attachment")
    class Attachment(BaseModel, frozen=True):
        model_config = ConfigDict(**JSON_CONFIG)
        data: bytes

    @node("attachment-size", title="Attachment size", description="Measures a file.")
    def attachment_size(file: Attachment) -> int:
        return len(file.data)

    graph = Graph.compile(make_graph(attachment_size), registry_of(attachment_size))
    assert graph.problems == []
    run = graph.run({"file": {"data": b"hello"}})
    assert run.outputs["result"] == 5
    assert graph.run({"file": {"data": "aGVsbG8="}}).outputs["result"] == 5


def test_json_config_can_not_be_changed():
    for change in (
        lambda: JSON_CONFIG.__setitem__("ser_json_bytes", "utf8"),
        lambda: JSON_CONFIG.update(ser_json_bytes="utf8"),
        lambda: JSON_CONFIG.pop("ser_json_bytes"),
        lambda: JSON_CONFIG.clear(),
    ):
        with pytest.raises(TypeError, match=r"ConfigDict\(\*\*JSON_CONFIG"):
            change()
    assert JSON_CONFIG["ser_json_bytes"] == "base64"


@pytest.mark.parametrize("typed_dict", [TypedDict, ExtensionsTypedDict])
def test_a_typed_dict_built_with_json_config_is_taken_and_reads_bytes_as_base64(typed_dict):
    """A ``TypedDict`` from ``typing`` or from ``typing_extensions`` is a row either way, so either takes a config."""

    @with_config(JSON_CONFIG)
    class Attachment(typed_dict):  # type: ignore[misc,valid-type]
        data: bytes

    @node("attachment-size", title="Attachment size", description="Measures a file.")
    def attachment_size(file: Attachment) -> int:
        return len(file["data"])

    graph = Graph.compile(make_graph(attachment_size), registry_of(attachment_size))
    assert graph.problems == []
    assert graph.run({"file": {"data": "__4="}}).outputs["result"] == 2
