"""Typed pydantic models for the serialized node schema.

``serialize_input`` / ``serialize_output`` (in :mod:`conductor.registry.schema`)
emit plain dicts for frontend consumption. These models are the *typed*
view of those dicts: the serializers are guaranteed to conform to them, so
a host that wants types can validate the wire payload into
:class:`SerializedInput` / :class:`SerializedOutput` instead of reading
loose ``dict[str, Any]`` shapes.

**Conformance contract.** Every key ``serialize_input`` can emit is a field
here: the eight base keys (:data:`_BASE_INPUT_KEYS`) plus the full widget-config
vocabulary (:data:`conductor.widgets.WIDGET_SCHEMA_KEYS`). The two sets are
pinned equal by ``tests/test_core/test_serialized_schema.py`` — a widget that
grows a new ``to_schema`` key fails that pin until the key is added here.
``model_config`` allows extras so a *host's* custom widget config never gets
rejected or silently dropped on validation.

This module is a deliberate leaf — it imports only pydantic — so any module
(including :mod:`conductor.widgets`) can depend on it without an import cycle
through the ``registry`` package initializer.
"""

from __future__ import annotations

import warnings
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# The non-widget-config fields of ``SerializedInput`` — the keys
# ``serialize_input`` writes directly from the ``InputMetadata`` rather than
# merging out of ``widget_config``. Everything else on the model mirrors
# ``WIDGET_SCHEMA_KEYS``.
_BASE_INPUT_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "type",
        "label",
        "description",
        "widget",
        "default",
        "optional",
        "disable_handle",
    }
)


with warnings.catch_warnings():
    # ``schema`` is a genuine widget-config key (``SchemaBuilder`` emits it),
    # so the field must be named ``schema`` for the wire dict to validate and
    # round-trip by name. That shadows pydantic's deprecated ``BaseModel.schema``
    # attribute; the shadowing is intentional and the warning is noise.
    warnings.filterwarnings(
        "ignore",
        message=r'Field name "schema" .*shadows an attribute',
        category=UserWarning,
    )

    class SerializedInput(BaseModel):
        """The typed shape of one entry in a node's serialized ``inputs``.

        Widget-config fields are all optional: a given widget emits only the
        keys its ``to_schema`` produces, so validating any single input dict
        leaves the rest unset.
        """

        model_config = ConfigDict(extra="allow")

        # Base keys — always emitted by ``serialize_input``.
        name: str
        type: str
        label: str
        description: str | None = None
        widget: str
        default: Any = None
        optional: bool = False
        disable_handle: bool = False

        # Widget-config keys — mirror ``WIDGET_SCHEMA_KEYS`` (pinned by test).
        accept: list[str] | None = None
        advanced: bool | None = None
        allow_additional: bool | None = None
        choices: list[str] | None = None
        choices_map: dict[str, list[str]] | None = None
        connection_input: str | None = None
        depends_on: str | None = None
        download: bool | None = None
        entity_type: str | None = None
        filename: str | None = None
        hidden_when: dict[str, list[str]] | None = None
        human_review: bool | None = None
        integer_only: bool | None = None
        item_widget: dict[str, Any] | None = None
        language: str | None = None
        max_date: str | None = None
        max_items: int | None = None
        max_length: int | None = None
        max_selected: int | None = None
        max_size_mb: float | None = None
        max_val: float | None = None
        min_columns: int | None = None
        min_date: str | None = None
        min_items: int | None = None
        min_length: int | None = None
        min_rows: int | None = None
        min_selected: int | None = None
        min_val: float | None = None
        multiple: bool | None = None
        pattern: str | None = None
        prompt: str | None = None
        range_max: float | None = None
        range_min: float | None = None
        rows: int | None = None
        schema: dict[str, Any] | None = None
        step: float | None = None
        variables: list[str] | None = None


class SerializedOutput(BaseModel):
    """The typed shape of one entry in a node's serialized ``outputs``."""

    model_config = ConfigDict(extra="allow")

    name: str
    type: str
    label: str
    description: str | None = None
    optional: bool = False
    download: bool = False
    filename: str | None = None


class SerializedNode(BaseModel):
    """The typed shape of one node's serialized catalog entry.

    The typed view of the dict :func:`conductor.registry.schema.serialize_node`
    emits. Base keys are always present; the process-standard keys (``actor``,
    ``timeout_seconds``, ``uses``, ``is_decision``, ``is_signal``,
    ``has_dynamic_outputs``, …) are only emitted when set on the node, so they
    default to ``None``. ``extra="allow"`` lets a host carry additional keys.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    base_id: str
    version: int
    name: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    category: str
    inputs: list[SerializedInput] = Field(default_factory=list)
    outputs: list[SerializedOutput] = Field(default_factory=list)
    width: int | None = None
    deprecated: bool = False
    latest_version: int
    docs: str | None = None

    # Process-standard metadata — present only when set on the node.
    actor: dict[str, Any] | None = None
    timeout_seconds: float | None = None
    idempotency_key: str | None = None
    uses: list[str] | None = None
    is_decision: bool | None = None
    is_signal: bool | None = None
    has_dynamic_outputs: bool | None = None
    has_dynamic_inputs: bool | None = None
