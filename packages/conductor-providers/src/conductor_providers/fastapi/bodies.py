"""The request bodies the runs routes read, and the JSON schema they declare for them.

A body that carries a graph is read as text, once, by conductor's ``read_json``. So a key written twice, such as two
nodes with one id, is refused rather than quietly dropped, and the graph inside is checked by ``Graph.compile``, the
one place a graph record is checked. FastAPI never parses these bodies, so their schema is only declared, through a
route's ``openapi_extra``, with ``declared``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from conductor import GraphRecord, read_json
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, WithJsonSchema


def declared(model: type[BaseModel]) -> dict[str, Any]:
    """A model's JSON schema that stands on its own, for a route's ``openapi_extra``.

    FastAPI adds to the document's ``components`` only the models a route's parameters or answer name, so the
    ``$ref`` entries of a schema given through ``openapi_extra`` would point at nothing. Here every ``$defs`` entry is
    written in where it is referred to. ``JsonValue`` refers to itself, so it is written as ``{}``: any JSON value.

    Args:
        model: The model.

    Returns:
        Its schema, with no ``$ref`` left in it.
    """
    schema = model.model_json_schema()
    definitions: dict[str, Any] = {}

    def lifted(value: Any) -> Any:
        if isinstance(value, dict):
            definitions.update(value.get("$defs", {}))
            return {key: lifted(item) for key, item in value.items() if key != "$defs"}
        if isinstance(value, list):
            return [lifted(item) for item in value]
        return value

    def written_in(value: Any) -> Any:
        if isinstance(value, dict):
            if "$ref" in value:
                name = value["$ref"].removeprefix("#/$defs/")
                return {} if name == "JsonValue" else written_in(definitions[name])
            return {key: written_in(item) for key, item in value.items()}
        if isinstance(value, list):
            return [written_in(item) for item in value]
        return value

    schema = lifted(schema)
    definitions = {name: lifted(entry) for name, entry in definitions.items()}
    return written_in(schema)


class StartRun(BaseModel):
    """The body of ``POST /runs`` and ``POST /estimate``: a graph, the values it starts on, and maybe a run to reuse."""

    # a key it doesn't know, such as "previus", is refused, never dropped
    model_config = ConfigDict(extra="forbid")

    graph: Annotated[dict[str, Any], WithJsonSchema(declared(GraphRecord))] = Field(
        description="The graph to run, as its record. Graph.compile checks it on the request's nodes."
    )
    inputs: dict[str, Any] = Field(
        default={}, description="Values for the graph's inputs, by name."
    )
    previous: str | None = Field(
        default=None,
        description="The id of a run whose work the new run reuses: a call that gets what it got there isn't made "
        "again.",
    )

    @classmethod
    def read(cls, text: bytes) -> StartRun:
        """Reads the body from its text.

        Args:
            text: The request's body.

        Returns:
            The body, with its graph left as plain data for ``Graph.compile``.

        Raises:
            InvalidCallError: The text isn't JSON, or writes a key twice (``invalid_record``).
            RequestValidationError: The body isn't an object of this shape, such as inputs that aren't an object.
                FastAPI answers it as it answers any body it refuses.
        """
        try:
            return cls.model_validate(read_json(text))
        except ValidationError as refused:
            errors = refused.errors(include_url=False)
            raise RequestValidationError(
                [{**error, "loc": ("body", *error["loc"])} for error in errors]
            ) from None


class Decline(BaseModel):
    """The body of ``POST /runs/{run_id}/declines``: the question a person declines to answer, and why.

    ``kind`` may be left out, since it can only be ``"decline"``.
    """

    # A run's state queues this body as it came, beside the answers, until the run takes it (see state.py).
    model_config = ConfigDict(extra="forbid")

    kind: Literal["decline"] = Field(
        default="decline", description='Always `"decline"`; it may be left out.'
    )
    waiting_id: str = Field(
        description="The `id` of the waiting call, as the run's `waiting` lists it."
    )
    reason: str | None = Field(
        default=None,
        description="Why the person declined, which the run's cause keeps in `ctx.reason`.",
    )


class Resume(BaseModel):
    """The body of ``POST /runs/{run_id}/resume``, which may be left out."""

    model_config = ConfigDict(extra="forbid")

    max_items: StrictInt | None = Field(
        default=None,
        description="A higher item limit for a run that stopped at its limit. It holds for the run from then on.",
    )
