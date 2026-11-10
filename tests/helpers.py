"""What several test modules need to read a stream."""

import asyncio
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any

from conductor import CompiledGraph, DType, NodeDefinition, NodeRegistry, RunState
from conductor.events import Event
from pydantic import TypeAdapter
from pydantic_core import core_schema


def waiting_id(state: RunState, name: str) -> str:
    """The id of the unit waiting at ``node`` or ``node@row``: what a person's answer is keyed by.
    A name no unit has comes back as it is, for a test of an unknown id."""
    return next((unit.id for unit in state.waiting if unit.id.split("#")[0] == name), name)


def answer(compiled: CompiledGraph, state: RunState, answers: Mapping[str, Any]) -> RunState:
    """``compiled.answer`` with the units named ``node`` or ``node@row``."""
    return compiled.answer(state, {waiting_id(state, name): value for name, value in answers.items()})


async def stream_of(compiled: CompiledGraph, state: RunState | None = None, answers: Mapping[str, Any] | None = None, **kwargs: Any) -> AsyncIterator[Event]:
    """Every event of one leg, as a plain async iterator, the answers written in first."""
    if answers:
        state = answer(compiled, state, answers)  # type: ignore[arg-type]
    async with compiled.stream(state, **kwargs) as events:
        async for event in events:
            yield event


def events_of(compiled: CompiledGraph, **kwargs: Any) -> list[Event]:
    """Every event of one leg as a list, for a test that is not async itself."""
    async def collect() -> list[Event]:
        return [event async for event in stream_of(compiled, **kwargs)]

    return asyncio.run(collect())


def read_table(table_type: Any, value: Any) -> Any:
    """``value`` as the frame a node's ``table_type`` input would receive."""
    return TypeAdapter(table_type).validate_python(value)


def registry_of(*definitions: NodeDefinition[..., Any], types: Iterable[type[DType]] = ()) -> NodeRegistry:
    """A registry holding ``definitions`` and the ``types`` named."""
    registry = NodeRegistry(types=types)
    registry.add(*definitions)
    return registry


def with_added(registry: NodeRegistry, *definitions: NodeDefinition[..., Any]) -> NodeRegistry:
    """A new registry holding what ``registry`` holds and ``definitions``; ``registry`` is left as it was."""
    extended = NodeRegistry()
    extended.add_subregistry(registry)
    extended.add(*definitions)
    return extended


class Json(DType):
    """Any JSON-shaped value held in ``value``, dumped as the value it holds: a type with a validator and serializer of its own."""

    id = "json"
    display_title = "JSON"

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Json) and other.value == self.value

    def __repr__(self) -> str:
        return f"Json({self.value!r})"

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            lambda value: value if isinstance(value, cls) else cls(value),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda json: json.value),
        )
