"""What several test modules need to read a stream."""

from collections.abc import AsyncIterator, Mapping
from typing import Any

from conductor import CompiledGraph, RunState
from conductor.events import Event
from pydantic import TypeAdapter


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


def read_table(table_type: Any, value: Any) -> Any:
    """``value`` as the frame a node's ``table_type`` input would receive."""
    return TypeAdapter(table_type).validate_python(value)
