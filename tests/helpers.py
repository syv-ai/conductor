"""What several test modules need to read a stream."""

from collections.abc import AsyncIterator, Mapping
from typing import Any

from conductor import CompiledGraph, RunState
from conductor.events import Event


async def stream_of(compiled: CompiledGraph, state: RunState | None = None, answers: Mapping[str, Any] | None = None, **kwargs: Any) -> AsyncIterator[Event]:
    """Every event of one leg, as a plain async iterator, the answers written in first."""
    if answers:
        state = compiled.answer(state, answers)  # type: ignore[arg-type]
    async with compiled.stream(state, **kwargs) as events:
        async for event in events:
            yield event
