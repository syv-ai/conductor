"""What several test modules need to read a stream."""

from typing import Any

from pydantic import TypeAdapter


def read_table(table_type: Any, value: Any) -> Any:
    """``value`` as the frame a node's ``table_type`` input would receive."""
    return TypeAdapter(table_type).validate_python(value)
