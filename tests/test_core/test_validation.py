"""Phase 1: Pydantic validation model generation from function signatures."""

from typing import Annotated

import pytest
from conductor.validation import create_validation_model
from conductor.widgets import Text
from pydantic import ValidationError


class TestValidationModelGeneration:
    def test_simple_function(self):
        def greet(
            name: Annotated[str, Text(label="Name")],
        ) -> str:
            return f"Hello {name}"

        model = create_validation_model(greet)
        validated = model(name="World")
        assert validated.model_dump() == {"name": "World"}

    def test_with_defaults(self):
        def greet(
            name: Annotated[str, Text(label="Name")],
            formal: Annotated[bool, Text(label="Formal")] = False,
        ) -> str:
            return f"Dear {name}" if formal else f"Hi {name}"

        model = create_validation_model(greet)
        validated = model(name="Alice")
        assert validated.model_dump() == {"name": "Alice", "formal": False}

    def test_type_coercion(self):
        """String values are coerced to the declared type."""

        def add(
            a: Annotated[int, Text(label="A")],
            b: Annotated[int, Text(label="B")],
        ) -> int:
            return a + b

        model = create_validation_model(add)
        validated = model(a="3", b="4")
        assert validated.model_dump() == {"a": 3, "b": 4}

    def test_validation_error_on_bad_input(self):
        def echo(text: Annotated[str, Text(label="Text")]) -> str:
            return text

        model = create_validation_model(echo)
        with pytest.raises(ValidationError):
            model()  # missing required field

    def test_optional_field(self):
        def greet(
            name: Annotated[str | None, Text(label="Name")] = None,
        ) -> str:
            return f"Hello {name or 'World'}"

        model = create_validation_model(greet)
        validated = model()
        assert validated.model_dump() == {"name": None}

    def test_list_type(self):
        def join(
            items: Annotated[list[str], Text(label="Items")],
        ) -> str:
            return ", ".join(items)

        model = create_validation_model(join)
        validated = model(items=["a", "b", "c"])
        assert validated.model_dump() == {"items": ["a", "b", "c"]}
