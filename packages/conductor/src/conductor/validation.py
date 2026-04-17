"""Pydantic model generation from function signatures."""

import inspect
from collections.abc import Callable
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from conductor.widgets import Widget


def _extract_type_string(annotation: Any) -> str:
    """Convert a Python type annotation to a JSON-serializable string."""
    if annotation == inspect.Parameter.empty:
        return "any"

    if isinstance(annotation, str):
        return annotation.lower()

    # NewType
    if hasattr(annotation, "__supertype__"):
        return str(annotation.__name__).lower()

    origin = get_origin(annotation)

    if origin is Literal:
        return "str"

    if origin is None:
        if hasattr(annotation, "__name__"):
            return str(annotation.__name__).lower()
        return str(annotation).lower()

    # Optional / Union — extract first non-None type
    args = get_args(annotation)
    non_none = [a for a in args if a is not type(None)]
    if hasattr(origin, "__name__") and "Union" in str(origin):
        return _extract_type_string(non_none[0]) if non_none else "any"

    origin_name = (
        str(origin.__name__).lower()
        if hasattr(origin, "__name__")
        else str(origin).lower()
    )

    if origin_name == "list" and args:
        return f"list[{_extract_type_string(args[0])}]"
    elif origin_name == "dict" and args:
        k = _extract_type_string(args[0]) if len(args) > 0 else "any"
        v = _extract_type_string(args[1]) if len(args) > 1 else "any"
        return f"dict[{k}, {v}]"
    elif origin_name == "tuple" and args:
        inner = ", ".join(_extract_type_string(a) for a in args)
        return f"tuple[{inner}]"

    return origin_name


def _is_injectable(annotation: Any) -> bool:
    """Check if an annotation is a framework-injectable type (e.g. FlowStore).

    These parameters are excluded from the validation model and node inputs.
    """
    # Avoid circular import — check by class name
    if isinstance(annotation, type) and annotation.__name__ == "FlowStore":
        return True
    # Handle Annotated[FlowStore, ...]
    if get_origin(annotation) is Annotated:
        base = get_args(annotation)[0]
        return _is_injectable(base)
    return False


def create_validation_model(func: Callable[..., Any]) -> type[BaseModel]:
    """Create a Pydantic model from a function signature for input validation."""
    sig = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)

    fields: dict[str, tuple[Any, Any]] = {}

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        annotation = type_hints.get(param_name, param.annotation)

        # Skip injectable framework types
        if _is_injectable(annotation):
            continue

        default = param.default if param.default != inspect.Parameter.empty else ...

        if get_origin(annotation) is Annotated:
            args = get_args(annotation)
            base_type = args[0]
            field_info = None

            for arg in args[1:]:
                if isinstance(arg, FieldInfo):
                    field_info = arg
                    break
                # Widget instances can produce FieldInfo
                if isinstance(arg, Widget):
                    field_info = arg.to_field_info() if hasattr(arg, "to_field_info") else None
                    break

            if field_info:
                if field_info.default is PydanticUndefined and default is not ...:
                    field_info = Field(
                        default=default,
                        description=field_info.description,
                        json_schema_extra=field_info.json_schema_extra,
                    )
                fields[param_name] = (base_type, field_info)
            else:
                fields[param_name] = (base_type, default)
        else:
            if annotation == inspect.Parameter.empty:
                annotation = Any
            fields[param_name] = (annotation, default)

    model_name = f"{func.__name__}_ValidationModel"
    return create_model(model_name, **fields)  # type: ignore[call-overload]
