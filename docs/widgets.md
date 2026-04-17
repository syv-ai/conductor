# Widgets

Widgets describe how a node parameter renders in a frontend UI. They also carry validation constraints (max length, min/max values, allowed choices, etc.) so the same annotation drives backend Pydantic validation and frontend rendering. One class per `WidgetType` enum value means a generic frontend can render any registered node by reading the registry.

If the widget you need is missing, jump to [Adding a new widget](#adding-a-new-widget).

## How to use a widget

Widgets attach to function parameters through `Annotated[T, Widget(...)]`:

```python
from typing import Annotated
from conductor.widgets import Text, Output

@registry.node("greet", version=1, name="Greet", description="Greets someone")
def greet(
    name: Annotated[str, Text(label="Name", description="Who to greet")],
) -> Annotated[str, Output(label="Greeting")]:
    return f"Hello, {name}!"
```

The widget's fields (`label`, `description`, `min_length`, `choices`, …) serialize into the JSON schema that the frontend reads to draw the node.

## Defaults — you don't always need a widget

If a parameter has only a type hint (or an `Annotated` without a `Widget` instance), the registry picks a default widget from the type:

| Python type | Default widget |
|---|---|
| `str` | `Text` |
| `int` | `Number(integer_only=True)` |
| `float` | `Number` |
| `bool` | `Checkbox` |
| `Date` | `DatePicker` |
| `Base64Str` / `NamedFile` / `MultiNamedFile` | `FileUpload` |
| `list[str]` | `List(item_widget=Text())` |
| `list[int]` | `List(item_widget=Number(integer_only=True))` |
| `list[T]` (bare or other `T`) | `List` (no `item_widget`) |
| `dict` or `dict[str, T]` | `SchemaBuilder` |
| anything else | no widget (rare; annotate explicitly) |

```python
# No Annotated needed — the registry picks Number and Checkbox
@registry.node("plan-trip", version=1, name="Plan Trip", description="...")
def plan_trip(days: int, include_weekends: bool) -> Annotated[str, Output(label="Plan")]:
    return f"{days}d, weekends={include_weekends}"
```

Explicit `Annotated[T, Widget(...)]` always wins. Defaults are a convenience; annotate when you want:
- A different widget (`Range` instead of `Number`, `Dropdown` instead of `Text`).
- Validation constraints (`min_length`, `max_val`, `pattern`).
- A human-friendly label (`label="Number of days"`) or description.

## Widget catalog

Every widget below corresponds 1:1 to a `WidgetType` enum value and has a concrete class in `conductor.widgets`.

### Text & code

- **`Text`** — single-line string. Options: `min_length`, `max_length`, `pattern`.
- **`Textarea`** — multi-line string. Options: `rows`, `min_length`, `max_length`.
- **`TemplateTextarea`** — textarea aware of a fixed set of interpolation variables. Options: `rows`, `variables: list[str]`. Interpolation is the node author's responsibility; the widget just declares the variables for editor hints.
- **`CodeEditor`** — syntax-highlighted code blob. Options: `language` (default `"python"`), `min_length`, `max_length`.

### Choice

- **`Dropdown`** — pick one from a fixed set. Options: `choices: list[str]`.
- **`DependentDropdown`** — choices depend on another field's value. Options: `depends_on: str`, `choices_map: dict[str, list[str]]`.
- **`Multiselect`** — pick many from a fixed set. Options: `choices`, `min_selected`, `max_selected`.
- **`EntityDropdown`** — host-loaded async choices. Options: `entity_kind: str` (host-defined), `multiple: bool`. Which data source a given `entity_kind` maps to is a host-application concern; the widget only declares intent.

### Numeric

- **`Number`** — free numeric input (text-style). Options: `min_val`, `max_val`, `step`, `integer_only`. Prefer `Range` for slider UX.
- **`Range`** — numeric slider. Options: `min_val`, `max_val`, `step`.

### Boolean

- **`Checkbox`** — boolean. Typically rendered as a square checkbox.
- **`Switch`** — boolean with toggle styling. Value is a plain `bool`; the widget type hints at visual treatment.

### Date & file

- **`DatePicker`** — ISO-8601 date (`"YYYY-MM-DD"`). Options: `min_date`, `max_date`.
- **`FileUpload`** — base64-encoded file. Options: `accept` (e.g. `".pdf,.csv"`), `max_size_mb`, `multiple`.

### Structured

- **`List`** — user-authored array. Each element is edited through `item_widget`. Options: `item_widget: Widget | None`, `min_items`, `max_items`. When `item_widget` is `None`, frontends default to a plain text input per row.
- **`SchemaBuilder`** — structured dict / object editor. Options: `schema: dict | None` (JSON-Schema-ish hint), `allow_additional: bool`. Values are plain Python dicts at runtime.
- **`IfElseBuilder`** — conditional-expression editor. Options: `variables: list[str]`. The concrete expression language is host-defined; the backend stores whatever the frontend emits.

### Special

- **`ConnectionList`** — aggregates *N upstream edges* into a single labeled dict input (keys are producer display names). Distinct from `List`: `ConnectionList` is for edge-driven aggregation, `List` is for user-authored values.
- **`Output`** — marker on return-value annotations. Options: `download` (offer as download in the UI), `filename`.

## Inspecting the schema

Every widget has a `to_schema()` method that emits the JSON dict the frontend reads:

```python
>>> Text(label="URL", pattern=r"https?://.*").to_schema()
{
  "widget": "text",
  "label": "URL",
  "description": None,
  "disable_handle": False,
  "pattern": "https?://.*"
}
```

At the node level, `serialize_registry(registry)` produces the full per-node schema (inputs with their widget configs, outputs, metadata). `conductor_providers.react.palette_from_registry(registry)` wraps that for ReactFlow frontends.

## Adding a new widget

If the catalog above doesn't cover your case, adding a widget is four steps. The built-in widgets are all ~30 lines each, so use them as templates.

### 1. Add the enum value

`packages/conductor/src/conductor/types.py`:

```python
class WidgetType(str, Enum):
    ...
    COLOR_PICKER = "color-picker"   # new
```

The string value is what ends up in the JSON schema (`"widget": "color-picker"`), so pick something your frontend can dispatch on.

### 2. Define the Widget class

`packages/conductor/src/conductor/widgets.py`:

```python
@dataclass
class ColorPicker(Widget):
    """Hex color input. Value is a string like ``"#3366ff"``."""

    disable_handle: bool = True
    default: str | None = None

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.COLOR_PICKER

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        if self.default is not None:
            schema["default"] = self.default
        return schema
```

Conventions the existing widgets follow:

- **Inherit from `Widget`.** You get `label`, `description`, `disable_handle`, `hidden_when`, `advanced`, and a base `to_schema()` for free.
- **Override `widget_type`** to return your new `WidgetType` member.
- **Override `to_schema()`** to add fields specific to your widget. Call `super().to_schema()` first. Emit optional fields only when set (`if self.foo is not None:`) so the JSON stays compact.
- **Decide `disable_handle`.** If the widget is for user input only (no edge makes sense), default to `True`. If it should also accept an edge from upstream, default to `False`.

### 3. (Optional) Register a default for a Python type

If your widget is the natural default for some built-in type, update the dispatch in `packages/conductor/src/conductor/registry/__init__.py`:

```python
def _default_widget(base_type, param_name):
    ...
    if base_type is MyColor:                # custom type alias
        return ColorPicker(label=param_name)
    ...
```

Skip this step if users should always opt in explicitly (which is the case for most domain-specific widgets).

### 4. Write tests

Add a test class to `tests/test_core/test_widget_defaults.py`:

```python
class TestColorPickerWidget:
    def test_emits_correct_type(self):
        s = ColorPicker(label="bg").to_schema()
        assert s["widget"] == WidgetType.COLOR_PICKER.value

    def test_default_surfaces_when_set(self):
        s = ColorPicker(label="bg", default="#fff").to_schema()
        assert s["default"] == "#fff"
```

If you added a default-widget rule, also test that `def f(x: MyColor)` produces a `ColorPicker` input.

That's it on the Python side. The frontend owes a matching component that reads the schema's `"widget": "color-picker"` dispatch key — but that's per-framework, not a backend concern.

## Troubleshooting

**"`NameError: name 'Text' is not defined`" when registering a node in a test.**
`get_type_hints` evaluates annotations in the function's `__globals__`, not the local scope where the decorator was applied. Import widgets at module level in test files — local imports inside a test method will fail at `@registry.node(...)` time.

**I want a widget the library has a `WidgetType` value for, but no class.**
That shouldn't happen anymore — every enum value has a concrete class. If you find a gap, open an issue or follow the four-step recipe above.

**My node's input renders as a plain text input even though I wrote `list[str]`.**
Check that the frontend dispatches on `"widget": "list"`. The backend is emitting it; older frontends may fall back to Text when they don't recognize a widget type.

## Related

- [`examples/08_widgets.ipynb`](../examples/08_widgets.ipynb) — hands-on tour of every widget.
- [`README.md`](../README.md) — user-facing widget table + default mapping.
- [`CLAUDE.md`](../CLAUDE.md) — convention notes for agent sessions.
