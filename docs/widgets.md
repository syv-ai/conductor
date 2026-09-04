# Widgets

A widget is the control an input is edited with, plus what that control needs: a dropdown's choices, a number's range. Every widget is a frozen, keyword-only record with a `kind` discriminator, and `AnyWidget` is the union of all of them, so pydantic dumps a widget and publishes a JSON schema per kind — a generic frontend renders any registered node by reading the palette.

## How to use a widget

An author writes the widget inside `Annotated` on a `run` parameter:

```python
from typing import Annotated
from conductor import NodeDefinition, Result
from conductor.widgets import Text as TextWidget
from conductor_nodes.types import Text

class Greet(NodeDefinition):
    id = "greet"
    title = "Greet"
    description = "Greets someone"
    category = "text"

    def run(
        self, name: Annotated[Text, TextWidget(title="Name", description="Who to greet")]
    ) -> Annotated[Text, Result(title="Greeting")]:
        return Text(f"Hello, {name}!")
```

Three things written on the widget belong to the field, not the control: `title`, `description` and `show_handle`. They sit on the widget because a parameter has one annotation object; `Interface.of` copies them onto the `Input` and they are left out of the widget's own dump, so each travels once.

## Every input declares its own

Conductor ships no default widget for any type. The same `Text` may be a textarea, a single line or a dropdown, so a parameter with no widget is a broken declaration and fails at import — not one that falls back to a default control.

A widget does not decide whether a cable can reach the input: `show_handle` defaults to `True` on the base and no control overrides it. A node closes one input by writing `show_handle=False` on that input's annotation; such an input may declare any pydantic-validatable type (a schema, a list of branches), since nothing travels on a wire to it. Where a cable *can* land, the parameter declares a `DType` — or `Any`, for a value the node routes without reading.

## Widget catalog

### Text & code

- **`Text`** — single-line string. Options: `min_length`, `max_length`, `pattern`.
- **`Textarea`** — multi-line string. Options: `rows`, `min_length`, `max_length`.
- **`TemplateTextarea`** — text with placeholders; each placeholder is an input. Options: `rows`.
- **`CodeEditor`** — source a person writes, highlighted for `language` (default `"python"`). Options: `min_length`, `max_length`.

### Choice

- **`Dropdown`** — pick one of a declared vocabulary of `Choice`s (`id`, `title`, optional `element` — a dtype description when the option fits only one element type). Options: `choices`.
- **`EntityDropdown`** — choices the host resolves (documents, say). Options: `entity_kind`, `multiple`.

### Numeric

- **`Number`** — a number typed in, optionally bounded and optionally whole. Options: `min_val`, `max_val`, `step`, `integer_only`.
- **`Range`** — a number picked on a slider. Options: `min_val`, `max_val`, `step`.

### Boolean

- **`Switch`** — on or off.

### Date & file

- **`DatePicker`** — a date from a calendar. Options: `min_date`, `max_date`, `seed` (`"today"` asks the editor to write today's date when the node is placed).
- **`FileUpload`** — files a person uploads. Options: `accept`, `max_size_mb`, `multiple` (births a series of files).

### Structured

- **`List`** — a list of values typed by hand; the per-item control is derived by the host from the element type. Options: `min_items`, `max_items`.
- **`Tags`** — free-form labels added one at a time.
- **`TableInput`** — a table typed or pasted in, column types and all. Options: `min_rows`, `min_columns`, `column_types` (the host's scalar types as `Choice`s).
- **`SchemaBuilder`** — a schema built field by field. Options: `schema`, `allow_additional`, `field_types` (the host's vocabulary as `Choice`s).
- **`IfElseBuilder`** — conditions built from the host's operators. Options: `operators` (a tuple of `OperatorChoice`: `id`, `title`, `category`, `arity`).

### Special

- **`ConnectionList`** — edited by wiring only: the value comes down a cable, so there is nothing to type. The control for a `Series[X]` input, an `Any` input and an open roster's rows.

The vocabulary inside a control — a dropdown's choices, a builder's operators, a table's column types — is the host's, declared on the widget where it declares the input, so it travels as data and no frontend list has to agree with a host table by hand.

## Inspecting the schema

A widget dumps through pydantic; the three lifted fields are excluded because they live on the `Input`:

```python
>>> from pydantic import TypeAdapter
>>> from conductor.widgets import AnyWidget, Text
>>> TypeAdapter(AnyWidget).dump_python(Text(title="URL", pattern=r"https?://.*"), mode="json")
{'kind': 'text', 'min_length': None, 'max_length': None, 'pattern': 'https?://.*'}
```

At the node level, `cls.describe()` is the whole palette entry — every version's `Input` and `Output` records, each `Input` carrying its widget — and `conductor_providers.react.palette_from_registry(registry)` is `[cls.describe() for cls in registry.definitions()]`.

## Adding a new widget

The set of controls is closed: `AnyWidget` is built from the subclasses declared in `conductor/widgets.py`, and an `Input` carrying a widget declared elsewhere is refused by pydantic. That is deliberate — conductor ships the controls and a host ships the vocabulary inside them as data — and a new control is a change here, since the component that renders each `kind` has to exist in the host's frontend anyway.

1. Add a frozen, keyword-only dataclass subclassing `Widget` in `widgets.py` with a `kind: Literal["color-picker"] = "color-picker"` field and whatever the control needs. `AnyWidget` picks it up at import.
2. Add a test in `tests/test_core/test_interface.py` that an input declaring it dumps with that `kind`.
3. The frontend owes a component dispatching on `"kind": "color-picker"`.

## Related

- [`examples/08_widgets.ipynb`](../examples/08_widgets.ipynb) — hands-on tour of every widget.
- [`README.md`](../README.md) — the widget table.
- [`CLAUDE.md`](../CLAUDE.md) — convention notes for agent sessions.
