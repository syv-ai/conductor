# Widgets

A widget is the control an input is edited with, plus what that control needs: a dropdown's choices, a number's range. Every widget is a frozen pydantic model with a `kind` discriminator, and `AnyWidget` is the union of all of them, so pydantic dumps a widget and publishes a JSON schema per kind — a generic frontend renders any registered node by reading the palette.

## How to use a widget

An author writes the widget inside `Annotated` on a `run` parameter:

```python
from typing import Annotated
from conductor import NodeDefinition, Param, Result
from conductor.widgets import TextWidget
from conductor_nodes.types import Text

class Greet(NodeDefinition):
    id = "greet"
    title = "Greet"
    description = "Greets someone"
    category = "text"

    def run(
        self, name: Annotated[Text, Param(title="Name", description="Who to greet", widget=TextWidget())]
    ) -> Annotated[Text, Result(title="Greeting")]:
        return Text(f"Hello, {name}!")
```

A widget is the control and nothing more. What a person reads about the input — `title`, `description` — and whether an edge can reach it — `show_handle` — are written on the `Param` the widget sits on, `Annotated[Text, Param(title="Text", widget=Textarea())]`, and travel once, on the `Input`; a widget's own dump is its configuration alone.

## Every input declares its own

Conductor ships no default widget for any type. The same `Text` may be a textarea, a single line or a dropdown, so nothing falls back to a default control: a `Param` with no widget is an input only an edge fills, and a widget written bare in `Annotated` fails at import naming the `Param` form.

A widget does not decide whether an edge can reach the input: no control carries `show_handle`. A node closes one input by writing `show_handle=False` on its `Param`; such an input may declare any pydantic-validatable type (a schema, a list of branches), since nothing travels on an edge to it. Where an edge *can* land, the parameter declares a `DType` — or `Any`, for a value the node routes without reading.

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
- **`SchemaBuilder`** — a schema built field by field. Options: `schema`, `allow_additional`, `field_types` (the host's vocabulary as `Choice`s). `schema` is the keyword and the wire key; the attribute is `schema_`, since `builder.schema` is pydantic's own method.
- **`IfElseBuilder`** — conditions built from the host's operators. Options: `operators` (a tuple of `OperatorChoice`: `id`, `title`, `category`, `arity`).

### No widget

An input filled by edges only — a `Series[X]` gathered from other nodes, an `Any` passed through, each connected name of an open interface — declares no widget: `Param(title=...)` alone. There is nothing to type, so there is no control.

The vocabulary inside a control — a dropdown's choices, a builder's operators, a table's column types — is the host's, declared on the widget where it declares the input, so it travels as data and no frontend list has to agree with a host table by hand.

## Inspecting the schema

A widget dumps through pydantic; the three lifted fields are excluded because they live on the `Input`. The dump is for an editor to read, not to load back: without its title it is not a widget any more.

```python
>>> from pydantic import TypeAdapter
>>> from conductor.widgets import AnyWidget, TextWidget
>>> TypeAdapter(AnyWidget).dump_python(TextWidget(pattern=r"https?://.*"), mode="json")
{'kind': 'text', 'min_length': None, 'max_length': None, 'pattern': 'https?://.*'}
```

At the node level, `cls.describe()` is the whole palette entry — every version's `Input` and `Output` records, each `Input` carrying its widget — and `registry.describe()` is the palette: every node's record and the vocabulary.

## Adding a new widget

The set of controls is closed: `AnyWidget` is built from the subclasses declared in `conductor/widgets.py`, and an `Input` carrying a widget declared elsewhere is refused by pydantic. That is deliberate — conductor ships the controls and a host ships the vocabulary inside them as data — and a new control is a change here, since the component that renders each `kind` has to exist in the host's frontend anyway.

1. Add a class subclassing `Widget` in `widgets.py` with a `kind: Literal["color-picker"] = "color-picker"` field and whatever the control needs. `AnyWidget` is built from `Widget.__subclasses__()`, so it picks the class up at import.
2. Add a test in `tests/test_core/test_interface.py` that an input declaring it dumps with that `kind`.
3. The frontend owes a component dispatching on `"kind": "color-picker"`.

## Related

- [`examples/08_widgets.ipynb`](../examples/08_widgets.ipynb) — hands-on tour of every widget.
- [`README.md`](../README.md) — the widget table.
- [`AGENTS.md`](../AGENTS.md) — convention notes for agent sessions.
