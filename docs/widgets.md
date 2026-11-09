# Widgets

A widget is the control a person edits an input with, plus what that control needs. Each is a frozen pydantic model with a `kind`, and `AnyWidget` is their union, so a frontend renders any node from the palette.

## Declaring one

The widget goes on the `Param` of a parameter. The title, the description and `connectable` belong to the `Param`:

```python
from typing import Annotated

from conductor import Param, Result, node
from conductor.widgets import TextWidget


@node("greet")
def greet(name: Annotated[str, Param(title="Name", description="Who to greet", widget=TextWidget())]) -> Annotated[str, Result(title="Greeting")]:
    return f"Hello, {name}!"
```

No type has a default widget, since the same `str` may be a textarea, a single line or a dropdown. An input with no widget gets no control in an editor: an edge fills it, or a value set in code. `Param(connectable=False)` closes an input to edges, and it may then declare any type pydantic can validate.

## Catalog

| Widget | For | Options |
|--------|-----|---------|
| `TextWidget` | One line of text | `min_length`, `max_length`, `pattern` |
| `Textarea` | Several lines | `rows`, `min_length`, `max_length` |
| `TemplateTextarea` | Text with placeholders, each an input | `rows` |
| `CodeEditor` | Source code | `language`, `min_length`, `max_length` |
| `Dropdown` | One of a list of `Choice`s | `choices` |
| `EntityDropdown` | Choices the host resolves | `entity_kind`, `multiple` |
| `NumberWidget` | A number typed in | `min_val`, `max_val`, `step`, `integer_only` |
| `Range` | A number on a slider | `min_val`, `max_val`, `step` |
| `Switch` | On or off | |
| `DatePicker` | A date | `min_date`, `max_date`, `seed` |
| `FileUpload` | Uploaded files | `accept`, `max_size_mb`, `multiple` |
| `ListWidget` | A list typed by hand | `min_items`, `max_items` |
| `Tags` | Free-form labels | |
| `TableInput` | A table typed or pasted in | `min_rows`, `min_columns`, `column_types` |
| `SchemaBuilder` | A schema built field by field | `schema`, `allow_additional`, `field_types` |
| `IfElseBuilder` | Conditions from the host's operators | `operators` |

The vocabulary inside a control, such as a dropdown's `choices` or a builder's `operators`, is the host's, declared as data on the widget.

## The schema

A widget dumps through pydantic. The dump is for an editor to read, not to load back:

```python
>>> from pydantic import TypeAdapter
>>> from conductor.widgets import AnyWidget, TextWidget
>>> TypeAdapter(AnyWidget).dump_python(TextWidget(pattern=r"https?://.*"), mode="json")
{'kind': 'text', 'min_length': None, 'max_length': None, 'pattern': 'https?://.*'}
```

`definition.describe()` is a node's palette entry, and `registry.describe()` is the whole palette.

## Adding a control

The set is closed: `AnyWidget` is built from the `Widget` subclasses in `conductor/widgets.py`. A new control is a class there with a `kind: Literal["..."]` field, a test that an input declaring it dumps with that `kind`, and a component in the host's frontend that renders it.

See [`examples/07_widgets.ipynb`](../examples/07_widgets.ipynb) for every control in use.
