# Widgets

A widget is the control a person edits an input with, plus what that control needs. Each is a frozen pydantic model with a `type`, so a frontend renders any node from the palette.

## Declaring one

The widget goes on the `Param` of a parameter, beside the title and the description:

```python
from typing import Annotated

from conductor import Param, node
from conductor.widgets import TextWidget


@node("greet")
def greet(name: Annotated[str, Param(title="Name", description="Who to greet", widget=TextWidget())]) -> Annotated[str, Param(title="Greeting")]:
    return f"Hello, {name}!"
```

No type has a default widget, since the same `str` may be a textarea, a single line or a dropdown. An input with no widget gets no control in an editor: an edge fills it, or a value set in code.

## Catalog

| Widget | For | Options |
|--------|-----|---------|
| `TextWidget` | One line of text | |
| `Textarea` | Several lines | `rows` |
| `Dropdown` | One of a list of `Choice`s | `choices` |
| `NumberWidget` | A number typed in | `step`, `integer_only` |
| `Range` | A number on a slider | `step` |
| `Switch` | On or off | |
| `DatePicker` | A date | `seed` |
| `FileUpload` | Uploaded files | `accept`, `max_size_mb`, `multiple` |
| `ListWidget` | A list typed by hand | |

The limits a value must keep to are the `Param`'s, in pydantic's words, and they are enforced when a graph is loaded for a typed-in value, and on every value a caller gives: `Param(ge=0, le=2, widget=Range())`, `Param(max_length=31, widget=TextWidget())`.

The vocabulary inside a control, such as a dropdown's `choices`, is the host's, declared as data on the widget.

## The schema

A widget dumps through pydantic. The dump is for an editor to read, not to load back:

```python
>>> from conductor.widgets import Textarea
>>> Textarea(rows=2).model_dump(mode="json")
{'type': 'textarea', 'rows': 2}
```

`definition.describe()` is a node's palette entry, and `registry.describe()` is the whole palette.

## Adding a control

Conductor ships the common controls and neither loading nor the engine reads a widget, so a control of your own needs no conductor release. Subclass `Widget` with a `type: Literal["..."]` field, beside the component in your frontend that renders it:

```python
from typing import Literal

from conductor.widgets import Widget


class EntityDropdown(Widget):
    type: Literal["entity-dropdown"] = "entity-dropdown"
    entity_kind: str = ""
```

A field's description carries its widget whole, so the editor receives every field of it. A host that wants a closed union of widgets for its frontend declares it over its own kinds.

See [`examples/07_widgets.ipynb`](../examples/07_widgets.ipynb) for every control in use.
