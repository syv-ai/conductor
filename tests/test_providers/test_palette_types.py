"""The published palette.d.ts stays in lockstep with the serialized schema.

The React provider ships hand-written TypeScript types (`palette.d.ts`) for
the palette payload. This pins them to the Python source of truth: every
`SerializedInput` / `SerializedOutput` field and every key `serialize_node`
can emit must be declared in the `.d.ts`, so a new serialized key can't land
in Python and silently leave the published types stale.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import conductor_providers.react as react_pkg
from conductor.registry import NodeRegistry
from conductor.registry.schema import serialize_registry
from conductor.registry.serialized import SerializedInput, SerializedOutput
from conductor.widgets import Output, Text

PALETTE_DTS = Path(react_pkg.__file__).parent / "palette.d.ts"


def _declared_keys() -> str:
    return PALETTE_DTS.read_text()


def _is_declared(key: str, text: str) -> bool:
    # A property declaration: `  key?:` or `  key:`, at any indentation.
    return re.search(rf"(?m)^\s*{re.escape(key)}\??:", text) is not None


def _maximal_node_keys() -> set[str]:
    """Every key serialize_node can emit — a node with all optional metadata set."""
    reg = NodeRegistry()

    @reg.node(
        "full",
        version=1,
        name="Full",
        description="d",
        actor="system",
        timeout=5,
        idempotency_key="k",
        uses=["thing"],
        is_decision=True,
        is_signal=True,
        compute_outputs=lambda **kw: [],
    )
    def full(x: Annotated[str, Text(label="X")] = "") -> Annotated[str, Output(label="O")]:
        return x

    return set(serialize_registry(reg)[0].keys())


def test_palette_dts_exists() -> None:
    assert PALETTE_DTS.is_file(), PALETTE_DTS


def test_serialized_input_fields_are_declared() -> None:
    text = _declared_keys()
    missing = [k for k in SerializedInput.model_fields if not _is_declared(k, text)]
    assert not missing, f"palette.d.ts missing PaletteInput keys: {sorted(missing)}"


def test_serialized_output_fields_are_declared() -> None:
    text = _declared_keys()
    missing = [k for k in SerializedOutput.model_fields if not _is_declared(k, text)]
    assert not missing, f"palette.d.ts missing PaletteOutput keys: {sorted(missing)}"


def test_serialize_node_keys_are_declared() -> None:
    text = _declared_keys()
    missing = [k for k in _maximal_node_keys() if not _is_declared(k, text)]
    assert not missing, f"palette.d.ts missing PaletteNode keys: {sorted(missing)}"
