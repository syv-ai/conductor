"""A class or an exception is imported by name; a module name stands before a dot only for its functions and constants.

`pydantic_core.PydanticSerializationError` is written `PydanticSerializationError`
with `from pydantic_core import PydanticSerializationError`; `json.dumps`,
`time.monotonic()` and `re.MULTILINE` stay as they are. The tables below are the
rule's exceptions.
"""

from __future__ import annotations

import ast
import importlib
from ast import AST, Attribute, Import, ImportFrom, Name
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]

# Modules whose classes stay written through the module: polars is `pl` everywhere
# (a bare `Series` is conductor's own class), and a bare `time` or `datetime` is a module.
CLASSES_THAT_STAY_THROUGH = {
    "polars": "*",
    "pandas": "*",
    "datetime": {"date", "time", "datetime"},
}

# A class in lower case reads as a function (`functools.partial`, `weakref.ref`), so it
# stays unless it is named here with the name it is imported under.
LOWER_CASE_CLASSES_IMPORTED_AS = {
    ("regex", "error"): "RegexError",
}

# Where the bare name is already taken in the file, or two modules want it, the class is
# imported under this name, which says where it is from.
NAME_WHEN_BARE_IS_TAKEN = {
    ("asyncio", "Event"): "AsyncioEvent",
    ("threading", "Event"): "ThreadEvent",
    ("dataclasses", "Field"): "DataclassField",
}

# `typing` constructs are not always `type` objects (`Optional`, `Callable`), so for these
# modules a name in capitals-then-lower-case is a class.
MODULES_OF_TYPING_CONSTRUCTS = {"typing", "typing_extensions", "collections.abc"}

SOURCE_AND_TESTS = ("packages/*/src", "tests")


class Site(NamedTuple):
    """One class written through its module: where it stands and what it should read."""

    line: int
    start: int
    end: int
    written: str
    binding: str
    module: str
    name: str


def _bindings(tree: AST) -> dict[str, str]:
    """Each name an `import` or `from x import module` gives a module, with that module's path."""
    bound: dict[str, str] = {}
    for found in ast.walk(tree):
        if isinstance(found, Import):
            for alias in found.names:
                if alias.asname:
                    bound[alias.asname] = alias.name
                else:
                    bound[alias.name.split(".")[0]] = alias.name.split(".")[0]
        elif isinstance(found, ImportFrom) and found.level == 0 and found.module:
            for alias in found.names:
                try:
                    imported = importlib.import_module(f"{found.module}.{alias.name}")
                except ImportError:
                    continue
                bound[alias.asname or alias.name] = imported.__name__
    return bound


def _is_class(module: ModuleType, name: str) -> bool:
    thing = getattr(module, name)
    if isinstance(thing, type):
        return True
    return module.__name__ in MODULES_OF_TYPING_CONSTRUCTS and name[0].isupper() and not name.isupper()


def _stays(module: str, name: str) -> bool:
    kept = CLASSES_THAT_STAY_THROUGH.get(module)
    if kept is not None and (kept == "*" or name in kept):
        return True
    return name[0].islower() and (module, name) not in LOWER_CASE_CLASSES_IMPORTED_AS


def sites_in(source: str, line_starts: list[int]) -> list[Site]:
    """Every class or exception in `source` written as `module.Name`, with its byte span.

    `line_starts` are the byte offsets at which each line of `source` begins.
    """
    tree = ast.parse(source)
    bound = _bindings(tree)
    inner = {id(found.value) for found in ast.walk(tree) if isinstance(found, Attribute)}
    found_sites: list[Site] = []
    for node in ast.walk(tree):
        if not isinstance(node, Attribute) or id(node) in inner:
            continue
        chain: list[Attribute] = []
        cursor: ast.expr = node
        while isinstance(cursor, Attribute):
            chain.append(cursor)
            cursor = cursor.value
        if not isinstance(cursor, Name) or cursor.id not in bound:
            continue
        chain.reverse()
        try:
            owner = importlib.import_module(bound[cursor.id])
        except ImportError:
            continue
        for step in chain:
            if isinstance(getattr(owner, step.attr, None), ModuleType):
                owner = getattr(owner, step.attr)
                continue
            if hasattr(owner, step.attr) and _is_class(owner, step.attr) and not _stays(owner.__name__, step.attr):
                found_sites.append(
                    Site(
                        line=step.lineno,
                        start=line_starts[step.lineno - 1] + step.col_offset,
                        end=line_starts[step.end_lineno - 1] + step.end_col_offset,
                        written=ast.unparse(step),
                        binding=cursor.id,
                        module=owner.__name__,
                        name=step.attr,
                    )
                )
            break
    return sorted(found_sites)


def line_starts_of(data: bytes) -> list[int]:
    """Byte offsets at which each line of `data` begins."""
    starts = [0]
    for index, byte in enumerate(data):
        if byte == 0x0A:
            starts.append(index + 1)
    return starts


def python_files(root: Path = ROOT) -> list[Path]:
    """Every library source file and every test file under `root`."""
    return sorted(path for pattern in SOURCE_AND_TESTS for path in root.glob(f"{pattern}/**/*.py"))


def classes_through_a_module(root: Path = ROOT) -> list[str]:
    """`file:line: written` for every class or exception written through its module."""
    found: list[str] = []
    for path in python_files(root):
        data = path.read_bytes()
        found += [
            f"{path.relative_to(root)}:{site.line}: {site.written}"
            for site in sites_in(data.decode("utf-8"), line_starts_of(data))
        ]
    return found


def test_a_class_is_imported_by_name_not_written_through_its_module() -> None:
    assert classes_through_a_module() == []
