"""Classes and exceptions are imported by name, not written through their module.

So the code says `PydanticSerializationError`, after `from pydantic_core import PydanticSerializationError`,
rather than `pydantic_core.PydanticSerializationError`. Functions and constants may still be written through their
module: `json.dumps`, `time.monotonic()` and `re.MULTILINE` are fine.

The tables below list the exceptions to the rule.
"""

from __future__ import annotations

import ast
import importlib
from ast import AST, Attribute, Import, ImportFrom, Name
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]

# Modules whose classes are still written through the module. polars is `pl.` everywhere, since a bare `Series`
# is conductor's own class. And a bare `time` or `datetime` looks like the module.
CLASSES_THAT_STAY_THROUGH = {
    "polars": "*",
    "pandas": "*",
    "datetime": {"date", "time", "datetime"},
}

# A class with a lower-case name looks like a function (`functools.partial`, `weakref.ref`). So it may be written
# through its module, unless it is listed here with the name to import it under.
LOWER_CASE_CLASSES_IMPORTED_AS = {
    ("regex", "error"): "RegexError",
}

# Classes whose bare name clashes, in a file or between two modules. Each is imported under the name given here,
# which says where it comes from.
NAME_WHEN_BARE_IS_TAKEN = {
    ("asyncio", "Event"): "AsyncioEvent",
    ("threading", "Event"): "ThreadEvent",
    ("dataclasses", "Field"): "DataclassField",
}

# Not every `typing` construct is a `type` object (`Optional`, `Callable`). So in these modules, a name that starts
# with a capital and isn't all capitals counts as a class.
MODULES_OF_TYPING_CONSTRUCTS = {"typing", "typing_extensions", "collections.abc"}

SOURCE_AND_TESTS = ("packages/*/src", "tests")


class Site(NamedTuple):
    """One place where a class is written through its module, such as `pydantic_core.PydanticSerializationError`.

    Attributes:
        line: The line it is on.
        start: The byte offset where it starts in the file.
        end: The byte offset where it ends.
        written: The text as written, such as `pydantic_core.PydanticSerializationError`.
        binding: The name the module is bound to in the file, such as `pydantic_core`.
        module: The module the class is defined in.
        name: The class's name, such as `PydanticSerializationError`.
    """

    line: int
    start: int
    end: int
    written: str
    binding: str
    module: str
    name: str


def _bindings(tree: AST) -> dict[str, str]:
    """Returns each name in the file that is bound to a module, with the module's full name.

    A name is bound to a module by `import x`, `import x as y`, or `from x import module`.
    """
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
    return (
        module.__name__ in MODULES_OF_TYPING_CONSTRUCTS and name[0].isupper() and not name.isupper()
    )


def _stays(module: str, name: str) -> bool:
    kept = CLASSES_THAT_STAY_THROUGH.get(module)
    if kept is not None and (kept == "*" or name in kept):
        return True
    return name[0].islower() and (module, name) not in LOWER_CASE_CLASSES_IMPORTED_AS


def sites_in(source: str, line_starts: list[int]) -> list[Site]:
    """Finds every class or exception in `source` written as `module.Name`.

    Args:
        source: The text of one Python file.
        line_starts: The byte offset where each line of `source` starts.

    Returns:
        One `Site` per place, sorted.
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
            if (
                hasattr(owner, step.attr)
                and _is_class(owner, step.attr)
                and not _stays(owner.__name__, step.attr)
            ):
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
    """Returns the byte offset where each line of `data` starts."""
    starts = [0]
    for index, byte in enumerate(data):
        if byte == 0x0A:
            starts.append(index + 1)
    return starts


def python_files(root: Path = ROOT) -> list[Path]:
    """Returns every Python file of the packages' source and the tests under `root`."""
    return sorted(path for pattern in SOURCE_AND_TESTS for path in root.glob(f"{pattern}/**/*.py"))


def classes_through_a_module(root: Path = ROOT) -> list[str]:
    """Lists every class or exception written through its module, as `file:line: written`."""
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
