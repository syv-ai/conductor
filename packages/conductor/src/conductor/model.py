"""``ConductorModel`` — the base of every record a host saves or sends.

A graph a host stores, a problem it shows, an input it renders, the cause
of a failure it reports: each crosses into JSON, so each is a frozen
pydantic model on this one base, and reads and writes itself::

    graph = Graph.from_path("approval.yaml")
    graph.to_yaml()
    graph.model_dump_json()          # JSON is pydantic's own, not renamed

What compile and the engine build on every call — ``CompiledGraph`` and
its node and field views, a version, an interface, the ledger's records —
is not a model: those are frozen dataclasses, built far more often than
they are ever parsed. A node's return declaration is not one either: a
``run`` that returns a model returns one value, not one output per field.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Self

from pydantic import BaseModel, ConfigDict


def _yaml() -> ModuleType:
    """PyYAML, which only the YAML methods need; missing, it names the extra that brings it."""
    try:
        import yaml
    except ImportError as missing:
        raise ImportError("reading or writing YAML needs PyYAML: uv add 'syv-conductor[yaml]'") from missing
    return yaml


class ConductorModel(BaseModel):
    """A frozen record that saves and loads itself as YAML, JSON or a file.

    JSON is pydantic's own ``model_dump_json`` / ``model_validate_json``.
    Beside it: ``to_yaml`` and ``from_yaml``, and ``to_path`` and
    ``from_path``, where the suffix decides — ``.json`` is JSON, ``.yaml``
    and ``.yml`` are YAML, and any other suffix, ``.JSON`` included, is
    refused rather than guessed.
    """

    model_config = ConfigDict(frozen=True)

    def to_yaml(self) -> str:
        return _yaml().safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        return cls.model_validate(_yaml().safe_load(text))

    def to_path(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix == ".json":
            text = self.model_dump_json(indent=2)
        elif path.suffix in (".yaml", ".yml"):
            text = self.to_yaml()
        else:
            raise ValueError(f"{path.name}: a record is saved as .json, .yaml or .yml")
        path.write_text(text, encoding="utf-8")

    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        path = Path(path)
        if path.suffix == ".json":
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        if path.suffix in (".yaml", ".yml"):
            return cls.from_yaml(path.read_text(encoding="utf-8"))
        raise ValueError(f"{path.name}: a record is read from .json, .yaml or .yml")
