"""``ConductorModel`` — the base of every record a host saves or sends.

A graph a host stores, a problem it shows, an input it renders, the cause
of a failure it reports: each crosses into JSON, so each is a frozen
pydantic model on this one base.

What a host saves reads back what it wrote — a ``Graph`` and its parts, a
``Problem``, an ``ErrorCause``, a ``Policy``, an ``Index``::

    graph = Graph.from_path("approval.yaml")
    graph.to_yaml()
    graph.model_dump_json()          # JSON is pydantic's own, not renamed

What describes a node — ``NodeDescription``, ``VersionDescription``,
``Input``, ``Output`` and the widgets — is written for an editor. It is
built from a ``run`` signature, so its ``dtype`` dumps as the type's
description and reads back as that description, not the class; the rest
of the record — the title, the widget — reads back as written. The class
is still there to ``describe()``.

What compile and the engine build on every call — ``CompiledGraph`` and
its node and field views, a version, an interface, the ledger's records —
is not a model: those are frozen dataclasses, built far more often than
they are ever parsed. A node's return declaration is not one either: a
``run`` that returns a model returns one value, not one output per field.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict


class ConductorModel(BaseModel):
    """A frozen record that writes itself as YAML, JSON or a file, and reads back what a host saves.

    JSON is pydantic's own ``model_dump_json`` / ``model_validate_json``.
    Beside it: ``to_yaml`` and ``from_yaml``, and ``to_path`` and
    ``from_path``, where the suffix decides — ``.json`` is JSON, ``.yaml``
    and ``.yml`` are YAML, and any other suffix, ``.JSON`` included, is
    refused rather than guessed. A record that describes a node writes and
    does not read back (the module docstring says why).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def __repr_args__(self) -> Iterator[tuple[str | None, Any]]:
        """Pydantic's repr arguments without the fields at their default, as scikit-learn prints an estimator.

        ``Input(name='text', dtype=Text, title='Text', widget=Textarea())``
        rather than every ``None`` and ``True`` the record carries.
        """
        fields = type(self).model_fields
        for name, value in super().__repr_args__():
            field = fields.get(name) if name is not None else None
            if field is not None and not field.is_required() and value == field.get_default(call_default_factory=True):
                continue
            yield name, value

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        return cls.model_validate(yaml.safe_load(text))

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
