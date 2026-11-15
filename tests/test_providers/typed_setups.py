"""The two setups a host writes, as the docs show them, for mypy to check: one with dict metadata, one with a typed
model. ``test_typed_setups.py`` runs mypy on this file. The lines marked ``type: ignore`` are mistakes mypy must
catch: if it stops catching one, the ignore is unused and mypy fails."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event

from conductor import NodeRegistry
from conductor_providers.fastapi import RunStore, Worker, run_router
from fastapi import FastAPI
from pydantic import BaseModel


@dataclass(frozen=True)
class User:
    id: str
    team_id: str


def current_user() -> User:
    return User("anna", "a")


def with_dict_metadata(store: RunStore, registry: NodeRegistry[None]) -> FastAPI:
    def run_metadata() -> dict[str, str]:
        user = current_user()
        return {"user": user.id}

    def own_runs() -> Callable[[dict[str, str]], bool]:
        user = current_user()
        return lambda metadata: metadata["user"] == user.id

    worker = Worker(store, nodes_for=lambda metadata: registry)
    app = FastAPI()
    app.include_router(run_router(worker, metadata=run_metadata, run_filter=own_runs))
    return app


class Member(BaseModel):
    user: str
    team: str


@dataclass(frozen=True)
class AppContext:
    team: str
    stop: Event


@dataclass(frozen=True)
class Other:
    pass


def with_a_typed_model(store: RunStore, registry: NodeRegistry[AppContext]) -> FastAPI:
    def run_metadata() -> Member:
        user = current_user()
        return Member(user=user.id, team=user.team_id)

    def team_runs() -> Callable[[Member], bool]:
        user = current_user()
        return lambda member: member.team == user.team_id

    worker = Worker(
        store,
        metadata_type=Member,
        nodes_for=lambda member: registry,
        context=lambda member, stop: AppContext(team=member.team, stop=stop),
    )
    app = FastAPI()
    app.include_router(run_router(worker, metadata=run_metadata, run_filter=team_runs))
    return app


def mistakes(store: RunStore, registry: NodeRegistry[AppContext]) -> None:
    Worker(  # type: ignore[misc]  # a context of another class than the registry's
        store,
        metadata_type=Member,
        nodes_for=lambda member: registry,
        context=lambda member, stop: Other(),
    )
    Worker(
        store,
        metadata_type=Member,
        nodes_for=lambda member: registry,
        context=lambda member, stop: AppContext(team=member.tenant, stop=stop),  # type: ignore[attr-defined]
    )
