"""Running a graph: calling its nodes, storing what each call gave, and going on later.

A ``Run`` is one run of a compiled graph. It has the graph inputs the caller gave and an outcome for every call
made so far. ``run.resume()`` calls every node that is ready, and stops when the run is complete, waits for a
person, fails, or hits a limit. A run can be saved with ``run.to_json()`` and read back with ``Run.from_json``,
even in another process, and it goes on from there.

The files, roughly in the order a run uses them:

- ``type.py``: ``Run`` itself, and ``make_run``, which ``graph.with_inputs`` calls.
- ``blueprint.py``: the function nodes a run calls, with what each call reads. It is built once per graph.
- ``outcome.py``: the outcome of one call, such as ``Done`` or ``Unanswered``.
- ``arguments.py``: works out the arguments of one call from the outcomes so far.
- ``scheduler.py``: calls every node that is ready, for ``run.resume()`` and ``run.stream()``.
- ``call.py``: one call of a node's function.
- ``stream.py``, ``events.py`` and ``node_events.py``: ``run.stream()`` and the events it yields.
- ``waiting.py``: ``Waiting``, a call waiting for a person, as a host is shown it.
- ``results.py``: ``run.results`` and ``run.outputs``.
- ``progress.py``: ``run.progress``, how far a run is.
- ``state.py``: ``run.status``.
- ``reuse.py``: reusing outcomes from a previous run, for ``run.again()``.
- ``validator.py``: ``EdgeValidator``, which checks a value an edge gives an input that is stricter than the
  output it is wired to; ``WaitingIdValidator``, which reads the id of a waiting call a host answers or declines;
  and ``ForgetValidator``, which checks the node ids a host gives ``run.with_forgotten``.
- ``json/``: saving a run as JSON and reading it back.
"""

from .blueprint import RunBlueprint
from .events import (
    Event,
    NodeComplete,
    NodeFailed,
    NodeProgress,
    NodeRetry,
    NodeStart,
    NodeWaiting,
    RunCheckpoint,
    RunEnd,
    RunStart,
)
from .json import SavedRun, codec_of
from .progress import CallCount, RunProgress
from .state import RunStatus
from .type import (
    NoneInputCall,
    Run,
    make_run,
)
from .waiting import Waiting

__all__ = [
    "CallCount",
    "Event",
    "NodeComplete",
    "NodeFailed",
    "NodeProgress",
    "NodeRetry",
    "NodeStart",
    "NodeWaiting",
    "NoneInputCall",
    "Run",
    "RunBlueprint",
    "RunCheckpoint",
    "RunEnd",
    "RunProgress",
    "RunStart",
    "RunStatus",
    "SavedRun",
    "Waiting",
    "codec_of",
    "make_run",
]
