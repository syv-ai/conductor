"""Running a graph: calling its nodes, storing what each call gave, and going on later.

A ``Run`` is one run of a compiled graph. It has the graph inputs the caller gave and an outcome for every call
made so far. ``run.resume()`` calls every node that is ready, and stops when the run is complete, waits for a
person, fails, or hits a limit. A run can be saved with ``run.to_json()`` and read back with ``Run.from_json``,
even in another process, and it goes on from there.

The files, roughly in the order a run uses them:

- ``type.py``: ``Run`` itself, and ``make_run``, which ``graph.with_inputs`` calls.
- ``blueprint.py``: the function nodes a run calls, with what each call reads. It is built once per graph.
- ``outcome.py``: the outcome of one call, such as ``Done`` or ``Unanswered``.
- ``outcomes.py``: ``RunOutcomes``, the outcome of every call so far, and the arguments of a call worked out from
  them.
- ``scheduler.py``: calls every node that is ready, for ``run.resume()`` and ``run.stream()``.
- ``call.py``: one call of a node's function.
- ``stream.py``, ``events.py`` and ``node_events.py``: ``run.stream()`` and the events it yields.
- ``waiting.py``: ``Question``, a call that asked as a person is shown it, and ``Waiting``, one still waiting, with the id to answer it by.
- ``asked.py``: ``Asked``, a call a person answered or declined, with their reply, for ``run.asked``.
- ``results.py``: ``run.results`` and ``run.outputs``.
- ``progress.py``: ``run.progress``, how far a run is.
- ``state.py``: ``run.status``.
- ``reuse.py``: reusing outcomes from a previous run, for ``run.again()``.
- ``validator.py``: ``EdgeValidator``, which checks a value an edge gives an input that is stricter than the
  output it is wired to; ``WaitingIdValidator``, which reads the id of a waiting call a host answers or declines;
  and ``ForgetValidator``, which checks the node ids a host gives ``run.with_forgotten``.
- ``json/``: saving a run as JSON and reading it back.
"""

from .asked import Asked
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
from .validator import WaitingIdValidator
from .waiting import Question, Waiting

__all__ = [
    "Asked",
    "CallCount",
    "Event",
    "NodeComplete",
    "NodeFailed",
    "NodeProgress",
    "NodeRetry",
    "NodeStart",
    "NodeWaiting",
    "NoneInputCall",
    "Question",
    "Run",
    "RunBlueprint",
    "RunCheckpoint",
    "RunEnd",
    "RunProgress",
    "RunStart",
    "RunStatus",
    "SavedRun",
    "Waiting",
    "WaitingIdValidator",
    "codec_of",
    "make_run",
]
