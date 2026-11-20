"""Saving a run as JSON and reading it back.

``run.to_json()`` calls ``run_to_json``, ``run.outputs_to_json()`` calls ``outputs_to_json``, and
``Run.from_json(saved, graph)`` calls ``run_from_json``. Code outside this package imports from here.

- ``type.py``: ``SavedRun`` and the classes inside it, the shape of a saved run. Each docstring shows its JSON.
- ``codec.py``: the graph's codec. For one graph, it makes a subclass of ``SavedRun`` that knows the graph's
  node names and value types, and checks a saved run against them. The ``Graph`` keeps it.
- ``write.py``: from a run to JSON, and a question a person declined, which its problem keeps.
- ``read.py``: from JSON back to a run.
"""

from .codec import codec_of
from .read import run_from_json
from .type import SavedRun
from .write import outputs_to_json, question_to_json, run_to_json

__all__ = [
    "SavedRun",
    "codec_of",
    "outputs_to_json",
    "question_to_json",
    "run_from_json",
    "run_to_json",
]
