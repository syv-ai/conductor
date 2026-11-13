"""Reads a run from JSON and writes a run as JSON.

Code outside this package imports from here only, and ``Run`` is its one caller: ``run.to_json()`` is
``run_to_json``, ``run.outputs_to_json()`` is ``outputs_to_json`` and ``Run.from_json(saved, graph)`` is
``run_from_json``. ``type.py`` has the shape of a saved run, with the JSON shown in each docstring. ``codec.py``
makes, for one graph, the subclass of that shape whose keys and value types are the graph's own: the graph's
codec, which reads and writes the JSON (``codec_of``, which ``Graph`` keeps). ``read.py`` and ``write.py`` do the two directions.
"""

from .codec import codec_of
from .read import run_from_json
from .type import SavedRun
from .write import outputs_to_json, run_to_json
