"""Reads a run from JSON and writes a run as JSON.

Code outside this package imports from here only, and ``Run`` is its one caller: ``run.to_json()`` is
``run_to_json``, ``run.outputs_to_json()`` is ``outputs_to_json`` and ``Run.from_json(saved, graph)`` is
``run_from_json``. ``type.py`` has the shape of a saved run, with the JSON shown in each docstring. ``codec.py``
builds one graph's validator and serializer for that shape. ``read.py`` and ``write.py`` do the two directions.
"""

from conductor._internal.run.json.codec import codec_of
from conductor._internal.run.json.read import run_from_json
from conductor._internal.run.json.write import outputs_to_json, run_to_json
