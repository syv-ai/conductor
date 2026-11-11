"""A run: a graph's inputs and what every unit came to, and what a leg asks of it.

Code outside this package imports from here only. ``type.py`` has ``Run``; ``call_inputs.py`` what a call
receives: the node rule, read for one item or for an item nobody knows.
"""

from conductor._internal.run.type import (
    NotCalled,
    Run,
    checked_max_items,
    no_outcomes,
    run_from_json,
)
