"""A run: a graph's inputs and what every unit came to, and what a leg asks of it.

Code outside this package imports from here only. ``type.py`` has ``Run``; ``arguments.py`` what a call
receives: the node rule, read for one item or for an item nobody knows.
"""

from conductor._internal.run.type import (
    NotCalled,
    Run,
    check_run_limits,
    no_outcomes,
)
