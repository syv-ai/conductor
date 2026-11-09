# ``Table[R]`` is a Polars frame to a type checker; ``R`` names the columns.
from typing import Any

import polars

type Table[Rows = Any] = polars.DataFrame
