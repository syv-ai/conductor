"""The inputs most of this package's nodes take: a value and the control a person sets it with."""

from typing import Annotated

from conductor import Param
from conductor.widgets import NumberInput, Switch, Textarea, TextInput

Number = Annotated[float, Param(widget=NumberInput())]
LongText = Annotated[str, Param(widget=Textarea())]
Line = Annotated[str, Param(widget=TextInput())]
Flag = Annotated[bool, Param(widget=Switch())]
