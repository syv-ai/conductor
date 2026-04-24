"""Compound node types — structured regions that sit in the DAG as super-nodes."""

from conductor.compound.for_each import FOR_EACH, ForEachNode
from conductor.compound.protocol import CompoundNodeType, NodeExecutor, Region
from conductor.compound.subprocess import SUBPROCESS, SubprocessNode
from conductor.compound.while_loop import WHILE, WhileNode

__all__ = [
    "FOR_EACH",
    "ForEachNode",
    "WHILE",
    "WhileNode",
    "SUBPROCESS",
    "SubprocessNode",
    "CompoundNodeType",
    "NodeExecutor",
    "Region",
]
