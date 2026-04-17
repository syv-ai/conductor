# Conductor — Design Specification

**Status:** Proposal
**Authors:** Rasmus Krebs, Claude (AI assistant)
**Date:** 2026-04-01
**Depends on:** [Graph Processing Framework Research](./graph-processing-framework-research.md)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Design Principles](#2-design-principles)
3. [Architecture Overview](#3-architecture-overview)
4. [Package Structure](#4-package-structure)
5. [Layer 1: Types, Widgets, and Metadata](#5-layer-1-types-widgets-and-metadata)
6. [Layer 2: Node Registry](#6-layer-2-node-registry)
7. [Layer 3: Graph Compilation](#7-layer-3-graph-compilation)
8. [Layer 4: Execution Engine](#8-layer-4-execution-engine)
9. [Layer 5: Compound Nodes](#9-layer-5-compound-nodes)
10. [Host Application Integration](#10-host-application-integration)
11. [Migration Path](#11-migration-path)
12. [Appendix: Research Summary](#12-appendix-research-summary)

---

## 1. Problem Statement

### Current State

The AKA Flows executor (`backend/app/services/flows/execution/executor.py`) is ~1,000 lines and growing. It has two core problems:

1. **The executor knows too much.** It special-cases for-each loop regions, app node sub-flows, and skip propagation inline. Every new control-flow pattern (while-loop, try/catch, parallel-split) requires modifying the executor loop itself. This makes the executor fragile and hard to contribute to.

2. **Two duplicated execution paths.** `_execute_flow_impl` (sync) and `_execute_flow_streaming_impl` (async) duplicate the same logic with minor variations for event emission. Bug fixes must be applied in both places.

Additionally, the execution engine is tightly coupled to AKA Flows' domain (user IDs, RBAC, app nodes), making it impossible to extract and reuse in other projects.

### What Works Well

The **node registration system** is excellent and must be preserved exactly:

- `@register_node` decorator converts plain Python functions into executable, validated, UI-renderable nodes.
- `Annotated[T, widgets.Widget(...)]` type hints are the single source of truth for backend logic, Pydantic validation, AND frontend widget rendering.
- The registry serializes to JSON, which the frontend reads to render node palettes, input forms, and output handles — making it trivial to add new nodes.
- Node functions remain plain, testable Python with no framework coupling.

This pattern is the crown jewel and the primary reason the framework is worth extracting.

### Goals

1. Extract a standalone, reusable graph execution library with zero application-specific dependencies.
2. Preserve the `@register_node` + `Annotated` + widgets pattern exactly as-is.
3. Eliminate the sync/streaming code duplication.
4. Make control-flow extensible without modifying the executor.
5. Establish a uniform execution contract (single DTO) for all node types.

### Non-Goals

- Distributed execution (single-process is sufficient for our scale).
- BSP/superstep execution model (topological ordering is simpler and sufficient for narrow DAGs).
- Shared-state model (edge-based data flow matches the visual wire-drawing paradigm).
- Cycle support (region-based loops cover current needs; true cycles add complexity without clear demand).

---

## 2. Design Principles

### 2.1 The Registry Is the API

A Python function with type hints and widget annotations is the single source of truth for:
- Backend execution logic
- Input validation rules (via auto-generated Pydantic model)
- Frontend UI rendering (via JSON-serialized registry)

No configuration files, no separate schema definitions, no manual sync between backend and frontend.

### 2.2 Edge-Based Data Flow

Data moves along explicit connections between node output handles and input handles. This matches the ReactFlow visual programming model where users draw wires between ports. Contrast with LangGraph's shared-state model, which doesn't map to a visual wire metaphor.

### 2.3 Compile Then Execute

Graph construction and execution are separate phases (adopted from Beam, LangGraph, Ray). The `compile()` step validates structure, discovers compound regions, and produces an immutable execution plan. All structural errors surface before any node runs.

### 2.4 Uniform Execution Contract

Every node — simple, compound, or extension — receives the same `NodeExecRequest` DTO. The executor dispatches without type-checking node kinds. New control-flow patterns are compound nodes, not executor modifications.

### 2.5 One Execution Path

Streaming (async generator yielding events) is the only execution path. Synchronous execution is a thin wrapper that collects events. This eliminates the current ~500 lines of duplication.

### 2.6 Host-Agnostic Core

The engine has no opinions about authentication, persistence, web frameworks, or application-specific node types. The host application integrates via:
- An `ExtensionResolver` protocol for custom node types (e.g., app nodes / sub-flows)
- An opaque `context: dict[str, Any]` on the run state for host-specific data

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                       HOST APPLICATION                              │
│                    (AKA Flows, or any other)                        │
│                                                                     │
│  ┌─────────────┐  ┌──────────────────┐  ┌────────────────────────┐ │
│  │ Web routes   │  │ App node resolver│  │ Node function files    │ │
│  │ (FastAPI)    │  │ (ExtensionRes.)  │  │ (@registry.node)       │ │
│  └──────┬───── ┘  └────────┬─────────┘  └────────┬───────────────┘ │
└─────────┼──────────────────┼─────────────────────┼─────────────────┘
          │                  │                     │
          ▼                  ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FLOWENGINE LIBRARY                           │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  LAYER 2: REGISTRY                                           │   │
│  │  NodeRegistry · @node decorator · to_json() serialization    │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │  LAYER 1: TYPES + WIDGETS + METADATA                   │  │   │
│  │  │  WidgetType · ResultFormat · widgets.* · InputMetadata  │  │   │
│  │  │  OutputMetadata · Pydantic validation model generation  │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌───────────────────┐    ┌─────────────────────────────────────┐   │
│  │  LAYER 3: GRAPH   │    │  LAYER 4: EXECUTION ENGINE          │   │
│  │  GraphNode/Edge   │───▶│  compile() → CompiledGraph          │   │
│  │  Topological sort │    │  execute() → AsyncGenerator[Event]  │   │
│  │  Region discovery │    │  FlowRunState · NodeExecRequest     │   │
│  └───────────────────┘    │  InputResolver · EventSink          │   │
│                           └──────────────┬──────────────────────┘   │
│                                          │                          │
│  ┌───────────────────────────────────────┴──────────────────────┐   │
│  │  LAYER 5: COMPOUND NODES                                     │   │
│  │  CompoundNodeType protocol · ForEachNode · (future: TryCatch)│   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Package Structure

```
conductor/
├── __init__.py                    # Public API surface
│
├── types.py                       # Core enums, type aliases, custom types
├── widgets.py                     # Widget functions (Annotated metadata)
├── metadata.py                    # Frozen dataclasses: InputMetadata, OutputMetadata, NodeMetadata
├── validation.py                  # Pydantic model generation from function signatures
├── errors.py                      # Exception hierarchy
│
├── registry/
│   ├── __init__.py                # NodeRegistry class, @node decorator
│   ├── definition.py              # NodeDefinition frozen dataclass
│   └── schema.py                  # JSON serialization for frontend consumption
│
├── graph/
│   ├── __init__.py
│   ├── model.py                   # GraphNode, GraphEdge (the wire format)
│   ├── compiler.py                # compile() → CompiledGraph
│   ├── topology.py                # Topological sort, cycle detection
│   └── regions.py                 # Region discovery for compound nodes
│
├── execution/
│   ├── __init__.py
│   ├── engine.py                  # execute() async generator + _dispatch_node
│   ├── state.py                   # FlowRunState (shared mutable run state)
│   ├── request.py                 # NodeExecRequest DTO
│   ├── resolver.py                # Input resolution from edges + static data
│   ├── results.py                 # normalize_result, extract_output, filter_skipped
│   ├── events.py                  # Event TypedDicts + thread-safe EventSink
│   └── skip.py                    # SKIPPED sentinel + should_skip_node logic
│
└── compound/
    ├── __init__.py
    ├── protocol.py                # CompoundNodeType, Region, NodeExecutor protocol
    └── for_each.py                # ForEachNode implementation
```

**Dependency rule:** No module in `conductor` imports from the host application. No FastAPI, SQLModel, or app-specific imports.

---

## 5. Layer 1: Types, Widgets, and Metadata

These modules are carried over from the current codebase with minimal changes. They are already clean, framework-agnostic, and well-designed.

### 5.1 Types (`types.py`)

```python
class WidgetType(str, Enum):
    """All widget types for node parameters. Maps 1:1 to frontend components."""
    TEXT = "text"
    TEXTAREA = "textarea"
    DROPDOWN = "dropdown"
    DEPENDENT_DROPDOWN = "dependent-dropdown"
    RANGE = "range"
    CHECKBOX = "checkbox"
    FILE = "file"
    SCHEMA_BUILDER = "schema-builder"
    DATEPICKER = "datepicker"
    NUMBER = "number"
    SWITCH = "switch"
    CONNECTION_LIST = "connection-list"
    TEMPLATE_TEXTAREA = "template-textarea"
    IF_ELSE_BUILDER = "if-else-builder"
    MULTISELECT = "multiselect"
    ENTITY_DROPDOWN = "entity-dropdown"
    TABLE_INPUT = "table-input"
    COLUMN_SELECT = "column-select"
    OUTPUT = "output"

class ResultFormat(str, Enum):
    """How node results are wrapped in the container format."""
    SINGLE = "single"       # {result: value}
    MULTI = "multi"         # {output_1: v1, output_2: v2}
    DICT_SPREAD = "dict"    # {result: dict, **dict}

RESULT_KEY = "result"
OUTPUT_PREFIX = "output_"
NodeResult = dict[str, Any]
```

Custom type aliases (`Base64Str`, `Date`, `Table`, etc.) and type label mappings are unchanged.

### 5.2 Widgets (`widgets.py`)

Widget functions return Pydantic `Field` objects with `json_schema_extra` metadata. The pattern is identical to today:

```python
def Text(label: str, description: str, disable_handle: bool = False, **kwargs) -> Any:
    json_schema_extra = {"widget": WidgetType.TEXT.value, "label": label, ...}
    return Field(description=description, json_schema_extra=json_schema_extra, **kwargs)

def ConnectionList(label: str, description: str, **kwargs) -> Any: ...
def Dropdown(label: str, description: str, choices: list[str], **kwargs) -> Any: ...
def Output(label: str, description: str | None = None, download: bool = False) -> dict: ...
# ... all existing widgets carry over unchanged
```

**Extensibility:** Host applications can define additional widget functions following the same pattern. The frontend must have a corresponding component for any new widget type.

### 5.3 Metadata (`metadata.py`)

Frozen dataclasses computed once at registration time:

```python
@dataclass(frozen=True)
class InputMetadata:
    name: str               # Parameter name (matches function signature)
    type_str: str           # "str", "int", "list[str]"
    type_label: str         # "Tekst", "Heltal", "Liste af Tekst"
    label: str              # Display label from widget
    description: str | None
    widget: WidgetType
    default: Any
    optional: bool
    expects_list: bool           # Pre-computed for multi-edge handling
    uses_connection_list: bool   # Pre-computed for input resolution
    disable_handle: bool
    widget_config: dict[str, Any]

@dataclass(frozen=True)
class OutputMetadata:
    name: str               # "result" or "output_N"
    type_str: str
    type_label: str
    label: str
    description: str | None
    optional: bool
    download: bool
    filename: str | None

@dataclass(frozen=True)
class NodeMetadata:
    id: str
    name: str
    description: str
    tags: tuple[str, ...]
    inputs: tuple[InputMetadata, ...]
    outputs: tuple[OutputMetadata, ...]
    result_format: ResultFormat
    validation_model: type | None    # Pydantic model for input validation
    width: int | None
```

### 5.4 Validation (`validation.py`)

`create_validation_model(func)` generates a Pydantic model from a function signature. `_extract_type_string(annotation)` converts Python type annotations to JSON-serializable strings for the frontend. Both are unchanged.

---

## 6. Layer 2: Node Registry

### 6.1 NodeDefinition

```python
@dataclass(frozen=True)
class NodeDefinition:
    id: str                                     # "echo@2" (full versioned ID)
    base_id: str                                # "echo"
    version: int                                # 2
    name: str                                   # Display name for UI
    description: str
    tags: tuple[str, ...]
    inputs: tuple[InputMetadata, ...]           # Pre-computed at registration
    outputs: tuple[OutputMetadata, ...]         # Pre-computed at registration
    result_format: ResultFormat
    validation_model: type[BaseModel] | None    # Pydantic model for input validation
    func: Callable[..., Any] | None             # The RAW function (not wrapped)
    width: int | None
    docs: str | None                            # Markdown documentation
```

### 6.2 NodeRegistry

```python
class NodeRegistry:
    """Versioned registry. Nodes identified as base_id@version."""

    def node(
        self,
        base_id: str,
        *,
        version: int = 1,
        name: str,
        description: str,
        tags: list[str] | None = None,
        width: int | None = None,
        docs: str | None = None,
    ) -> Callable:
        """Decorator to register a function as a node."""
        ...

    def get(self, full_id: str) -> NodeDefinition | None: ...
    def get_latest(self, base_id: str) -> NodeDefinition | None: ...
    def is_deprecated(self, full_id: str) -> bool: ...
    def all(self) -> list[NodeDefinition]: ...
    def all_current(self) -> list[NodeDefinition]: ...

    def to_json(self) -> list[dict[str, Any]]:
        """Serialize all current nodes for frontend consumption."""
        ...
```

### 6.3 Key Change: Raw Function Storage

**Current behavior:** The `@register_node` decorator wraps the function in a validation + exception-catching closure. The wrapper is stored in the registry and called at execution time.

**New behavior:** The decorator stores the **raw function** on `NodeDefinition.func`. Validation and exception wrapping move to the execution engine's `_dispatch_node()`. This means:

- Node functions remain plain, testable Python: `assert greeting("World", False) == "Hello World"`
- No hidden wrapper in the call stack during debugging.
- Validation is still performed (using the pre-built Pydantic model), just at a different layer.

### 6.4 Frontend Serialization (`schema.py`)

Replaces the current `introspection.py`. Produces the same JSON structure the frontend already consumes:

```python
def serialize_node_definition(nd: NodeDefinition, registry: NodeRegistry) -> dict:
    """Convert a NodeDefinition to the frontend JSON schema.

    Output matches the current NodePublic Pydantic model structure.
    """
    return {
        "id": nd.id,
        "base_id": nd.base_id,
        "version": nd.version,
        "name": nd.name,
        "description": nd.description,
        "tags": list(nd.tags),
        "inputs": [_serialize_input(inp) for inp in nd.inputs],
        "outputs": [_serialize_output(out) for out in nd.outputs],
        "width": nd.width,
        "deprecated": registry.is_deprecated(nd.id),
        "latest_version": (registry.get_latest(nd.base_id) or nd).version,
        "docs": nd.docs,
    }

def _serialize_input(inp: InputMetadata) -> dict:
    """Serialize an input parameter with full widget configuration."""
    data = {
        "name": inp.name,
        "type": inp.type_str,
        "type_label": inp.type_label,
        "label": inp.label,
        "description": inp.description,
        "widget": inp.widget.value,
        "default": inp.default,
        "optional": inp.optional,
        "disable_handle": inp.disable_handle,
    }
    data.update(inp.widget_config)  # Merge choices, range_min, etc.
    return data
```

---

## 7. Layer 3: Graph Compilation

### 7.1 Graph Model (Wire Format)

These are the structures sent by the frontend (matching ReactFlow's data model):

```python
@dataclass(frozen=True)
class GraphNode:
    id: str                          # Unique instance ID (UUID from ReactFlow)
    type: str                        # Registry type ("echo@2") or extension type ("app:uuid@1")
    data: dict[str, Any] | None      # Static configuration from UI widgets

@dataclass(frozen=True)
class GraphEdge:
    id: str
    source: str                      # Source node instance ID
    target: str                      # Target node instance ID
    source_handle: str | None        # Output handle ("result", "output_1")
    target_handle: str | None        # Input handle ("text", "items")
```

### 7.2 CompiledGraph

```python
@dataclass(frozen=True)
class CompiledGraph:
    """Immutable, validated, ready-to-execute graph."""

    execution_order: tuple[str, ...]
    edge_map: dict[tuple[str, str], tuple[tuple[str, str], ...]]  # (target, handle) → sources
    node_map: dict[str, GraphNode]
    compound_nodes: dict[str, NodeExecutor]       # start_id → executor instance
    managed_ids: frozenset[str]                    # Body/end nodes managed by compound nodes
    registry: NodeRegistry
    extension_resolver: ExtensionResolver | None
```

### 7.3 The `compile()` Function

```python
def compile(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    registry: NodeRegistry,
    compound_types: list[CompoundNodeType] | None = None,
    extension_resolver: ExtensionResolver | None = None,
) -> CompiledGraph:
    """Validate and compile a graph into an immutable execution plan.

    Steps:
        1. Validate all node types exist in registry or extension resolver.
        2. Validate all edge endpoints reference existing nodes.
        3. Topological sort (raises CycleDetectionError on cycles).
        4. For each registered CompoundNodeType, run its discover() function
           to find regions and instantiate compound node executors.
        5. Freeze everything into a CompiledGraph.

    Raises:
        CompilationError: On any structural validation failure.
    """
```

This replaces the validation + setup code currently duplicated at the top of both `_execute_flow_impl` and `_execute_flow_streaming_impl`.

### 7.4 Extension Resolver Protocol

The hook for host-app-specific node types:

```python
class ExtensionResolver(Protocol):
    """Implemented by host applications for custom node types."""

    def is_known_type(self, node_type: str) -> bool:
        """Return True if this resolver handles the given node type."""
        ...

    def create_executor(self, node_type: str) -> NodeExecutor:
        """Create an executor instance for the given node type."""
        ...
```

---

## 8. Layer 4: Execution Engine

### 8.1 NodeExecRequest (The Universal DTO)

```python
@dataclass(frozen=True)
class NodeExecRequest:
    """Passed to every node execution — simple, compound, or extension."""

    node_id: str                     # Instance ID (UUID from frontend)
    node_type: str                   # Registry type ("echo@2")
    inputs: dict[str, Any]           # Resolved inputs (edges + static data)
    data: dict[str, Any]             # Raw node.data from UI configuration
    state: FlowRunState              # Shared mutable run state
```

### 8.2 NodeExecutor Protocol

```python
class NodeExecutor(Protocol):
    """The contract every executable unit implements."""

    def execute(self, req: NodeExecRequest) -> Any:
        """Execute the node and return a raw result (not yet normalized)."""
        ...
```

### 8.3 FlowRunState

```python
@dataclass
class FlowRunState:
    """Mutable state for a single flow execution run."""

    # Graph structure (immutable references)
    compiled: CompiledGraph
    resolver: InputResolver

    # Mutable execution state
    results: dict[str, NodeResult]

    # Event emission
    _event_sink: EventSink

    # Control
    _cancelled: threading.Event
    _started_at: float
    _timeout_seconds: int

    # Host-app context (opaque to the engine)
    context: dict[str, Any]

    def emit(self, event: ExecutionEvent) -> None:
        """Thread-safe event emission. Used by compound nodes."""
        self._event_sink.push(event)

    def is_cancelled(self) -> bool: ...
    def is_timed_out(self) -> bool: ...
    def remaining_seconds(self) -> float | None: ...

    def execute_subgraph(
        self,
        node_ids: list[str],
        overlay: dict[str, NodeResult] | None = None,
    ) -> dict[str, NodeResult]:
        """Execute a subset of nodes with result isolation.

        Creates a snapshot of parent results, applies overlay, executes
        the given nodes in order, and returns the local results dict.
        The parent results are NOT modified.

        This is the method compound nodes call to run their body.
        """
        ...
```

**Why `context: dict[str, Any]` instead of typed fields?**

The current `ExecutionContext` has AKA-Flows-specific fields: `user_id`, `flow_id`, `parent_app_ids`, `app_node_cache`, `user_team_ids`, `user_role`. These are meaningless to other projects using the engine. By making context opaque, the engine stays host-agnostic while the host app stores whatever it needs.

### 8.4 Event Types

```python
# All events are TypedDicts for zero-overhead serialization.

class NodeStartEvent(TypedDict):
    type: Literal["node_start"]
    node_id: str

class NodeCompleteEvent(TypedDict, total=False):
    type: Literal["node_complete"]
    node_id: str
    result: Any
    cached: bool

class NodeSkippedEvent(TypedDict):
    type: Literal["node_skipped"]
    node_id: str

class NodeErrorEvent(TypedDict):
    type: Literal["node_error"]
    node_id: str
    error: str
    is_validation: bool

class NodeProgressEvent(TypedDict):
    type: Literal["node_progress"]
    node_id: str
    current: int
    total: int

class FlowCompleteEvent(TypedDict):
    type: Literal["flow_complete"]
    results: dict[str, Any]

class FlowErrorEvent(TypedDict):
    type: Literal["flow_error"]
    error: str
    is_validation: bool

class FlowCancelledEvent(TypedDict):
    type: Literal["flow_cancelled"]
    completed_nodes: list[str]

class FlowTimeoutEvent(TypedDict):
    type: Literal["flow_timeout"]
    completed_nodes: list[str]
    elapsed_seconds: float
    timeout_seconds: int

ExecutionEvent = (
    NodeStartEvent | NodeCompleteEvent | NodeSkippedEvent | NodeErrorEvent
    | NodeProgressEvent | FlowCompleteEvent | FlowErrorEvent
    | FlowCancelledEvent | FlowTimeoutEvent
)
```

### 8.5 EventSink

Thread-safe bridge between sync compound node execution and the async generator:

```python
class EventSink:
    """Compound nodes push events here; the engine drains them."""

    def __init__(self):
        self._queue: deque[ExecutionEvent] = deque()
        self._lock: threading.Lock = threading.Lock()

    def push(self, event: ExecutionEvent) -> None:
        with self._lock:
            self._queue.append(event)

    def pop(self) -> ExecutionEvent | None:
        with self._lock:
            return self._queue.popleft() if self._queue else None
```

### 8.6 The Executor

The complete engine — one async generator, one dispatch function:

```python
async def execute(
    compiled: CompiledGraph,
    *,
    timeout_seconds: int = 300,
    context: dict[str, Any] | None = None,
    cache: dict[str, Any] | None = None,
) -> AsyncGenerator[ExecutionEvent, None]:
    """Execute a compiled graph, yielding streaming events.

    This is the ONLY execution entry point.
    """
    sink = EventSink()
    state = FlowRunState(
        compiled=compiled,
        resolver=InputResolver(compiled.registry),
        results={},
        _event_sink=sink,
        _cancelled=threading.Event(),
        _started_at=time.monotonic(),
        _timeout_seconds=timeout_seconds,
        context=context or {},
    )

    cache = cache or {}
    completed: list[str] = []

    for node_id in compiled.execution_order:
        # Skip nodes managed by compound nodes (loop body/end)
        if node_id in compiled.managed_ids:
            continue

        # Control checks
        if state.is_cancelled():
            yield FlowCancelledEvent(...)
            return
        if state.is_timed_out():
            yield FlowTimeoutEvent(...)
            return

        node = compiled.node_map[node_id]

        # Skip propagation
        if should_skip_node(node, compiled.edge_map, state.results):
            state.results[node_id] = SKIPPED
            yield NodeSkippedEvent(type="node_skipped", node_id=node_id)
            completed.append(node_id)
            continue

        yield NodeStartEvent(type="node_start", node_id=node_id)
        await asyncio.sleep(0)

        # Cache hit
        if node_id in cache:
            state.results[node_id] = cache[node_id]
            yield NodeCompleteEvent(..., cached=True)
            completed.append(node_id)
            continue

        # Resolve inputs
        try:
            inputs = state.resolver.resolve(
                node, compiled.edge_map, state.results, compiled.node_map
            )
        except InputResolutionError as e:
            yield NodeErrorEvent(...)
            yield FlowErrorEvent(...)
            return

        # Build DTO
        req = NodeExecRequest(
            node_id=node_id,
            node_type=node.type,
            inputs=inputs,
            data=node.data or {},
            state=state,
        )

        # Execute in thread pool
        try:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = loop.run_in_executor(pool, _dispatch_node, req, compiled)
                remaining = state.remaining_seconds()
                result = await asyncio.wait_for(
                    future,
                    timeout=max(0.1, remaining) if remaining is not None else None,
                )

            result = normalize_result(result)
            state.results[node_id] = result
            completed.append(node_id)
            yield NodeCompleteEvent(
                type="node_complete", node_id=node_id,
                result=filter_skipped(result),
            )
        except asyncio.TimeoutError:
            yield FlowTimeoutEvent(...)
            return
        except NodeValidationException as e:
            yield NodeErrorEvent(..., is_validation=True)
            yield FlowErrorEvent(..., is_validation=True)
            return
        except NodeExecutionException as e:
            yield NodeErrorEvent(..., is_validation=False)
            yield FlowErrorEvent(..., is_validation=False)
            return

        # Drain events emitted by compound nodes
        while event := sink.pop():
            yield event

    yield FlowCompleteEvent(
        type="flow_complete",
        results=filter_all_skipped(state.results),
    )
```

### 8.7 Node Dispatch

```python
def _dispatch_node(req: NodeExecRequest, compiled: CompiledGraph) -> Any:
    """Route execution to the right handler. Three-way lookup, no special cases."""

    # 1. Compound node (for-each, future try-catch, etc.)
    if req.node_id in compiled.compound_nodes:
        return compiled.compound_nodes[req.node_id].execute(req)

    # 2. Extension node (app nodes, host-specific)
    ext = compiled.extension_resolver
    if ext and ext.is_known_type(req.node_type):
        return ext.create_executor(req.node_type).execute(req)

    # 3. Registry node — validate + call
    node_def = compiled.registry.get(req.node_type)
    if not node_def or not node_def.func:
        raise NodeExecutionException(f"Node type '{req.node_type}' not found")

    if node_def.validation_model:
        try:
            validated = node_def.validation_model(**req.inputs)
            inputs = validated.model_dump()
        except ValidationError as e:
            raise NodeValidationException(
                format_validation_error(e, node_def)
            ) from e
    else:
        inputs = req.inputs

    try:
        return node_def.func(**inputs)
    except (NodeValidationException, NodeExecutionException):
        raise
    except Exception as e:
        raise NodeExecutionException(
            f"Execution failed for {req.node_type}: {type(e).__name__}: {e}"
        ) from e
```

### 8.8 Sync Convenience Wrappers

```python
async def collect(events: AsyncGenerator[ExecutionEvent, None]) -> dict[str, Any]:
    """Consume all events, return final results. Raises on error/cancel/timeout."""
    results = {}
    async for event in events:
        if event["type"] == "flow_complete":
            results = event["results"]
        elif event["type"] in ("flow_error", "flow_cancelled", "flow_timeout"):
            raise FlowExecutionException(event.get("error", "Flow did not complete"))
    return results

def execute_sync(compiled: CompiledGraph, **kwargs) -> dict[str, Any]:
    """Blocking entry point. Runs the async engine and collects results."""
    return asyncio.run(collect(execute(compiled, **kwargs)))
```

---

## 9. Layer 5: Compound Nodes

### 9.1 CompoundNodeType Protocol

```python
@dataclass(frozen=True)
class Region:
    """A group of related nodes managed by a compound node."""
    start_id: str
    end_id: str
    body_ids: frozenset[str]

    @property
    def all_ids(self) -> frozenset[str]:
        return self.body_ids | {self.start_id, self.end_id}

@dataclass(frozen=True)
class CompoundNodeType:
    """Registration for a compound node type.

    The compiler calls discover() to find regions in the graph,
    then calls factory() to create an executor for each region.
    """
    start_type_prefix: str                          # e.g., "for-each-start"
    end_type_prefix: str                            # e.g., "for-each-end"
    discover: Callable[
        [list[GraphNode], list[GraphEdge]],
        list[Region],
    ]
    factory: Callable[
        [Region, tuple[str, ...]],                  # (region, execution_order)
        NodeExecutor,
    ]
```

### 9.2 ForEachNode

```python
class ForEachNode:
    """Compound node for for-each loop iteration."""

    def __init__(self, region: Region, execution_order: tuple[str, ...]):
        self.region = region
        self.body_order = [nid for nid in execution_order if nid in region.body_ids]

    def execute(self, req: NodeExecRequest) -> Any:
        items = prepare_loop_items(req.inputs.get("items", []))
        items = items[:MAX_ITERATIONS]
        parallel = req.inputs.get("execution_mode", "Sekventiel") == "Parallel"
        state = req.state

        state.emit(NodeStartEvent(type="node_start", node_id=self.region.end_id))

        def run_one(item: Any, idx: int) -> Any:
            overlay = {self.region.start_id: normalize_result((item, idx + 1))}
            local = state.execute_subgraph(self.body_order, overlay=overlay)
            return resolve_end_inputs(local, self.region.end_id, state)

        if parallel:
            with ThreadPoolExecutor(max_workers=min(len(items), 8)) as pool:
                collected = list(pool.map(
                    lambda pair: run_one(pair[1], pair[0]),
                    enumerate(items),
                ))
            state.emit(NodeProgressEvent(
                type="node_progress", node_id=self.region.end_id,
                current=len(items), total=len(items),
            ))
        else:
            collected = []
            for idx, item in enumerate(items):
                if state.is_cancelled():
                    break
                collected.append(run_one(item, idx))
                state.emit(NodeProgressEvent(
                    type="node_progress", node_id=self.region.end_id,
                    current=idx + 1, total=len(items),
                ))

        end_result = normalize_result(collect_loop_results(collected))
        state.results[self.region.end_id] = end_result
        state.emit(NodeCompleteEvent(
            type="node_complete", node_id=self.region.end_id,
            result=filter_skipped(end_result),
        ))

        return items  # Start node's own result


# Registration constant:
FOR_EACH = CompoundNodeType(
    start_type_prefix="for-each-start",
    end_type_prefix="for-each-end",
    discover=discover_for_each_regions,
    factory=lambda region, order: ForEachNode(region, order),
)
```

### 9.3 Adding New Compound Nodes

Adding a new control-flow pattern requires **zero changes** to the engine. Example: a hypothetical try/catch node:

```python
class TryCatchNode:
    def __init__(self, region: TryCatchRegion, execution_order):
        self.region = region
        self.try_body = [nid for nid in execution_order if nid in region.try_ids]
        self.catch_body = [nid for nid in execution_order if nid in region.catch_ids]

    def execute(self, req: NodeExecRequest) -> Any:
        try:
            results = req.state.execute_subgraph(self.try_body)
            return results.get(self.try_body[-1])
        except (NodeExecutionException, NodeValidationException) as e:
            overlay = {"__error__": normalize_result(str(e))}
            results = req.state.execute_subgraph(self.catch_body, overlay=overlay)
            return results.get(self.catch_body[-1])

TRY_CATCH = CompoundNodeType(
    start_type_prefix="try",
    end_type_prefix="catch-end",
    discover=discover_try_catch_regions,
    factory=lambda region, order: TryCatchNode(region, order),
)
```

The host application registers it alongside `FOR_EACH`:

```python
compiled = compile(
    nodes=nodes,
    edges=edges,
    registry=registry,
    compound_types=[FOR_EACH, TRY_CATCH],
)
```

---

## 10. Host Application Integration

### 10.1 AKA Flows Integration Example

```python
# backend/app/services/flows/engine.py

from conductor import NodeRegistry, compile, execute
from conductor.compound.for_each import FOR_EACH

# --- Registry (replaces current global NODE_REGISTRY) ---
registry = NodeRegistry()

# --- Node registration (unchanged pattern) ---
@registry.node("greeting", version=1, name="Hilsen", description="Returnerer en hilsen")
def greeting(
    name: Annotated[str, widgets.Text(label="Navn", description="Navn")],
    formal: Annotated[bool, widgets.Checkbox(label="Formel")] = False,
) -> Annotated[str, widgets.Output(label="Hilsen")]:
    return f"Kære {name}" if formal else f"Hej {name}"


# --- Extension resolver for app nodes ---
class AppNodeResolver:
    def __init__(self, app_cache: dict[str, AppNodeData]):
        self.app_cache = app_cache

    def is_known_type(self, node_type: str) -> bool:
        return node_type.startswith("app:")

    def create_executor(self, node_type: str) -> NodeExecutor:
        return AppNodeExecutor(node_type, self.app_cache)

class AppNodeExecutor:
    def __init__(self, node_type: str, cache: dict):
        self.node_type = node_type
        self.cache = cache

    def execute(self, req: NodeExecRequest) -> Any:
        # Access host-specific context
        ctx = req.state.context
        parent_ids = ctx.get("parent_app_ids", [])
        # ... recursion detection, RBAC, sub-flow execution via execute_sync
```

### 10.2 Route Handler

```python
@router.post("/flows/{flow_id}/execute")
async def execute_flow(flow_id: UUID, request: FlowExecuteRequest, user: User):
    # Pre-load app nodes (host-specific)
    app_cache = await preload_app_nodes(request.nodes, user)

    # Compile (validation happens here, before execution)
    compiled = compile(
        nodes=[GraphNode(id=n.id, type=n.type, data=n.data) for n in request.nodes],
        edges=[GraphEdge(...) for e in request.edges],
        registry=registry,
        compound_types=[FOR_EACH],
        extension_resolver=AppNodeResolver(app_cache),
    )

    # Execute with streaming
    async def event_stream():
        async for event in execute(
            compiled,
            timeout_seconds=300,
            context={
                "user_id": user.id,
                "flow_id": flow_id,
                "parent_app_ids": [],
                "app_node_cache": app_cache,
                "user_team_ids": user.team_ids,
                "user_role": user.role,
            },
            cache=request.cache,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

### 10.3 Frontend Node Registry Endpoint

```python
@router.get("/nodes")
def get_node_types():
    """Serve the full registry as JSON for the frontend."""
    return registry.to_json()
```

The frontend consumes this unchanged — the JSON structure is identical to today's `NodePublic` format.

### 10.4 Responsibility Boundary

| Concern | Conductor | Host Application |
|---|---|---|
| Node registration | `@registry.node()` decorator | Defines all node functions |
| Widget system | `widgets.py` (Text, Dropdown, etc.) | Can define additional widget types |
| Type validation | Pydantic model from signature | N/A (automatic) |
| Graph compilation | `compile()` | Calls it, provides extension resolver |
| Topological sort | Built-in | N/A |
| Region discovery | Framework for compound types | Registers compound types |
| Execution loop | `execute()` async generator | Wraps in SSE/WebSocket endpoint |
| Input resolution | `InputResolver` | N/A |
| Result normalization | `normalize_result()` | N/A |
| Skip propagation | `SKIPPED` sentinel | N/A |
| Event streaming | `EventSink` + event types | Consumes events for transport |
| Timeout/cancellation | Built-in | Sets limits, triggers cancel |
| Extension nodes | `ExtensionResolver` protocol | Implements for app nodes, sub-flows |
| Registry JSON | `registry.to_json()` | Serves via API endpoint |
| Auth/RBAC | N/A (not its concern) | Enforced before calling engine |
| Persistence | N/A | Database, caching, etc. |
| Error message i18n | English error messages | Host translates or wraps |

---

## 11. Migration Path

### 11.1 File-Level Mapping

| Current (AKA Flows) | New (Conductor) | Change Level |
|---|---|---|
| `services/flows/types.py` | `conductor/types.py` | Copy, remove app-specific types |
| `services/flows/widgets.py` | `conductor/widgets.py` | Copy unchanged |
| `services/flows/metadata.py` | `conductor/metadata.py` | Copy unchanged |
| `services/flows/validation.py` | `conductor/validation.py` | Copy unchanged |
| `services/flows/registry.py` | `conductor/registry/__init__.py` | Refactor: split class, remove legacy dicts |
| `services/flows/introspection.py` | `conductor/registry/schema.py` | Refactor: work from NodeDefinition |
| `execution/executor.py` (1015 lines) | `conductor/execution/engine.py` (~150 lines) | Rewrite |
| `execution/context.py` | `conductor/execution/state.py` | Replace with FlowRunState |
| `execution/graph.py` | `conductor/graph/topology.py` + `regions.py` | Split |
| `execution/inputs.py` | `conductor/execution/resolver.py` | Copy, adjust imports |
| `execution/results.py` | `conductor/execution/results.py` | Copy unchanged |
| `execution/app_node.py` | Host app's `AppNodeResolver` | Host-side code |
| `nodes/loop.py` (prepare/collect) | `conductor/compound/for_each.py` | Move helpers |
| `nodes/control.py` (if_else) | Stays a simple registered node | No change |
| `nodes/*.py` (all others) | Host app's node definitions | Change decorator import only |

### 11.2 Migration Steps

**Phase 1: Extract the library**

1. Create `conductor/` package with the structure above.
2. Copy types, widgets, metadata, validation unchanged.
3. Refactor registry (split `NodeDefinition` out, remove legacy dicts, raw function storage).
4. Implement `compile()` from existing validation logic.
5. Implement `execute()` as the single streaming path.
6. Implement `ForEachNode` compound node from existing `_execute_for_each_region` logic.
7. Port input resolution and result normalization.

**Phase 2: Integrate into AKA Flows**

8. Create `AppNodeResolver` implementing `ExtensionResolver`.
9. Update route handlers to use `compile()` + `execute()`.
10. Update node files to import from `conductor` instead of `app.services.flows`.
11. Update frontend node registry endpoint to use `registry.to_json()`.

**Phase 3: Clean up**

12. Remove old `execution/executor.py`, `execution/context.py`.
13. Remove legacy `NODE_REGISTRY` and `NODE_METADATA_REGISTRY` dicts.
14. Remove `introspection.py` (replaced by `schema.py`).

### 11.3 Risk Mitigation

- **Existing node functions require zero changes** to their logic. Only the decorator import path changes.
- **Frontend receives identical JSON** — `registry.to_json()` produces the same structure as the current `NodePublic` model.
- **Tests can run against both implementations** during migration by routing through a compatibility shim.
- **App node behavior is preserved** — the `AppNodeResolver` encapsulates the same `execute_app_node` logic.

---

## 12. Appendix: Research Summary

The [Graph Processing Framework Research](./graph-processing-framework-research.md) evaluated five frameworks. This section summarizes which patterns were adopted and which were explicitly rejected.

### Adopted Patterns

| Pattern | Source | How Applied |
|---|---|---|
| Two-phase construction/execution | Beam, LangGraph, Ray | `compile()` → `execute()` separation |
| Nodes as plain functions | LangGraph, Ray | `@registry.node()` stores raw functions |
| Decorator-based registration | Ray | Already in place, preserved |
| Compile-time validation | Beam, LangGraph | All structural errors caught in `compile()` |
| Conditional edges (SKIPPED) | LangGraph | Existing pattern preserved via skip propagation |

### Explicitly Rejected Patterns

| Pattern | Source | Reason |
|---|---|---|
| BSP/superstep execution | Pregel, LangGraph | Most flows are narrow and deep; superstep barriers add latency for the common case (linear chains) while only helping the rare case (wide parallelism) |
| Shared state with reducers | LangGraph | Edge-based data flow matches the visual wire-drawing metaphor. Shared state creates a mismatch between what users see (wires) and how it executes (global dict) |
| Transactional rollback | LangGraph | Nodes have side effects (API calls, emails). You cannot roll back a sent email. Transactional semantics only protect state, not external effects |
| Operator chaining/fusion | Flink | Premature optimization. Flows have 5–30 nodes processing one request. Scheduling overhead is negligible |
| True cycle support | LangGraph | Region-based loops (for-each) cover current needs. True cycles require convergence detection and add complexity without demonstrated demand |

### Deferred Patterns

| Pattern | Source | When to Revisit |
|---|---|---|
| Checkpointing/resume | LangGraph, Pregel, Flink | When there is demand for pause/resume, human-in-the-loop, or long-running flows that survive process restarts |
| Error routing | Beam | When there is demand for flows that continue past failures and collect errors for review. Requires frontend UI for error channels |
| State backends | Flink | When flows outgrow in-memory state (unlikely at current scale) |
