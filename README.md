<p align="center">
  <img src="logo-white-background.png" alt="Conductor logo" width="140">
</p>

<h1 align="center">Conductor</h1>

<p align="center">
  A reusable, host-agnostic graph execution engine for building DAG-based workflow systems. Register nodes as plain Python functions with type annotations, compile them into a validated execution plan, and run them with streaming events.
</p>


Built to be the shared core behind visual flow builders — define nodes once, get backend execution, input validation, and frontend UI metadata for free.

## Features

- **Decorator-based node registration** — `@registry.node()` turns any function into a validated, UI-renderable node
- **Widget annotations** — `Annotated[str, Text(label="Input")]` is the single source of truth for validation, execution, and frontend rendering
- **Compile-then-execute** — structural errors caught before any node runs
- **Eager parallel scheduling** — nodes start as soon as their dependencies finish; independent branches run concurrently
- **Retry** — per-node `max_retries`/`retry_delay` with exponential backoff, or a global `RetryConfig`
- **Structured error hierarchy** — `NodeValidationError`, `NodeExecutionError`, `NodeConnectionError`, `NodeTimeoutError`, and more, all carrying `node_id`/`node_type` context
- **Streaming execution** — async generator yields events (node_start, node_complete, node_retry, flow_complete, etc.)
- **Conditional branching** — SKIPPED sentinel propagates through inactive branches
- **For-each loops** — compound node regions with sequential or parallel execution
- **Human-in-the-loop** — `HumanInputRequired` pauses to a JSON-serializable checkpoint; resume later
- **Class-based nodes** — `BaseNode` ABC for complex nodes needing state
- **FlowStore** — side-channel key-value cache for cross-node data sharing
- **Auto-discovery** — scan a package to register all `@node`-decorated functions
- **Extension resolver** — protocol for host-app-specific node types (sub-flows, etc.)
- **Zero app dependencies** — no FastAPI, no database, no auth in the core

## Quick start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
git clone <repo-url> conductor
cd conductor
uv sync
uv run pre-commit install   # strip notebook outputs on commit
```

### Run tests

```bash
uv run pytest tests/ -v
```

### Run the demo playground

```bash
uv sync --group demo
uv run uvicorn demo.app:app --port 8765 --reload
```

Open http://localhost:8765 — drag nodes onto the canvas, connect them, and click "Run Flow" to see streaming execution.

## Usage

### 1. Create a registry and register nodes

```python
from typing import Annotated
from conductor import NodeRegistry
from conductor.widgets import Text, Textarea, Dropdown, Range, Output

registry = NodeRegistry()

@registry.node("echo", version=1, name="Echo", description="Returns input unchanged")
def echo(
    text: Annotated[str, Text(label="Input", description="Text to echo")],
) -> Annotated[str, Output(label="Output")]:
    return text

@registry.node("uppercase", version=1, name="Uppercase", description="Converts to uppercase")
def uppercase(
    text: Annotated[str, Text(label="Input")],
) -> Annotated[str, Output(label="Result")]:
    return text.upper()
```

### 2. Build and execute a flow

```python
from conductor import GraphNode, GraphEdge, compile
from conductor.execution.engine import execute_sync

compiled = compile(
    nodes=[
        GraphNode("n1", "echo@1", {"text": "hello world"}),
        GraphNode("n2", "uppercase@1", None),
    ],
    edges=[
        GraphEdge("e1", "n1", "n2", "result", "text"),
    ],
    registry=registry,
)

results = execute_sync(compiled)
print(results["n2"]["result"])  # "HELLO WORLD"
```

### 3. Stream execution events

```python
from conductor.execution.engine import execute

async for event in execute(compiled):
    match event["type"]:
        case "node_start":
            print(f"Starting {event['node_id']}")
        case "node_complete":
            print(f"Done {event['node_id']}: {event['result']}")
        case "node_retry":
            print(f"Retry {event['node_id']} ({event['attempt']}/{event['max_retries']}): {event['error']}")
        case "flow_complete":
            print(f"Flow done: {event['results']}")
```

## Project structure

```
conductor/
├── packages/
│   └── conductor/                 # Core library
│       ├── pyproject.toml          # pip install conductor
│       └── src/conductor/
│           ├── types.py            # Enums: WidgetType, ResultFormat, NodeCategory
│           ├── widgets.py          # Widget ABC + Text, Dropdown, Range, Output, etc.
│           ├── metadata.py         # InputMetadata, OutputMetadata
│           ├── validation.py       # Pydantic model generation from signatures
│           ├── errors.py           # Exception hierarchy (ConductorError, NodeError, ...)
│           ├── node.py             # BaseNode ABC for class-based nodes
│           ├── _sentinel.py        # SKIPPED singleton
│           ├── registry/
│           │   ├── __init__.py     # NodeRegistry + @node decorator
│           │   ├── definition.py   # NodeDefinition dataclass
│           │   ├── discovery.py    # Auto-discovery via importlib
│           │   └── schema.py       # JSON serialization for frontends
│           ├── graph/
│           │   ├── model.py        # GraphNode, GraphEdge
│           │   ├── topology.py     # Topological sort, cycle detection
│           │   ├── compiler.py     # compile() -> CompiledGraph
│           │   └── regions.py      # Compound node region discovery
│           ├── execution/
│           │   ├── engine.py       # execute(), execute_sync(), eager scheduler, retry loop
│           │   ├── retry.py        # RetryConfig
│           │   ├── state.py        # FlowRunState
│           │   ├── store.py        # FlowStore (cross-node cache)
│           │   ├── request.py      # NodeExecRequest DTO
│           │   ├── resolver.py     # Input resolution from edges
│           │   ├── results.py      # Result normalization
│           │   ├── events.py       # Event TypedDicts + EventSink
│           │   ├── skip.py         # Skip propagation
│           │   └── checkpoint.py   # FlowCheckpoint for human-in-the-loop
│           └── compound/
│               ├── protocol.py     # CompoundNodeType, Region
│               └── for_each.py     # ForEachNode + FOR_EACH constant
├── examples/                       # Usage examples (6 examples)
├── demo/                           # Interactive playground (FastAPI + browser UI)
├── tests/                          # pytest test suite (131 tests)
└── docs/                           # Design spec, llms.txt, MkDocs site
```

## Concepts

### Nodes

Nodes are the building blocks. Register them as decorated functions or BaseNode subclasses:

```python
# Function-based (most nodes)
@registry.node("add", version=1, name="Add", description="Adds two numbers")
def add(
    a: Annotated[float, Text(label="A")],
    b: Annotated[float, Text(label="B")],
) -> Annotated[float, Output(label="Sum")]:
    return a + b

# Multi-output
@registry.node("split", version=1, name="Split", description="Splits text")
def split(
    text: Annotated[str, Text(label="Input")],
) -> tuple[
    Annotated[str, Output(label="First half")],
    Annotated[str, Output(label="Second half")],
]:
    mid = len(text) // 2
    return text[:mid], text[mid:]

# Class-based (complex nodes)
class MyNode(BaseNode):
    node_id = "my-node"
    node_name = "My Node"
    node_description = "A complex node"

    def execute(self, req: NodeExecRequest) -> str:
        return req.inputs["text"].upper()

registry.register_class(MyNode)
```

### Versioning

Nodes are versioned as `base_id@version`. When you register a new version, the old one becomes deprecated but continues to work for existing flows:

```python
@registry.node("echo", version=2, name="Echo v2", description="Echo with prefix")
def echo_v2(text: Annotated[str, Text(label="Input")], prefix: Annotated[str, Text(label="Prefix")] = "") -> ...:
    return f"{prefix}{text}"

registry.get("echo@1")          # Old version (deprecated)
registry.get("echo@2")          # Current version
registry.get_latest("echo")     # Returns echo@2
registry.is_deprecated("echo@1") # True
```

### FlowStore

Side-channel key-value store for sharing data between nodes outside of edges:

```python
from conductor.execution.store import FlowStore

@registry.node("cache-doc", version=1, name="Cache Document", description="Parses and caches")
def cache_doc(
    file: Annotated[str, Text(label="File")],
    store: FlowStore,  # Auto-injected by the engine
) -> Annotated[str, Output(label="Text")]:
    parsed = expensive_parse(file)
    store.set("parsed_doc", parsed)  # Available to downstream nodes
    return parsed.text
```

### Auto-discovery

Scan a Python package to register all `@node`-decorated functions:

```python
registry.discover("myapp.nodes")  # Imports all modules, triggering decorators
```

### Extension resolver

Let host applications handle custom node types:

```python
class AppNodeResolver:
    def is_known_type(self, node_type: str) -> bool:
        return node_type.startswith("app:")

    def create_executor(self, node_type: str):
        return MyAppNodeExecutor(node_type)

compiled = compile(nodes, edges, registry, extension_resolver=AppNodeResolver())
```

### Eager parallel execution

The engine schedules nodes eagerly: as soon as all of a node's dependencies finish, its task is dispatched. Independent branches run concurrently without any configuration. Sync node functions are offloaded to `asyncio.to_thread`, so they don't block the event loop.

```
  A (0.3s) ──> C (0.3s) ──┐
                           ├──> E (0.3s)
  B (0.3s) ──> D (0.3s) ──┘
```

Sequential would be 5 × 0.3 s = 1.5 s. Eager execution: `A + B` in parallel (0.3 s), `C + D` in parallel (0.3 s), `E` (0.3 s) = ~0.9 s.

No flag is needed — this is the default and only execution mode.

### Retry

Nodes can retry automatically on failure. Configure retries at the **node level** (preferred) or the **flow level**:

```python
from conductor.execution.retry import RetryConfig

# Node-level — wins over any global config
@registry.node("fetch-url", version=1, name="Fetch", description="HTTP GET",
               max_retries=3, retry_delay=0.5)
def fetch_url(url: Annotated[str, Text(label="URL")]) -> Annotated[str, Output(label="Body")]:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.text

# Or flow-level — applies to every node that doesn't set its own max_retries
results = execute_sync(
    compiled,
    retry=RetryConfig(max_retries=2, delay=1.0, backoff_factor=2.0),
)
```

Delay between attempts is `retry_delay * backoff_factor ** (attempt - 1)` — e.g., `1s, 2s, 4s, ...` with defaults. Node-level retry uses a backoff factor of 2.0.

**What gets retried:**
- `NodeExecutionError` (anything raised from a node function)
- `NodeConnectionError` (raise this from nodes for transient network/API failures)

**What never gets retried:**
- `NodeValidationError` — pydantic rejected the inputs; retrying with the same inputs is pointless
- `HumanInputRequired` — pauses immediately

Each retry emits a `node_retry` streaming event with `{attempt, max_retries, error, delay}`.

```python
async for event in execute(compiled, retry=RetryConfig(max_retries=2, delay=0.5)):
    if event["type"] == "node_retry":
        print(f"Retrying {event['node_id']} in {event['delay']}s — {event['error']}")
```

### Error types

All exceptions inherit from `ConductorError` and are importable from `conductor.errors`. Node-level errors carry `node_id`, `node_type`, and the `original` exception so they propagate with enough context to log, display, or route to an error handler.

```
ConductorError                     # Base — catch-all for any engine error
├── CompilationError                # Graph structure is invalid
│   ├── CycleDetectionError         # Graph contains a cycle
│   └── TypeCheckError              # Edge type mismatch (strict mode)
├── NodeError                       # Something went wrong with a specific node
│   ├── NodeValidationError         # Input validation failed (Pydantic) — never retried
│   ├── NodeExecutionError          # Node function raised — retried if configured
│   ├── NodeTimeoutError            # Node exceeded its timeout
│   └── NodeConnectionError         # External service / network failure inside a node
├── InputResolutionError            # Could not resolve inputs from edges
├── FlowExecutionError              # Flow-level failure (raised by execute_sync)
├── FlowPausedError                 # Flow paused for human input (carries checkpoint)
└── HumanInputRequired              # Signal raised by nodes to request human input
```

Use `NodeConnectionError` from your node code to mark a failure as transient and retry-worthy:

```python
from conductor.errors import NodeConnectionError

@registry.node("fetch-api", version=1, name="Fetch", description="HTTP GET",
               max_retries=3, retry_delay=1.0)
def fetch_api(url: Annotated[str, Text(label="URL")]) -> Annotated[str, Output(label="Body")]:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        raise NodeConnectionError(f"API call failed: {e}") from e
```

Legacy aliases (`NodeValidationException`, `NodeExecutionException`, `FlowExecutionException`, `FlowPausedException`) remain importable from `conductor.errors` and map to the new `*Error` names.

### Human-in-the-loop

Nodes can pause execution to request human input. The engine checkpoints state (JSON-serializable), and execution resumes later with the human's response:

```python
from conductor.errors import HumanInputRequired, FlowPausedException
from conductor.execution.engine import execute_sync, resume_sync

# A node that needs approval
@registry.node("approve", version=1, name="Approval", description="Needs approval")
def approve(text: Annotated[str, Text(label="Content")]) -> Annotated[str, Output(label="Approved")]:
    raise HumanInputRequired(
        prompt=f"Please approve: {text}",
        schema={"approved": "bool", "comment": "str"},
    )

# Execute — pauses at the approval node
try:
    results = execute_sync(compiled)
except FlowPausedException as e:
    checkpoint = e.checkpoint  # JSON-serializable dict, store in DB

# Resume later with the human's response
results = resume_sync(compiled, checkpoint, response="Approved!")
```

Key features:
- Checkpoints are plain dicts — serialize to JSON, store in a database, resume hours/days later
- FlowStore data survives the checkpoint/resume cycle
- A flow can pause multiple times (sequential approval gates)
- Works with both streaming (`flow_paused` event) and sync (`FlowPausedException`) APIs
- Both function-based and class-based nodes can raise `HumanInputRequired`

### Custom data types

Define custom types using `NewType` — at runtime they're their base type, but the type string surfaces in the frontend JSON schema:

```python
from typing import NewType, TypedDict

# Simple alias — shows as "base64str" in the schema
Base64Str = NewType("Base64Str", str)

# Structured type
class NamedFile(TypedDict):
    content: str   # Base64-encoded
    filename: str

# Use in node signatures
@registry.node("upload", version=1, name="Upload", description="Accepts a file")
def upload(file: Annotated[Base64Str, FileUpload(label="File")]) -> ...:
    ...
```

Built-in types: `Base64Str`, `Date`, `NamedFile`, `MultiNamedFile`. Host apps can define additional types following the same pattern.

## Widgets

Widgets define how inputs render in the frontend and what validation to apply:

| Widget | Description | Key options |
|--------|-------------|-------------|
| `Text` | Single-line text | `min_length`, `max_length`, `pattern` |
| `Textarea` | Multi-line text | `rows`, `min_length`, `max_length` |
| `Dropdown` | Select from choices | `choices` |
| `Range` | Numeric slider | `min_val`, `max_val`, `step` |
| `Checkbox` | Boolean toggle | |
| `FileUpload` | File upload | `accept`, `max_size_mb`, `multiple` |
| `ConnectionList` | Multiple connections | |
| `Output` | Return value marker | `download`, `filename` |

## Execution events

The `execute()` async generator yields these events:

| Event | When |
|-------|------|
| `node_start` | Node begins execution |
| `node_complete` | Node finished (includes result) |
| `node_skipped` | Node skipped (all inputs SKIPPED) |
| `node_error` | Node raised an unretryable (or final) exception |
| `node_retry` | Node failed and will be retried (includes attempt, max_retries, error, delay) |
| `node_progress` | Loop iteration progress |
| `flow_complete` | All nodes done (includes all results) |
| `flow_paused` | Node requested human input (includes checkpoint) |
| `flow_error` | Unrecoverable error |
| `flow_timeout` | Execution exceeded timeout |
| `flow_cancelled` | Execution was cancelled |

## Using in other projects

### AI context (llms.txt)

The repo includes an `docs/llms.txt` file — a comprehensive AI-readable reference for the entire library. Import it as context when using Conductor in other projects with AI assistants:

```
# In another project's CLAUDE.md or AI context:
See /path/to/conductor/docs/llms.txt for Conductor API reference.
```

### Documentation

For full documentation, we recommend [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). To set it up:

```bash
uv add --group docs mkdocs-material mkdocstrings[python]
uv run mkdocs serve  # Local preview at http://localhost:8000
uv run mkdocs gh-deploy  # Deploy to GitHub Pages
```

## Examples

The examples are Jupyter notebooks under `examples/` — open them in VS Code, JupyterLab, or any notebook UI and run the cells interactively.

| Notebook | What it covers |
|----------|---------------|
| `01_basic_nodes.ipynb` | Widgets, multi-output, optional params |
| `02_build_and_run_flow.ipynb` | Graph building, collecting results, streaming |
| `03_class_nodes_and_store.ipynb` | BaseNode ABC, FlowStore injection |
| `04_control_flow.ipynb` | Conditionals (SKIPPED), for-each loops |
| `05_auto_discovery.ipynb` | Package scanning, JSON schema for frontends |
| `06_human_in_the_loop.ipynb` | Pause, checkpoint, resume |

```bash
uv sync                       # includes the ipykernel used by the notebooks
uv run jupyter lab examples/  # or open the .ipynb files in VS Code
```

The notebooks use `await collect(execute(compiled))` because the kernel already owns an event loop. From a plain `.py` script, use `execute_sync(compiled)` instead.

## License

TBD
