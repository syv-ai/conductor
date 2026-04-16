# Graph Processing Framework Research

**Purpose:** Inform the design of a maintainable, scalable graph execution framework for AKA Flows.
**Primary concern:** The current executor is bloated, hard to contribute to, difficult to manage, and error-prone. We need a framework that is easy to reason about, extend, and maintain — while being robust enough to scale.
**Date:** 2026-04-01

---

## Table of Contents

1. [Pregel — Foundational Model](#1-pregel--foundational-model)
2. [Apache Beam — Pipeline-as-DAG](#2-apache-beam--pipeline-as-dag)
3. [Apache Flink — Stateful Streaming over DAGs](#3-apache-flink--stateful-streaming-over-dags)
4. [LangGraph — Pregel-Inspired Graph Orchestration](#4-langgraph--pregel-inspired-graph-orchestration)
5. [Ray — Distributed Execution with Decorators](#5-ray--distributed-execution-with-decorators)
6. [Cross-Framework Comparison](#6-cross-framework-comparison)
7. [Patterns and Concepts to Adopt](#7-patterns-and-concepts-to-adopt)
8. [Recommendations for AKA Flows](#8-recommendations-for-aka-flows)

---

## 1. Pregel — Foundational Model

**Source:** Malewicz et al., "Pregel: A System for Large-Scale Graph Processing" (SIGMOD 2010, Google)

### Core Idea

Pregel introduces a vertex-centric, message-passing model built on the Bulk Synchronous Parallel (BSP) paradigm. Computation proceeds in discrete **supersteps**. During each superstep, every active vertex executes a user-defined `Compute()` function in parallel. Vertices communicate by sending messages along edges; messages sent in superstep S are delivered at superstep S+1.

### Key Concepts

**Supersteps and synchronization barriers.** The entire computation is a sequence of supersteps separated by global barriers. Within a superstep, vertices run in parallel with no coordination. Between supersteps, the system synchronizes and delivers messages. This gives you deterministic execution semantics — the same input always produces the same computation trace.

**Vertex state machine.** Each vertex is either Active or Inactive. A vertex votes to halt when it has no more work to do. It reactivates only when it receives a message. The algorithm terminates when all vertices are inactive and no messages are in transit.

**Message passing over shared memory.** Pregel deliberately chose message passing over shared memory or remote reads. Messages are batched and delivered asynchronously, amortizing network latency. This is a key design choice — it makes the programming model simpler (no data races, no locks) at the cost of requiring developers to think in terms of messages.

**Combiners.** An optimization where messages destined for the same vertex can be combined (e.g., summing integers) before delivery. Reduces network traffic by up to 4× in practice. Only works for commutative, associative operations.

**Aggregators.** A mechanism for global coordination. Each vertex contributes a value to an aggregator (e.g., min, max, sum); the reduced result is broadcast to all vertices in the next superstep. Useful for convergence detection, global statistics, and phase coordination.

**Topology mutations.** Vertices can add/remove edges and vertices during computation. Mutations take effect between supersteps with a defined resolution order (removals before additions, edge before vertex).

### Implementation Architecture

The system uses a master-worker architecture. The master partitions the graph across workers (default: `hash(vertex_id) mod N`), coordinates supersteps, and handles fault tolerance through checkpointing. Workers maintain their partition in memory as a map from vertex ID to (value, outgoing edges, message queue, active flag).

Fault tolerance is checkpoint-based: at configurable intervals, workers save their partition state to persistent storage. On failure, the master reassigns lost partitions and workers reload from the latest checkpoint.

### Relevance to AKA Flows

Pregel was designed for billion-vertex web graphs — a scale AKA Flows doesn't need. But several of its design principles transfer directly:

- **Superstep model** gives deterministic, debuggable execution
- **Message passing** eliminates shared state bugs
- **Vertex-centric programming** ("think like a node") maps naturally to user-defined workflow nodes
- **Vote-to-halt** provides clean termination semantics
- **Aggregators** solve the "how do I get global state" problem without breaking encapsulation

What Pregel does NOT solve for us: it assumes a static graph topology (mutations are awkward), it has no built-in support for conditional branching or human-in-the-loop, and its fault tolerance model (full checkpoint) is heavy for our use case.

---

## 2. Apache Beam — Pipeline-as-DAG

### Core Computational Model

Beam represents pipelines as DAGs where PCollections (distributed datasets) are nodes and PTransforms (operations) are edges. Pipelines are constructed declaratively in a driver program, then submitted to a **runner** (Dataflow, Flink, Spark, Direct) for execution.

The key insight is the **two-phase model**: graph construction is separate from graph execution. This enables validation, optimization, and runner-agnostic portability.

### Key Abstractions

**PCollection:** An immutable, unordered bag of elements. Can be bounded (batch) or unbounded (streaming). Each PCollection is owned by a single Pipeline.

**PTransform:** A composable unit of computation. Takes zero or more PCollections as input, produces zero or more as output. The `expand()` method defines the transformation logic.

**DoFn:** The workhorse — a function that processes one element at a time within a PTransform. Has a defined lifecycle: `Setup → StartBundle → ProcessElement (×N) → FinishBundle → Teardown`.

### Execution Model

Beam processes elements in **bundles** — groups of elements that form the atomic unit of retry. This is a sweet spot between per-element checkpointing (too expensive) and full-pipeline retry (too coarse). The runner decides bundle sizes based on the execution context.

Topological ordering is enforced by the runner: it traverses the DAG from sources to sinks, ensuring inputs are available before transforms execute. Within a stage, elements are processed in parallel across workers.

### State and Windowing

Beam's State API provides per-key, mutable state within transforms. State is partitioned by key and window, and is automatically cleaned up when windows expire. Three state types: ValueState, BagState, MapState.

The windowing model divides unbounded data into finite chunks for aggregation: fixed windows, sliding windows, sessions, and custom implementations.

### API Design Patterns

```python
# Composite transform — the recommended abstraction boundary
class ProcessRecords(beam.PTransform):
    def expand(self, pcoll):
        return (pcoll
                | "Validate" >> beam.ParDo(ValidateDoFn())
                | "Transform" >> beam.Map(transform_fn)
                | "Group" >> beam.GroupByKey())

# Side inputs for broadcast data
lookup = side_data | beam.AsDict()
results = main | beam.ParDo(LookupDoFn(), beam.AsSideInput(lookup))

# Multiple outputs via side outputs (error routing)
valid, invalid = (records
    | beam.ParDo(ClassifyDoFn())
      .with_outputs("invalid", main="valid"))
```

### Error Handling

Beam uses bundle-level retry: if any element in a bundle fails, the entire bundle is retried. This means DoFn side effects must be idempotent. For more granular error handling, Beam supports error routing — failed elements are directed to a separate PCollection for analysis rather than causing pipeline failure.

### Trade-offs for AKA Flows

**Strengths:**
- Clean separation of graph definition and execution
- Composable transforms reduce code duplication
- Runner abstraction provides execution flexibility
- Error routing prevents silent data loss
- Mature, well-documented Python SDK

**Weaknesses:**
- Designed for data processing pipelines, not workflow orchestration
- No built-in support for cycles or conditional branching on state
- Bundle-based execution model doesn't map well to node-by-node workflow execution
- Windowing model is irrelevant for our use case
- Heavy infrastructure dependency (needs a runner)

**Key patterns to borrow:**
- Two-phase construction/execution
- Composite transforms as the abstraction boundary
- Error routing instead of exception swallowing
- Type-safe extensibility with automatic serialization inference

---

## 3. Apache Flink — Stateful Streaming over DAGs

### Core Computational Model

Flink represents programs as dataflow graphs: streams of records flow through operators arranged in a DAG. The graph goes through four compilation stages: StreamGraph → JobGraph (with operator chaining) → ExecutionGraph (with parallelism) → Physical Plan.

**Operator chaining** is a key optimization: consecutive operators that don't require data shuffling are fused into a single task, running in one thread. This dramatically reduces serialization and context-switching overhead.

### Stateful Processing

This is Flink's standout feature. Two state scopes:

**Keyed state:** Partitioned by key, stored in an embedded key-value store. Each key has exactly one state partition. Supports ValueState, ListState, MapState, ReducingState, AggregatingState.

**Operator state:** Scoped per parallel operator instance (not per key). Used for things like source offset tracking.

**State backends** determine storage characteristics:
- HashMapStateBackend — in-memory, fast, memory-limited
- EmbeddedRocksDBStateBackend — disk-backed, handles larger state, slightly slower

### Checkpointing and Fault Tolerance

Flink uses **asynchronous barrier snapshotting**, a variant of the Chandy-Lamport algorithm adapted for DAGs. Checkpoint barriers flow through the dataflow graph; when an operator receives a barrier from all inputs, it snapshots its state. This provides exactly-once processing semantics without stopping the pipeline.

Recovery replays from the latest checkpoint: operator state is restored, source offsets are reset, and processing resumes. No data loss, no duplicates.

### Backpressure

Flink uses credit-based flow control: downstream operators grant "credits" (buffer slots) to upstream operators. When a downstream operator is slow, it stops granting credits, and backpressure propagates naturally upstream to the sources. This prevents out-of-memory situations without explicit rate limiting.

### API Design

```python
# PyFlink DataStream API
env = StreamExecutionEnvironment.get_execution_environment()

stream = (env.from_source(source)
    .map(process_fn)
    .filter(filter_fn)
    .key_by(key_selector)
    .window(TumblingEventTimeWindows.of(Time.seconds(10)))
    .reduce(reduce_fn)
    .sink_to(sink))

env.execute("my-job")
```

### Trade-offs for AKA Flows

**Strengths:**
- Best-in-class stateful processing with multiple state backends
- Exactly-once semantics via barrier-based checkpointing
- Operator chaining reduces execution overhead
- Credit-based backpressure is elegant and proven
- Event time processing enables correct ordering regardless of arrival time

**Weaknesses:**
- Designed for continuous streaming, not discrete workflow execution
- Heavy operational footprint (JVM-based, requires cluster management)
- PyFlink is a bridge to the JVM — adds complexity and overhead
- DAG-only (no cycles), and the DAG is more about data flow than control flow
- Windowing and watermark concepts add complexity we don't need

**Key patterns to borrow:**
- Operator chaining (fuse sequential nodes that don't need intermediate state)
- Barrier-based checkpointing (lightweight, non-blocking snapshots)
- Keyed state with pluggable backends
- Credit-based backpressure for async message passing
- Four-stage graph compilation (logical → optimized → physical → runtime)

---

## 4. LangGraph — Pregel-Inspired Graph Orchestration

### Core Computational Model

LangGraph is the framework most directly inspired by Pregel, and the closest to what AKA Flows needs. Its runtime is explicitly named "Pregel" and adopts the BSP model with important adaptations.

Graphs consist of **nodes** (computation functions) and **edges** (transition rules). Nodes receive the current state, perform computation, and return state updates. Edges can be fixed or conditional (decision functions that inspect state and return the next node).

**Unlike Pregel and traditional DAG systems, LangGraph supports cycles.** This is critical for iterative workflows (retry loops, agent self-correction, approval cycles).

### State Management with Pydantic

State is the central abstraction. A graph is parameterized by a state schema, which can be a TypedDict, dataclass, or Pydantic BaseModel.

```python
from typing import Annotated
from typing_extensions import TypedDict
import operator

class WorkflowState(TypedDict):
    input: str
    results: Annotated[list[str], operator.add]  # Reducer: append, don't overwrite
    status: str
```

**Reducers** define how concurrent state updates merge. Without a reducer, the last write wins. With `operator.add`, list values are concatenated. This is directly analogous to Pregel's combiners.

**Pydantic vs TypedDict:** LangGraph recommends TypedDict for internal graph state (zero overhead) and Pydantic BaseModel at API boundaries (full validation). This dual approach is worth adopting — validate at the edges, trust internally.

### Execution Model: Supersteps

LangGraph's execution proceeds in supersteps, exactly like Pregel:

1. Mark entrypoint node as active
2. Execute all active nodes in parallel (superstep completes when all finish)
3. Identify nodes that received state updates → activate them
4. Repeat until all nodes are inactive

**Transactional semantics:** If any node in a superstep fails, none of that superstep's updates are applied. This all-or-nothing guarantee ensures state consistency.

**Fan-out / fan-in:** A single node can activate multiple downstream nodes (parallel branches). A node with multiple incoming edges can optionally wait for all predecessors (join semantics) before executing.

### Checkpointing and Human-in-the-Loop

Every superstep produces a checkpoint — a snapshot of the full graph state. Checkpoints are indexed and retrievable, enabling:

- **Time-travel debugging**: Inspect any previous state
- **Human-in-the-loop**: Pause at a checkpoint, let a human inspect/modify state, resume
- **Fault tolerance**: Resume from last good checkpoint on failure
- **Conversation memory**: Maintain state across interactions

```python
# Compile with checkpointing and interrupt points
graph = builder.compile(
    checkpointer=SqliteSaver.from_conn_string(":memory:"),
    interrupt_before=["human_review_node"]
)
```

### API Design

```python
from langgraph.graph import START, END, StateGraph

# Define state
class State(TypedDict):
    data: str
    processed: bool

# Define nodes as plain functions
def process_node(state: State) -> dict:
    return {"data": state["data"].upper(), "processed": True}

def route_node(state: State) -> str:
    return "done" if state["processed"] else "process"

# Build graph
builder = StateGraph(State)
builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_conditional_edges("process", route_node, {"done": END, "process": "process"})

# Compile and run
graph = builder.compile()
result = graph.invoke({"data": "hello", "processed": False})
```

### Error Handling

LangGraph supports per-node retry policies with configurable backoff:

```python
from langgraph.pregel import RetryPolicy

retry = RetryPolicy(max_retries=3, backoff_type="exponential")
builder.add_node("unreliable", unreliable_fn, retry=retry)
```

Errors within a superstep prevent that superstep's state updates from committing (transactional rollback). This is much cleaner than exception propagation through a call stack.

### Trade-offs for AKA Flows

**Strengths:**
- Directly addresses our problem space (workflow graph orchestration)
- Superstep model gives deterministic, debuggable execution
- First-class support for cycles, conditional routing, and human-in-the-loop
- Checkpointing enables pause/resume, time-travel, and fault tolerance
- Clean API: nodes are plain functions, state is typed
- Pydantic integration for validation at boundaries
- Transactional superstep semantics prevent partial state corruption

**Weaknesses:**
- Tightly coupled to LangChain ecosystem and LLM agent use cases
- State is a single shared object (no per-node private state)
- No built-in distribution/parallelism across machines (single-process)
- Young project — API still evolving, fewer production battle scars
- Limited observability and monitoring compared to Beam/Flink

**Key patterns to borrow:**
- Superstep execution with transactional semantics
- State schema with reducers for merge conflict resolution
- Conditional edges for dynamic routing
- Checkpoint-per-superstep for debuggability and resilience
- Nodes as plain functions (minimal framework coupling)
- Compile step that validates graph structure before execution

---

## 5. Ray — Distributed Execution with Decorators

### Core Computational Model

Ray provides three primitives: **tasks** (stateless remote functions), **actors** (stateful remote classes), and **objects** (immutable distributed values). The `@ray.remote` decorator transforms ordinary Python functions and classes into distributed units of work.

```python
@ray.remote
def process(data):
    return transform(data)

# Async execution returning an ObjectRef (future)
ref = process.remote(input_data)
result = ray.get(ref)  # Block and retrieve
```

### DAG Execution

The Ray DAG API uses `.bind()` for lazy graph construction (vs `.remote()` for immediate execution):

```python
from ray.dag import InputNode

with InputNode() as user_input:
    step1 = validate.bind(user_input)
    step2 = process.bind(step1)
    step3 = finalize.bind(step2)
    dag = step3

# Execute the full DAG
result = ray.get(dag.execute(input_data))
```

Dependencies are resolved automatically through ObjectRefs. When task B depends on the output of task A, Ray ensures A completes before B starts. Independent tasks run in parallel without explicit scheduling.

### Stateful Nodes via Actors

```python
@ray.remote
class NodeState:
    def __init__(self):
        self.execution_count = 0
        self.cache = {}

    def execute(self, data):
        self.execution_count += 1
        result = expensive_computation(data)
        self.cache[data.id] = result
        return result
```

Actor methods execute serially on a single worker, so there are no concurrency issues within an actor. State is mutable and persistent across calls but lost on actor death (unless using Ray Workflows for durability).

### Ray Workflows (Durable Execution)

Ray Workflows add checkpointing on top of Ray tasks for exactly-once execution:

```python
from ray import workflow

@workflow.step
def step1(x):
    return expensive_compute(x)

@workflow.step
def step2(result):
    return refine(result)

dag = step2.bind(step1.bind(input_data))
workflow.execute(dag, workflow_id="flow-123")

# Resume after failure
output = workflow.get_output("flow-123")
```

Each step's result is persisted before the next step starts. On failure, execution resumes from the last completed step.

### Fault Tolerance

**Tasks:** Lineage-based reconstruction — if an object is lost, Ray re-executes the task that created it (and recursively, its dependencies). Requires deterministic, idempotent tasks.

**Actors:** Can auto-restart with `max_restarts` parameter, but state is lost on restart.

**Workflows:** Checkpoint-based, exactly-once semantics.

### Scalability

Ray scales from single-machine to cluster with no code changes — only the `ray.init()` call changes. The architecture is a head node (scheduler + global control store) plus worker nodes, each with a local object store (shared memory). Objects transfer via TCP when needed across nodes.

### Trade-offs for AKA Flows

**Strengths:**
- Decorator pattern makes distribution nearly transparent (`@ray.remote`)
- Automatic dependency resolution — no manual scheduling
- Seamless single-machine to cluster scaling
- Actors provide clean stateful node abstraction
- Ray Workflows add durability when needed
- `.bind()` for lazy graph construction enables validation before execution
- Mature, widely adopted in ML/AI production workloads

**Weaknesses:**
- DAG API is relatively basic — no built-in conditional routing or cycles
- Workflow scheduling logic must be built on top (it's a toolkit, not a framework)
- No built-in graph validation, topological analysis, or execution visualization
- Actor state loss on failure is a footgun
- Cluster management adds operational complexity
- Python-only (though this isn't a weakness for us)

**Key patterns to borrow:**
- Decorator-based node registration (`@ray.remote` → `@node`)
- ObjectRef/future pattern for async dependency resolution
- `.bind()` for lazy graph construction and validation
- Actor pattern for stateful nodes
- Lineage-based reconstruction for fault tolerance
- Resource annotations on decorators (`num_cpus`, `memory`)

---

## 6. Cross-Framework Comparison

### Execution Model

| Aspect | Pregel | Beam | Flink | LangGraph | Ray |
|--------|--------|------|-------|-----------|-----|
| **Core model** | BSP / supersteps | Pipeline DAG | Dataflow DAG | BSP / supersteps | Task DAG |
| **Supports cycles** | Via messages | No | No (iterations are special) | Yes (first-class) | No (manual) |
| **Conditional routing** | Via messages | Side outputs | Not natively | Conditional edges | Manual |
| **Parallelism** | Vertex-parallel | Bundle-parallel | Operator-parallel | Node-parallel per superstep | Task-parallel |
| **Deterministic** | Yes (BSP) | Runner-dependent | Yes (with event time) | Yes (supersteps) | No (async) |

### State Management

| Aspect | Pregel | Beam | Flink | LangGraph | Ray |
|--------|--------|------|-------|-----------|-----|
| **State scope** | Per-vertex value | Per-key per-window | Per-key + operator | Global shared state | Per-actor |
| **State validation** | Templated types | Coder inference | Serialization | Pydantic / TypedDict | Python types |
| **State persistence** | Checkpoint | Runner-managed | State backends | Checkpointer | Workflows |
| **Merge semantics** | Combiners | Combine transforms | Reduce/Aggregate | Reducers | Manual |

### Fault Tolerance

| Aspect | Pregel | Beam | Flink | LangGraph | Ray |
|--------|--------|------|-------|-----------|-----|
| **Mechanism** | Checkpoint + replay | Bundle retry | Barrier snapshots | Checkpoint per superstep | Lineage reconstruction |
| **Granularity** | Superstep | Bundle | Operator | Superstep | Task |
| **Exactly-once** | Yes | Yes | Yes | Yes (within superstep) | Workflows only |
| **Recovery cost** | Replay from checkpoint | Retry bundle | Replay from checkpoint | Replay from checkpoint | Re-execute lineage |

### Developer Experience

| Aspect | Pregel | Beam | Flink | LangGraph | Ray |
|--------|--------|------|-------|-----------|-----|
| **Node definition** | Subclass Vertex | Subclass PTransform/DoFn | Functions/classes | Plain functions | `@ray.remote` decorated |
| **Graph construction** | Implicit (data) | Pipeline builder | Environment builder | StateGraph builder | `.bind()` DAG |
| **Validation** | Compile-time (C++) | Runner submission | Job compilation | `compile()` | None built-in |
| **Learning curve** | Medium | High | High | Low | Low-Medium |
| **Python-native** | No (C++) | Yes (SDK) | Partial (PyFlink) | Yes | Yes |

### Operational Complexity

| Aspect | Pregel | Beam | Flink | LangGraph | Ray |
|--------|--------|------|-------|-----------|-----|
| **Infrastructure** | Google cluster | Runner cluster | Flink cluster | Single process | Single → cluster |
| **Dependencies** | C++ toolchain | Runner + SDK | JVM + PyFlink | pip install | pip install + ray |
| **Monitoring** | Custom HTTP | Runner UI | Flink dashboard | LangSmith | Ray dashboard |

---

## 7. Patterns and Concepts to Adopt

Based on the research, these are the patterns most relevant to making AKA Flows' executor maintainable, extensible, and robust.

### 7.1 Two-Phase Construction/Execution (from Beam, Ray, LangGraph)

Separate graph definition from execution. This enables validation (are all edges connected? are there orphan nodes? do types match?), optimization (can we fuse sequential nodes?), and debugging (inspect the graph before running it).

```
Phase 1: Build graph → validate → optimize
Phase 2: Execute with runtime context
```

All three of Beam, Ray, and LangGraph use this pattern. LangGraph's `compile()` step and Ray's `.bind()` are the cleanest implementations.

### 7.2 Superstep Execution with Transactional Semantics (from Pregel, LangGraph)

Execute the graph in discrete supersteps. Within each superstep, all eligible nodes run in parallel. State updates are applied atomically — if any node fails, the entire superstep rolls back.

This gives us: deterministic execution, easy debugging (inspect state between supersteps), clean error semantics (no partial state corruption), and natural checkpointing boundaries.

### 7.3 Nodes as Plain Functions with Typed State (from LangGraph, Ray)

Nodes should be plain Python functions that receive state and return updates. No framework superclass, no special lifecycle methods. This minimizes framework coupling and makes nodes easy to test independently.

```python
# Good: plain function with typed input/output
def my_node(state: WorkflowState) -> dict:
    return {"result": compute(state.input)}

# Avoid: framework-coupled class
class MyNode(FrameworkNode):
    def compute(self, context):
        ...
```

### 7.4 State Schema with Reducers (from LangGraph, Pregel)

Define state as a typed schema with explicit merge rules for concurrent updates. This eliminates the "who writes last wins" problem when parallel branches converge.

### 7.5 Conditional Edges for Dynamic Routing (from LangGraph)

Support edges that are functions of state, not just static connections. This enables workflows that branch based on intermediate results without requiring the node itself to know about routing.

### 7.6 Checkpoint-per-Superstep (from LangGraph, Pregel, Flink)

Save a state snapshot after each superstep. This enables pause/resume, human-in-the-loop, fault tolerance, and time-travel debugging — all from a single mechanism.

### 7.7 Error Routing over Exception Propagation (from Beam)

Instead of letting exceptions bubble up and crash the workflow, route errors to a dedicated channel. This lets the workflow continue on the happy path while collecting errors for review.

### 7.8 Operator Chaining / Node Fusion (from Flink)

Sequential nodes that don't need intermediate checkpointing or parallel execution can be fused into a single execution unit. This reduces scheduling overhead without changing semantics.

### 7.9 Decorator-Based Registration (from Ray)

Use Python decorators to register functions as nodes. This keeps node definitions clean and framework coupling minimal.

```python
@flow_node(retry=3, timeout=30)
def validate_input(state: FlowState) -> dict:
    ...
```

### 7.10 Compile-Time Graph Validation (from Beam, LangGraph)

Before execution, validate the graph structure: check for disconnected nodes, type mismatches between connected nodes, missing required state keys, cycles where not allowed, and unreachable nodes.

---

## 8. Recommendations for AKA Flows

### What to build toward

Based on this research, the ideal execution framework for AKA Flows would combine:

1. **LangGraph's execution model** (supersteps, transactional semantics, conditional edges, cycle support) as the core runtime
2. **Ray's decorator pattern** (clean node registration, minimal framework coupling) for developer experience
3. **Beam's error routing** (dedicated error channels, not exception propagation) for robustness
4. **Flink's operator chaining** (fuse simple sequential nodes) as an optimization
5. **Pydantic at the boundaries** (validate inputs/outputs) with lightweight internal state passing

### What NOT to do

- Don't adopt Beam or Flink wholesale — they solve the wrong problem (data pipelines, not workflow orchestration)
- Don't build custom distribution from scratch — if you need distribution later, Ray or Celery can be layered on
- Don't couple node implementations to the framework — keep nodes as plain functions
- Don't skip the compile/validate step — catching structural errors before execution prevents the hardest-to-debug failures

### Suggested next steps

1. **Extract the core execution model:** Implement a superstep executor that takes a validated graph and runs it in superstep cycles with transactional semantics
2. **Define the node contract:** A plain function that takes typed state and returns a partial state update
3. **Define the graph builder API:** StateGraph-like builder with `add_node()`, `add_edge()`, `add_conditional_edges()`, and a `compile()` step
4. **Implement state schema with reducers:** TypedDict/Pydantic state with explicit merge rules for fan-in scenarios
5. **Add checkpointing:** Persist state after each superstep for debuggability and resilience
6. **Add error routing:** Dedicated error channels that collect failures without stopping the workflow

---

## Sources

### Papers
- Malewicz et al., "Pregel: A System for Large-Scale Graph Processing," SIGMOD 2010. https://doi.org/10.1145/1807167.1807184

### Apache Beam
- Beam Programming Guide: https://beam.apache.org/documentation/programming-guide/
- Beam Execution Model: https://beam.apache.org/documentation/runtime/model/
- PTransform Style Guide: https://beam.apache.org/contribute/ptransform-style-guide/

### Apache Flink
- Flink Architecture: https://nightlies.apache.org/flink/flink-docs-master/docs/concepts/flink-architecture/
- Stateful Stream Processing: https://nightlies.apache.org/flink/flink-docs-master/docs/concepts/stateful-stream-processing/
- State Backends: https://nightlies.apache.org/flink/flink-docs-master/docs/ops/state/state_backends/
- Fault Tolerance: https://nightlies.apache.org/flink/flink-docs-master/docs/learn-flink/fault_tolerance/

### LangGraph
- Building LangGraph (design rationale): https://blog.langchain.com/building-langgraph/
- LangGraph Pregel and Supersteps: https://medium.com/@maksymilian.pilzys/langgraph-transactions-pregel-message-passing-and-super-steps-0e101e620f10
- Evolution from Pregel to LangGraph: https://medium.com/@pur4v/the-evolution-of-graph-processing-from-pregel-to-langgraph-6e8c2063df98

### Ray
- Ray Core: https://docs.ray.io/en/latest/ray-core/walkthrough.html
- Ray DAG API: https://docs.ray.io/en/latest/ray-core/ray-dag.html
- Ray Workflows: https://docs.ray.io/en/latest/workflows/basics.html
- Ray Fault Tolerance: https://docs.ray.io/en/latest/ray-core/fault-tolerance.html
