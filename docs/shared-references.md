# Shared references (producers and consumers)

**Status:** Proposal — v1 design
**Authors:** Rasmus Krebs, Claude
**Date:** 2026-04-17

---

## 1. Problem

Today the only way to move data between nodes is an explicit `GraphEdge`. This works for most flows but has three gaps:

1. **Fan-out is noisy.** A node that produces a single mapping used by ten downstream nodes requires ten edges.
2. **Edges cannot cross compound region boundaries.** A for-each body cannot consume a value produced by a node outside the loop. The canonical example: a system prompt defined once at the top of a flow and fed into an LLM node inside a for-each iterating over 1,000 inputs.
3. **`FlowStore` exists but is invisible in the graph.** It's imperative (`store.set("key", value)` inside the node body, `store.get("key")` in the consumer), so the dependency is not visible in the UI, not validated at compile time, and not part of the dependency graph used for scheduling.

Two concrete flows drive this work:

- **Pseudonymisation** — one node computes a `{name: pseudonym}` mapping; many downstream redaction nodes consume it.
- **LLM loop with shared system prompt** — a `system_prompt` node produces a string consumed by an `llm-chat` node inside a `for-each` iterating over records.

## 2. Goals

1. Let any node *instance* mark an output as a **shared reference** via a per-instance setting in the flow builder — no per-node-type engineering.
2. Let any node *instance* consume a shared reference from another node — including across compound region boundaries.
3. Compile-time validation: type checking, cycle detection, single producer per handle, no collision with explicit edges.
4. Zero required changes to existing node functions or classes. Nodes are unaware that their outputs may be shared.
5. Preserve determinism: a shared reference is an edge under the hood, declared differently and resolved against producer identity rather than handle position.

## 3. Non-goals (v1)

- **Producers inside compound regions.** In v1 a producer must be a top-level node. For-each bodies and conditional branches cannot define shared references. Follow-up work may add region-scoped producers (e.g., "value of X at end of region") once semantics are proven in practice.
- **Fan-in / multiple producers per reference.** A shared reference has exactly one producer.
- **Event-stream / last-writer-wins semantics.** A shared reference is computed once per flow run.
- **Dynamic / reactive updates.** Consumers do not re-run when a producer re-runs mid-flow. (There is no such mid-flow re-run today.)
- **Cross-flow references.** Shared references live inside a single `CompiledGraph`.

## 4. User-facing model

A **shared reference** is a runtime value exposed by a **producer** node and readable by any number of **consumer** nodes.

- **Producer:** a node that has opted in to sharing one or more of its outputs. Opting in is a per-instance setting in the flow builder, not a per-node-type concern.
- **Consumer:** a node that has one of its inputs bound to a producer's output by reference identity.
- **Reference identity:** the pair `(producer_node_id, output_handle)`. The user-visible label is display-only; references are bound by identity so that renaming the label never breaks subscribers.

Producers use the verb **"produce"**. Consumers use the verb **"consume"**. We call the value itself a **"shared reference"** (or just *reference* in context).

## 5. Data model

Two optional fields on `GraphNode`:

```python
@dataclass(frozen=True)
class GraphNode:
    id: str
    type: str
    data: dict[str, Any] | None
    # --- new ---
    produces: dict[str, str] | None = None
    consumes: dict[str, tuple[str, str]] | None = None
```

- `produces` maps **output handle → display label**. Presence of a handle in this dict is the opt-in signal that the output is a shared reference. The label is for UI rendering (node palettes, subscribe menus); it does not participate in resolution.
- `consumes` maps **input handle → (producer_node_id, output_handle)**. Each entry declares that the consumer's input should be filled by the producer's shared output.

**Both fields default to `None`**. A node with neither field is indistinguishable from a node authored before this feature existed — backward compatible.

### Example

```python
nodes = [
    GraphNode(
        "n1", "build-map@1", data={...},
        produces={"result": "pseudonym map"},
    ),
    GraphNode(
        "n2", "redact@1", data={"text": "Alice met Bob."},
        consumes={"mapping": ("n1", "result")},
    ),
]

edges = []   # no explicit edges required
```

### JSON wire format

When serialized to the flow JSON that host applications read and write, tuples become lists:

```json
{
  "id": "n2",
  "type": "redact@1",
  "data": {"text": "Alice met Bob."},
  "consumes": {"mapping": ["n1", "result"]}
}
```

Host applications converting flow JSON into `GraphNode` instances must turn the list back into a tuple. The core library accepts only tuples.

## 6. Compile-time semantics

All validation happens inside `compile()`. The following checks are added after edge validation (step 2) and before topological sort (step 3):

### 6.1 Producer validation

For each node with a non-empty `produces`:

1. The node must not be a managed node of a compound region (i.e., `node.id not in managed_ids`). Violation → `CompilationError("Node '{id}' cannot produce a shared reference from inside a compound region (v1 limitation)")`.
2. Every output handle in `produces` must exist on the node's registered outputs. Violation → `CompilationError("Node '{id}' produces unknown handle '{handle}' — not declared in node type '{type}'")`.
3. *(Optional UX check)* Labels should be unique across the graph. Duplicate labels are allowed but emit a `TypeWarning("shared-label-collision", ...)`.

### 6.2 Consumer validation

For each node with a non-empty `consumes`:

1. For each entry `input_handle → (producer_id, output_handle)`:
    1. `producer_id` must exist in `node_map`. Violation → `CompilationError`.
    2. The producer's `produces` dict must include `output_handle`. Violation → `CompilationError("Node '{id}' consumes '{producer_id}.{output_handle}' but that output is not produced as a shared reference")`.
    3. `input_handle` must exist on the consumer's registered inputs. Violation → `CompilationError`.
    4. The consumer input handle must not already have an explicit edge targeting it. Violation → `CompilationError("Input '{node_id}.{input_handle}' is both consumed and connected by an edge — choose one")`.

### 6.3 Implicit dependency edges

Consume bindings contribute to the dependency graph used for scheduling and cycle detection. Two equivalent implementations are acceptable:

**Option A — fold into edge_map.**
Before topological sort, inject a synthetic edge for every consume entry:

```python
for target_node in nodes:
    for target_handle, (producer_id, output_handle) in (target_node.consumes or {}).items():
        synthetic_edge = GraphEdge(
            id=f"__consume_{target_node.id}_{target_handle}",
            source=producer_id,
            target=target_node.id,
            source_handle=output_handle,
            target_handle=target_handle,
        )
        edges.append(synthetic_edge)
```

**Option B — track a parallel `consume_map`.**
Keep `edge_map` limited to explicit edges; build a second map that `_build_dep_graph` consults alongside `edge_map`.

**Decision:** Option B. It preserves the semantic distinction between explicit and implicit edges in `CompiledGraph` (useful for diagnostics, UI debugging, and future work). The implementation cost is marginally higher: `_build_dep_graph` and `InputResolver` read two maps instead of one.

`CompiledGraph` gains a field:

```python
consume_map: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)
# (target_id, target_handle) -> (producer_id, output_handle)
```

### 6.4 Topological sort

Topological sort consumes the combined dependencies (`edge_map` ∪ `consume_map`). A cycle that passes through a shared reference surfaces as `CycleDetectionError`. The error message must name the shared reference that closes the cycle, not just the two nodes involved.

### 6.5 Type checking

Type compatibility of each consume binding is checked using the existing `check_edge_types` machinery. A type-mismatched consume produces a `TypeWarning` (default) or raises `CompilationError` in `strict_types=True` mode — identical to explicit edge behavior.

## 7. Runtime semantics

### 7.1 Input resolution

`InputResolver.resolve` gains a single additional lookup, placed *after* explicit edges and *before* static data:

```
1. Explicit edge targeting this input  →  use edge value
2. consume_map has entry for this input →  use value from results[producer_id][output_handle]
3. Fall through to static data (node.data[input_handle])
4. Fall through to widget default (if any)
```

The engine already disallows (1) and (2) simultaneously at compile time (§6.2.1.4), so the first-match semantics are unambiguous.

### 7.2 Scheduling

`_build_dep_graph` in `execution/engine.py` reads `compiled.edge_map` to compute in-degree. It is updated to read `compiled.consume_map` as well, adding each producer → consumer pair as a dependency. Consumers wait for their producers the same way edge targets wait for edge sources.

### 7.3 SKIPPED propagation

If a producer is skipped, its consume binding resolves to SKIPPED — the consumer receives SKIPPED for that input, and the existing skip propagation logic applies (if all of the consumer's resolved inputs are SKIPPED, the consumer is skipped). This is consistent with how explicit edges handle SKIPPED sources.

### 7.4 Execution state

No new state container is required. Published values already live in `state.results[producer_id][output_handle]`, which is where the resolver reads them from.

### 7.5 Node runtime contract

Nodes are unaware of sharing. A producer returns its output normally — the runtime caches it in `state.results` just as it does today. A consumer receives the shared value in its normal `inputs[handle]` parameter. Neither the node function nor the `BaseNode` subclass sees anything new.

## 8. Compound region interaction

A consumer inside a compound region (for-each body, conditional branch) may consume a top-level producer. The compound node's internal executor reads from the shared `state.results` dict, so the value is available by the time any body node runs (the producer is ordered before the compound node by topological sort).

**Broadcast, not iteration.** The same producer value is read on every iteration. This is the intended behavior (system prompt broadcast into loop body). If a consumer wants per-iteration values, it uses the for-each loop's own `output_1` (current item) instead.

**Producers inside compound regions are disallowed in v1** (§6.1.1). Motivations for the v1 restriction:

- Semantics of "what value" are ambiguous: the value from iteration N? The last iteration? A list?
- Compound regions manage their own sub-execution; exposing their internal node results as graph-level shared references requires additional plumbing in `CompoundNodeType`.
- Real use cases (pseudonymisation, system prompt) don't need it. Deferring keeps v1 small.

When the restriction is relaxed, the natural semantics are "the last iteration's value" for for-each producers and "the active branch's value" for conditional producers — but those are v2 decisions.

## 9. Checkpoint and resume

Shared reference values survive checkpoint/resume automatically because they live in `state.results`, which is already part of `FlowCheckpoint.results`. No additional fields are needed on the checkpoint.

`consumes` and `produces` are part of the flow graph definition, not the checkpoint — the host application reconstructs the `CompiledGraph` from its stored flow definition on resume, just as it does today.

## 10. Host application / UI considerations

This library does not render UI. The host application is responsible for:

- A per-node "Share output as [label]" setting, one toggle per output handle.
- A per-input "Bind to shared reference" affordance, offering a dropdown of all type-compatible producers in the current flow. Selecting a producer from the dropdown should disable the drawn-edge affordance for that input handle (enforcing §6.2.1.4 in the UI).
- A distinct visual treatment for shared reference handles versus regular edge handles (different color, badge, etc.).
- Renaming a producer's display label updates only the UI label; the reference identity is the `(producer_id, output_handle)` tuple, which does not change.

The JSON returned by `serialize_registry(registry)` is unchanged — it describes node *types*, and shared references are a *per-instance* concept. Host applications surface shared references from the per-flow JSON alongside `GraphNode` and `GraphEdge`.

## 11. Explicit error cases (for testing)

| Scenario | Behavior |
|---|---|
| Consumer references a nonexistent `producer_id` | `CompilationError` |
| Consumer references a handle the producer does not list in `produces` | `CompilationError` |
| Consumer uses an input handle that does not exist on the consumer's node type | `CompilationError` |
| Input has both an explicit edge and a consume binding | `CompilationError` |
| Producer is inside a for-each or conditional region | `CompilationError` (v1 limitation) |
| Producer declares an output handle that does not exist on its node type | `CompilationError` |
| Cycle through shared references | `CycleDetectionError` |
| Type-incompatible consume binding, default mode | `TypeWarning` on `compiled.type_warnings` |
| Type-incompatible consume binding, `strict_types=True` | `CompilationError` |
| Skipped producer, consumer depends only on that producer | Consumer is skipped |
| Two producers publish the same display label | `TypeWarning` (non-fatal) |

## 12. Out of scope for v1 — revisit later

- **Producers inside compound regions** (see §8).
- **Multiple producers per reference** / fan-in.
- **Reference deprecation flows** — producer removed while subscribers exist. Host app concern; core validates strictly.
- **Named-reference namespacing** across nested flows / sub-flows. When sub-flows land, references may need a scope concept.
- **Schema JSON for labels** — we might want `serialize_registry` output or a separate endpoint to surface display labels, but that's a host-app decision.

## 13. Implementation checklist

When building this, in order:

1. Add `produces` and `consumes` fields to `GraphNode`.
2. Add `consume_map` to `CompiledGraph`.
3. Add validation passes (§6.1, §6.2) to `compile()`.
4. Extend `_build_dep_graph` in `execution/engine.py` to include `consume_map`.
5. Extend `InputResolver.resolve` with the consume lookup (§7.1).
6. Extend `check_edge_types` (or add a sibling helper) to cover consume bindings (§6.5).
7. Update error messages for `CycleDetectionError` to mention shared references when applicable (§6.4).
8. Remove the skip marker on `tests/test_core/test_shared_references.py`.

All changes stay in `packages/conductor/src/conductor/` — no changes to demo, examples, or registry schema.

## 14. Open questions

1. **Display-label uniqueness:** warn or enforce? Proposal warns; could be tightened later without breaking flows.
2. **Error message conventions:** exact wording of the new error messages. Proposal sets a baseline; refine in code review.
3. **Do we surface `consumes`/`produces` on `CompiledGraph` debug output?** Probably yes — useful for host app diagnostics — but not critical for v1.
