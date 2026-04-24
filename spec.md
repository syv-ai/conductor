Implement all of this and write your progress in PROGRESS.md - I want ALL features IMPLEMENTED and well
     tested.  Conductor Process-Standard Change Spec


       Ten changes, grouped by effort and depth. Each section covers what the change is, why it matters, how it interacts with existing Conductor, key design decisions that need a call, and what's explicitly out of scope.


     ---  Tier 1 — Metadata and modeling primitives


       These changes add declarative surface area without rewriting the engine. They're cheap to build and have outsized impact on what processes can express.

     1. Decision node with edge guards

       What it is. A new first-class node type (let's call it decision) whose only purpose is branching. Unlike a regular node, a decision node does no computation — it evaluates expressions on its outgoing edges and
     selectively
        emits SKIPPED down the branches that don't match. Each outgoing edge from a decision node carries an optional when expression and a priority integer. Evaluation walks edges in priority order; the first edge whose
     when
       evaluates true is the "taken" branch, and every other outgoing edge is marked SKIPPED. An edge with no when is the else fallback, and there must be exactly one per decision node.


       Why we need it. Conductor already supports branching through the SKIPPED sentinel — nodes like logic-if-empty and logic-if-equals return SKIPPED on one side and a real value on the other, and downstream nodes with
       all-SKIPPED inputs are themselves skipped. This works, but it has two problems for a process standard. First, the branching logic is hidden inside node Python; you can't look at the graph and see "this is where the
       decision happens" — you have to read code. Second, a decision that depends on fifteen conditions becomes fifteen connected logic-if-* nodes, which is unreadable. A first-class decision node makes branching visible as
      a
       shape on the diagram, with the conditions written as data on the edges. This is how BPMN, AWS Step Functions, and Temporal all model decisions, and it's what makes a process diagram legible to a non-programmer.

       How it fits into Conductor. The runtime mechanism stays the same — SKIPPED propagation is battle-tested and doesn't need replacing. What changes is how SKIPPED is produced. Today, a user-authored function decides;
       tomorrow, the engine evaluates a CEL expression on an edge and decides. The scheduler, cycle detector, and type system don't need to know a decision node is special. The compiler gets a new check: every decision node
      must
        have exactly one else-edge and at least one guarded edge.


       Key design decisions.
     - What does a decision node "produce" as its result? Probably pass its inputs through unchanged, so downstream nodes can still read the data that drove the decision. Alternative: it produces nothing, and consumers must
       read from the original producer upstream.

     - Priority ordering on edges: explicit integer, or implicit by edge-creation order? Explicit is more robust (stable under graph reshuffles) but adds clutter. I'd lean explicit with a sensible default.
     - Can a regular (non-decision) node's outgoing edges also carry when guards? This would let any node be a mini-decision. Cleaner to keep it restricted to decision for now — the shape on the diagram is the point.

       Non-goals. Not a replacement for existing logic-if-* nodes, which are still useful when branching depends on complex computation. The decision node handles the common case of "branch on a simple expression against
       already-computed values."


     2. CEL as the expression language

       What it is. Adopt Common Expression Language (CEL) as the single expression language used across the standard: edge when guards, while-loop conditions, idempotency keys, input-mapping expressions, trigger filters.
     CEL is
       Google's standardized expression language, sandboxed by design, used inside Kubernetes admission controllers, Envoy, and gRPC. There's a mature Python implementation (cel-python), and the syntax is familiar
       (invoice.amount > 1000 && customer.tier == "gold").


       Why we need it. Any declarative process standard needs to evaluate expressions at points where user-written Python doesn't belong — on edges, in loop conditions, in idempotency keys. Embedding full Python there is
     unsafe
       (sandboxing Python is a nightmare) and makes the flow format non-portable. A tiny homegrown DSL is tempting but ends up reinventing CEL badly. CEL has three properties we need: it's sandboxed (no I/O, no side
     effects,
       bounded execution), it's typed (we can statically check that a when returns a boolean), and it's standardized (future you doesn't have to explain or maintain a parser). JSONLogic is the main alternative but stores
       expressions as nested JSON, which is painful to read and write.

       How it fits into Conductor. Add cel-python as the only new dep (or keep it optional behind an extra, if minimizing the core footprint matters). The compiler gains a small expression-validation pass: for every
       when/condition/idempotency_key, parse the CEL expression and check its types against the available context. Runtime, the engine evaluates expressions at the relevant points (edge-taken decision, loop iteration, node
       start).


       Key design decisions.
     - The context schema must be standardized and documented: something like $.inputs.* for process inputs, $..* for a node's results, $.ctx.* for a FlowStore-like global scratchpad. This is the contract users write
       against.

     - Type-check expressions at compile time? Yes if we can — it catches typos like invoice.amont > 1000 before runtime. Requires knowing output types, which Conductor already tracks via widget annotations.
     - Error handling at runtime: if a when expression throws (e.g., missing field), does the branch count as false, raise, or take the else edge? Probably raise — silent fallthrough hides bugs.

       Non-goals. CEL is for routing and gating, not business logic. If you need to compute a value, that still happens in a node. The line: if it fits on one line and doesn't touch I/O, it's an expression; otherwise it's a
       node.


     3. Actor metadata on nodes

       What it is. A new metadata field on node registration that declares who performs the step: system (an automated service), human (a person taking an action), agent (an AI agent making a decision), or external_service
     (a
       third-party system the flow calls out to). Can be a simple string or a structured form like {kind: "human", role: "finance_manager"} to specify the human role.

       Why we need it. A process-as-code standard has to answer "who does this?" for every step. Conductor today treats every node as "a function" — there's no declared distinction between a node that runs a Python function
      and
       a node that asks a human for approval (other than HumanInputRequired, which is a runtime signal, not a declaration). Actors matter for three reasons. First, frontends can render human steps differently from system
     steps,
       making process diagrams legible to stakeholders who don't care about the code. Second, audit trails can attribute actions: "this approval was performed by finance_manager." Third, estimation and SLA tracking care
     deeply —
        a human step takes hours to days, a system step takes milliseconds.

       How it fits into Conductor. Pure metadata — the engine itself doesn't care. HumanInputRequired remains the runtime primitive for pausing on human actors; the actor tag is the declaration that composes with it. The
     schema
       exporter includes actor in the JSON metadata, so frontends can read it without a bespoke protocol.

       Key design decisions.

     - Fixed enum or open set? Fixed is safer for tooling; open lets projects define their own actor taxonomies. Compromise: fixed set of kinds, free-form role inside each.
     - Should a human actor automatically imply a pause? No — a flow might record "the doctor signed off" without actually blocking on a human; the signing happened out of band. The pause semantics belong to the runtime
     node,
       not the declaration.

       Non-goals. Not an access control system. "Who can trigger this flow" and "who can approve step X" are permission questions; the actor tag is a description, not a rule. A permission layer sits above and consumes the
     actor
       metadata.


     4. Top-level dependencies section

       What it is. A declaration block at the flow level listing every dependency the flow touches: databases, external APIs, subprocess flows, notification channels, message queues. Each dependency gets a stable id, a
     kind, and
        config metadata (endpoint, auth method, etc.). Nodes that touch a dependency declare it via a uses: list on their registration.

       Why we need it. A process standard should answer "what does this flow touch?" without executing it. That question comes up constantly: SOC2 audits, change-impact analysis ("we're migrating Stripe — which flows need
       updating?"), rate-limit coordination ("this flow makes three calls to Stripe; limit concurrent instances"), and onboarding ("what services does this process depend on?"). Today, the only way to answer it is to read
     every
       node's Python. A declarative dependencies block turns the answer into a query.


       How it fits into Conductor. Purely additive. The engine doesn't enforce dependency rules — it just stores and surfaces them. The compile step gains a validation: every uses: entry on a node must reference a declared
       top-level dependency. Hosts consume the list and implement semantics: credential injection, connection pooling, circuit breakers, rate limiting.

       Key design decisions.

     - Where does dependency config live? Inside the flow (static), or referenced by id into a host-managed registry (dynamic)? Referenced is more realistic for production: the flow declares "I use stripe," and the host's
     dep
       registry says what "stripe" means in each environment.

     - Should dependencies be typed (api, db, queue, subprocess, notification)? Yes — it lets tooling filter and reason (e.g., "all flows that hit a database, for the DBA to review").

       Non-goals. Not an IoC container. Not credential storage. The dependencies block is a manifest that answers "what," not "how."


     5. Top-level triggers section

       What it is. A declarative list of what can start the flow: manual (UI button / API call), schedule (cron), event (named signal), webhook (HTTP endpoint). Each trigger is a small config blob that a host can read to
     wire up
        external machinery.

       Why we need it. Triggers are the other half of "portable process definitions." If the flow lives in Conductor but the cron job lives in the host's config, moving the flow means rewiring. Declaring triggers in the
     flow
       makes it a self-contained artifact: you can hand the flow file to a new host, and it knows how the flow gets started. It's also documentation — a reader learns from the flow itself that "this runs every Monday at 9
     AM."

       How it fits into Conductor. Zero runtime impact. Conductor does not implement cron, does not expose webhook endpoints, does not listen on message queues — and should not. The flow declares its triggers as metadata;
     the
       host inspects them and wires the external machinery. The Conductor CLI or a helper library can surface a "collect all triggers from these flow files" command to help hosts.

       Key design decisions.
     - Trigger-specific config shapes: schedule needs a cron expression and timezone; webhook needs a path and optional auth spec; event needs a name and correlation filter. Each kind has its own schema, all under a common
       triggers: list.

     - Input mapping: triggers produce data (cron payload, webhook body), which becomes the flow's inputs. A map: CEL expression per trigger lets the flow accept one shape regardless of trigger kind.

       Non-goals. Not a scheduler. Not an HTTP router. Not a message bus client. This is declaration only — the host implements all three.


     6. Per-node timeout and idempotency_key

       What it is. Two new kwargs on @registry.node():

     - timeout: an ISO 8601 duration or seconds value. The engine wraps the node's execution in asyncio.wait_for; if it exceeds the budget, the engine raises NodeTimeoutError (which already exists in the hierarchy but isn't
       user-configurable).

     - idempotency_key: a CEL expression evaluated against the node's inputs. The resulting string is surfaced on the node_start event and passed into the node function if it accepts an idempotency_key: parameter. The node
       uses it when calling external systems (e.g., Stripe's Idempotency-Key header).

       Why we need it. Timeout: without a per-node cap, a hung external call can stall a flow indefinitely. Retry machinery can't rescue a node that never returns — the retry only kicks in after the current attempt raises.
     The
       retry config is about transient failures; the timeout is about liveness. You need both, and they're orthogonal. Idempotency: when a node retries (or when a flow is resumed from a checkpoint), external side effects
     must
       not happen twice. The idempotency key is how downstream services dedupe requests. It has to be stable across retries, so the engine must compute it once from inputs and pass the same key every attempt — which is why
     it
       belongs in the framework, not in ad-hoc node code.


       How it fits into Conductor. Both are additive. Timeout wires into the existing async dispatch path. Idempotency key is advisory metadata from the engine's perspective — the engine computes and exposes it, but the
     node
       function is responsible for actually using it when calling external services. The NodeTimeoutError is already in the error hierarchy, so the error-handling path is in place.

       Key design decisions.
     - On timeout, retry or fail? Should follow the existing retry config. Usually yes — timeouts often reflect transient network issues.
     - Should idempotency keys auto-include a retry counter? No — then every retry has a different key, defeating the purpose. The whole point is stability across retries.

     - What about cancellation? Timeout should cleanly cancel the node task and run any compensation (once we have it). Clean cancellation in asyncio.to_thread is tricky; probably best-effort.

       Non-goals. Idempotency key is not a distributed lock. It's advisory: the engine surfaces it, but doesn't prevent double-execution on its own. That's the job of the downstream service.


     ---  Tier 2 — New compound regions


       These extend Conductor's existing compound-region machinery (already used for for-each) to cover loop and composition patterns. They're structural but don't require rewriting the scheduler.

     7. While / until compound region

       What it is. A new compound region type, while-start / while-end, mirroring the existing for-each-start / for-each-end pattern. The start node holds a CEL condition; before each iteration, it evaluates the condition
       against the loop's state; if true, the body runs; if false, the loop exits. The end node collects the final body result (not a list — while loops typically care about the last iteration, not the history). A
     max_iterations
        safety cap prevents runaway loops.


       Why we need it. For-each handles "iterate over this collection" — a known, bounded set. But many real processes loop on conditions, not collections: retry-with-backoff until success, poll-until-ready,
       paginate-until-empty, wait-until-threshold-hit. None of these know their iteration count up front. Today you'd have to fake it with a giant for-each over a synthetic range, which is awkward and still bounded. A while
       compound is the natural shape.


       How it fits into Conductor. Reuses the entire region machinery: discovery by type-name prefix, compilation, scheduler integration, cancellation checks between iterations, node_progress events. The only new piece is
     the
       condition evaluation. Shared references work identically — a consumer inside the body sees the same producer value every iteration (broadcast semantics), which matches intuition for "use the same config on every
     retry."

       Key design decisions.
     - Loop protocol: each iteration, what's in scope? For-each gives (item, index). While should give (iteration_count, last_body_result). Needs to be a fixed, documented protocol.
     - Until vs while: are these separate compounds, or is until just while with a negate: true flag and body-first ordering? Single compound with flags is simpler.

     - Can the body be empty (zero iterations)? Yes — if the condition is false initially, the loop runs zero times and emits no body events. The end node produces a null or a sentinel.
     - What happens at max_iterations? Raise a LoopRunawayError — silent truncation masks real bugs. For-each has similar logic; align the behavior.

     - Cancellation: check state.is_cancelled() between iterations, same as for-each's sequential mode.

       Non-goals. Not a general goto. Not arbitrary cycles. Still a structured region with a single entry and exit, which preserves Conductor's DAG invariant (the region is a "super node" in the outer DAG).


     8. Subprocess as a first-class compound

       What it is. A subprocess-call node type that references another flow by id and version, maps inputs from the caller's scope into the sub-flow's inputs, and binds the sub-flow's outputs back into the caller. Default
     mode:
       sync — the caller waits for the sub-flow to complete. Optional mode: async — the caller gets a handle, and the sub-flow's result is collected later.

       Why we need it. Composition is non-negotiable for a process standard. Real organizations have shared sub-flows: fraud checks, notification fan-outs, approval chains, payment processing. Without first-class
     subprocesses,
       every caller inlines the sub-flow or rebuilds it — both are bad. The ExtensionResolver protocol lets hosts hack subprocess-like behavior today, but it's host-specific and doesn't participate in Conductor's type
     system.
       Making subprocess native means type checking sub-flow inputs and outputs against caller bindings at compile time, versioning (callers pin to a specific sub-flow version), and unified observability.


       How it fits into Conductor. The subprocess node compiles the referenced sub-flow at the caller's compile time (closed-world mode — full type checking, cycle detection across flow boundaries) or at runtime (open-world
      —
       flows discovered dynamically). Both modes should be supported. HITL inside a sub-flow propagates up: if the sub-flow pauses, the outer flow pauses too, and the checkpoint captures both. Sub-flow events appear in the
     outer
        event stream tagged with a parent_node_id so consumers can reconstruct the call hierarchy.


       Key design decisions.
     - Closed-world vs open-world compilation: closed is stricter (catches more errors early); open is more flexible (flows can be added without recompile). Support both via a flag; default to closed.
     - Cycle detection: A calling B calling A is a real risk. In closed-world compile, detect at compile time. In open-world, enforce a runtime depth limit.

     - Error propagation: a sub-flow error bubbles as what in the outer flow? Probably a new SubprocessFailedError that wraps the inner error and preserves the failing node id inside the sub-flow, so error messages are
     useful.
     - Shared references across flow boundaries: inbound only? Outbound probably not — otherwise sub-flows leak state back into callers. Keep it one-way.

       Non-goals. Not remote procedure call. Sub-flows run in the same Conductor engine by default. Remote execution is a host-extension concern (via the existing ExtensionResolver or a new executor hook).


     ---  Tier 3 — Hard, high-value additions


       These require real engineering and have design questions with no cheap answers. They're what separates a toy process engine from one that runs real business-critical flows.

     9. Compensation / saga support

       What it is. Per-node compensation: field pointing to another node that can "undo" the first node's work. When a flow fails, the engine walks all completed nodes in reverse topological order and runs each one's
       compensation node. Compensation nodes are regular nodes with special access to their target's inputs and outputs. Per-node on_error: policy (fail, continue, compensate) decides what happens when a given node fails —
       compensate is the one that triggers the cascade.


       Why we need it. Partial failure is the hardest problem in distributed processes. Without compensation, a half-completed flow leaves external systems in inconsistent state: money charged but order never saved, email
     sent
       but record never persisted, inventory reserved but payment failed. Every serious orchestrator — Temporal, AWS Step Functions, Camunda, Zeebe — supports compensation, because it's the only sensible answer to "how do
     we
       undo what we've already done?" The saga pattern (compensating transactions in reverse order) is industry-standard.


       How it fits into Conductor. This is the biggest change of the ten. The engine gains a new phase: after a failed flow, walk the completed subgraph in reverse topological order, dispatch each completed node's
     compensation
       node with access to the original inputs and outputs, and emit new events (compensation_start, compensation_complete, compensation_failed). The existing state.results dict already holds what compensations need to
     read. The
        error hierarchy needs a new type for "compensation also failed" scenarios.


       Key design decisions.

     - Best-effort or strict? Best-effort (continue compensating even if one fails) is the conventional answer — you want to roll back as much as possible. Strict (halt on first failure) leaves partial state and is rarely
     what
        anyone wants.

     - Compensating a loop: per-iteration, or whole-loop? Per-iteration is more correct but complicated (each iteration's compensation needs its own iteration-scoped state). Whole-loop is simpler but often wrong.
     - Compensating a subprocess: recursively run the sub-flow's compensation plan. The sub-flow defines its own compensation boundary.

     - Compensating HITL: you can't un-ask a human. The compensation for a HITL node is typically "notify that we're rolling back" — a different node that sends a message rather than reverses the decision.

     - What compensations receive: the node's inputs, its outputs, and the error that triggered compensation. All three matter — undoing an email requires knowing which email; undoing a charge requires knowing the charge
     id.
     - Idempotency: compensations should be idempotent (running twice is safe), because compensation itself can fail and be retried.

       Non-goals. Not a distributed transaction. No two-phase commit. No consensus protocol. Compensation is semantic undo — the business logic of "how to reverse this" — not a low-level transactional guarantee.


     10. Signal / event node (external wait)

       What it is. A new node type (event or signal) that pauses the flow until an external event arrives. It's the general-purpose cousin of HITL: HITL waits for a human response, event waits for any external signal — a
     webhook
        from a payment processor, a file arrival in S3, a timer expiry, a message from a queue. Each event node declares a name, an optional correlation expression (CEL) to match incoming signals against, an optional
     timeout,
       and an optional on_timeout: edge for fallback.


       Why we need it. Real processes wait on non-human events. A payment flow waits for a webhook confirming the charge. A data pipeline waits for a file. A reminder system waits for a timer. Today, Conductor has HITL for
       humans but no general async event primitive. Users end up bodging it — either misusing HITL for non-human events, or polling in a busy-loop node. A proper event node unifies the "pause and wait" abstraction and lets
     hosts
        wire up their own signal sources cleanly.


       How it fits into Conductor. Reuses most of HITL's infrastructure: checkpoint the flow to JSON, persist state, resume on signal. The new pieces are (a) a signal registry — when a flow pauses on an event, the host
     needs to
       know "which flows are waiting on which signals, with which correlation keys" so it can route incoming signals correctly; and (b) a timer subsystem — for pure time-based waits, Conductor can manage the timer
     internally
       without a host callback. Correlation is important: "payment_received" signals often target a specific invoice, and the flow supplies a correlation expression like invoice.id == $.signal.invoice_id that the host
     evaluates
       against incoming signals to decide which waiting flow to resume.

       Key design decisions.
     - Signal registry API: how does the host query "which flows are waiting on what"? Probably a function on the paused checkpoint that returns (signal_name, correlation_expression, timeout_deadline), which the host
     persists
       and queries on signal arrival.

     - Timer vs external signal: conceptually one node type with a kind: discriminator, or two separate nodes? One is cleaner; two is clearer. Lean toward one with a kind.
     - Correlation expression evaluation: the host passes the candidate signal and its payload to Conductor, which evaluates the correlation expression; if true, Conductor resumes the flow. This keeps the evaluation in
       Conductor (consistent with CEL semantics) without making Conductor a message bus client.

     - On timeout: the flow takes the on_timeout: edge, which points at a fallback path (often "escalate to human" or "emit a failure event"). Timeout is orthogonal to compensation.

       Non-goals. Not a message bus. Not a queue consumer. Not a webhook server. Conductor surfaces "I'm waiting on X with correlation Y"; the host wires X and Y to its message bus / webhook endpoint / timer daemon and
     calls
       back when they fire.


     ---  One cross-cutting concern: the process file format

       None of these changes matter unless we also commit to a canonical file format — YAML, JSON, or a Python DSL — for expressing flows declaratively. Conductor today is Python-API-first: you instantiate GraphNode and
       GraphEdge in code. That's great for programmatic flow building, but a process standard needs a serialization format that non-programmers can read and write, that version control can diff, and that tooling can
     validate
       against a schema.

       The React Flow provider already does half the job — it serializes to a React Flow JSON shape. But React Flow JSON is for rendering, not for authorship. The standard needs a layered format: a high-level author-facing
     layer
        (probably YAML) that compiles down to the React Flow wire format, which is what the engine executes. Without this, the ten changes above are library features, not a standard.
