// Shape of `/api/nodes` — matches conductor.registry.schema.serialize_registry.
export type Widget =
  | "text"
  | "textarea"
  | "template-textarea"
  | "dropdown"
  | "dependent-dropdown"
  | "range"
  | "number"
  | "checkbox"
  | "switch"
  | "datepicker"
  | "file"
  | "schema-builder"
  | "multiselect"
  | "entity-dropdown"
  | "if-else-builder"
  | "connection-list"
  | "output"
  | "list"
  | "code-editor";

export type NodeCategory = "io" | "control";

export interface NodeParam {
  name: string;
  type: string;
  label: string;
  description?: string | null;
  widget: Widget;
  default?: unknown;
  optional: boolean;
  disable_handle?: boolean;
  // widget-specific:
  rows?: number;
  choices?: string[];
  min_val?: number;
  max_val?: number;
  step?: number;
  integer_only?: boolean;
}

export interface NodeOutput {
  name: string;
  type: string;
  label: string;
  description?: string | null;
  optional: boolean;
  download?: boolean;
  filename?: string | null;
}

export interface NodeType {
  id: string; // "text@1"
  base_id: string;
  version: number;
  name: string;
  description: string;
  tags: string[];
  category: NodeCategory;
  inputs: NodeParam[];
  outputs: NodeOutput[];
  width?: number | null;
  deprecated?: boolean;
  latest_version?: number;
  docs?: string | null;
}

// What we send to /api/execute-stream.
export interface FlowNodePayload {
  id: string;
  type: string;
  data: Record<string, unknown>;
  produces?: Record<string, string>; // handle → label
  consumes?: Record<string, [string, string]>; // input handle → [producer_id, output_handle]
}

export interface FlowEdgePayload {
  id: string;
  source: string;
  target: string;
  source_handle: string | null;
  target_handle: string | null;
}

// Execution events (matches conductor.execution.events).
export type ExecutionEvent =
  | { type: "node_start"; node_id: string }
  | { type: "node_complete"; node_id: string; result: unknown; cached?: boolean }
  | { type: "node_skipped"; node_id: string }
  | { type: "node_error"; node_id: string; error: string; is_validation: boolean }
  | { type: "node_progress"; node_id: string; current: number; total: number }
  | { type: "node_retry"; node_id: string; attempt: number; max_retries: number; error: string; delay: number }
  | { type: "flow_complete"; results: Record<string, Record<string, unknown>> }
  | { type: "flow_error"; error: string; is_validation: boolean }
  | { type: "flow_cancelled"; completed_nodes: string[] }
  | { type: "flow_timeout"; completed_nodes: string[]; elapsed_seconds: number; timeout_seconds: number }
  | { type: "flow_paused"; node_id: string; prompt: string; schema: unknown; checkpoint: unknown };

// ReactFlow-side state we attach to each node.
export type ExecutionState = "idle" | "running" | "completed" | "error" | "skipped";

export interface LoopProgress {
  current: number;
  total: number;
}

// ReactFlow v12 requires node data to satisfy Record<string, unknown>.
export type CustomNodeData = {
  nodeType: NodeType;
  values: Record<string, unknown>;
  // Shared references:
  produces: Record<string, string>; // handle → label
  consumes: Record<string, [string, string]>;
  // Runtime:
  executionState: ExecutionState;
  result?: Record<string, unknown>;
  error?: string;
  progress?: LoopProgress;
  // For the shared-ref picker — all published refs in the flow.
  availableRefs: Array<{ nodeId: string; nodeLabel: string; handle: string; label: string }>;
  // Callbacks from the parent canvas.
  onValueChange: (nodeId: string, name: string, value: unknown) => void;
  onTogglePublish: (nodeId: string, handle: string, label: string | null) => void;
  onSetConsume: (nodeId: string, input: string, ref: [string, string] | null) => void;
  onDelete: (nodeId: string) => void;
  // True when a drawn edge already fills this input handle.
  connectedInputs: Set<string>;
} & { [key: string]: unknown };
