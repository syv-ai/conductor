"use client";
import * as React from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  addEdge,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Connection,
  type Edge,
  type Node,
  type OnConnect,
  type XYPosition,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type {
  CustomNodeData,
  ExecutionEvent,
  ExecutionState,
  FlowEdgePayload,
  FlowNodePayload,
  LoopProgress,
  NodeType,
} from "@/lib/types";
import { CustomNode } from "./CustomNode";

export type LogItem = { id: number; event: ExecutionEvent; time: number };

interface Props {
  nodeTypes: NodeType[];
  onEventLog: (log: LogItem[]) => void;
  runSignalRef: React.MutableRefObject<() => Promise<void>>;
  clearSignalRef: React.MutableRefObject<() => void>;
  loadExampleSignalRef: React.MutableRefObject<(example: "basic" | "shared-refs" | "for-each") => void>;
  runningRef: React.MutableRefObject<boolean>;
}

const rfNodeTypes = { conductor: CustomNode };

interface GraphNode extends Node {
  data: CustomNodeData;
}

function FlowCanvasInner({
  nodeTypes,
  onEventLog,
  runSignalRef,
  clearSignalRef,
  loadExampleSignalRef,
  runningRef,
}: Props) {
  const [nodes, setNodes, onNodesChange] = useNodesState<GraphNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const nextIdRef = React.useRef(1);
  const logRef = React.useRef<LogItem[]>([]);
  const { screenToFlowPosition } = useReactFlow();

  const typeMap = React.useMemo(() => {
    const m = new Map<string, NodeType>();
    nodeTypes.forEach((t) => m.set(t.id, t));
    return m;
  }, [nodeTypes]);

  // =================================================================
  // Node/edge mutators, passed to CustomNode via data
  // =================================================================

  const updateData = React.useCallback(
    (nodeId: string, patch: (d: CustomNodeData) => Partial<CustomNodeData>) => {
      setNodes((prev) =>
        prev.map((n) => (n.id === nodeId ? { ...n, data: { ...n.data, ...patch(n.data) } } : n)),
      );
    },
    [setNodes],
  );

  const onValueChange = React.useCallback(
    (nodeId: string, name: string, value: unknown) => {
      updateData(nodeId, (d) => ({ values: { ...d.values, [name]: value } }));
    },
    [updateData],
  );

  const onTogglePublish = React.useCallback(
    (nodeId: string, handle: string, label: string | null) => {
      updateData(nodeId, (d) => {
        const nextProduces = { ...d.produces };
        if (label === null) delete nextProduces[handle];
        else nextProduces[handle] = label;
        return { produces: nextProduces };
      });
    },
    [updateData],
  );

  const onSetConsume = React.useCallback(
    (nodeId: string, input: string, ref: [string, string] | null) => {
      updateData(nodeId, (d) => {
        const nextConsumes = { ...d.consumes };
        if (ref === null) delete nextConsumes[input];
        else nextConsumes[input] = ref;
        return { consumes: nextConsumes };
      });
    },
    [updateData],
  );

  const onDelete = React.useCallback(
    (nodeId: string) => {
      setNodes((prev) => prev.filter((n) => n.id !== nodeId));
      setEdges((prev) => prev.filter((e) => e.source !== nodeId && e.target !== nodeId));
      // Also clear any consume references pointing at the deleted node.
      setNodes((prev) =>
        prev.map((n) => {
          const newConsumes = { ...n.data.consumes };
          let changed = false;
          for (const [inp, [pid]] of Object.entries(newConsumes)) {
            if (pid === nodeId) {
              delete newConsumes[inp];
              changed = true;
            }
          }
          return changed ? { ...n, data: { ...n.data, consumes: newConsumes } } : n;
        }),
      );
    },
    [setNodes, setEdges],
  );

  // =================================================================
  // Published-refs index, injected into every node
  // =================================================================

  const availableRefs = React.useMemo(
    () =>
      nodes.flatMap((n) =>
        Object.entries(n.data.produces).map(([handle, label]) => ({
          nodeId: n.id,
          nodeLabel: n.data.nodeType.name,
          handle,
          label,
        })),
      ),
    [nodes],
  );

  // connectedInputs per node (derived from edges)
  const connectedByNode = React.useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const e of edges) {
      if (!map.has(e.target)) map.set(e.target, new Set());
      if (e.targetHandle) map.get(e.target)!.add(e.targetHandle);
    }
    return map;
  }, [edges]);

  // Keep data callbacks + availableRefs fresh on every node.
  const nodesWithCallbacks = React.useMemo<GraphNode[]>(
    () =>
      nodes.map((n) => ({
        ...n,
        data: {
          ...n.data,
          availableRefs,
          connectedInputs: connectedByNode.get(n.id) ?? new Set(),
          onValueChange,
          onTogglePublish,
          onSetConsume,
          onDelete,
        },
      })),
    [nodes, availableRefs, connectedByNode, onValueChange, onTogglePublish, onSetConsume, onDelete],
  );

  // =================================================================
  // Drag/drop from palette
  // =================================================================

  const onDragOver = React.useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const addNodeAt = React.useCallback(
    (typeId: string, position: XYPosition) => {
      const nt = typeMap.get(typeId);
      if (!nt) return;
      const id = `n${nextIdRef.current++}`;
      const values: Record<string, unknown> = {};
      nt.inputs.forEach((inp) => {
        if (inp.default !== null && inp.default !== undefined) values[inp.name] = inp.default;
      });
      const newNode: GraphNode = {
        id,
        type: "conductor",
        position,
        dragHandle: ".drag-handle",
        data: {
          nodeType: nt,
          values,
          produces: {},
          consumes: {},
          executionState: "idle",
          availableRefs: [],
          connectedInputs: new Set(),
          onValueChange,
          onTogglePublish,
          onSetConsume,
          onDelete,
        },
      };
      setNodes((prev) => [...prev, newNode]);
      return id;
    },
    [typeMap, setNodes, onValueChange, onTogglePublish, onSetConsume, onDelete],
  );

  const onDrop = React.useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const typeId = e.dataTransfer.getData("application/conductor-node-type");
      if (!typeId) return;
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      addNodeAt(typeId, position);
    },
    [addNodeAt, screenToFlowPosition],
  );

  // =================================================================
  // Edge creation
  // =================================================================

  const onConnect: OnConnect = React.useCallback(
    (conn: Connection) => {
      setEdges((prev) =>
        addEdge(
          { ...conn, id: `e${nextIdRef.current++}`, type: "default" },
          prev,
        ),
      );
    },
    [setEdges],
  );

  // =================================================================
  // Execution
  // =================================================================

  const pushEvent = React.useCallback(
    (ev: ExecutionEvent) => {
      logRef.current = [...logRef.current, { id: logRef.current.length + 1, event: ev, time: Date.now() }];
      onEventLog(logRef.current);
    },
    [onEventLog],
  );

  const clearState = React.useCallback(() => {
    setNodes((prev) =>
      prev.map((n) => ({
        ...n,
        data: {
          ...n.data,
          executionState: "idle" as ExecutionState,
          result: undefined,
          error: undefined,
          progress: undefined,
        },
      })),
    );
    logRef.current = [];
    onEventLog([]);
  }, [setNodes, onEventLog]);

  const setNodeExecutionState = React.useCallback(
    (nodeId: string, patch: Partial<CustomNodeData>) => {
      setNodes((prev) =>
        prev.map((n) => (n.id === nodeId ? { ...n, data: { ...n.data, ...patch } } : n)),
      );
    },
    [setNodes],
  );

  const runFlow = React.useCallback(async () => {
    if (runningRef.current) return;
    if (nodes.length === 0) return;
    runningRef.current = true;
    clearState();

    const payload: { nodes: FlowNodePayload[]; edges: FlowEdgePayload[] } = {
      nodes: nodes.map((n) => ({
        id: n.id,
        type: n.data.nodeType.id,
        data: n.data.values,
        produces: Object.keys(n.data.produces).length ? n.data.produces : undefined,
        consumes: Object.keys(n.data.consumes).length ? n.data.consumes : undefined,
      })),
      edges: edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        source_handle: e.sourceHandle ?? null,
        target_handle: e.targetHandle ?? null,
      })),
    };

    try {
      const { executeStream } = await import("@/lib/api");
      for await (const ev of executeStream(payload.nodes, payload.edges)) {
        pushEvent(ev);
        switch (ev.type) {
          case "node_start":
            setNodeExecutionState(ev.node_id, { executionState: "running" });
            break;
          case "node_complete":
            setNodeExecutionState(ev.node_id, {
              executionState: "completed",
              result: ev.result as Record<string, unknown>,
            });
            break;
          case "node_skipped":
            setNodeExecutionState(ev.node_id, { executionState: "skipped" });
            break;
          case "node_error":
            setNodeExecutionState(ev.node_id, {
              executionState: "error",
              error: ev.error,
            });
            break;
          case "node_progress":
            setNodeExecutionState(ev.node_id, {
              progress: { current: ev.current, total: ev.total } as LoopProgress,
            });
            break;
          default:
            break;
        }
      }
    } catch (err) {
      pushEvent({
        type: "flow_error",
        error: err instanceof Error ? err.message : String(err),
        is_validation: false,
      });
    } finally {
      runningRef.current = false;
    }
  }, [nodes, edges, clearState, pushEvent, setNodeExecutionState, runningRef]);

  const clearAll = React.useCallback(() => {
    setNodes([]);
    setEdges([]);
    logRef.current = [];
    onEventLog([]);
    nextIdRef.current = 1;
  }, [setNodes, setEdges, onEventLog]);

  // =================================================================
  // Examples
  // =================================================================

  const loadExample = React.useCallback(
    (example: "basic" | "shared-refs" | "for-each") => {
      clearAll();
      // We use a microtask to let the clear settle before building.
      queueMicrotask(() => {
        const make = (typeId: string, x: number, y: number, values: Record<string, unknown> = {}) => {
          const nt = typeMap.get(typeId);
          if (!nt) return null;
          const id = `n${nextIdRef.current++}`;
          const defaults: Record<string, unknown> = {};
          nt.inputs.forEach((inp) => {
            if (inp.default !== null && inp.default !== undefined) defaults[inp.name] = inp.default;
          });
          const node: GraphNode = {
            id,
            type: "conductor",
            position: { x, y },
            dragHandle: ".drag-handle",
            data: {
              nodeType: nt,
              values: { ...defaults, ...values },
              produces: {},
              consumes: {},
              executionState: "idle",
              availableRefs: [],
              connectedInputs: new Set(),
              onValueChange,
              onTogglePublish,
              onSetConsume,
              onDelete,
            },
          };
          return node;
        };

        const link = (source: string, sourceHandle: string, target: string, targetHandle: string) => ({
          id: `e${nextIdRef.current++}`,
          source,
          sourceHandle,
          target,
          targetHandle,
          type: "default",
        });

        if (example === "basic") {
          const a = make("text@1", 60, 60, { value: "hello world" });
          const b = make("uppercase@1", 380, 60);
          const c = make("template@1", 700, 60, { template: "shouted: {input}" });
          if (!a || !b || !c) return;
          setNodes([a, b, c]);
          setEdges([link(a.id, "result", b.id, "text"), link(b.id, "result", c.id, "input")]);
          return;
        }

        if (example === "shared-refs") {
          // One "source" text; two consumers read it through a shared ref.
          const src = make("text@1", 60, 60, { value: "Alice met Bob at 42 Main St" });
          const r1 = make("regex@1", 380, 20, { pattern: "\\b[A-Z][a-z]+\\b" });
          const r2 = make("regex@1", 380, 200, { pattern: "\\d+" });
          if (!src || !r1 || !r2) return;
          src.data.produces = { result: "source text" };
          r1.data.consumes = { text: [src.id, "result"] };
          r2.data.consumes = { text: [src.id, "result"] };
          setNodes([src, r1, r2]);
          setEdges([]);
          return;
        }

        if (example === "for-each") {
          const lister = make("make-list@1", 60, 60, {
            text: "apple\nbanana\ncherry",
            separator: "\\n",
            trim: true,
          });
          const start = make("for-each-start@1", 380, 60);
          const upper = make("uppercase@1", 720, 60);
          const end = make("for-each-end@1", 1040, 60);
          if (!lister || !start || !upper || !end) return;
          setNodes([lister, start, upper, end]);
          setEdges([
            link(lister.id, "result", start.id, "items"),
            link(start.id, "output_1", upper.id, "text"),
            link(upper.id, "result", end.id, "item"),
          ]);
          return;
        }
      });
    },
    [clearAll, typeMap, setNodes, setEdges, onValueChange, onTogglePublish, onSetConsume, onDelete],
  );

  // Expose actions to the parent.
  React.useEffect(() => {
    runSignalRef.current = runFlow;
    clearSignalRef.current = clearAll;
    loadExampleSignalRef.current = loadExample;
  }, [runFlow, clearAll, loadExample, runSignalRef, clearSignalRef, loadExampleSignalRef]);

  return (
    <ReactFlow
      nodes={nodesWithCallbacks}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      onDrop={onDrop}
      onDragOver={onDragOver}
      nodeTypes={rfNodeTypes}
      fitView
      proOptions={{ hideAttribution: true }}
      defaultEdgeOptions={{ type: "default" }}
      deleteKeyCode={["Backspace", "Delete"]}
    >
      <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
      <MiniMap pannable zoomable className="!bg-card !border-border" />
      <Controls className="!bg-card !border-border" />
    </ReactFlow>
  );
}

export function FlowCanvas(props: Props) {
  return (
    <ReactFlowProvider>
      <FlowCanvasInner {...props} />
    </ReactFlowProvider>
  );
}
