import type { ExecutionEvent, FlowEdgePayload, FlowNodePayload, NodeType } from "./types";

export async function fetchNodeTypes(): Promise<NodeType[]> {
  const res = await fetch("/api/nodes");
  if (!res.ok) throw new Error(`GET /api/nodes failed: ${res.status}`);
  return res.json();
}

export async function* executeStream(
  nodes: FlowNodePayload[],
  edges: FlowEdgePayload[],
  signal?: AbortSignal,
): AsyncGenerator<ExecutionEvent> {
  const res = await fetch("/api/execute-stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ nodes, edges }),
    signal,
  });
  if (!res.ok) {
    let detail: string;
    try {
      detail = JSON.stringify(await res.json());
    } catch {
      detail = await res.text();
    }
    throw new Error(`POST /api/execute-stream failed (${res.status}): ${detail}`);
  }
  if (!res.body) throw new Error("execute-stream returned an empty body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const raw = line.slice(6).trim();
      if (!raw) continue;
      yield JSON.parse(raw) as ExecutionEvent;
    }
  }
}
